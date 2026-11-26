"""Standalone Trainer & Pure-BF16 Evaluator for Native V_nl Three-Phase Thinking + 8-Slot Block Diffusion.

Runs strictly apples-to-apples against Gate 2 arms (same Qwen3-4B-Base dense checkpoint, same 252 r=16 LoRA
+ FP32MasterAdamW, same 16,000 N^0.5 source-balanced math+logic problems in identical seed=2026 order, same
100 strictly unseen Val problems evaluated in pure native BF16 without 4-bit overhead or autocast).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "stage3_cloud"))

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.model import AVLPMLModel, AblationConfig
from avl_pmlv1.engine.nl_block_dllm import NL_MASK_ID, NLBlockDiffusionWrapper
from avl_pmlv1.engine.q_avl_pml import (
    build_five_group_optimizer,
    build_warmup_cosine_scheduler,
    merge_lora_delta_into_bf16_model,
    quantize_to_q_avl_pml,
)
from avl_pmlv1.vocab.nl_three_think import format_nl_three_think_example
from eval_gate2_suite import (
    gate2_step_bucket,
    grade_prediction,
    load_eval_problems,
    render_markdown_table,
    summarize_arm,
)
from train_stage3_warmup import (
    SourceBalancedSampler,
    load_jsonl_traces,
    resolve_tokenizer,
    select_stratified_val,
    step_bucket,
)


def prepare_nl_bd3_example(
    rec: Dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    mask_rng: random.Random,
    max_prompt_len: int = 256,
    max_target_len: int = 384,
    block_size: int = 8,
    counterfactual_post_prob: float = 0.20,
    is_val: bool = False,
) -> Dict[str, Any]:
    """Convert a Three-Phase NL example into exact 8-slot padded BD3-LM [Prompt | Y_clean | Y_noisy] tensors."""
    ex = format_nl_three_think_example(
        rec,
        tokenizer=tokenizer,
        device=device,
        rng=mask_rng,
        max_prompt_len=max_prompt_len,
        max_target_len=max_target_len,
        counterfactual_post_prob=counterfactual_post_prob,
        is_val=is_val,
    )
    p_len = ex["p_len"]
    full_ids = ex["full_ids"]  # [1, p_len + t_len]
    prompt_ids = full_ids[:, :p_len]
    raw_tgt = full_ids[:, p_len:]  # [1, t_len]
    t_len = int(raw_tgt.shape[1])

    eos_id = int(getattr(tokenizer, "eos_token_id", 151643) or 151643)
    rem = t_len % block_size
    pad_n = (block_size - rem) % block_size
    if pad_n > 0:
        pad_t = torch.full((1, pad_n), eos_id, dtype=torch.long, device=device)
        y_clean = torch.cat([raw_tgt, pad_t], dim=1)
    else:
        y_clean = raw_tgt
    s_len = int(y_clean.shape[1])

    # Eligible supervision range: [0..s_len) for clean traces; [post_rel..s_len) for 20% counterfactual [Post] traces!
    post_rel = max(0, ex["post_start"] - p_len)
    ans_rel = max(0, ex["ans_start"] - p_len)
    is_cf = bool(ex["is_counterfactual"])
    sup_start = post_rel if is_cf else 0

    y_noisy = y_clean.clone()
    loss_mask = torch.zeros_like(y_clean, dtype=torch.bool)
    beat_cycle = (1.0, 0.67, 0.33)
    num_blocks = s_len // block_size
    for b_i in range(num_blocks):
        b_s = b_i * block_size
        b_e = b_s + block_size
        if b_e <= sup_start:
            # Counterfactual corrupted [Formal] block: keep clean in Y_noisy & never supervise!
            continue
        if is_val:
            p_b = beat_cycle[b_i % len(beat_cycle)]
        else:
            # Exact Beat-0..2 BD3-LM trajectory coverage: 25% full-block 1.0 + 75% U[0.20, 0.95]
            p_b = 1.0 if mask_rng.random() < 0.25 else mask_rng.uniform(0.20, 0.95)
        for pos in range(max(b_s, sup_start), b_e):
            if p_b >= 0.999 or mask_rng.random() < p_b:
                y_noisy[0, pos] = NL_MASK_ID
                loss_mask[0, pos] = True

    if not loss_mask.any():
        pick = min(s_len - 1, max(sup_start, t_len - 1))
        y_noisy[0, pick] = NL_MASK_ID
        loss_mask[0, pick] = True

    ans_mask = torch.zeros_like(y_clean, dtype=torch.bool)
    if ans_rel < t_len:
        ans_mask[0, ans_rel:t_len] = True

    return {
        "prompt_ids": prompt_ids,
        "y_clean": y_clean,
        "y_noisy": y_noisy,
        "loss_mask": loss_mask,
        "ans_mask": ans_mask,
        "domain": ex["domain"],
        "num_steps": ex["num_steps"],
    }


@torch.no_grad()
def evaluate_bd3_val(
    wrapper: NLBlockDiffusionWrapper,
    val_subset: List[Dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    max_prompt_len: int = 256,
    max_target_len: int = 384,
    block_size: int = 8,
) -> Dict[str, Any]:
    was_train = wrapper.training
    wrapper.eval()
    val_rng = random.Random(777)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    tot_losses, ans_losses, tok_accs, ans_accs = [], [], [], []
    by_dom: Dict[str, Dict[str, List[float]]] = {}
    by_bkt: Dict[str, Dict[str, List[float]]] = {}

    for rec in val_subset:
        ex = prepare_nl_bd3_example(
            rec,
            tokenizer=tokenizer,
            device=device,
            mask_rng=val_rng,
            max_prompt_len=max_prompt_len,
            max_target_len=max_target_len,
            block_size=block_size,
            counterfactual_post_prob=0.0,
            is_val=True,
        )
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, st = wrapper.forward_bd3_train(
                ex["prompt_ids"], ex["y_clean"], ex["y_noisy"], ex["loss_mask"], block_size=block_size, ans_mask=ex["ans_mask"]
            )
        tot_losses.append(st["loss_total"])
        tok_accs.append(st["tok_acc"])
        if st["ans_acc"] > 0.0 or st["loss_ans"] > 0.0:
            ans_losses.append(st["loss_ans"])
            ans_accs.append(st["ans_acc"])
        d_r = by_dom.setdefault(ex["domain"], {"loss": [], "acc": []})
        d_r["loss"].append(st["loss_total"])
        d_r["acc"].append(st["tok_acc"])
        b_r = by_bkt.setdefault(step_bucket(ex["num_steps"]), {"loss": [], "acc": []})
        b_r["loss"].append(st["loss_total"])
        b_r["acc"].append(st["tok_acc"])

    if was_train:
        wrapper.train()

    def _m(xs: List[float]) -> float:
        return round(sum(xs) / max(1, len(xs)), 4)

    return {
        "n_val": len(val_subset),
        "val_loss_total": _m(tot_losses),
        "val_tok_acc": round(sum(tok_accs) / max(1, len(tok_accs)), 2),
        "val_loss_ans": _m(ans_losses) if ans_losses else 0.0,
        "val_ans_tok_acc": round(sum(ans_accs) / max(1, len(ans_accs)), 2) if ans_accs else 0.0,
        "by_domain": {
            d: {"n": len(v["loss"]), "loss": _m(v["loss"]), "tok_acc": round(sum(v["acc"]) / max(1, len(v["acc"])), 2)}
            for d, v in sorted(by_dom.items())
        },
        "by_step_bucket": {
            b: {"n": len(v["loss"]), "loss": _m(v["loss"]), "tok_acc": round(sum(v["acc"]) / max(1, len(v["acc"])), 2)}
            for b, v in sorted(by_bkt.items())
        },
    }


def save_wrapper_trainable(wrapper: NLBlockDiffusionWrapper, out_path: Path) -> None:
    sd_save: Dict[str, torch.Tensor] = {}
    for n, p in wrapper.model.named_parameters():
        if p.requires_grad and n != "nl_head.weight":
            sd_save[n] = p.detach().cpu().contiguous()
    sd_save["nl_mask_delta"] = wrapper.nl_mask_delta.detach().cpu().contiguous()
    save_file(sd_save, str(out_path))


def run_train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AVLPMLConfig.from_yaml(args.config)
    tokenizer = resolve_tokenizer(args.tokenizer_dir or None)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    arm_name = args.arm_name

    print(
        f"=== [Native V_nl Three-Think 8-Slot Block-dLLM Trainer | arm={arm_name} | steps={args.total_steps} x accum={args.grad_accum} | "
        f"block_size={args.block_size} | balance={args.balance_sources}] ===",
        flush=True,
    )

    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16 if device.type == "cuda" else torch.float32)
    base_model = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=True))
    torch.set_default_dtype(prev_dt)

    if Path(args.init_ckpt).exists():
        sd = load_file(args.init_ckpt, device="cpu")
        base_model.load_state_dict(sd, strict=False)
        base_model.nl_head.weight = base_model.nl_embed.weight
        del sd
        gc.collect()
        print(f"[Loaded Init Ckpt] {args.init_ckpt}", flush=True)

    # Freeze all symbolic-only modules so ONLY native Qwen3 GQA+FFN LoRA, Norms, & nl_mask_delta train!
    freeze_tokens = (
        "sym_embed_head.", "chunk_slot_embed.", "block_idx_embed.", "sym_final_norm.",
        "engrams.", "dir_head.", ".pml.", ".alpha_calm", ".film_proj.", ".film_tau.",
        "hyper_router.", "sc_proj.", "in_span_tag.", "alpha_span_tag", "copy_gate.",
    )
    for name, p in base_model.named_parameters():
        if any(tok in name for tok in freeze_tokens):
            p.requires_grad_(False)

    q_stats = quantize_to_q_avl_pml(base_model, lora_rank_gqa=16, lora_rank_dense=16, lora_rank_routed=0, group_size=64)
    wrapper = NLBlockDiffusionWrapper(base_model, mask_token_id=NL_MASK_ID)
    print(f"[Q-AVL-PML + NLBlockDiffusion Ready] {q_stats}", flush=True)

    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        sd_res = load_file(args.resume_ckpt, device="cpu")
        if "nl_mask_delta" in sd_res:
            wrapper.nl_mask_delta.data.copy_(sd_res.pop("nl_mask_delta").to(wrapper.nl_mask_delta.dtype))
        base_model.load_state_dict(sd_res, strict=False)
        base_model.nl_head.weight = base_model.nl_embed.weight
        del sd_res
        gc.collect()
        print(f"[Resumed Ckpt] {args.resume_ckpt} (start_step={args.start_step})", flush=True)

    base_model.gradient_checkpointing = True
    wrapper.to(device)

    opt, opt_summary = build_five_group_optimizer(wrapper, stage="B", is_lora_hybrid=True)
    sched = build_warmup_cosine_scheduler(opt, total_steps=args.total_steps, warmup_ratio=0.10)
    if args.start_step > 0:
        for _ in range(args.start_step):
            sched.step()
    print("[5-Group Optimizer]", json.dumps(opt_summary, indent=2), flush=True)

    allowed_doms = {d.strip().lower() for d in args.domains.split(",") if d.strip()}
    train_dataset = load_jsonl_traces(
        args.data, max_sym_len=args.max_target_len, allowed_domains=allowed_doms, require_answer=True, hydrate_all_refs=True
    )
    sampler = SourceBalancedSampler(train_dataset, mode=args.balance_sources)
    print(f"[Train Dataset Loaded] {len(train_dataset)} rows | balance={args.balance_sources}", flush=True)
    print("[Source Mix]", json.dumps(sampler.summary(), indent=2), flush=True)

    raw_val = load_jsonl_traces(
        args.val_data,
        max_sym_len=args.max_target_len,
        max_samples=max(args.val_samples * 5, 2000),
        allowed_domains=allowed_doms,
        require_answer=True,
        hydrate_all_refs=True,
    )
    val_subset = select_stratified_val(raw_val, n_target=args.val_samples, seed=args.seed)
    print(f"[Val Dataset Loaded] {len(val_subset)} stratified unseen Val rows", flush=True)

    sample_rng = random.Random(args.seed)
    mask_rng = random.Random(args.seed + 10007)
    if args.start_step > 0:
        for _ in range(args.start_step * args.grad_accum):
            _ = sampler.sample(sample_rng)
            _ = mask_rng.random()
    torch.manual_seed(args.seed + args.start_step)

    wrapper.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    history_train: List[Dict[str, Any]] = []
    history_val: List[Dict[str, Any]] = []
    best_val_loss = float(args.init_best_val)
    best_out = save_dir / f"stage3_{arm_name}_best_val.safetensors"
    t0 = time.perf_counter()

    for step in range(args.start_step + 1, args.total_steps + 1):
        opt.zero_grad(set_to_none=True)
        last_st: Dict[str, float] = {}
        for micro in range(args.grad_accum):
            rec = sampler.sample(sample_rng)
            ex = prepare_nl_bd3_example(
                rec,
                tokenizer=tokenizer,
                device=device,
                mask_rng=mask_rng,
                max_prompt_len=args.max_prompt_len,
                max_target_len=args.max_target_len,
                block_size=args.block_size,
                counterfactual_post_prob=0.20,
                is_val=False,
            )
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, st = wrapper.forward_bd3_train(
                    ex["prompt_ids"], ex["y_clean"], ex["y_noisy"], ex["loss_mask"], block_size=args.block_size, ans_mask=ex["ans_mask"]
                )
                (loss / float(args.grad_accum)).backward()
            if micro == args.grad_accum - 1:
                last_st = st

        if not math.isfinite(last_st.get("loss_total", float("nan"))):
            raise RuntimeError(f"Non-finite loss at step {step}")

        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        opt.step()
        sched.step()
        if device.type == "cuda" and step % 25 == 0:
            torch.cuda.empty_cache()

        if step == 1 or step % args.log_every == 0 or step == args.total_steps:
            vram = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
            resv = round(torch.cuda.memory_reserved(device) / (1024**3), 2) if device.type == "cuda" else 0.0
            row = {
                "step": step,
                "loss_total": round(last_st["loss_total"], 4),
                "tok_acc": last_st["tok_acc"],
                "loss_ans": last_st["loss_ans"],
                "ans_acc": last_st["ans_acc"],
                "peak_vram_gb": vram,
                "reserved_vram_gb": resv,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            history_train.append(row)
            print(
                f"  [{arm_name} | Step {step:05d}/{args.total_steps}] tot={row['loss_total']:.4f}({row['tok_acc']:.1f}%) "
                f"ans={row['loss_ans']:.4f}({row['ans_acc']:.1f}%) | vram={vram}GB(resv={resv}GB) | {row['elapsed_sec']}s",
                flush=True,
            )

        if val_subset and (step % args.eval_every == 0 or step == args.total_steps):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            t_v = time.perf_counter()
            val_res = evaluate_bd3_val(
                wrapper, val_subset, tokenizer, device,
                max_prompt_len=args.max_prompt_len, max_target_len=args.max_target_len, block_size=args.block_size,
            )
            val_res["step"] = step
            val_res["val_sec"] = round(time.perf_counter() - t_v, 2)
            history_val.append(val_res)
            print(
                f"  [VAL {arm_name} @ Step {step}] tot={val_res['val_loss_total']:.4f}({val_res['val_tok_acc']:.1f}%) "
                f"ans={val_res['val_loss_ans']:.4f}({val_res['val_ans_tok_acc']:.1f}%) | {val_res['val_sec']}s",
                flush=True,
            )
            if val_res["val_loss_total"] < best_val_loss:
                best_val_loss = val_res["val_loss_total"]
                save_wrapper_trainable(wrapper, best_out)
                print(f"  [New Best Val Ckpt] {best_out} (val_loss_total={best_val_loss:.4f})", flush=True)

        if step % args.save_every == 0 or step == args.total_steps:
            ck = save_dir / f"stage3_{arm_name}_q_avl_5080_step{step}.safetensors"
            save_wrapper_trainable(wrapper, ck)
            print(f"  [Saved Checkpoint] {ck}", flush=True)

    rep = {
        "arm_name": arm_name,
        "gen_mode": "nl_three_think_dllm",
        "block_size": args.block_size,
        "balance_sources": args.balance_sources,
        "total_steps": args.total_steps,
        "grad_accum": args.grad_accum,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0,
        "best_val_loss": round(best_val_loss, 4) if math.isfinite(best_val_loss) else None,
        "best_ckpt": str(best_out),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "train_history": history_train,
        "val_history": history_val,
    }
    rep_path = save_dir / f"{arm_name}_train_report.json"
    rep_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved Train Report] {rep_path}", flush=True)

    del wrapper, base_model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_out


@torch.no_grad()
def run_eval_pure_bf16(args: argparse.Namespace, ckpt_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AVLPMLConfig.from_yaml(args.config)
    tokenizer = resolve_tokenizer(args.tokenizer_dir or None)

    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16 if device.type == "cuda" else torch.float32)
    base_model = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=True))
    torch.set_default_dtype(prev_dt)

    sd_base = load_file(args.init_ckpt, device="cpu")
    base_model.load_state_dict(sd_base, strict=False)
    base_model.nl_head.weight = base_model.nl_embed.weight
    del sd_base
    gc.collect()

    wrapper = NLBlockDiffusionWrapper(base_model, mask_token_id=NL_MASK_ID)
    sd_arm = load_file(str(ckpt_path), device="cpu")
    if "nl_mask_delta" in sd_arm:
        wrapper.nl_mask_delta.data.copy_(sd_arm.pop("nl_mask_delta").to(wrapper.nl_mask_delta.dtype))
    m_info = merge_lora_delta_into_bf16_model(base_model, sd_arm, lora_scaling=2.0)
    del sd_arm
    gc.collect()
    wrapper.eval().to(device)
    print(f"[Pure-BF16 Merged NL-dLLM Ready] {m_info}", flush=True)

    suites = [
        (args.eval_file_40, 40, args.out_prefix + "_val40"),
        (args.eval_file_60, 60, args.out_prefix + "_val60"),
    ]
    allowed_doms = {d.strip().lower() for d in args.domains.split(",") if d.strip()}
    beats_list = [int(x.strip()) for x in args.eval_beats.split(",") if x.strip()]
    budget_list = [int(x.strip()) for x in args.eval_budgets.split(",") if x.strip()]

    for eval_file, n_samples, out_base in suites:
        problems = load_eval_problems(eval_file, max_samples=n_samples, seed=args.seed, allowed_domains=allowed_doms)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        arms_dict: Dict[str, Any] = {}
        for max_tok in budget_list:
            for beats in beats_list:
                tag = f"{args.arm_name}_b{beats}_t{max_tok}"
                t_a = time.perf_counter()
                recs: List[Dict[str, Any]] = []
                for prob in problems:
                    prompt_text = (
                        f"Problem ({prob['domain']}):\n{prob['question']}\n"
                        f"Use three-phase thinking ([Pre] plan, [Formal] derivation, [Post] verification) "
                        f"and end with 'Answer: <final answer>'.\n"
                    )
                    p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[: args.max_prompt_len]
                    p_t = torch.tensor([p_ids or [tokenizer.eos_token_id or 151643]], dtype=torch.long, device=device)
                    ans_text, n_toks, pre_ms, tot_ms = wrapper.generate_nl_block_dllm(
                        p_t, tokenizer, max_new_tokens=max_tok, block_size=args.block_size, denoise_beats=beats
                    )
                    gradable = bool(prob["ground_truth_answer"]) and prob["domain"].lower() in ("math", "logic")
                    ok = grade_prediction(ans_text, prob["ground_truth_answer"], prob["domain"]) if gradable else False
                    recs.append(
                        {
                            "id": prob["id"],
                            "domain": prob["domain"],
                            "source": prob["source"],
                            "step_bucket": gate2_step_bucket(prob["num_steps"]),
                            "gradable": gradable,
                            "correct": ok,
                            "draft_correct": ok,
                            "think_tokens": n_toks,
                            "prefill_ms": pre_ms,
                            "think_ms": round(max(0.0, tot_ms - pre_ms), 2),
                            "repair_ms": 0.0,
                            "verbalize_ms": 0.0,
                            "total_e2e_ms": tot_ms,
                            "repair_accepted": False,
                        }
                    )
                arms_dict[tag] = summarize_arm(recs, ref_nl_tokens=None)
                print(
                    f"  [Done {tag} | {Path(eval_file).name} n={len(problems)}] "
                    f"EM={arms_dict[tag]['exact_match_pct']:.2f}% | avg_toks={arms_dict[tag]['avg_think_tokens']:.1f} | "
                    f"avg_e2e={arms_dict[tag]['avg_total_e2e_ms']:.1f}ms | wall={time.perf_counter() - t_a:.1f}s",
                    flush=True,
                )

        peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
        report = {
            "eval_file": eval_file,
            "mode": "bf16_merged",
            "n_problems": len(problems),
            "peak_vram_gb": peak_vram,
            "arms": arms_dict,
        }
        out_json = Path(out_base + ".json")
        out_md = Path(out_base + ".md")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        out_md.write_text(render_markdown_table(report), encoding="utf-8")
        print(f"[Saved Eval] {out_json} & {out_md} (peak_vram={peak_vram} GB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--arm-name", type=str, default="nl_three_think_dllm")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "model_4b_dense_pml.yaml"))
    parser.add_argument("--init-ckpt", type=str, default=str(ROOT / "checkpoints" / "avl_pmlv12_4b_dense_step0_bf16.safetensors"))
    parser.add_argument("--tokenizer-dir", type=str, default="")
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "stage1_mega" / "mega_train_fsq.jsonl"))
    parser.add_argument("--val-data", type=str, default=str(ROOT / "data" / "stage1_mega" / "mega_val_fsq.jsonl"))
    parser.add_argument("--eval-file-40", type=str, default=str(ROOT / "data" / "stage1_mega" / "mega_val_gsm8k_pw_kk_40.jsonl"))
    parser.add_argument("--eval-file-60", type=str, default=str(ROOT / "data" / "stage1_mega" / "mega_val_fsq.jsonl"))
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "checkpoints" / "gate2_dense_2000"))
    parser.add_argument("--out-prefix", type=str, default=str(ROOT / "outputs" / "gate2_dense_2000_nl_three_think_dllm"))
    parser.add_argument("--eval-ckpt", type=str, default="")
    parser.add_argument("--domains", type=str, default="math,logic")
    parser.add_argument("--balance-sources", choices=["none", "sqrt"], default="sqrt")
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=384)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=120)
    parser.add_argument("--eval-beats", type=str, default="3,4", help="Comma-separated denoise beat counts to benchmark on same ckpt.")
    parser.add_argument("--eval-budgets", type=str, default="256,384", help="Comma-separated max_new_tokens budgets (256 fair match + 384 speedup match).")
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--init-best-val", type=float, default=float("inf"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ckpt = Path(args.eval_ckpt) if args.eval_ckpt else (Path(args.save_dir) / f"stage3_{args.arm_name}_best_val.safetensors")
    if args.action in ("train", "both"):
        ckpt = run_train(args)
    if args.action in ("eval", "both"):
        run_eval_pure_bf16(args, ckpt)


if __name__ == "__main__":
    main()
