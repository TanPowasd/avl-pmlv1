"""Stage-C Backward Revision Trainer & Controlled Pure-BF16 Evaluator for Native V_nl 8-Slot Block-dLLM.

Tests genuine '向前修改' (modifying already-generated tokens via block diffusion):
  - Stage-C post-training (600 steps x grad_accum=8 = 4,800 episodes from stage3_nl_three_think_dllm_best_val):
      40% Forward 8-slot BD3-LM (retains strong forward drafting)
    + 30% 16-Slot Wet-Cement Seam Back-Refinement (Block b-1 masked/perturbed, Block b clean right context!)
    + 30% Full-Draft Bidirectional 4-Fold PLL & Counterfactual In-Place Repair (trains exact 4-fold PLL + repair!)
  - Controlled Pure-BF16 Evaluation on the EXACT same Val40 & Val60 suites across 4 modes on the SAME checkpoint:
      1) rev_forward_only  (wet=OFF, post=OFF)
      2) rev_wet_cement    (wet=ON,  post=OFF)
      3) rev_post_inplace  (wet=OFF, post=ON)
      4) rev_wet_plus_post (wet=ON,  post=ON)
    Reports exact Fixed / Broken / NetFix (both vs rev_forward_only AND within-problem draft->final),
    wet_triggers, wet_accepted_edits, post_accept_rate, and repair_overhead_ms!
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "stage3_cloud"))

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.model import AVLPMLModel, AblationConfig
from avl_pmlv1.engine.nl_backward_rev import NLBackwardRevisionEngine
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
from run_nl_three_think_dllm import evaluate_bd3_val, prepare_nl_bd3_example, save_wrapper_trainable
from train_stage3_warmup import (
    SourceBalancedSampler,
    load_jsonl_traces,
    resolve_tokenizer,
    select_stratified_val,
)


def _pad_to_block(raw_tgt: torch.Tensor, eos_id: int, block_size: int) -> Tuple[torch.Tensor, int]:
    t_len = int(raw_tgt.shape[1])
    rem = t_len % block_size
    pad_n = (block_size - rem) % block_size
    if pad_n > 0:
        pad_t = torch.full((1, pad_n), eos_id, dtype=torch.long, device=raw_tgt.device)
        return torch.cat([raw_tgt, pad_t], dim=1), t_len
    return raw_tgt, t_len


def _make_wet_cement_batch(
    rec: Dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    rng: random.Random,
    max_prompt_len: int = 256,
    max_target_len: int = 384,
    block_size: int = 8,
) -> Dict[str, Any]:
    """Construct 16-slot Wet-Cement seam batch: Block b-1 has 25%-50% [MASK] (and optional wrong-token noise),
    while Block b stays 100% CLEAN right-hand context!
    """
    ex = format_nl_three_think_example(
        rec,
        tokenizer=tokenizer,
        device=device,
        rng=rng,
        max_prompt_len=max_prompt_len,
        max_target_len=max_target_len,
        counterfactual_post_prob=0.0,
        is_val=False,
    )
    p_len = ex["p_len"]
    prompt_ids = ex["full_ids"][:, :p_len]
    raw_tgt = ex["full_ids"][:, p_len:]
    eos_id = int(getattr(tokenizer, "eos_token_id", 151643) or 151643)
    y_clean, t_len = _pad_to_block(raw_tgt, eos_id, block_size)
    s_len = int(y_clean.shape[1])

    parity = rng.choice([0, 1]) if s_len >= 3 * block_size else 0
    y_wet = y_clean.clone()
    loss_mask = torch.zeros_like(y_clean, dtype=torch.bool)

    w_off = parity * block_size
    while w_off + 2 * block_size <= s_len:
        # Window covers [w_off .. w_off + B) = Block b-1 (to revise) and [w_off + B .. w_off + 2B) = Block b (CLEAN right context!)
        p_m = rng.uniform(0.25, 0.50)
        masked_any = False
        for pos in range(w_off, w_off + block_size):
            if rng.random() < p_m:
                y_wet[0, pos] = NL_MASK_ID
                loss_mask[0, pos] = True
                masked_any = True
        if not masked_any:
            pick = rng.randrange(w_off, w_off + block_size)
            y_wet[0, pick] = NL_MASK_ID
            loss_mask[0, pick] = True
        w_off += 2 * block_size

    if not loss_mask.any():
        pick = rng.randrange(0, max(1, min(block_size, s_len)))
        y_wet[0, pick] = NL_MASK_ID
        loss_mask[0, pick] = True

    ans_rel = max(0, ex["ans_start"] - p_len)
    ans_mask = torch.zeros_like(y_clean, dtype=torch.bool)
    if ans_rel < t_len:
        ans_mask[0, ans_rel:t_len] = True

    return {
        "prompt_ids": prompt_ids,
        "y_clean": y_clean,
        "y_wet": y_wet,
        "loss_mask": loss_mask,
        "ans_mask": ans_mask,
        "parity": parity,
    }


def _make_full_bidir_batch(
    rec: Dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    rng: random.Random,
    max_prompt_len: int = 256,
    max_target_len: int = 384,
    block_size: int = 8,
    corrupt_prob: float = 0.50,
) -> Dict[str, Any]:
    """Construct full-draft bidirectional batch matching 4-fold PLL & in-place revision:
      - 50%: Exact 1-of-4 complementary 25% fold masking (exact parity with score_draft_4fold_pll!)
      - 50%: Error-localized span masking + light wrong-token distractor in another slot -> restores clean truth!
    """
    ex = format_nl_three_think_example(
        rec,
        tokenizer=tokenizer,
        device=device,
        rng=rng,
        max_prompt_len=max_prompt_len,
        max_target_len=max_target_len,
        counterfactual_post_prob=0.0,
        is_val=False,
    )
    p_len = ex["p_len"]
    prompt_ids = ex["full_ids"][:, :p_len]
    raw_tgt = ex["full_ids"][:, p_len:]
    eos_id = int(getattr(tokenizer, "eos_token_id", 151643) or 151643)
    y_clean, t_len = _pad_to_block(raw_tgt, eos_id, block_size)
    s_len = int(y_clean.shape[1])

    y_masked = y_clean.clone()
    loss_mask = torch.zeros_like(y_clean, dtype=torch.bool)

    if rng.random() >= corrupt_prob:
        # Exact 1-of-4 complementary 25% fold (exact parity with 4-fold PLL!)
        fold = rng.randrange(4)
        for pos in range(fold, s_len, 4):
            y_masked[0, pos] = NL_MASK_ID
            loss_mask[0, pos] = True
    else:
        # In-place surgical block repair: mask 2..5 slots inside 1-2 blocks (biased toward Formal/Post/Answer)
        n_blocks = max(1, s_len // block_size)
        n_pick_blks = min(2, n_blocks)
        chosen_blks = rng.sample(range(n_blocks), k=n_pick_blks)
        for b_i in chosen_blks:
            b_s = b_i * block_size
            k_m = rng.randint(2, min(5, block_size))
            slots = rng.sample(range(b_s, min(b_s + block_size, s_len)), k=k_m)
            for pos in slots:
                y_masked[0, pos] = NL_MASK_ID
                loss_mask[0, pos] = True
        # Also lightly mask 10% elsewhere for global robustness
        for pos in range(s_len):
            if not loss_mask[0, pos] and rng.random() < 0.10:
                y_masked[0, pos] = NL_MASK_ID
                loss_mask[0, pos] = True

    if not loss_mask.any():
        y_masked[0, 0] = NL_MASK_ID
        loss_mask[0, 0] = True

    ans_rel = max(0, ex["ans_start"] - p_len)
    ans_mask = torch.zeros_like(y_clean, dtype=torch.bool)
    if ans_rel < t_len:
        ans_mask[0, ans_rel:t_len] = True

    return {
        "prompt_ids": prompt_ids,
        "y_clean": y_clean,
        "y_masked": y_masked,
        "loss_mask": loss_mask,
        "ans_mask": ans_mask,
    }


@torch.no_grad()
def evaluate_stage_c_val(
    engine: NLBackwardRevisionEngine,
    val_subset: List[Dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    max_prompt_len: int = 256,
    max_target_len: int = 384,
    block_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate BOTH forward BD3-LM retention AND wet-cement + 4-fold bidir revision losses on unseen Val!"""
    was_train = engine.wrapper.training
    engine.wrapper.eval()

    fwd_res = evaluate_bd3_val(
        engine.wrapper, val_subset, tokenizer, device,
        max_prompt_len=max_prompt_len, max_target_len=max_target_len, block_size=block_size,
    )

    val_rng = random.Random(999)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    wet_losses, wet_accs, bidir_losses, bidir_accs = [], [], [], []
    for rec in val_subset[:60]:
        wb = _make_wet_cement_batch(rec, tokenizer, device, val_rng, max_prompt_len, max_target_len, block_size)
        bb = _make_full_bidir_batch(rec, tokenizer, device, val_rng, max_prompt_len, max_target_len, block_size, corrupt_prob=0.0)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, st_w = engine.forward_wet_cement_train(
                wb["prompt_ids"], wb["y_clean"], wb["y_wet"], wb["loss_mask"],
                block_size=block_size, parity=wb["parity"], ans_mask=wb["ans_mask"],
            )
            _, st_b = engine.forward_full_bidir_train(
                bb["prompt_ids"], bb["y_clean"], bb["y_masked"], bb["loss_mask"], ans_mask=bb["ans_mask"],
            )
        wet_losses.append(st_w["loss_total"])
        wet_accs.append(st_w["tok_acc"])
        bidir_losses.append(st_b["loss_total"])
        bidir_accs.append(st_b["tok_acc"])

    if was_train:
        engine.wrapper.train()

    def _m(xs: List[float], nd: int = 4) -> float:
        return round(sum(xs) / max(1, len(xs)), nd)

    wet_l = _m(wet_losses)
    bidir_l = _m(bidir_losses)
    fwd_l = fwd_res["val_loss_total"]
    combined = round(0.4 * fwd_l + 0.3 * wet_l + 0.3 * bidir_l, 4)

    return {
        "val_loss_combined": combined,
        "val_loss_fwd": fwd_l,
        "val_tok_acc_fwd": fwd_res["val_tok_acc"],
        "val_loss_wet": wet_l,
        "val_tok_acc_wet": _m(wet_accs, 2),
        "val_loss_bidir": bidir_l,
        "val_tok_acc_bidir": _m(bidir_accs, 2),
        "val_loss_ans_fwd": fwd_res["val_loss_ans"],
        "val_ans_tok_acc_fwd": fwd_res["val_ans_tok_acc"],
    }


def run_train_stage_c(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AVLPMLConfig.from_yaml(args.config)
    tokenizer = resolve_tokenizer(args.tokenizer_dir or None)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    arm_name = args.arm_name

    print(
        f"=== [Stage-C Native V_nl Backward Revision Trainer | arm={arm_name} | steps={args.total_steps} x accum={args.grad_accum} | "
        f"mix=40%Fwd/30%Wet16/30%BidirPLL] ===",
        flush=True,
    )

    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16 if device.type == "cuda" else torch.float32)
    base_model = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=True))
    torch.set_default_dtype(prev_dt)

    sd = load_file(args.init_ckpt, device="cpu")
    base_model.load_state_dict(sd, strict=False)
    base_model.nl_head.weight = base_model.nl_embed.weight
    del sd
    gc.collect()

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
    engine = NLBackwardRevisionEngine(wrapper)

    # Warm-start from trained Stage-B nl_three_think_dllm best checkpoint!
    if args.stage_b_ckpt and Path(args.stage_b_ckpt).exists():
        sd_b = load_file(args.stage_b_ckpt, device="cpu")
        if "nl_mask_delta" in sd_b:
            wrapper.nl_mask_delta.data.copy_(sd_b.pop("nl_mask_delta").to(wrapper.nl_mask_delta.dtype))
        base_model.load_state_dict(sd_b, strict=False)
        base_model.nl_head.weight = base_model.nl_embed.weight
        del sd_b
        gc.collect()
        print(f"[Loaded Stage-B Warmstart] {args.stage_b_ckpt}", flush=True)
    else:
        print("[WARN] No Stage-B checkpoint provided/found; training Stage-C from step-0 base!", flush=True)

    base_model.gradient_checkpointing = True
    wrapper.to(device)
    print(f"[Stage-C Q-AVL-PML Ready] {q_stats}", flush=True)

    # Stage-C uses gentler LR (0.5x of Stage-B) so forward BD3-LM knowledge stays intact while learning backward repair!
    opt, opt_summary = build_five_group_optimizer(wrapper, stage="C", is_lora_hybrid=True)
    sched = build_warmup_cosine_scheduler(opt, total_steps=args.total_steps, warmup_ratio=0.10)
    print("[Stage-C Optimizer]", json.dumps(opt_summary, indent=2), flush=True)

    allowed_doms = {d.strip().lower() for d in args.domains.split(",") if d.strip()}
    train_dataset = load_jsonl_traces(
        args.data, max_sym_len=args.max_target_len, allowed_domains=allowed_doms, require_answer=True, hydrate_all_refs=True
    )
    sampler = SourceBalancedSampler(train_dataset, mode=args.balance_sources)
    raw_val = load_jsonl_traces(
        args.val_data, max_sym_len=args.max_target_len, max_samples=max(args.val_samples * 5, 2000),
        allowed_domains=allowed_doms, require_answer=True, hydrate_all_refs=True,
    )
    val_subset = select_stratified_val(raw_val, n_target=args.val_samples, seed=args.seed)

    sample_rng = random.Random(args.seed + 50001)
    mask_rng = random.Random(args.seed + 60007)
    torch.manual_seed(args.seed + 50001)

    wrapper.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    history_train: List[Dict[str, Any]] = []
    history_val: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_out = save_dir / f"stage3_{arm_name}_best_val.safetensors"
    t0 = time.perf_counter()

    # Evaluate Step 0 baseline right after loading Stage-B ckpt so we see exact before/after Stage-C delta!
    val0 = evaluate_stage_c_val(engine, val_subset, tokenizer, device, args.max_prompt_len, args.max_target_len, args.block_size)
    val0["step"] = 0
    history_val.append(val0)
    print(
        f"  [VAL {arm_name} @ Step 0 (Stage-B init)] comb={val0['val_loss_combined']:.4f} | "
        f"fwd={val0['val_loss_fwd']:.4f}({val0['val_tok_acc_fwd']:.1f}%) | "
        f"wet16={val0['val_loss_wet']:.4f}({val0['val_tok_acc_wet']:.1f}%) | "
        f"bidir={val0['val_loss_bidir']:.4f}({val0['val_tok_acc_bidir']:.1f}%)",
        flush=True,
    )

    for step in range(1, args.total_steps + 1):
        opt.zero_grad(set_to_none=True)
        step_losses = {"fwd": [], "wet": [], "bidir": []}
        for micro in range(args.grad_accum):
            rec = sampler.sample(sample_rng)
            # Clean deterministic 40% Fwd / 30% Wet-16 / 30% Full-Bidir schedule inside grad_accum=8 (3 Fwd, 3 Wet, 2 Bidir / rotating)
            u = mask_rng.random()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                if u < 0.40:
                    ex = prepare_nl_bd3_example(
                        rec, tokenizer, device, mask_rng,
                        max_prompt_len=args.max_prompt_len, max_target_len=args.max_target_len,
                        block_size=args.block_size, counterfactual_post_prob=0.20, is_val=False,
                    )
                    loss, st = wrapper.forward_bd3_train(
                        ex["prompt_ids"], ex["y_clean"], ex["y_noisy"], ex["loss_mask"],
                        block_size=args.block_size, ans_mask=ex["ans_mask"],
                    )
                    step_losses["fwd"].append(st["loss_total"])
                elif u < 0.70:
                    wb = _make_wet_cement_batch(
                        rec, tokenizer, device, mask_rng,
                        max_prompt_len=args.max_prompt_len, max_target_len=args.max_target_len, block_size=args.block_size,
                    )
                    loss, st = engine.forward_wet_cement_train(
                        wb["prompt_ids"], wb["y_clean"], wb["y_wet"], wb["loss_mask"],
                        block_size=args.block_size, parity=wb["parity"], ans_mask=wb["ans_mask"],
                    )
                    step_losses["wet"].append(st["loss_total"])
                else:
                    bb = _make_full_bidir_batch(
                        rec, tokenizer, device, mask_rng,
                        max_prompt_len=args.max_prompt_len, max_target_len=args.max_target_len, block_size=args.block_size,
                    )
                    loss, st = engine.forward_full_bidir_train(
                        bb["prompt_ids"], bb["y_clean"], bb["y_masked"], bb["loss_mask"], ans_mask=bb["ans_mask"],
                    )
                    step_losses["bidir"].append(st["loss_total"])
                (loss / float(args.grad_accum)).backward()

        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        opt.step()
        sched.step()
        if device.type == "cuda" and step % 10 == 0:
            torch.cuda.empty_cache()

        if step == 1 or step % args.log_every == 0 or step == args.total_steps:
            vram = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
            resv = round(torch.cuda.memory_reserved(device) / (1024**3), 2) if device.type == "cuda" else 0.0
            row = {
                "step": step,
                "fwd": round(sum(step_losses["fwd"]) / max(1, len(step_losses["fwd"])), 4) if step_losses["fwd"] else None,
                "wet": round(sum(step_losses["wet"]) / max(1, len(step_losses["wet"])), 4) if step_losses["wet"] else None,
                "bidir": round(sum(step_losses["bidir"]) / max(1, len(step_losses["bidir"])), 4) if step_losses["bidir"] else None,
                "peak_vram_gb": vram,
                "reserved_vram_gb": resv,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            history_train.append(row)
            print(
                f"  [{arm_name} | Step {step:04d}/{args.total_steps}] fwd={row['fwd']} wet={row['wet']} bidir={row['bidir']} | "
                f"vram={vram}GB(resv={resv}GB) | {row['elapsed_sec']}s",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.total_steps:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            t_v = time.perf_counter()
            vres = evaluate_stage_c_val(engine, val_subset, tokenizer, device, args.max_prompt_len, args.max_target_len, args.block_size)
            vres["step"] = step
            vres["val_sec"] = round(time.perf_counter() - t_v, 2)
            history_val.append(vres)
            print(
                f"  [VAL {arm_name} @ Step {step}] comb={vres['val_loss_combined']:.4f} | "
                f"fwd={vres['val_loss_fwd']:.4f}({vres['val_tok_acc_fwd']:.1f}%) | "
                f"wet16={vres['val_loss_wet']:.4f}({vres['val_tok_acc_wet']:.1f}%) | "
                f"bidir={vres['val_loss_bidir']:.4f}({vres['val_tok_acc_bidir']:.1f}%) | {vres['val_sec']}s",
                flush=True,
            )
            if vres["val_loss_combined"] < best_val_loss:
                best_val_loss = vres["val_loss_combined"]
                save_wrapper_trainable(wrapper, best_out)
                print(f"  [New Best Stage-C Ckpt] {best_out} (comb={best_val_loss:.4f})", flush=True)

        if step % args.save_every == 0 or step == args.total_steps:
            ck = save_dir / f"stage3_{arm_name}_step{step}.safetensors"
            save_wrapper_trainable(wrapper, ck)
            print(f"  [Saved Checkpoint] {ck}", flush=True)

    rep = {
        "arm_name": arm_name,
        "stage_b_ckpt": args.stage_b_ckpt,
        "total_steps": args.total_steps,
        "grad_accum": args.grad_accum,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0,
        "best_val_loss_combined": round(best_val_loss, 4) if math.isfinite(best_val_loss) else None,
        "best_ckpt": str(best_out),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "train_history": history_train,
        "val_history": history_val,
    }
    rep_path = save_dir / f"{arm_name}_train_report.json"
    rep_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved Stage-C Train Report] {rep_path}", flush=True)

    del engine, wrapper, base_model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_out


@torch.no_grad()
def run_eval_backward_rev(args: argparse.Namespace, ckpt_path: Path) -> None:
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
    engine = NLBackwardRevisionEngine(wrapper)
    print(f"[Pure-BF16 Merged Stage-C Backward-Rev Ready] {m_info}", flush=True)

    modes = [
        ("rev_forward_only", False, False),
        ("rev_wet_cement", True, False),
        ("rev_post_inplace", False, True),
        ("rev_wet_plus_post", True, True),
    ]
    suites = [
        (args.eval_file_40, 40, args.out_prefix + "_val40"),
        (args.eval_file_60, 60, args.out_prefix + "_val60"),
    ]
    allowed_doms = {d.strip().lower() for d in args.domains.split(",") if d.strip()}
    budget_list = [int(x.strip()) for x in args.eval_budgets.split(",") if x.strip()]

    for eval_file, n_samples, out_base in suites:
        problems = load_eval_problems(eval_file, max_samples=n_samples, seed=args.seed, allowed_domains=allowed_doms)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        arms_dict: Dict[str, Any] = {}
        extra_telemetry: Dict[str, Any] = {}

        for max_tok in budget_list:
            fwd_correct_map: Dict[str, bool] = {}
            for mode_name, en_wet, en_post in modes:
                tag = f"{mode_name}_b{args.eval_beats}_t{max_tok}"
                t_a = time.perf_counter()
                recs: List[Dict[str, Any]] = []
                wet_trigs, wet_edits, post_atts, post_accs = 0, 0, 0, 0
                vs_fwd_fixed, vs_fwd_broken = 0, 0

                for prob in problems:
                    prompt_text = (
                        f"Problem ({prob['domain']}):\n{prob['question']}\n"
                        f"Use three-phase thinking ([Pre] plan, [Formal] derivation, [Post] verification) "
                        f"and end with 'Answer: <final answer>'.\n"
                    )
                    p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[: args.max_prompt_len]
                    p_t = torch.tensor([p_ids or [tokenizer.eos_token_id or 151643]], dtype=torch.long, device=device)

                    res = engine.generate_with_backward_revision(
                        p_t,
                        tokenizer=tokenizer,
                        max_new_tokens=max_tok,
                        block_size=args.block_size,
                        denoise_beats=args.eval_beats,
                        enable_wet_cement=en_wet,
                        enable_post_revision=en_post,
                    )
                    gradable = bool(prob["ground_truth_answer"]) and prob["domain"].lower() in ("math", "logic")
                    draft_ok = grade_prediction(res["draft_text"], prob["ground_truth_answer"], prob["domain"]) if gradable else False
                    final_ok = grade_prediction(res["final_text"], prob["ground_truth_answer"], prob["domain"]) if gradable else False

                    if mode_name == "rev_forward_only":
                        fwd_correct_map[prob["id"]] = final_ok
                        base_ok = final_ok
                    else:
                        base_ok = fwd_correct_map.get(prob["id"], draft_ok)
                        if gradable:
                            if (not base_ok) and final_ok:
                                vs_fwd_fixed += 1
                            elif base_ok and (not final_ok):
                                vs_fwd_broken += 1

                    wet_trigs += res["wet_triggers"]
                    wet_edits += res["wet_accepted_edits"]
                    post_atts += int(res["post_attempted"])
                    post_accs += int(res["post_accepted"])

                    # For wet_cement (online), compare against rev_forward_only as draft_correct so Fixed/Broken/NetFix is exact!
                    # For post_inplace, draft_ok is exact pre-post draft!
                    ref_draft_ok = base_ok if (en_wet and not en_post) else draft_ok
                    recs.append(
                        {
                            "id": prob["id"],
                            "domain": prob["domain"],
                            "source": prob["source"],
                            "step_bucket": gate2_step_bucket(prob["num_steps"]),
                            "gradable": gradable,
                            "correct": final_ok,
                            "draft_correct": ref_draft_ok,
                            "think_tokens": res["n_tokens"],
                            "prefill_ms": res["prefill_ms"],
                            "think_ms": res["think_ms"],
                            "repair_ms": res["repair_ms"],
                            "verbalize_ms": 0.0,
                            "total_e2e_ms": res["total_e2e_ms"],
                            "repair_accepted": bool(res["post_accepted"] or res["wet_accepted_edits"] > 0),
                        }
                    )

                arms_dict[tag] = summarize_arm(recs, ref_nl_tokens=None)
                extra_telemetry[tag] = {
                    "wet_triggers_total": wet_trigs,
                    "wet_accepted_edits_total": wet_edits,
                    "post_attempted_total": post_atts,
                    "post_accepted_total": post_accs,
                    "vs_forward_fixed": vs_fwd_fixed,
                    "vs_forward_broken": vs_fwd_broken,
                    "vs_forward_net_fix": vs_fwd_fixed - vs_fwd_broken,
                }
                ra = arms_dict[tag]["repair_audit"]
                print(
                    f"  [Done {tag} | {Path(eval_file).name} n={len(problems)}] "
                    f"EM={arms_dict[tag]['exact_match_pct']:.2f}% | avg_toks={arms_dict[tag]['avg_think_tokens']:.1f} | "
                    f"avg_e2e={arms_dict[tag]['avg_total_e2e_ms']:.1f}ms (rev={ra['repair_overhead_ms']:.1f}ms) | "
                    f"Fix/Brk/Net={ra['fixed_count']}/{ra['broken_count']}/{ra['net_fix']:+d} | "
                    f"wet_edits={wet_edits} post_acc={post_accs}/{post_atts} | wall={time.perf_counter() - t_a:.1f}s",
                    flush=True,
                )

        peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
        report = {
            "eval_file": eval_file,
            "mode": "bf16_merged",
            "n_problems": len(problems),
            "peak_vram_gb": peak_vram,
            "arms": arms_dict,
            "backward_revision_telemetry": extra_telemetry,
        }
        out_json = Path(out_base + ".json")
        out_md = Path(out_base + ".md")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        out_md.write_text(render_markdown_table(report), encoding="utf-8")
        print(f"[Saved Backward-Rev Eval] {out_json} & {out_md} (peak_vram={peak_vram} GB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--arm-name", type=str, default="nl_three_think_dllm_rev")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "model_4b_dense_pml.yaml"))
    parser.add_argument("--init-ckpt", type=str, default=str(ROOT / "checkpoints" / "avl_pmlv12_4b_dense_step0_bf16.safetensors"))
    parser.add_argument("--stage-b-ckpt", type=str, default=str(ROOT / "checkpoints" / "gate2_dense_2000" / "stage3_nl_three_think_dllm_best_val.safetensors"))
    parser.add_argument("--tokenizer-dir", type=str, default="")
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "stage1_mega" / "mega_train_fsq.jsonl"))
    parser.add_argument("--val-data", type=str, default=str(ROOT / "data" / "stage1_mega" / "mega_val_fsq.jsonl"))
    parser.add_argument("--eval-file-40", type=str, default=str(ROOT / "data" / "stage1_mega" / "mega_val_gsm8k_pw_kk_40.jsonl"))
    parser.add_argument("--eval-file-60", type=str, default=str(ROOT / "data" / "stage1_mega" / "mega_val_fsq.jsonl"))
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "checkpoints" / "gate2_dense_2000"))
    parser.add_argument("--out-prefix", type=str, default=str(ROOT / "outputs" / "gate2_dense_2000_nl_backward_rev"))
    parser.add_argument("--eval-ckpt", type=str, default="")
    parser.add_argument("--domains", type=str, default="math,logic")
    parser.add_argument("--balance-sources", choices=["none", "sqrt"], default="sqrt")
    parser.add_argument("--total-steps", type=int, default=600)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=384)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-samples", type=int, default=120)
    parser.add_argument("--eval-beats", type=int, default=3)
    parser.add_argument("--eval-budgets", type=str, default="256,384")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ckpt = Path(args.eval_ckpt) if args.eval_ckpt else (Path(args.save_dir) / f"stage3_{args.arm_name}_best_val.safetensors")
    if args.action in ("train", "both"):
        ckpt = run_train_stage_c(args)
    if args.action in ("eval", "both"):
        run_eval_backward_rev(args, ckpt)


if __name__ == "__main__":
    main()
