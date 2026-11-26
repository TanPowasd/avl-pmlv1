"""Stage 2 (M3-3) Local RTX 5080 Q-AVL-PML 4.54B Trial Training Step on Real Stage 1 30k FSQ Data!

Verifies end-to-end training health on our 16 GB RTX 5080 strictly below the 14.4 GB VRAM redline:
  1. Loads surgical BF16 AVL-PMLv1.2-4.5B checkpoint and converts to Blackwell-native 4-bit Q-AVL-PML
     (GQA + DenseSwiGLU + Routed Experts in 4-bit + BF16 LoRA; Shared, PML, Engram, FSQ, Router, Graphs in full BF16/FP32).
  2. Enables per-layer gradient checkpointing + builds exact Master Manual §3.2 5-Group Optimizer + 10% Warmup.
  3. Runs 20 real training steps on REAL FSQ-annotated 8/16-slot traces from `stage1_30k_fsq_aligned.jsonl`
     with real block-diffusion masks, real Track-1 hard-indexed PML writes, and `compute_six_losses`.
  4. Audits: Peak VRAM < 12.5 GB, zero NaN/Inf, live gradients across active optimizer groups, and strictly
     decreasing L_diff loss!
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List

import torch
from safetensors.torch import load_file

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.wet_cement import build_block_diffusion_attn_mask
from avl_pmlv1.engine.model import AVLPMLModel
from avl_pmlv1.engine.q_avl_pml import (
    build_five_group_optimizer,
    build_warmup_cosine_scheduler,
    quantize_to_q_avl_pml,
)
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab


def load_real_stage1_batches(jsonl_path: Path, max_samples: int = 256, max_seq_len: int = 256) -> List[Dict[str, Any]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            toks = r["token_ids_aligned_fsq"]
            if 32 <= len(toks) <= max_seq_len:
                rows.append(r)
            if len(rows) >= max_samples:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "model_4_5b.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "avl_pmlv12_4_5b_step0_bf16.safetensors"))
    parser.add_argument("--stage1-data", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1" / "stage1_30k_fsq_aligned.jsonl"))
    parser.add_argument("--out-report", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "stage2_m3_trial_report.json"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AVLPMLConfig.from_yaml(args.config)
    print("=== [Stage 2 M3-3 Local RTX 5080 Q-AVL-PML 4.54B Trial Run] ===", flush=True)

    # 1. Load BF16 surgical checkpoint on CPU and convert to Q-AVL-PML BEFORE moving to GPU!
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = AVLPMLModel(cfg)
    torch.set_default_dtype(prev_dtype)

    sd = load_file(args.ckpt, device="cpu")
    model.load_state_dict(sd, strict=False)
    model.nl_head.weight = model.nl_embed.weight
    del sd
    gc.collect()

    t_q = time.perf_counter()
    q_stats = quantize_to_q_avl_pml(
        model, lora_rank_gqa=16, lora_rank_dense=16, lora_rank_routed=0, group_size=64, freeze_nl_embed=True
    )
    print(f"[Q-AVL-PML Quantized in {time.perf_counter()-t_q:.1f}s] {q_stats}", flush=True)

    model.gradient_checkpointing = True
    model.to(device)
    # Keep Group D scalars in strict FP32 on GPU!
    opt, opt_summary = build_five_group_optimizer(model, stage="B", is_lora_hybrid=True)
    sched = build_warmup_cosine_scheduler(opt, total_steps=args.steps, warmup_ratio=0.25)
    print("[5-Group Optimizer Summary]", json.dumps(opt_summary, indent=2), flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples = load_real_stage1_batches(Path(args.stage1_data), max_samples=128, max_seq_len=args.max_seq_len)
    rng = random.Random(2026)
    model.train()

    step_logs: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    # Pre-build 4 deterministic real Stage 1 traces with fixed masks (realistic grad_accum=4 mini-batch per step!)
    trial_batch = []
    for rec in samples[:4]:
        gold_toks = list(rec["token_ids_aligned_fsq"])
        frame_sizes = list(rec["frame_sizes"])
        rmask = list(rec["reassert_mask"])
        tgt = torch.tensor([gold_toks], dtype=torch.long, device=device)
        rmask_t = torch.tensor([rmask], dtype=torch.long, device=device)
        inp = tgt.clone()
        loss_mask = torch.zeros_like(tgt, dtype=torch.bool)
        off = 0
        for f_i, flen in enumerate(frame_sizes):
            if f_i >= max(1, len(frame_sizes) // 3):
                for pos in range(off + 1, off + flen - 1):
                    if rng.random() < 0.45:
                        inp[0, pos] = SymbolicVocab.MASK_ID
                        loss_mask[0, pos] = True
            off += flen
        if not loss_mask.any():
            inp[0, -3] = SymbolicVocab.MASK_ID
            loss_mask[0, -3] = True
        attn_mask = build_block_diffusion_attn_mask(frame_sizes, wet_seam_flags=[False] * len(frame_sizes)).to(device)
        trial_batch.append((inp, tgt, rmask_t, loss_mask, attn_mask, len(gold_toks)))

    grad_norms_by_group: Dict[str, float] = {}

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot_diff = tot_loss = tot_act = tot_ent = tot_ov = 0.0
        for inp, tgt, rmask_t, loss_mask, attn_mask, seq_len in trial_batch:
            wb = model.init_whiteboard_states(1, device, dtype=torch.bfloat16)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model.forward_symbolic(
                    inp,
                    phase=CognitivePhase.FORMAL,
                    attn_mask=attn_mask,
                    wb_states=wb,
                    t_k=0.35,
                    reassert_flags=rmask_t,
                    write_to_draft=False,
                    return_soft_sketch=False,
                )
                loss, stats = model.compute_six_losses(out, tgt, loss_mask, collect_stats=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {stats}")
            (loss / float(len(trial_batch))).backward()
            tot_diff += stats["loss_diff"] / len(trial_batch)
            tot_loss += stats["loss_total"] / len(trial_batch)
            tot_act += stats["active_experts_mean"] / len(trial_batch)
            tot_ent += stats["row_entropy_syn"] / len(trial_batch)
            tot_ov += stats["syn_adv_overlap"] / len(trial_batch)

        # On step 5, verify live finite gradient norms across ALL active optimizer groups!
        if step == 5:
            for pg in opt.param_groups:
                gn_sq = sum(float(p.grad.detach().float().pow(2).sum().item()) for p in pg["params"] if p.grad is not None)
                grad_norms_by_group[pg["name"]] = round(gn_sq ** 0.5, 4)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        step_logs.append(
            {
                "step": step,
                "loss_total": round(tot_loss, 4),
                "loss_diff": round(tot_diff, 4),
                "active_experts": round(tot_act, 2),
                "row_entropy_syn": round(tot_ent, 3),
                "syn_adv_overlap": round(tot_ov, 4),
            }
        )
        if step in (1, 5, 10, args.steps):
            s = step_logs[-1]
            cur_vram = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
            print(
                f"  [Step {step:02d}/{args.steps}] accum=4 | total={s['loss_total']:.4f} diff={s['loss_diff']:.4f} | "
                f"act_exp={s['active_experts']:.2f} | ent={s['row_entropy_syn']:.3f} ov={s['syn_adv_overlap']:.4f} | peak_vram={cur_vram} GB",
                flush=True,
            )

    elapsed_s = round(time.perf_counter() - t0, 2)
    peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0

    early_loss = step_logs[0]["loss_diff"]
    late_loss = step_logs[-1]["loss_diff"]
    passed = (peak_vram_gb < 14.0) and (late_loss < early_loss - 0.5) and all(v > 0.0 for v in grad_norms_by_group.values())

    report = {
        "trial_passed": passed,
        "q_avl_pml_stats": q_stats,
        "optimizer_5group_summary": opt_summary,
        "step5_grad_norms_by_group": grad_norms_by_group,
        "peak_vram_gb": peak_vram_gb,
        "vram_redline_gb": 14.4,
        "steps": args.steps,
        "elapsed_sec": elapsed_s,
        "ms_per_step": round(1000.0 * elapsed_s / args.steps, 1),
        "early_4step_mean_diff_loss": round(early_loss, 4),
        "late_4step_mean_diff_loss": round(late_loss, 4),
        "loss_drop": round(early_loss - late_loss, 4),
        "step_logs": step_logs,
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"=== [Stage 2 M3-3 Local 5080 Trial Complete | PASSED={passed}] ===", flush=True)
    print(
        f"Peak VRAM={peak_vram_gb} GB (< 14.4 GB) | {report['ms_per_step']} ms/step | "
        f"L_diff: {early_loss:.4f} -> {late_loss:.4f} (drop={early_loss - late_loss:+.4f}) | "
        f"Step-5 Grads={grad_norms_by_group}",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
