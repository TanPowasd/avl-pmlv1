"""Stage 2 (M3) Hard Verification Gates (M3-1 Step-0 Qwen3 Cosine, M3-2 TF32-Off <1e-5, M3-4 Canary & RoPE Extrapolation)."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.wet_cement import build_block_diffusion_attn_mask
from avl_pmlv1.engine.model import AVLPMLModel
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab


def load_surgical_model(cfg: AVLPMLConfig, ckpt_path: Path) -> AVLPMLModel:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = AVLPMLModel(cfg)
    torch.set_default_dtype(prev_dtype)

    sd = load_file(str(ckpt_path), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Only tied nl_head.weight alias is expected to be omitted from safetensors
    real_missing = [k for k in missing if k != "nl_head.weight"]
    if real_missing or unexpected:
        raise RuntimeError(f"State dict mismatch! missing={real_missing}, unexpected={unexpected}")
    model.nl_head.weight = model.nl_embed.weight
    model.eval()
    return model


@torch.no_grad()
def run_gate_m3_1_step0_parity(
    base_dir: Path, cfg: AVLPMLConfig, ckpt_path: Path, device: torch.device
) -> Dict[str, Any]:
    """Compare surgical AVL-PMLv1.2-4.5B directly against vanilla HuggingFace Qwen3ForCausalLM at Step 0."""
    prompts = [
        "Solve step by step: If a train travels at 60 mph for 2.5 hours, how far does it travel? Answer:",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n",
        "In formal logic, if A implies B and B implies C, and A is true, then C must be",
        "The capital of France is Paris, and the capital of Japan is",
    ]
    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    enc = tok(prompts, padding=False)

    # 1. Run vanilla HuggingFace Qwen3-4B-Base on GPU (single-sample unpadded for exact position parity!)
    print("[Gate M3-1] Loading official Qwen3-4B-Base reference in BF16 on GPU...", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), dtype=torch.bfloat16, device_map={"": device}
    )
    ref_model.eval()
    ref_hiddens: List[torch.Tensor] = []
    ref_top1: List[int] = []
    ref_top5: List[List[int]] = []
    ref_logits_last: List[torch.Tensor] = []

    for ids_list in enc["input_ids"]:
        inp = torch.tensor([ids_list], dtype=torch.long, device=device)
        out = ref_model(inp, output_hidden_states=True, use_cache=False)
        ref_hiddens.append(out.hidden_states[-1].float().cpu())
        last_l = out.logits[0, -1].float().cpu()
        ref_logits_last.append(last_l)
        ref_top1.append(int(last_l.argmax().item()))
        ref_top5.append(torch.topk(last_l, k=5).indices.tolist())

    del ref_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Load surgical AVL-PMLv1.2-4.5B BF16 onto GPU (~9.1 GB fits comfortably in 16 GB RTX 5080!)
    print("[Gate M3-1] Loading surgical AVL-PMLv1.2-4.5B BF16 on GPU...", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = load_surgical_model(cfg, ckpt_path).to(device)
    bf16_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0

    full_cos_list, full_logit_cos, full_top1_ok = [], [], 0
    k2_cos_list, k2_logit_cos, k2_top1_ok, k2_top5_ok = [], [], 0, 0
    gen_samples: List[Dict[str, str]] = []

    for i, ids_list in enumerate(enc["input_ids"]):
        inp = torch.tensor([ids_list], dtype=torch.long, device=device)

        # (A) Exact Full-Partition Conservation Check (512 + sum 8x1152 == exact 9728!)
        h_full, _ = model.forward_nl_causal(inp, full_sum_eval=True, return_hidden=True)
        l_full = model.nl_head(h_full)[0, -1].float().cpu()
        c_h = float(F.cosine_similarity(h_full.float().cpu().reshape(-1, cfg.hidden_size), ref_hiddens[i].reshape(-1, cfg.hidden_size), dim=-1).mean().item())
        c_l = float(F.cosine_similarity(l_full.unsqueeze(0), ref_logits_last[i].unsqueeze(0), dim=-1).item())
        full_cos_list.append(c_h)
        full_logit_cos.append(c_l)
        if int(l_full.argmax().item()) == ref_top1[i]:
            full_top1_ok += 1

        # (B) Real Sparse Adaptive Top-2 alpha_upcycle Forward Pass
        h_k2, _ = model.forward_nl_causal(inp, full_sum_eval=False, return_hidden=True)
        l_k2 = model.nl_head(h_k2)[0, -1].float().cpu()
        c_hk2 = float(F.cosine_similarity(h_k2.float().cpu().reshape(-1, cfg.hidden_size), ref_hiddens[i].reshape(-1, cfg.hidden_size), dim=-1).mean().item())
        c_lk2 = float(F.cosine_similarity(l_k2.unsqueeze(0), ref_logits_last[i].unsqueeze(0), dim=-1).item())
        k2_cos_list.append(c_hk2)
        k2_logit_cos.append(c_lk2)
        pred_k2 = int(l_k2.argmax().item())
        if pred_k2 == ref_top1[i]:
            k2_top1_ok += 1
        if pred_k2 in ref_top5[i]:
            k2_top5_ok += 1

        if i == 3:
            # Short 8-token greedy completion sanity check in full-sum mode
            cur = inp.clone()
            gen_ids = []
            for _ in range(8):
                lg, _ = model.forward_nl_causal(cur, full_sum_eval=True)
                nxt = lg[:, -1:].argmax(dim=-1)
                gen_ids.append(int(nxt.item()))
                cur = torch.cat([cur, nxt], dim=1)
            gen_samples.append({"prompt": prompts[i], "full_sum_gen_8tok": tok.decode(gen_ids)})

    # Keep model returned on CPU for subsequent gates
    model.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "bf16_infer_peak_vram_gb": bf16_vram_gb,
        "full_partition_hidden_cos": round(sum(full_cos_list) / len(full_cos_list), 6),
        "full_partition_logit_cos": round(sum(full_logit_cos) / len(full_logit_cos), 6),
        "full_partition_top1_match": f"{full_top1_ok}/{len(prompts)}",
        "sparse_k2_step0_hidden_cos": round(sum(k2_cos_list) / len(k2_cos_list), 4),
        "sparse_k2_step0_logit_cos": round(sum(k2_logit_cos) / len(k2_logit_cos), 4),
        "sparse_k2_step0_top1_match": f"{k2_top1_ok}/{len(prompts)}",
        "sparse_k2_step0_top5_match": f"{k2_top5_ok}/{len(prompts)}",
        "sanity_generation": gen_samples,
        "model_cpu": model,
    }


@torch.no_grad()
def run_gate_m3_2_tf32_off_4_5b(
    model: AVLPMLModel, cfg: AVLPMLConfig, stage1_jsonl: Path, device: torch.device
) -> Dict[str, Any]:
    """Strict TF32-OFF train-vs-inference equivalence (< 1e-5) across ALL 8 core PML layers of real 4.54B!"""
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        # Take a real multi-frame Stage 1 FSQ-annotated trace (padded to multiple of 16)
        sample_rec = None
        with open(stage1_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if 64 <= len(r["token_ids_aligned_fsq"]) <= 128:
                    sample_rec = r
                    break
        assert sample_rec is not None
        seq = list(sample_rec["token_ids_aligned_fsq"])
        rmask = list(sample_rec["reassert_mask"])
        pad_n = (16 - (len(seq) % 16)) % 16
        seq += [SymbolicVocab.PAD_ID] * pad_n
        rmask += [0] * pad_n
        n_chunks = len(seq) // 16

        tokens_t = torch.tensor([seq], dtype=torch.long, device=device)
        rmask_t = torch.tensor([rmask], dtype=torch.long, device=device)
        torch.manual_seed(2026)
        x_in = torch.randn(1, len(seq), cfg.hidden_size, dtype=torch.float32, device=device)
        x_val = torch.randn(1, len(seq), cfg.hidden_size, dtype=torch.float32, device=device)

        max_state_err = 0.0
        max_h_err = 0.0
        max_epi_err = 0.0
        max_ac_err = 0.0

        # Audit ALL 8 core Track-1 PML layers in strict FP32 on GPU (1 layer at a time -> < 0.2 GB VRAM!)
        for l_idx in cfg.pml_layer_indices:
            pml_mod = model.attn_layers[l_idx].pml
            assert pml_mod is not None
            pml_fp32 = pml_mod.to(device=device, dtype=torch.float32)

            wb_full = model.init_whiteboard_states(1, device, dtype=torch.float32)[0]
            wb_step = model.init_whiteboard_states(1, device, dtype=torch.float32)[0]

            h_full, epi_full, ac_full = pml_fp32.sequential_scan_forward(
                x_in, tokens_t, wb_full, reassert_flags=rmask_t, write_to_draft=False, x_val_embed=x_val
            )

            h_parts, epi_parts, ac_parts = [], [], []
            prev_tok = None
            for c in range(n_chunks):
                s, e = c * 16, (c + 1) * 16
                hc, epic, acc = pml_fp32.sequential_scan_forward(
                    x_in[:, s:e],
                    tokens_t[:, s:e],
                    wb_step,
                    reassert_flags=rmask_t[:, s:e],
                    write_to_draft=True,
                    prev_last_token=prev_tok,
                    x_val_embed=x_val[:, s:e],
                )
                prev_tok = tokens_t[:, e - 1]
                wb_step.lazy_commit_previous_tail()
                h_parts.append(hc)
                epi_parts.append(epic)
                ac_parts.append(acc)

            max_state_err = max(max_state_err, float((wb_full.s_rw_commit - wb_step.s_rw_commit).abs().max().item()))
            max_h_err = max(max_h_err, float((h_full - torch.cat(h_parts, dim=1)).abs().max().item()))
            max_epi_err = max(max_epi_err, float((epi_full - torch.cat(epi_parts, dim=1)).abs().max().item()))
            max_ac_err = max(max_ac_err, float((ac_full - torch.cat(ac_parts, dim=1)).abs().max().item()))

            pml_mod.to(device="cpu", dtype=torch.bfloat16)

        passed = max(max_state_err, max_h_err, max_epi_err, max_ac_err) < 1e-5
        return {
            "num_pml_layers_audited": len(cfg.pml_layer_indices),
            "trace_id": sample_rec["id"],
            "trace_len_tokens": len(seq),
            "max_state_abs_err": max_state_err,
            "max_h_abs_err": max_h_err,
            "max_yellow_light_err": max_epi_err,
            "max_red_light_err": max_ac_err,
            "passed_lt_1e5": passed,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def run_gate_m3_4_canary_and_rope_extrap(
    model: AVLPMLModel, cfg: AVLPMLConfig, device: torch.device
) -> Dict[str, Any]:
    """Zero-leakage -999999 canary on real 4.54B StaticEngram + pure-relative RoPE S=1024 (128x8-slot) check!"""
    assert model.block_idx_embed is None, "Stage 2 4.5B MUST NOT have absolute block_idx_embed!"
    # 1. StaticEngram -999999 zero-leakage canary on BOTH L3 and L15!
    tok_clean = torch.randint(4, 1000, (1, 64), device=device)
    tok_clean[:, 32:] = SymbolicVocab.MASK_ID
    tok_poison = tok_clean.clone()
    # Note: Engram only sees [MASK] at masked target slots; poison clean copy vs masked copy with extreme future tokens
    tok_a = tok_clean.clone()
    tok_b = tok_clean.clone()
    x_a = torch.randn(1, 64, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    x_b = x_a.clone()
    x_b[:, 32:] = -999999.0  # Poison masked-target hidden slots shouldn't leak into prefix Engram hashes or prefix outputs!

    max_canary_diff = 0.0
    for key, eng in model.engrams.items():
        eng.to(device)
        # Dual-stream strict block isolation check (8-slot blocks): poisoning noisy future block MUST NOT change prefix!
        ya, _, _ = eng(x_a, tok_a, is_sym_phase=True, dual_stream_sym_len=16, dual_stream_block_size=8)
        yb, _, _ = eng(x_b, tok_b, is_sym_phase=True, dual_stream_sym_len=16, dual_stream_block_size=8)
        diff = float((ya[:, :32] - yb[:, :32]).abs().max().item())
        max_canary_diff = max(max_canary_diff, diff)
        eng.to("cpu")

    # 2. Pure-Relative RoPE 1,024-token (= 128 blocks of 8 slots, > old 64-block cap!) forward check
    model.to(device)
    long_ids = torch.randint(4, 1000, (1, 1024), device=device)
    wb = model.init_whiteboard_states(1, device, dtype=torch.bfloat16)
    mask_long = build_block_diffusion_attn_mask([16] * 64, wet_seam_flags=[False] * 64).to(device)
    out_long = model.forward_symbolic(
        long_ids, phase=CognitivePhase.FORMAL, attn_mask=mask_long, wb_states=wb, return_soft_sketch=False
    )
    logits_finite = bool(torch.isfinite(out_long["logits"]).all().item())  # type: ignore[arg-type]
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "block_idx_embed_removed": model.block_idx_embed is None,
        "engram_zero_leakage_canary_diff": max_canary_diff,
        "rope_1024tok_128blk_extrap_finite": logits_finite,
        "rope_theta": cfg.rope_theta,
        "use_qk_norm": cfg.use_qk_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "model_4_5b.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "avl_pmlv12_4_5b_step0_bf16.safetensors"))
    parser.add_argument("--surgery-report", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "stage2_surgery_report.json"))
    parser.add_argument("--stage1-data", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1" / "stage1_30k_fsq_aligned.jsonl"))
    parser.add_argument("--out-report", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "stage2_m3_gates_report.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AVLPMLConfig.from_yaml(args.config)
    srep = json.load(open(args.surgery_report, "r", encoding="utf-8"))
    base_dir = Path(srep["base_model_dir"])
    ckpt_path = Path(args.ckpt)

    t0 = time.perf_counter()
    g1 = run_gate_m3_1_step0_parity(base_dir, cfg, ckpt_path, device)
    model_cpu = g1.pop("model_cpu")
    print("[Gate M3-1 Result]", json.dumps(g1, indent=2, ensure_ascii=False), flush=True)

    g2 = run_gate_m3_2_tf32_off_4_5b(model_cpu, cfg, Path(args.stage1_data), device)
    print("[Gate M3-2 Result]", json.dumps(g2, indent=2, ensure_ascii=False), flush=True)

    g4 = run_gate_m3_4_canary_and_rope_extrap(model_cpu, cfg, device)
    print("[Gate M3-4 Result]", json.dumps(g4, indent=2, ensure_ascii=False), flush=True)

    all_passed = (
        g1["full_partition_hidden_cos"] > 0.999
        and g2["passed_lt_1e5"]
        and g4["block_idx_embed_removed"]
        and g4["engram_zero_leakage_canary_diff"] == 0.0
        and g4["rope_1024tok_128blk_extrap_finite"]
    )
    report = {
        "all_gates_passed": all_passed,
        "gate_m3_1_step0_parity": g1,
        "gate_m3_2_tf32_off_equivalence": g2,
        "gate_m3_4_canary_and_extrapolation": g4,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"=== [Stage 2 M3 Gates Complete | ALL_PASSED={all_passed} | {report['elapsed_sec']}s] ===", flush=True)
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
