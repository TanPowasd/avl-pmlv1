"""AVL-PMLv1.2 Hard-Indexed Register Memory Battle Suite (RTX 5080, 0 RMB, < 14.0 GB VRAM):

Directly re-challenges the EXACT two tasks where v1.1 Soft PML, Pure Block-dLLM, and Mainstream Causal AR collapsed:
  Battle 1: 24-Step x 8-Register Interleaved ALU (mod 16, 304 toks, same seed=9101 test set)
            + Zero-Shot 36-Step x 8-Register Memory Extrapolation (448 toks — 50% longer than training!)
            + Runtime Zero-Leakage Canary (-999999 poisoned targets) + K_max=0 Knock-Out!
  Battle 2: Variable-Length Autonomous <EOT> Halting (L in [2..8], mod 16, same seed=9303 test set)
            vs archived v1.1 Soft PML (4.69%) & Mainstream Causal KV-AR (22.66%)!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.model import AblationConfig, AVLPMLModel
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab
from run_gate1_ablation import (
    amp_ctx,
    build_optimizers,
    build_warmup_cosine_scheduler,
    set_u_post_warmup,
    sync_timer,
)
from run_stage0_hard_stress import (
    HardALU24Batch,
    VarLenBatch,
    evaluate_hard_alu24_dllm,
    evaluate_varlen_autonomous_dllm,
    sample_hard_alu24_batch,
    sample_varlen_batch,
    train_hard_alu24_dllm_step,
    train_varlen_dllm_step,
    unmask_8slot_3beat_alu,
)
from run_v11_verification import check_vram_guard, free_vram, save_checkpoint


@torch.no_grad()
def verify_alu24_zero_leakage_canary(model: AVLPMLModel, batch: HardALU24Batch, device: torch.device) -> Dict[str, Any]:
    """Poison target_sym_ids -> -999999 and verify 0.00% diff on final_step24_em (impossible if any target leaked!)."""
    model.eval()
    # Evaluate on normal vs poisoned target_sym_ids (only evaluate final_step24_em against batch.final_ans_tok!)
    pois = HardALU24Batch(
        prompt_ids=batch.prompt_ids.clone(),
        target_sym_ids=torch.full_like(batch.target_sym_ids, -999999),
        rw_gaps=batch.rw_gaps.clone(),
        final_ans_tok=batch.final_ans_tok.clone(),
    )
    r_clean = evaluate_hard_alu24_dllm(model, [batch], device)
    r_pois = evaluate_hard_alu24_dllm(model, [pois], device)
    return {
        "em_normal": r_clean["final_step24_em"],
        "em_with_poisoned_targets": r_pois["final_step24_em"],
        "abs_em_diff": round(abs(r_clean["final_step24_em"] - r_pois["final_step24_em"]), 4),
        "zero_leakage_verified": r_clean["final_step24_em"] == r_pois["final_step24_em"],
    }


@torch.no_grad()
def evaluate_36step_memory_extrapolation(
    model: AVLPMLModel, batches: List[HardALU24Batch], device: torch.device
) -> Dict[str, float]:
    """Pure Zero-Shot 36-Step x 8-Register Memory Extrapolation (trained ONLY on 24 steps!).
    Each block's instruction header [0..3] ([op, REG_src, k, REG_dst]) is supplied as the program stream,
    while Beat 1 [4] (memory read of REG_src) & Beat 2 [5..7] (ALU + BIND REG_dst) start 100% [MASK]!
    Isolates true 36-step S^RW register tracking without Prompt-length position-shift confounding.
    """
    model.eval()
    final_ok = read_ok = read_late_ok = read_late_tot = lg_ok = lg_tot = alu_ok = 0
    tot_probs = tot_steps = 0

    with amp_ctx(device):
        for batch in batches:
            prompt = batch.prompt_ids.to(device)
            tgt = batch.target_sym_ids.to(device)
            gaps = batch.rw_gaps.to(device)
            bsz, p_len = prompt.shape
            num_blocks = tgt.shape[1] // 8  # 36 blocks!

            wb = model.init_whiteboard_states(bsz, device)
            if not model.ablation.disable_pml:
                model.populate_ro_from_prompt(prompt, wb)
            exp_state = model.init_experience_state(bsz, device)

            # Prefill ONLY the 16-token initial register bank [IN_0..7, v0..7] (fixed length 16, zero prompt-length shift!)
            init_p = prompt[:, :16]
            m_pref = torch.ones(1, 1, 16, 16, device=device, dtype=torch.bool)
            out_p = model.forward_symbolic(
                init_p, phase=CognitivePhase.FORMAL, attn_mask=m_pref, wb_states=wb, exp_state=exp_state,
                t_k=0.0, read_only_pml=True, dual_stream_block_size=8, past_kvs=None, start_pos=0, return_soft_sketch=False,
            )
            kvs = out_p["present_kvs"]  # type: ignore[assignment]

            gen_blocks: List[torch.Tensor] = []
            for b_i in range(num_blocks):
                t_blk = tgt[:, b_i * 8 : (b_i + 1) * 8]
                # Instruction header [0..3] given by program stream; slots [4..7] (read val, new val, BIND, REG_dst) 100% [MASK]!
                cur = torch.full((bsz, 8), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
                cur[:, :4] = t_blk[:, :4]
                s_pos = 16 + b_i * 8
                for beat, t_k in [(1, 0.50), (2, 0.375)]:
                    out = model.forward_symbolic(
                        cur, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb, exp_state=exp_state,
                        t_k=t_k, read_only_pml=True, dual_stream_block_size=8, past_kvs=kvs, start_pos=s_pos, return_soft_sketch=False,
                    )
                    cur = unmask_8slot_3beat_alu(cur, out["logits"], beat)  # type: ignore[arg-type]

                gen_blocks.append(cur)
                r_m = cur[:, 4] == t_blk[:, 4]
                read_ok += int(r_m.sum().item())
                if b_i >= 24:
                    read_late_ok += int(r_m.sum().item())
                    read_late_tot += bsz
                is_lg = gaps[:, b_i] >= 8
                lg_ok += int((r_m & is_lg).sum().item())
                lg_tot += int(is_lg.sum().item())
                alu_ok += int((cur[:, 5] == t_blk[:, 5]).sum().item())
                tot_steps += bsz

                if b_i < num_blocks - 1:
                    out_c = model.forward_symbolic(
                        cur, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb, exp_state=exp_state,
                        t_k=0.0, read_only_pml=model.ablation.disable_pml, dual_stream_block_size=8,
                        past_kvs=kvs, start_pos=s_pos, return_soft_sketch=False,
                    )
                    kvs = out_c["present_kvs"]  # type: ignore[assignment]

            full_gen = torch.cat(gen_blocks, dim=1)
            final_ok += int((full_gen[:, -3] == batch.final_ans_tok.to(device)).sum().item())
            tot_probs += bsz

    return {
        "final_step36_em": round(100.0 * final_ok / max(1, tot_probs), 2),
        "all_36steps_read_acc": round(100.0 * read_ok / max(1, tot_steps), 2),
        "extrap_steps24_35_read_acc": round(100.0 * read_late_ok / max(1, read_late_tot), 2),
        "long_gap_ge8_read_acc": round(100.0 * lg_ok / max(1, lg_tot), 2),
        "all_36steps_alu_acc": round(100.0 * alu_ok / max(1, tot_steps), 2),
    }


def run_preflight_profile(cfg: AVLPMLConfig, optim_path: Path, device: torch.device) -> Dict[str, Any]:
    print("=== [AVL-PMLv1.2 HARD-INDEXED PML 5-STEP PRE-FLIGHT PROFILE] ===", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vocab = SymbolicVocab(cfg)
    rng = random.Random(123)
    prof: Dict[str, Any] = {}

    def _p(name: str, fn, steps: int) -> None:
        m = AVLPMLModel(cfg).to(device)
        opt, _ = build_optimizers(m, optim_path, stabilized_v11=True)
        for w in range(2):
            fn(m, opt, w + 1)
        t0 = sync_timer(device)
        for s in range(5):
            fn(m, opt, s + 3)
        ms = (sync_timer(device) - t0) * 1000.0 / 5.0
        vram = check_vram_guard(f"v12_prof_{name}")
        est = round(ms * steps / 1000.0, 1)
        prof[name] = {"ms_per_step": round(ms, 2), "planned_steps": steps, "est_sec": est, "vram": vram}
        print(f"  [{name:22s}] {ms:6.2f} ms/step | est={est:5.1f}s | VRAM={vram}", flush=True)
        del m, opt
        free_vram()

    _p("battle1_alu24_v12", lambda m, o, s: train_hard_alu24_dllm_step(m, o, sample_hard_alu24_batch(vocab, rng, 32, "train", 24), device, s, 100), 1200)
    _p("battle2_varlen_v12", lambda m, o, s: train_varlen_dllm_step(m, o, sample_varlen_batch(vocab, rng, 64, "train", 2, 8, 8), device, s, 100), 1100)

    tot_s = sum(v["est_sec"] for v in prof.values()) + 60.0
    prof["total_estimated_minutes"] = round(tot_s / 60.0, 2)
    print(f"=== [PROFILE SUMMARY] Peak VRAM={check_vram_guard('v12_end')['peak_alloc_gb']} GB (< 14.0 GB SAFE!) | Projected Total = {prof['total_estimated_minutes']} min ===", flush=True)
    return prof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    optim_path = root / "configs" / "optim_5groups.yaml"
    out_path = root / "outputs" / "stage0_v12_hard_pml_battle.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prof = run_preflight_profile(cfg, optim_path, device)
    if args.profile_only:
        save_checkpoint(out_path, {"status": "profile_only", "profile": prof})
        return

    t_global = time.perf_counter()
    vocab = SymbolicVocab(cfg)
    report: Dict[str, Any] = {
        "architecture_version": "AVL-PMLv1.2-HardIndexed-RegisterMemory",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "preflight_profile": prof,
        "battle1_alu24_and_36extrap": {},
        "battle2_varlen_autonomous_halt": {},
    }

    # Load archived baselines from Stage 0-Hard for apples-to-apples table
    prev_path = root / "outputs" / "stage0_hard_stress.json"
    if prev_path.exists():
        with open(prev_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        report["archived_baselines_exact_same_testsets"] = {
            "alu24_v11_soft_pml": prev["stress1_alu24_8reg_mod16"]["v11_full_diff_pml"],
            "alu24_pure_block_dllm_no_pml": prev["stress1_alu24_8reg_mod16"]["pure_block_dllm_no_pml"],
            "alu24_mainstream_causal_kv_ar": prev["stress1_alu24_8reg_mod16"]["mainstream_causal_kv_ar"],
            "varlen_v11_soft_pml": prev["stress3_varlen_autonomous_halt"]["v11_autonomous_prefix_kv"],
            "varlen_mainstream_causal_kv_ar": prev["stress3_varlen_autonomous_halt"]["mainstream_causal_kv_ar"],
        }
    save_checkpoint(out_path, report)

    # =========================================================================
    # BATTLE 1: 24-Step x 8-Register Interleaved ALU (seed=9101) + 36-Step Extrap
    # =========================================================================
    rng_ev1 = random.Random(9101)
    test_alu24 = [sample_hard_alu24_batch(vocab, rng_ev1, 32, "test", 24) for _ in range(5)]  # exact same 160 programs!
    rng_ev36 = random.Random(9136)
    test_alu36 = [sample_hard_alu24_batch(vocab, rng_ev36, 32, "test", 36) for _ in range(4)]  # 128 unseen 36-step programs!

    torch.manual_seed(42)
    m1 = AVLPMLModel(cfg).to(device)
    opt1, _ = build_optimizers(m1, optim_path, stabilized_v11=True)
    sched1 = build_warmup_cosine_scheduler(opt1, total_steps=1200, warmup_ratio=0.10, min_lr_ratio=0.03)
    rng_tr1 = random.Random(4101)
    t0 = sync_timer(device)
    for st in range(1, 1201):
        b = sample_hard_alu24_batch(vocab, rng_tr1, 32, "train", 24)
        train_hard_alu24_dllm_step(m1, opt1, b, device, st, 1200)
        sched1.step()
    tr1_s = round(sync_timer(device) - t0, 1)
    del opt1, sched1
    set_u_post_warmup(m1, 1.0)

    ev24 = evaluate_hard_alu24_dllm(m1, test_alu24, device)
    canary1 = verify_alu24_zero_leakage_canary(m1, test_alu24[0], device)
    ev36 = evaluate_36step_memory_extrapolation(m1, test_alu36, device)
    orig_k = m1.cfg.k_max
    m1.cfg.k_max = 0
    ev24_k0 = evaluate_hard_alu24_dllm(m1, test_alu24, device)
    m1.cfg.k_max = orig_k
    alpha_rms = round(
        float(torch.stack([l.alpha_calm.detach().float().pow(2).mean().sqrt() for l in m1.attn_layers if l.enable_pml]).mean().item()),
        4,
    )
    report["battle1_alu24_and_36extrap"] = {
        "v12_hard_indexed_pml_24step": ev24,
        "v12_zero_shot_36step_extrap": ev36,
        "k0_knockout_24step_em": ev24_k0["final_step24_em"],
        "k0_knockout_drop": round(ev24["final_step24_em"] - ev24_k0["final_step24_em"], 2),
        "zero_leakage_canary": canary1,
        "learned_alpha_calm_rms": alpha_rms,
        "train_sec": tr1_s,
        "vram": check_vram_guard("B1_v12"),
    }
    print(f"[Battle1 ALU24 v1.2] EM={ev24['final_step24_em']}% | read={ev24['all_steps_read_acc']}% | gap>=8={ev24['long_gap_ge8_read_acc']}% | alu={ev24['all_steps_alu_acc']}% | K0_drop={report['battle1_alu24_and_36extrap']['k0_knockout_drop']}% | Canary={canary1['zero_leakage_verified']} | {tr1_s}s", flush=True)
    print(f"[Battle1 ALU36 Extrap] EM36={ev36['final_step36_em']}% | all_read={ev36['all_36steps_read_acc']}% | steps24-35_read={ev36['extrap_steps24_35_read_acc']}% | gap>=8={ev36['long_gap_ge8_read_acc']}%", flush=True)
    del m1
    free_vram()
    save_checkpoint(out_path, report)

    # =========================================================================
    # BATTLE 2: Variable-Length Autonomous <EOT> Halting (seed=9303, L in 2..8 + 9..10)
    # =========================================================================
    rng_ev2 = random.Random(9303)
    test_2_8 = [sample_varlen_batch(vocab, rng_ev2, 32, "test", 2, 8, 8) for _ in range(8)]     # exact same 256 problems!
    test_9_10 = [sample_varlen_batch(vocab, rng_ev2, 32, "test", 9, 10, 10) for _ in range(4)]  # exact same 128 extrap!

    torch.manual_seed(42)
    m2 = AVLPMLModel(cfg).to(device)
    opt2, _ = build_optimizers(m2, optim_path, stabilized_v11=True)
    sched2 = build_warmup_cosine_scheduler(opt2, total_steps=1100, warmup_ratio=0.10, min_lr_ratio=0.03)
    rng_tr2 = random.Random(4303)
    t0 = sync_timer(device)
    for st in range(1, 1101):
        b = sample_varlen_batch(vocab, rng_tr2, 64, "train", 2, 8, 8)
        train_varlen_dllm_step(m2, opt2, b, device, st, 1100)
        sched2.step()
    tr2_s = round(sync_timer(device) - t0, 1)
    del opt2, sched2
    set_u_post_warmup(m2, 1.0)

    ev_in = evaluate_varlen_autonomous_dllm(m2, test_2_8, device, max_gen_blocks=10)
    ev_ex = evaluate_varlen_autonomous_dllm(m2, test_9_10, device, max_gen_blocks=12)
    orig_k = m2.cfg.k_max
    m2.cfg.k_max = 0
    ev_k0 = evaluate_varlen_autonomous_dllm(m2, test_2_8, device, max_gen_blocks=10)
    m2.cfg.k_max = orig_k
    pois2 = VarLenBatch(test_2_8[0].prompt_ids.clone(), torch.full_like(test_2_8[0].target_sym_ids, -999999), test_2_8[0].true_lengths.clone(), test_2_8[0].final_ans_tok.clone())
    c2_ok = evaluate_varlen_autonomous_dllm(m2, [test_2_8[0]], device)["exact_halt_and_ans_em"] == evaluate_varlen_autonomous_dllm(m2, [pois2], device)["exact_halt_and_ans_em"]

    report["battle2_varlen_autonomous_halt"] = {
        "v12_in_dist_L2_8": ev_in,
        "v12_extrap_L9_10": ev_ex,
        "k0_knockout_em": ev_k0["exact_halt_and_ans_em"],
        "k0_knockout_drop": round(ev_in["exact_halt_and_ans_em"] - ev_k0["exact_halt_and_ans_em"], 2),
        "zero_leakage_canary": c2_ok,
        "train_sec": tr2_s,
        "vram": check_vram_guard("B2_v12"),
    }
    print(f"[Battle2 VarLen v1.2] L2-8 EM={ev_in['exact_halt_and_ans_em']}% (halt={ev_in['exact_halt_step_acc']}%, {ev_in['ms_per_problem']}ms) | L9-10={ev_ex['exact_halt_and_ans_em']}% | K0_drop={report['battle2_varlen_autonomous_halt']['k0_knockout_drop']}% | Canary={c2_ok} | {tr2_s}s", flush=True)
    del m2
    free_vram()

    report["total_wall_clock_minutes"] = round((time.perf_counter() - t_global) / 60.0, 2)
    report["peak_vram_gb"] = check_vram_guard("v12_final")["peak_alloc_gb"]
    save_checkpoint(out_path, report)
    print(f"\n[DONE] AVL-PMLv1.2 Battle completed in {report['total_wall_clock_minutes']} min! Peak VRAM={report['peak_vram_gb']} GB -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
