"""Stage 0 Gate 1 Final Closure Verification (~17 min, RTX 5080, < 6 GB Peak VRAM):
Closes the last 2 Gate 1 items cleanly with 100% memory isolation & incremental JSON checkpointing:
  A. Sparse-MoE Seed Variance Closure (10% Linear Warmup + Fixed Group A/C/E LRs):
     - L3-Hard (20% Unseen 4-Op AST, N=640) across ALL 3 Seeds [42, 1024, 2026] + K_max=0 Knock-Out + Canary
     - Level 1 Seed 42 (N=640) v1.1_full (+ K_max=0 Knock-Out) vs Static Top-2 Baseline (both with Warmup!)
  B. Level 2a/b S^RO Unseen Rule Table Closure (8-Slot x 3-Beat Causal Dereferencing, N=320):
     - Beat 0 [0..2]: <IN_q>, q -> Beat 1 [3]: mapped = S^RO[q] -> Beat 2 [4..7]: acc + BIND + REG
     - v1.1_full (Differentiable S^RO + Track-1 PML) vs pure_block_dllm (disable_pml=True), with per-hop diagnostics!
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.wet_cement import hierarchical_three_beat_unmask
from avl_pmlv1.engine.model import AblationConfig, AVLPMLModel
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab
from gen_synth_tasks import Stage0SynthGenerator, TaskBatch
from run_four_point_audit import mean_std
from run_gate1_ablation import (
    amp_ctx,
    build_optimizers,
    build_warmup_cosine_scheduler,
    evaluate_symbolic_exact_match,
    set_u_post_warmup,
    sync_timer,
    train_dual_stream_step,
)
from run_v11_verification import (
    check_vram_guard,
    evaluate_symbolic_prefix_kv,
    free_vram,
    sample_level2_rule_chain_8slot,
    sample_level3_dynamic_ast_8slot,
    save_checkpoint,
    verify_zero_leakage_canary_8slot,
)


@torch.no_grad()
def evaluate_l2ab_3beat_diagnostics(
    model: AVLPMLModel, batches: List[TaskBatch], device: torch.device
) -> Dict[str, float]:
    """Prefix-KV 3-beat evaluation for Level 2a/b reporting per-hop accuracy: Beat 0 (q), Beat 1 (mapped=S^RO[q]), Beat 2 (acc)."""
    model.eval()
    q_ok = mapped_ok = step_acc_ok = final_ok = 0
    total_steps = total_probs = 0
    active_accum = 0.0
    n_fwds = 0

    with amp_ctx(device):
        for batch in batches:
            prompt = batch.prompt_ids.to(device)
            tgt = batch.target_sym_ids.to(device)
            bsz, p_len = prompt.shape
            num_blocks = tgt.shape[1] // 8

            wb_states = model.init_whiteboard_states(bsz, device)
            if not model.ablation.disable_pml:
                model.populate_ro_from_prompt(prompt, wb_states)
            exp_state = model.init_experience_state(bsz, device)

            m_pref = torch.ones(1, 1, p_len, p_len, device=device, dtype=torch.bool)
            out_p = model.forward_symbolic(
                prompt, phase=CognitivePhase.FORMAL, attn_mask=m_pref, wb_states=wb_states, exp_state=exp_state,
                t_k=0.0, read_only_pml=True, dual_stream_block_size=8, past_kvs=None, start_pos=0,
            )
            h_p_mean = out_p["hidden"].mean(dim=1)  # type: ignore[union-attr]
            committed_kvs = out_p["present_kvs"]  # type: ignore[assignment]

            gen_blocks: List[torch.Tensor] = []
            for b_idx in range(num_blocks):
                cur_blk = torch.full((bsz, 8), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
                cur_start = p_len + b_idx * 8
                for beat, t_k in [(0, 1.0), (1, 0.625), (2, 0.50)]:
                    out = model.forward_symbolic(
                        cur_blk, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb_states, exp_state=exp_state,
                        t_k=t_k, read_only_pml=True, dual_stream_block_size=8, past_kvs=committed_kvs, start_pos=cur_start,
                    )
                    if b_idx == 0 and beat == 0 and not model.ablation.static_top2_baseline:
                        exp_state.delta_b_pre = model.hyper_router(h_p_mean, out["hidden"].mean(dim=1))  # type: ignore[union-attr]
                    moe_aux: List[Dict[str, torch.Tensor]] = out["moe_aux"]  # type: ignore[assignment]
                    if moe_aux:
                        active_accum += float(torch.stack([m["active_count"].mean() for m in moe_aux]).mean().item())
                        n_fwds += 1
                    cur_blk, _ = hierarchical_three_beat_unmask(cur_blk, out["logits"], beat, 3)  # type: ignore[arg-type]

                gen_blocks.append(cur_blk)
                t_blk = tgt[:, b_idx * 8 : (b_idx + 1) * 8]
                q_ok += int((cur_blk[:, 2] == t_blk[:, 2]).sum().item())
                mapped_ok += int((cur_blk[:, 3] == t_blk[:, 3]).sum().item())
                step_acc_ok += int((cur_blk[:, 4] == t_blk[:, 4]).sum().item())
                total_steps += bsz

                if b_idx < num_blocks - 1:
                    out_c = model.forward_symbolic(
                        cur_blk, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb_states, exp_state=exp_state,
                        t_k=0.0, read_only_pml=model.ablation.disable_pml, dual_stream_block_size=8, past_kvs=committed_kvs, start_pos=cur_start,
                    )
                    committed_kvs = out_c["present_kvs"]  # type: ignore[assignment]

            full_gen = torch.cat(gen_blocks, dim=1)
            pred_ans = full_gen.gather(1, batch.answer_pos.to(device).unsqueeze(-1)).squeeze(-1)
            final_ok += int((pred_ans == batch.answer_tok_ids.to(device)).sum().item())
            total_probs += bsz

    return {
        "final_exact_match": round(100.0 * final_ok / max(1, total_probs), 2),
        "hop1_query_q_acc": round(100.0 * q_ok / max(1, total_steps), 2),
        "hop2_sro_mapped_acc": round(100.0 * mapped_ok / max(1, total_steps), 2),
        "hop3_step_acc_acc": round(100.0 * step_acc_ok / max(1, total_steps), 2),
        "avg_active_experts": round(active_accum / max(1, n_fwds), 3),
    }


def run_closure_l3_seed(
    cfg: AVLPMLConfig,
    optim_path: Path,
    device: torch.device,
    seed: int,
    steps: int,
    eval_tuples: List[Any],
    run_canary: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = AVLPMLModel(cfg).to(device)
    opt, sopt = build_optimizers(model, optim_path, stabilized_v11=True)
    sched = build_warmup_cosine_scheduler(opt, total_steps=steps, warmup_ratio=0.10, min_lr_ratio=0.03)
    gen = Stage0SynthGenerator(cfg, seed=300 + seed)
    t0 = sync_timer(device)
    for st in range(1, steps + 1):
        b, _, _, _ = sample_level3_dynamic_ast_8slot(gen, 64, "train")
        _ = train_dual_stream_step(model, opt, sopt, b, device, st, steps, block_size=8, beats_per_block=2)
        sched.step()
    train_s = round(sync_timer(device) - t0, 2)
    del opt, sopt, sched
    set_u_post_warmup(model, 1.0)

    sym_batches = [t[0] for t in eval_tuples]
    ev = evaluate_symbolic_prefix_kv(model, sym_batches, device, beats_per_block=2, block_size=8)
    orig_k = model.cfg.k_max
    model.cfg.k_max = 0
    ev_k0 = evaluate_symbolic_prefix_kv(model, sym_batches, device, beats_per_block=2, block_size=8)
    model.cfg.k_max = orig_k

    canary = verify_zero_leakage_canary_8slot(model, sym_batches[0], device) if run_canary else None
    vram = check_vram_guard(f"closure_L3_seed{seed}")
    del model
    free_vram()
    out: Dict[str, Any] = {
        "train_sec": train_s,
        "em_prefix_kv": ev["exact_match"],
        "avg_active_experts": ev["avg_active_experts"],
        "k0_knockout_em": ev_k0["exact_match"],
        "k0_knockout_drop": round(ev["exact_match"] - ev_k0["exact_match"], 2),
        "vram": vram,
    }
    if canary is not None:
        out["zero_leakage_canary"] = canary
    return out


def run_closure_l1_variant(
    cfg: AVLPMLConfig,
    ab_cfg: AblationConfig,
    optim_path: Path,
    device: torch.device,
    seed: int,
    steps: int,
    test_batches: List[TaskBatch],
    run_k0: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = AVLPMLModel(cfg, ablation=ab_cfg).to(device)
    opt, sopt = build_optimizers(model, optim_path, stabilized_v11=True)
    sched = build_warmup_cosine_scheduler(opt, total_steps=steps, warmup_ratio=0.10, min_lr_ratio=0.03)
    gen = Stage0SynthGenerator(cfg, seed=101 + seed)
    std_eng_hist: List[float] = []
    t0 = sync_timer(device)
    for st in range(1, steps + 1):
        b = gen.sample_level1_batch(64, "train")
        m = train_dual_stream_step(model, opt, sopt, b, device, st, steps, block_size=16)
        sched.step()
        if m.get("std_alpha_eng", 0.0) > 0:
            std_eng_hist.append(m["std_alpha_eng"])
    train_s = round(sync_timer(device) - t0, 2)
    del opt, sopt, sched
    set_u_post_warmup(model, 1.0)

    ev = evaluate_symbolic_exact_match(model, test_batches, device, beats_per_block=3, block_size=16)
    ev["train_sec"] = train_s
    ev["std_alpha_eng"] = round(sum(std_eng_hist[-30:]) / max(1, len(std_eng_hist[-30:])), 3) if std_eng_hist else 0.0
    if run_k0:
        orig_k = model.cfg.k_max
        model.cfg.k_max = 0
        ev_k0 = evaluate_symbolic_exact_match(model, test_batches, device, beats_per_block=3, block_size=16)
        model.cfg.k_max = orig_k
        ev["k0_knockout_em"] = ev_k0["exact_match"]
        ev["k0_knockout_drop"] = round(ev["exact_match"] - ev_k0["exact_match"], 2)
    ev["vram"] = check_vram_guard(f"closure_L1_seed{seed}")
    del model
    free_vram()
    return ev


def run_closure_l2ab_3beat(
    cfg: AVLPMLConfig,
    ab_cfg: AblationConfig,
    optim_path: Path,
    device: torch.device,
    steps: int,
    test_4s: List[TaskBatch],
    test_6s: List[TaskBatch],
) -> Dict[str, Any]:
    torch.manual_seed(42)
    model = AVLPMLModel(cfg, ablation=ab_cfg).to(device)
    opt, sopt = build_optimizers(model, optim_path, stabilized_v11=True)
    sched = build_warmup_cosine_scheduler(opt, total_steps=steps, warmup_ratio=0.10, min_lr_ratio=0.03)
    gen = Stage0SynthGenerator(cfg, seed=202)
    t0 = sync_timer(device)
    for st in range(1, steps + 1):
        b = sample_level2_rule_chain_8slot(gen, 64, "train", num_chain_steps=4)
        _ = train_dual_stream_step(model, opt, sopt, b, device, st, steps, block_size=8, beats_per_block=3)
        sched.step()
    train_s = round(sync_timer(device) - t0, 2)
    del opt, sopt, sched
    set_u_post_warmup(model, 1.0)

    ev4 = evaluate_l2ab_3beat_diagnostics(model, test_4s, device)
    ev6 = evaluate_l2ab_3beat_diagnostics(model, test_6s, device)

    orig_k = model.cfg.k_max
    model.cfg.k_max = 0
    ev4_k0 = evaluate_l2ab_3beat_diagnostics(model, test_4s, device)
    model.cfg.k_max = orig_k

    alpha_calm_rms = (
        round(
            float(
                torch.stack([l.alpha_calm.detach().float().pow(2).mean().sqrt() for l in model.attn_layers if l.enable_pml]).mean().item()
            ),
            4,
        )
        if not ab_cfg.disable_pml
        else 0.0
    )
    vram = check_vram_guard("closure_L2ab")
    del model
    free_vram()
    return {
        "train_sec": train_s,
        "unseen_rule_4step": ev4,
        "unseen_rule_6step_extrap": ev6,
        "k0_knockout_4step_em": ev4_k0["final_exact_match"],
        "k0_knockout_drop": round(ev4["final_exact_match"] - ev4_k0["final_exact_match"], 2),
        "learned_alpha_calm_rms": alpha_calm_rms,
        "vram": vram,
    }


def run_preflight_profile(cfg: AVLPMLConfig, optim_path: Path, device: torch.device) -> Dict[str, Any]:
    print("=== [CLOSURE PRE-FLIGHT 5-STEP VRAM & LATENCY PROFILE] ===", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gen = Stage0SynthGenerator(cfg, seed=777)
    prof: Dict[str, Any] = {}

    def _prof(name: str, ab: AblationConfig, batch_fn, blk_sz: int, beats: int, planned_steps: int, n_runs: int) -> None:
        m = AVLPMLModel(cfg, ablation=ab).to(device)
        opt, sopt = build_optimizers(m, optim_path, stabilized_v11=True)
        for w in range(2):
            train_dual_stream_step(m, opt, sopt, batch_fn(), device, w + 1, 100, block_size=blk_sz, beats_per_block=beats)
        t0 = sync_timer(device)
        for s in range(5):
            train_dual_stream_step(m, opt, sopt, batch_fn(), device, s + 3, 100, block_size=blk_sz, beats_per_block=beats)
        ms = (sync_timer(device) - t0) * 1000.0 / 5.0
        vram = check_vram_guard(f"prof_{name}")
        est = round(ms * planned_steps * n_runs / 1000.0, 1)
        prof[name] = {"ms_per_step": round(ms, 2), "planned": f"{planned_steps}x{n_runs}", "est_sec": est, "vram": vram}
        print(f"  [{name:22s}] {ms:6.2f} ms/step | est={est:5.1f}s | VRAM={vram}", flush=True)
        del m, opt, sopt
        free_vram()

    _prof("l3_hard_warmup", AblationConfig(), lambda: sample_level3_dynamic_ast_8slot(gen, 64, "train")[0], 8, 2, 1100, 3)
    _prof("l1_full_warmup", AblationConfig(), lambda: gen.sample_level1_batch(64, "train"), 16, 3, 1100, 1)
    _prof("l1_top2_warmup", AblationConfig(static_top2_baseline=True), lambda: gen.sample_level1_batch(64, "train"), 16, 3, 1100, 1)
    _prof("l2ab_3beat_full", AblationConfig(), lambda: sample_level2_rule_chain_8slot(gen, 64, "train", 4), 8, 3, 900, 1)
    _prof("l2ab_3beat_pure", AblationConfig(disable_pml=True), lambda: sample_level2_rule_chain_8slot(gen, 64, "train", 4), 8, 3, 900, 1)

    tot_s = sum(v["est_sec"] for v in prof.values()) + 25.0
    prof["total_estimated_minutes"] = round(tot_s / 60.0, 2)
    print(f"=== [PROFILE SUMMARY] Peak VRAM={check_vram_guard('prof_end')['peak_alloc_gb']} GB (< 14.0 GB SAFE!) | Projected Total = {prof['total_estimated_minutes']} min ===", flush=True)
    return prof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    optim_path = root / "configs" / "optim_5groups.yaml"
    out_path = root / "outputs" / "stage0_gate1_closure.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prof = run_preflight_profile(cfg, optim_path, device)
    if args.profile_only:
        save_checkpoint(out_path, {"status": "profile_only", "profile": prof})
        return

    t_global = time.perf_counter()
    report: Dict[str, Any] = {
        "architecture_version": "AVL-PMLv1.1-Gate1-Closure",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "stabilized_v11_lrs": {"warmup_ratio": 0.10, "A": 8e-4, "B": 1.1e-3, "C": 1.0e-3, "D": 6e-4, "E": 4e-3},
        "preflight_profile": prof,
        "closure_a_l3_hard_3seeds_n640": {"per_seed": {}},
        "closure_b_l1_seed42_n640": {},
        "closure_c_l2ab_3beat_n320": {},
    }
    save_checkpoint(out_path, report)

    # A. L3-Hard (20% Unseen 4-Op AST, N=640) x 3 Seeds with 10% Warmup + Fixed Group LRs
    gen_l3_ev = Stage0SynthGenerator(cfg, seed=9009)
    eval_l3_640 = [sample_level3_dynamic_ast_8slot(gen_l3_ev, 32, "test") for _ in range(20)]
    l3_ems, l3_k0_ems, l3_acts = [], [], []
    for idx, sd in enumerate([42, 1024, 2026]):
        res = run_closure_l3_seed(cfg, optim_path, device, sd, 1100, eval_l3_640, run_canary=(idx == 0))
        report["closure_a_l3_hard_3seeds_n640"]["per_seed"][str(sd)] = res
        l3_ems.append(res["em_prefix_kv"])
        l3_k0_ems.append(res["k0_knockout_em"])
        l3_acts.append(res["avg_active_experts"])
        print(f"[Closure A L3-Hard Seed {sd}] EM={res['em_prefix_kv']}% | act={res['avg_active_experts']} | K0_KO={res['k0_knockout_em']}% (drop={res['k0_knockout_drop']}%) | {res['train_sec']}s | elapsed={(time.perf_counter()-t_global)/60:.1f}m", flush=True)
        save_checkpoint(out_path, report)

    report["closure_a_l3_hard_3seeds_n640"]["summary"] = {
        "v11_warmup_em": mean_std(l3_ems),
        "k0_knockout_em": mean_std(l3_k0_ems),
        "k0_mean_drop_pct": round(mean_std(l3_ems)["mean"] - mean_std(l3_k0_ems)["mean"], 2),
        "avg_active_experts": mean_std(l3_acts),
    }
    save_checkpoint(out_path, report)

    # B. Level 1 Seed 42 (N=640) with 10% Warmup + Fixed Group LRs
    gen_l1_ev = Stage0SynthGenerator(cfg, seed=9001)
    eval_l1_640 = [gen_l1_ev.sample_level1_batch(32, "test") for _ in range(20)]
    l1_full = run_closure_l1_variant(cfg, AblationConfig(), optim_path, device, 42, 1100, eval_l1_640, run_k0=True)
    print(f"[Closure B L1 Seed 42 Full] EM={l1_full['exact_match']}% | act={l1_full['avg_active_experts']} | K0_drop={l1_full['k0_knockout_drop']}% | elapsed={(time.perf_counter()-t_global)/60:.1f}m", flush=True)
    l1_top2 = run_closure_l1_variant(cfg, AblationConfig(static_top2_baseline=True), optim_path, device, 42, 1100, eval_l1_640, run_k0=False)
    print(f"[Closure B L1 Seed 42 Top2] EM={l1_top2['exact_match']}% | act={l1_top2['avg_active_experts']} | elapsed={(time.perf_counter()-t_global)/60:.1f}m", flush=True)
    report["closure_b_l1_seed42_n640"] = {
        "v11_full_warmup": l1_full,
        "static_top2_warmup": l1_top2,
        "em_gain_vs_top2_pct": round(l1_full["exact_match"] - l1_top2["exact_match"], 2),
        "active_routed_saving_pct": round(100.0 * (2.0 - l1_full["avg_active_experts"]) / 2.0, 2),
    }
    save_checkpoint(out_path, report)

    # C. Level 2a/b (20% Unseen Rule Tables, N=320, 8-Slot x 3-Beat Causal Dereferencing)
    gen_l2_ev = Stage0SynthGenerator(cfg, seed=9002)
    test_4s = [sample_level2_rule_chain_8slot(gen_l2_ev, 32, "test", num_chain_steps=4) for _ in range(10)]
    test_6s = [sample_level2_rule_chain_8slot(gen_l2_ev, 32, "test", num_chain_steps=6) for _ in range(10)]

    l2_full = run_closure_l2ab_3beat(cfg, AblationConfig(), optim_path, device, 900, test_4s, test_6s)
    print(f"[Closure C L2a/b 3-Beat Full] 4step={l2_full['unseen_rule_4step']} | 6step={l2_full['unseen_rule_6step_extrap']['final_exact_match']}% | K0_drop={l2_full['k0_knockout_drop']}% | alpha_calm={l2_full['learned_alpha_calm_rms']}", flush=True)
    l2_pure = run_closure_l2ab_3beat(cfg, AblationConfig(disable_pml=True), optim_path, device, 900, test_4s, test_6s)
    print(f"[Closure C L2a/b 3-Beat Pure] 4step={l2_pure['unseen_rule_4step']} | 6step={l2_pure['unseen_rule_6step_extrap']['final_exact_match']}%", flush=True)
    report["closure_c_l2ab_3beat_n320"] = {
        "v11_full_diff_sro": l2_full,
        "pure_block_dllm_no_pml": l2_pure,
        "pml_gain_4step_pct": round(l2_full["unseen_rule_4step"]["final_exact_match"] - l2_pure["unseen_rule_4step"]["final_exact_match"], 2),
    }

    report["total_wall_clock_minutes"] = round((time.perf_counter() - t_global) / 60.0, 2)
    report["peak_vram_gb"] = check_vram_guard("closure_final")["peak_alloc_gb"]
    save_checkpoint(out_path, report)
    print(f"\n[DONE] Gate 1 Final Closure completed in {report['total_wall_clock_minutes']} min! Peak VRAM={report['peak_vram_gb']} GB -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
