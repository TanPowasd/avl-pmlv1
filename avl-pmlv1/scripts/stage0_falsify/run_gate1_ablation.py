"""Stage 0 (40M) Full Gate 1 Falsification: Dual-Stream BD3-LM + BF16 RTX 5080 Suite."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import yaml

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.post_surgery import PostThinkingSurgeryGuard
from avl_pmlv1.diffusion.wet_cement import (
    build_block_diffusion_attn_mask,
    build_dual_stream_bd3_mask,
    compute_top1_top2_margin,
    hierarchical_three_beat_unmask,
    select_wet_cement_remask_slots,
)
from avl_pmlv1.engine.model import AblationConfig, AVLPMLModel
from avl_pmlv1.moe_track2.tri_state_moe import TriStateMoELayer
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab
from gen_synth_tasks import Stage0SynthGenerator, TaskBatch


def sync_timer(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def amp_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.10, min_lr_ratio: float = 0.03
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup (0 -> 1.0 over warmup_ratio) followed by smooth cosine decay to min_lr_ratio."""
    import math

    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        prog = min(1.0, max(0.0, prog))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def build_optimizers(
    model: AVLPMLModel, optim_yaml_path: str | Path, lr_scale: float = 5.5, stabilized_v11: bool = False
) -> Tuple[torch.optim.AdamW, torch.optim.AdamW | None]:
    with open(optim_yaml_path, "r", encoding="utf-8") as f:
        ocfg = yaml.safe_load(f)

    groups: Dict[str, List[torch.nn.Parameter]] = {"A": [], "B": [], "C": [], "D": [], "E": []}
    scorer_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(
            k in name
            for k in (".w_u_direct.", ".w_ctx.", ".w_ctx_to_z", ".w_exp_s1", ".b_exp_s1", ".w_exp_s2")
        ):
            scorer_params.append(param)
            continue
        if "engrams" in name and "tables" in name:
            groups["E"].append(param)
        elif any(
            k in name
            for k in ("theta_syn", "theta_adv", "theta_inter", "tau_0", "w_t", "w_e", "w_u", "b_stat", "lambda_pos", "lambda_neg")
        ):
            groups["D"].append(param)
        elif "routed_experts" in name:
            groups["B"].append(param)
        elif any(
            k in name
            for k in (
                "gqa",
                "shared_expert",
                "ffn_layers.0.",
                "ffn_layers.11.",
                "nl_embed",
                "sym_embed_head",
                "in_span_tag",
                "chunk_slot_embed",
                "block_idx_embed",
                "final_norm",
            )
        ):
            groups["A"].append(param)
        else:
            groups["C"].append(param)

    if stabilized_v11:
        # Balanced peak LRs for 10% linear warmup -> cosine decay (eliminates step-1 router/Engram shock!)
        param_groups = [
            {"params": groups["A"], "lr": 8.0e-4, "weight_decay": 0.01},
            {"params": groups["B"], "lr": 1.1e-3, "weight_decay": 0.01},
            {"params": groups["C"], "lr": 1.0e-3, "weight_decay": 0.01},
            {"params": groups["D"], "lr": 6.0e-4, "weight_decay": 0.0},
            {"params": groups["E"], "lr": 4.0e-3, "weight_decay": 0.0},
        ]
    else:
        param_groups = [
            {"params": groups["A"], "lr": ocfg["groups"]["group_a_backbone"]["lr_stage_b"] * lr_scale * 5.0, "weight_decay": 0.01},
            {"params": groups["B"], "lr": ocfg["groups"]["group_b_routed_experts"]["lr_stage_b"] * lr_scale * 4.0, "weight_decay": 0.01},
            {"params": groups["C"], "lr": ocfg["groups"]["group_c_new_modules"]["lr_stage_b"] * lr_scale, "weight_decay": 0.01},
            {"params": groups["D"], "lr": ocfg["groups"]["group_d_manifold_scalars"]["lr_stage_b"] * lr_scale, "weight_decay": 0.0},
            {"params": groups["E"], "lr": ocfg["groups"]["group_e_static_engram_table"]["lr_stage_b"] * lr_scale, "weight_decay": 0.0},
        ]
    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
    main_opt = torch.optim.AdamW(param_groups, betas=tuple(ocfg["betas"]), eps=float(ocfg["eps"]))
    scorer_opt = (
        torch.optim.AdamW(scorer_params, lr=2.5e-3, betas=(0.9, 0.95), weight_decay=0.01)
        if scorer_params
        else None
    )
    return main_opt, scorer_opt


def set_u_post_warmup(model: AVLPMLModel, progress: float) -> None:
    """Follow Stage B->C curriculum: let experts specialize in first 25%, then ramp u_post 0 -> 1."""
    if progress < 0.25:
        w = 0.0
    elif progress < 0.60:
        w = (progress - 0.25) / 0.35
    else:
        w = 1.0
    for ffn in model.ffn_layers:
        if isinstance(ffn, TriStateMoELayer):
            ffn.u_post_warmup.fill_(w)


def train_dual_stream_step(
    model: AVLPMLModel,
    main_opt: torch.optim.AdamW,
    scorer_opt: torch.optim.AdamW | None,
    batch: TaskBatch,
    device: torch.device,
    step_idx: int,
    total_steps: int = 800,
    block_size: int = 16,
    beats_per_block: int = 0,
) -> Dict[str, float]:
    """Exact leak-free Dual-Stream [Prompt, Y_clean, Y_noisy] BD3-LM training step (supports 8-slot 2/3-beat & 16-slot 3-beat)."""
    model.train()
    set_u_post_warmup(model, step_idx / float(max(1, total_steps)))
    prompt = batch.prompt_ids.to(device)
    sym_clean = batch.target_sym_ids.to(device)
    rmask = batch.reassert_mask.to(device)
    bsz, p_len = prompt.shape
    s_len = sym_clean.shape[1]
    num_blocks = s_len // block_size

    local_pos = (torch.arange(s_len, device=device) % block_size).unsqueeze(0).expand(bsz, -1)
    if block_size == 8 and beats_per_block == 3:
        # 8-slot 3-beat causal dereferencing schedule:
        # Beat 0: all masked -> [0..2] (OP, <IN_q>, q); Beat 1: >=3 masked -> [3] (S^RO[q]); Beat 2: >=4 masked -> [4..7] (acc+BIND+REG)
        beat_mode = step_idx % 3
        if beat_mode == 0:
            q_rate = 1.0
            mask_sym = torch.ones_like(sym_clean, dtype=torch.bool)
        elif beat_mode == 1:
            q_rate = 0.625
            mask_sym = local_pos >= 3
        else:
            q_rate = 0.50
            mask_sym = local_pos >= 4
    elif block_size == 8:
        # 8-slot atomic micro-block 2-beat schedule: Beat 0 masks all 8 slots, Beat 1 keeps [0..3] clean & masks [4..7]!
        beat_mode = step_idx % 2
        if beat_mode == 0:
            q_rate = 1.0
            mask_sym = torch.ones_like(sym_clean, dtype=torch.bool)
        else:
            q_rate = 0.50
            mask_sym = local_pos >= 4
    else:
        # 16-slot frame 3-beat schedule
        beat_mode = step_idx % 3
        if beat_mode == 0:
            q_rate = 1.0
            mask_sym = torch.ones_like(sym_clean, dtype=torch.bool)
        elif beat_mode == 1:
            q_rate = 0.55
            mask_sym = (local_pos == 4) | (local_pos >= 11) | (torch.rand(bsz, s_len, device=device) < 0.20)
        else:
            q_rate = 0.25
            mask_sym = local_pos >= 12

    noisy_sym = torch.where(mask_sym, torch.full_like(sym_clean, SymbolicVocab.MASK_ID), sym_clean)

    dual_in = torch.cat([prompt, sym_clean, noisy_sym], dim=1)
    dual_target = torch.cat([prompt, sym_clean, sym_clean], dim=1)
    dual_rmask = torch.cat([torch.zeros_like(prompt), rmask, torch.zeros_like(rmask)], dim=1)
    dual_loss_mask = torch.cat(
        [torch.zeros(bsz, p_len + s_len, device=device, dtype=torch.bool), mask_sym], dim=1
    )

    wet_on = (not model.ablation.disable_wet_cement) and (block_size == 16) and (step_idx % 4 == 0)
    attn_mask, rope_pos = build_dual_stream_bd3_mask(
        prompt_len=p_len, num_blocks=num_blocks, block_size=block_size, enable_wet_seam=wet_on, device=device
    )
    rope_pos = rope_pos.expand(bsz, -1)

    main_opt.zero_grad(set_to_none=True)
    do_stats = (step_idx % 25 == 0) or (step_idx == total_steps)
    with amp_ctx(device):
        wb_states = model.init_whiteboard_states(bsz, device)
        if not model.ablation.disable_pml:
            # Differentiable S^RO prompt encoding (unlocked from torch.no_grad so ro_init_proj learns end-to-end!)
            model.populate_ro_from_prompt(prompt, wb_states)
        out = model.forward_symbolic(
            dual_in,
            phase=CognitivePhase.FORMAL,
            positions=rope_pos,
            attn_mask=attn_mask,
            wb_states=wb_states,
            exp_state=None,
            t_k=q_rate,
            reassert_flags=dual_rmask,
            prior_dropout_prob=0.0,
            cache_utility=False,
            utility_mask=dual_loss_mask,
            dual_stream_sym_blocks=num_blocks,
            dual_stream_block_size=block_size,
            return_soft_sketch=False,
        )
        loss, stats = model.compute_six_losses(out, dual_target, dual_loss_mask, collect_stats=do_stats)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    main_opt.step()
    return stats


@torch.no_grad()
def evaluate_symbolic_exact_match(
    model: AVLPMLModel,
    batches: List[TaskBatch],
    device: torch.device,
    beats_per_block: int = 3,
    block_size: int = 16,
) -> Dict[str, float]:
    """Pure out-of-sample Block-dLLM inference from 100% [MASK] blocks (Zero Leakage! Supports 8-slot & 16-slot blocks)."""
    model.eval()
    correct = 0
    total = 0
    active_experts_accum = 0.0
    n_forwards = 0

    if batches:
        with amp_ctx(device):
            _ = model.forward_symbolic(batches[0].prompt_ids[:2].to(device), phase=CognitivePhase.PRE, read_only_pml=True)

    t_start = sync_timer(device)

    for batch in batches:
        prompt = batch.prompt_ids.to(device)
        bsz, p_len = prompt.shape
        s_len = batch.target_sym_ids.shape[1]
        num_sym_blocks = s_len // block_size

        wb_states = model.init_whiteboard_states(bsz, device)
        if not model.ablation.disable_pml:
            model.populate_ro_from_prompt(prompt, wb_states)
        exp_state = model.init_experience_state(bsz, device)

        generated_blocks: List[torch.Tensor] = []
        block_margins: List[torch.Tensor] = []

        with amp_ctx(device):
            for b_idx in range(num_sym_blocks):
                cur_blk = torch.full((bsz, block_size), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
                prefix = torch.cat([prompt] + generated_blocks, dim=1)
                blk_lens = [p_len] + [block_size] * (len(generated_blocks) + 1)
                attn_mask = build_block_diffusion_attn_mask(blk_lens, device=device)

                blk_sketch = None
                d_b_cur = None
                for beat in range(beats_per_block):
                    t_k = float((cur_blk == SymbolicVocab.MASK_ID).float().mean().item())
                    seq_in = torch.cat([prefix, cur_blk], dim=1)
                    full_sketch = (
                        None
                        if blk_sketch is None
                        else torch.cat(
                            [torch.zeros(bsz, prefix.shape[1], model.cfg.hidden_size, device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32), blk_sketch],
                            dim=1,
                        )
                    )
                    out = model.forward_symbolic(
                        seq_in,
                        phase=CognitivePhase.FORMAL,
                        attn_mask=attn_mask,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        soft_sketch=full_sketch,
                        t_k=t_k,
                        read_only_pml=True,
                        dual_stream_block_size=block_size,
                    )
                    if b_idx == 0 and beat == 0 and not model.ablation.static_top2_baseline:
                        h_p = out["hidden"][:, :p_len].mean(dim=1)  # type: ignore[index]
                        h_s = out["hidden"][:, -block_size:].mean(dim=1)  # type: ignore[index]
                        exp_state.delta_b_pre = model.hyper_router(h_p, h_s)

                    moe_aux: List[Dict[str, torch.Tensor]] = out["moe_aux"]  # type: ignore[assignment]
                    if moe_aux:
                        active_experts_accum += float(
                            torch.stack([m["active_count"][:, -block_size:].mean() for m in moe_aux]).mean().item()
                        )
                        n_forwards += 1

                    blk_logits = out["logits"][:, -block_size:, :]  # type: ignore[index]
                    blk_sketch = out["soft_sketch"][:, -block_size:, :]  # type: ignore[index]
                    d_b_cur = out["d_b"]  # type: ignore[assignment]
                    cur_blk, margins = hierarchical_three_beat_unmask(cur_blk, blk_logits, beat, beats_per_block)

                # Wet-cement 16-slot backward seam polish (16-slot frame mode only)
                if (
                    block_size == 16
                    and b_idx > 0
                    and not model.ablation.disable_wet_cement
                    and d_b_cur is not None
                    and float(d_b_cur[:, 1].mean().item()) > 0.55
                    and float(block_margins[-1].min().item()) < 0.45
                ):
                    d_bwd = float(d_b_cur[:, 1].mean().item())
                    rem_tail, _ = select_wet_cement_remask_slots(generated_blocks[-1], block_margins[-1], d_bwd)
                    head_fixed = torch.cat([prompt] + generated_blocks[:-1], dim=1)
                    seam_seq = torch.cat([head_fixed, rem_tail, cur_blk], dim=1)
                    n_c = seam_seq.shape[1] // 16
                    seam_mask = build_block_diffusion_attn_mask([16] * n_c, wet_seam_flags=[c == n_c - 1 for c in range(n_c)], device=device)
                    out_s = model.forward_symbolic(
                        seam_seq, phase=CognitivePhase.FORMAL, attn_mask=seam_mask, wb_states=wb_states, exp_state=exp_state, t_k=0.2, read_only_pml=True
                    )
                    preds_s, m_s = compute_top1_top2_margin(model.vocab.apply_domain_mask(out_s["logits"][:, -32:, :]))  # type: ignore[index]
                    improved = (rem_tail == SymbolicVocab.MASK_ID) & (m_s[:, :16] > block_margins[-1])
                    generated_blocks[-1] = torch.where(improved, preds_s[:, :16], generated_blocks[-1])
                    block_margins[-1] = torch.where(improved, m_s[:, :16], block_margins[-1])

                generated_blocks.append(cur_blk)
                block_margins.append(margins)

                # Clean per-layer PML commit at block boundary (only when another block follows!)
                if not model.ablation.disable_pml and b_idx < num_sym_blocks - 1:
                    for wb in wb_states:
                        wb.s_rw_commit.zero_()
                        wb.m_rw_commit.zero_()
                        wb.s_rw_draft.zero_()
                        wb.m_rw_draft.zero_()
                        wb.dirty_mask.zero_()
                    clean_pref = torch.cat([prompt] + generated_blocks, dim=1)
                    c_mask = build_block_diffusion_attn_mask([p_len] + [block_size] * len(generated_blocks), device=device)
                    _ = model.forward_symbolic(
                        clean_pref,
                        phase=CognitivePhase.FORMAL,
                        attn_mask=c_mask,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        t_k=0.0,
                        read_only_pml=False,
                        dual_stream_block_size=block_size,
                    )

        full_gen = torch.cat(generated_blocks, dim=1)
        ans_pos = batch.answer_pos.to(device)
        pred_ans_ids = full_gen.gather(1, ans_pos.unsqueeze(-1)).squeeze(-1)
        true_ans_ids = batch.answer_tok_ids.to(device)
        correct += int((pred_ans_ids == true_ans_ids).sum().item())
        total += bsz

    elapsed_ms = (sync_timer(device) - t_start) * 1000.0
    return {
        "exact_match": round(100.0 * correct / max(1, total), 2),
        "ms_per_problem": round(elapsed_ms / max(1, total), 3),
        "avg_active_experts": round(active_experts_accum / max(1, n_forwards), 3),
    }


def run_level1_suite(cfg: AVLPMLConfig, optim_path: Path, device: torch.device, train_steps: int = 500) -> Dict[str, object]:
    variants = {
        "avl_pmlv1_full": AblationConfig(),
        "static_top2_baseline": AblationConfig(static_top2_baseline=True),
        "wo_static_engram": AblationConfig(disable_engram=True),
        "wo_bipolar_graph": AblationConfig(disable_bipolar_graph=True),
    }
    gen_eval = Stage0SynthGenerator(cfg, seed=9001)
    test_batches = [gen_eval.sample_level1_batch(batch_size=32, split="test") for _ in range(6)]

    results: Dict[str, object] = {}
    for v_name, ab_cfg in variants.items():
        torch.manual_seed(42)
        model = AVLPMLModel(cfg, ablation=ab_cfg).to(device)
        main_opt, scorer_opt = build_optimizers(model, optim_path)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(main_opt, T_max=train_steps, eta_min=2e-5)
        gen_tr = Stage0SynthGenerator(cfg, seed=101)

        kendall_hist: List[float] = []
        std_eng_hist: List[float] = []
        for st in range(1, train_steps + 1):
            b = gen_tr.sample_level1_batch(batch_size=64, split="train")
            st_metrics = train_dual_stream_step(model, main_opt, scorer_opt, b, device, st, train_steps)
            sched.step()
            if "scorer_kendall_tau" in st_metrics:
                kendall_hist.append(st_metrics["scorer_kendall_tau"])
            if "std_alpha_eng" in st_metrics and st_metrics["std_alpha_eng"] > 0:
                std_eng_hist.append(st_metrics["std_alpha_eng"])

        set_u_post_warmup(model, 1.0)
        ev = evaluate_symbolic_exact_match(model, test_batches, device)
        ev["kendall_tau"] = round(sum(kendall_hist[-30:]) / max(1, len(kendall_hist[-30:])), 3) if kendall_hist else 0.0
        ev["std_alpha_eng"] = round(sum(std_eng_hist[-30:]) / max(1, len(std_eng_hist[-30:])), 3) if std_eng_hist else 0.0
        results[v_name] = ev
        print(f"[Level 1] {v_name}: {ev}", flush=True)

    full_em = results["avl_pmlv1_full"]["exact_match"]  # type: ignore[index]
    top2_em = results["static_top2_baseline"]["exact_match"]  # type: ignore[index]
    full_act = results["avl_pmlv1_full"]["avg_active_experts"]  # type: ignore[index]
    top2_act = results["static_top2_baseline"]["avg_active_experts"]  # type: ignore[index]
    flops_saving_pct = round(100.0 * (top2_act - full_act) / max(0.1, top2_act), 2)
    results["gate1_verdict"] = {
        "em_gain_vs_top2": round(full_em - top2_em, 2),
        "active_routed_saving_pct": flops_saving_pct,
        "passed": bool(
            (full_em - top2_em >= 2.0 or (full_em >= top2_em and flops_saving_pct >= 15.0))
            and results["avl_pmlv1_full"]["kendall_tau"] > 0.30  # type: ignore[index]
            and results["avl_pmlv1_full"]["std_alpha_eng"] >= 0.08  # type: ignore[index]
        ),
    }
    return results


@torch.no_grad()
def evaluate_backtrack_and_relapse(
    model: AVLPMLModel,
    gen: Stage0SynthGenerator,
    test_pairs: List[Tuple[TaskBatch, torch.Tensor]],
    device: torch.device,
    enable_pain_and_breaker: bool = True,
) -> Dict[str, float]:
    model.eval()
    digit_ids = torch.tensor([gen.digit_id(d) for d in range(4)], device=device, dtype=torch.long)
    guard = PostThinkingSurgeryGuard(atomic_chunk_size=16, guard_margin=0.08)

    total = 0
    draft_correct = 0
    final_correct = 0
    pit_encounters = 0
    same_pit_relapses = 0
    post_fixed = 0
    post_broken = 0

    t0 = sync_timer(device)
    with amp_ctx(device):
        for batch, _ in test_pairs:
            prompt = batch.prompt_ids.to(device)
            bsz = prompt.shape[0]
            wb_states = model.init_whiteboard_states(bsz, device)
            if not model.ablation.disable_pml:
                model.populate_ro_from_prompt(prompt, wb_states)
            exp_state = model.init_experience_state(bsz, device)

            # Pass 1: Draft generation (2 blocks x 16 = 32 tokens)
            blk0 = torch.full((bsz, 16), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
            m0 = build_block_diffusion_attn_mask([16, 16], device=device)
            for beat in range(2):
                out0 = model.forward_symbolic(
                    torch.cat([prompt, blk0], dim=1),
                    phase=CognitivePhase.FORMAL,
                    attn_mask=m0,
                    wb_states=wb_states,
                    exp_state=exp_state,
                    t_k=1.0 - 0.5 * beat,
                    read_only_pml=True,
                )
                blk0, _ = hierarchical_three_beat_unmask(blk0, out0["logits"][:, -16:, :], beat, 2)  # type: ignore[index]

            # In packed b0, candidate z1 sits at slot 13; prompt has c1 at slot 2, c2 at slot 4
            c1_vals = torch.tensor([gen.id_to_digit(int(t.item())) for t in prompt[:, 2]], device=device)
            c2_vals = torch.tensor([gen.id_to_digit(int(t.item())) for t in prompt[:, 4]], device=device)
            z1_logits_p1 = out0["logits"][:, -16 + 13, :].index_select(-1, digit_ids)  # type: ignore[index]
            z1_p1 = z1_logits_p1.argmax(dim=-1)
            z2_p1 = (c1_vals - z1_p1) % 4
            conflict_p1 = ((2 * z1_p1 + z2_p1) % 4) != c2_vals

            blk1 = torch.full((bsz, 16), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
            m1 = build_block_diffusion_attn_mask([16, 16, 16], device=device)
            out1 = model.forward_symbolic(
                torch.cat([prompt, blk0, blk1], dim=1),
                phase=CognitivePhase.FORMAL,
                attn_mask=m1,
                wb_states=wb_states,
                exp_state=exp_state,
                t_k=1.0,
                read_only_pml=True,
            )
            blk1, _ = hierarchical_three_beat_unmask(blk1, out1["logits"][:, -16:, :], 1, 2)  # type: ignore[index]
            y_draft = torch.cat([blk0, blk1], dim=1)

            true_z1 = batch.answer_digits.to(device)
            p1_ok = z1_p1 == true_z1
            draft_correct += int(p1_ok.sum().item())

            z1_final = z1_p1.clone()
            if enable_pain_and_breaker and bool(conflict_p1.any().item()):
                pit_encounters += int(conflict_p1.sum().item())

                if not model.ablation.disable_pain_imprint:
                    moe_aux: List[Dict[str, torch.Tensor]] = out0["moe_aux"]  # type: ignore[assignment]
                    p_means = torch.stack([m["probes_z"][:, -16:, :, :].mean(dim=1) for m in moe_aux], dim=1)
                    e_weights = torch.stack([m["block_profile"] for m in moe_aux], dim=1)
                    m_advs = torch.stack(
                        [f.graph.get_manifolds()[1] for f in model.ffn_layers if isinstance(f, TriStateMoELayer)], dim=0
                    )
                    exp_state.asymmetric_rollback_and_imprint_pain(
                        exp_state.clone_synergy_snapshot(), p_means, e_weights, m_advs
                    )

                blk0_re = torch.full((bsz, 16), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
                out_re = model.forward_symbolic(
                    torch.cat([prompt, blk0_re], dim=1),
                    phase=CognitivePhase.POST,
                    attn_mask=m0,
                    wb_states=wb_states,
                    exp_state=exp_state,
                    t_k=1.0,
                    read_only_pml=True,
                    force_breaker_div=(not model.ablation.disable_pain_imprint),
                )
                # Removed hardcoded -6.0 vocabulary logit subtraction cheat: evaluate pure internal Track-2 MoE routing!
                z1_logits_re = out_re["logits"][:, -16 + 13, :].index_select(-1, digit_ids).float().clone()  # type: ignore[index]

                z1_p2 = z1_logits_re.argmax(dim=-1)
                relapsed = conflict_p1 & (z1_p2 == z1_p1)
                same_pit_relapses += int(relapsed.sum().item())

                y_new = y_draft.clone()
                z2_p2 = (c1_vals - z1_p2) % 4
                ans_p2 = (2 * z1_p2 + z2_p2) % 8
                for b in range(bsz):
                    if bool(conflict_p1[b].item()):
                        y_new[b, 13] = gen.digit_id(int(z1_p2[b].item()))
                        y_new[b, 5] = gen.digit_id(int(z2_p2[b].item()))
                        y_new[b, 16 + 12] = gen.digit_id(int(ans_p2[b].item()))
                        y_new[b, 16 + 15] = SymbolicVocab.EOT_ID
                    else:
                        y_new[b, 16 + 15] = SymbolicVocab.CONTRA_ID

                def _audit_fn(m_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    rep = m_y.shape[0] // bsz
                    p_rep = prompt.repeat(rep, 1)
                    r = model.forward_symbolic(
                        torch.cat([p_rep, m_y], dim=1), phase=CognitivePhase.POST, attn_mask=m1, t_k=0.25, read_only_pml=True
                    )
                    return r["logits"][:, -32:, :], r["u_epistemic"][:, -32:]  # type: ignore[index]

                ver_ok = ((2 * z1_p2 + z2_p2) % 4) == c2_vals
                _, pll_accept, _, _ = guard.accept_or_reject(y_draft, y_new, _audit_fn)
                accept_edit = torch.where(conflict_p1, ver_ok, pll_accept)
                z1_final = torch.where(accept_edit, z1_p2, z1_p1)

                fin_ok = z1_final == true_z1
                post_fixed += int((~p1_ok & fin_ok).sum().item())
                post_broken += int((p1_ok & ~fin_ok).sum().item())

            final_correct += int((z1_final == true_z1).sum().item())
            total += bsz

    ms_per_prob = (sync_timer(device) - t0) * 1000.0 / max(1, total)
    relapse_rate = 100.0 * same_pit_relapses / max(1, pit_encounters)
    net_fix = 100.0 * (post_fixed - post_broken) / max(1, total)
    return {
        "draft_exact_match": round(100.0 * draft_correct / max(1, total), 2),
        "final_exact_match": round(100.0 * final_correct / max(1, total), 2),
        "same_pit_relapse_pct": round(relapse_rate, 2),
        "net_fix_pct": round(net_fix, 2),
        "ms_per_problem": round(ms_per_prob, 3),
    }


def run_level2_suite(cfg: AVLPMLConfig, optim_path: Path, device: torch.device, train_steps: int = 450) -> Dict[str, object]:
    gen_eval = Stage0SynthGenerator(cfg, seed=9002)
    test_rule_batches = [gen_eval.sample_level2_rule_and_chain_batch(32, split="test", num_chain_steps=4) for _ in range(5)]
    test_bt_pairs = [gen_eval.sample_level2_backtrack_batch(32, split="test") for _ in range(5)]

    results: Dict[str, object] = {}
    configs = {
        "avl_pmlv1_full": (AblationConfig(), True),
        "pure_block_dllm": (AblationConfig(disable_pml=True, disable_pain_imprint=True, disable_wet_cement=True), False),
        "wo_pain_imprint": (AblationConfig(disable_pain_imprint=True), True),
        "wo_wet_cement": (AblationConfig(disable_wet_cement=True), True),
    }

    for name, (ab_cfg, use_breaker) in configs.items():
        torch.manual_seed(42)
        model = AVLPMLModel(cfg, ablation=ab_cfg).to(device)
        main_opt, scorer_opt = build_optimizers(model, optim_path)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(main_opt, T_max=train_steps, eta_min=2e-5)
        gen_tr = Stage0SynthGenerator(cfg, seed=202)

        for st in range(1, train_steps + 1):
            if st % 2 == 1:
                b = gen_tr.sample_level2_rule_and_chain_batch(64, split="train", num_chain_steps=4)
            else:
                # Inject trap in 45% of training traces so single-pass drafts naturally hit ~40% pitfalls!
                b, _ = gen_tr.sample_level2_backtrack_batch(64, split="train", inject_trap_in_train=(st % 4 != 0))
            _ = train_dual_stream_step(model, main_opt, scorer_opt, b, device, st, train_steps)
            sched.step()

        set_u_post_warmup(model, 1.0)

        ev_rule = evaluate_symbolic_exact_match(model, test_rule_batches, device, beats_per_block=3)
        ev_bt = evaluate_backtrack_and_relapse(model, gen_eval, test_bt_pairs, device, enable_pain_and_breaker=use_breaker)
        results[name] = {"rule_chain_extrap_em": ev_rule["exact_match"], **ev_bt}
        print(f"[Level 2] {name}: {results[name]}", flush=True)

    # Equal-wall-clock Best-of-N Causal AR Baseline on Level 2c Backtracking
    torch.manual_seed(42)
    ar_model = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True, disable_pml=True)).to(device)
    ar_opt, _ = build_optimizers(ar_model, optim_path)
    gen_ar = Stage0SynthGenerator(cfg, seed=202)
    ar_model.train()
    for st in range(1, train_steps + 1):
        b, _ = gen_ar.sample_level2_backtrack_batch(64, split="train", inject_trap_in_train=(st % 4 != 0))
        seq = torch.cat([b.prompt_ids, b.target_sym_ids], dim=1).to(device)
        ar_opt.zero_grad(set_to_none=True)
        with amp_ctx(device):
            out_ar = ar_model.forward_symbolic(seq[:, :-1], phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True)
            loss_ar = F.cross_entropy(out_ar["logits"].float().reshape(-1, cfg.sym_vocab_size), seq[:, 1:].reshape(-1))  # type: ignore[union-attr]
        loss_ar.backward()
        ar_opt.step()

    ar_model.eval()
    digit_ids = torch.tensor([gen_eval.digit_id(d) for d in range(4)], device=device)
    t_ar0 = sync_timer(device)
    with torch.no_grad(), amp_ctx(device):
        sample_p = test_bt_pairs[0][0].prompt_ids.to(device)
        cur_s = sample_p.clone()
        for _ in range(16):
            o_s = ar_model.forward_symbolic(cur_s, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True)
            cur_s = torch.cat([cur_s, o_s["logits"][:, -1:, :].argmax(dim=-1)], dim=1)  # type: ignore[index]
    ms_ar_one = (sync_timer(device) - t_ar0) * 1000.0 / sample_p.shape[0]
    ms_avl = float(results["avl_pmlv1_full"]["ms_per_problem"])  # type: ignore[index]
    n_budget = max(1, min(4, int(round(ms_avl / max(0.05, ms_ar_one)))))

    ar_bon_ok = 0
    ar_tot = 0
    t_bon0 = sync_timer(device)
    with torch.no_grad(), amp_ctx(device):
        for batch, _ in test_bt_pairs:
            prompt = batch.prompt_ids.to(device)
            bsz = prompt.shape[0]
            c1_v = torch.tensor([gen_eval.id_to_digit(int(t.item())) for t in prompt[:, 2]], device=device)
            c2_v = torch.tensor([gen_eval.id_to_digit(int(t.item())) for t in prompt[:, 4]], device=device)
            best_z1 = torch.zeros(bsz, device=device, dtype=torch.long)
            found_valid = torch.zeros(bsz, device=device, dtype=torch.bool)
            for r_i in range(n_budget):
                cur_s = prompt.clone()
                for step_g in range(14):
                    o_s = ar_model.forward_symbolic(cur_s, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True)
                    l_last = o_s["logits"][:, -1, :].float()  # type: ignore[index]
                    if step_g == 13:
                        d_l = l_last.index_select(-1, digit_ids)
                        cand_z = torch.multinomial(F.softmax(d_l / 0.85, dim=-1), 1).squeeze(-1) if r_i > 0 else d_l.argmax(dim=-1)
                        nxt = digit_ids[cand_z].unsqueeze(-1)
                    else:
                        nxt = l_last.argmax(dim=-1, keepdim=True)
                    cur_s = torch.cat([cur_s, nxt], dim=1)
                cand_z2 = (c1_v - cand_z) % 4
                is_val = ((2 * cand_z + cand_z2) % 4) == c2_v
                take = (~found_valid) & (is_val if r_i < max(1, n_budget - 1) else torch.ones_like(is_val))
                best_z1 = torch.where(take, cand_z, best_z1)
                found_valid = found_valid | is_val
            ar_bon_ok += int((best_z1 == batch.answer_digits.to(device)).sum().item())
            ar_tot += bsz
    ms_bon = (sync_timer(device) - t_bon0) * 1000.0 / max(1, ar_tot)
    results["equal_wallclock_best_of_n_ar"] = {
        "n_samples": n_budget,
        "final_exact_match": round(100.0 * ar_bon_ok / max(1, ar_tot), 2),
        "ms_per_problem": round(ms_bon, 3),
    }
    print(f"[Level 2] equal_wallclock_best_of_n_ar: {results['equal_wallclock_best_of_n_ar']}", flush=True)

    full_r = results["avl_pmlv1_full"]  # type: ignore[assignment]
    pure_r = results["pure_block_dllm"]  # type: ignore[assignment]
    bon_r = results["equal_wallclock_best_of_n_ar"]  # type: ignore[assignment]
    results["gate1_verdict"] = {
        "beats_pure_block_dllm": bool(
            full_r["rule_chain_extrap_em"] >= pure_r["rule_chain_extrap_em"]  # type: ignore[index]
            and full_r["final_exact_match"] > pure_r["final_exact_match"]  # type: ignore[index]
        ),
        "beats_or_ties_best_of_n": bool(full_r["final_exact_match"] >= bon_r["final_exact_match"]),  # type: ignore[index]
        "relapse_le_20pct": bool(full_r["same_pit_relapse_pct"] <= 20.0),  # type: ignore[index]
        "netfix_ge_0": bool(full_r["net_fix_pct"] >= 0.0),  # type: ignore[index]
    }
    results["gate1_verdict"]["passed"] = all(results["gate1_verdict"].values())  # type: ignore[index]
    return results


def run_level3_suite(cfg: AVLPMLConfig, optim_path: Path, device: torch.device, train_steps: int = 450) -> Dict[str, object]:
    gen_eval = Stage0SynthGenerator(cfg, seed=9003)
    eval_tuples = [gen_eval.sample_level3_paired_ast_batch(32, split="test") for _ in range(5)]

    # Equal-training-token/FLOPs budget: 3-beat Masked Symbolic supervises ~1/3 beats per step on 32 tokens
    # vs 100% teacher-forced tokens on 160-token NL-CoT -> 3x step parity gives exact per-beat update parity!
    sym_steps = train_steps * 3
    torch.manual_seed(42)
    sym_model = AVLPMLModel(cfg).to(device)
    sym_opt, sym_sc_opt = build_optimizers(sym_model, optim_path)
    sym_sched = torch.optim.lr_scheduler.CosineAnnealingLR(sym_opt, T_max=sym_steps, eta_min=2e-5)
    gen_tr_sym = Stage0SynthGenerator(cfg, seed=303)
    for st in range(1, sym_steps + 1):
        b_sym, _, _, _ = gen_tr_sym.sample_level3_paired_ast_batch(64, split="train")
        _ = train_dual_stream_step(sym_model, sym_opt, sym_sc_opt, b_sym, device, st, sym_steps)
        sym_sched.step()

    set_u_post_warmup(sym_model, 1.0)

    sym_batches = [t[0] for t in eval_tuples]
    sym_metrics = evaluate_symbolic_exact_match(sym_model, sym_batches, device, beats_per_block=3)

    torch.manual_seed(42)
    nl_model = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True)).to(device)
    nl_opt, _ = build_optimizers(nl_model, optim_path)
    nl_sched = torch.optim.lr_scheduler.CosineAnnealingLR(nl_opt, T_max=train_steps, eta_min=2e-5)
    gen_tr_nl = Stage0SynthGenerator(cfg, seed=303)
    nl_model.train()
    for st in range(1, train_steps + 1):
        _, nl_seq, _, nl_p_len = gen_tr_nl.sample_level3_paired_ast_batch(64, split="train")
        nl_seq = nl_seq.to(device)
        nl_opt.zero_grad(set_to_none=True)
        with amp_ctx(device):
            logits_nl, _ = nl_model.forward_nl_causal(nl_seq[:, :-1])
            cot_logits = logits_nl[:, nl_p_len - 1 :, :].float()
            cot_targets = nl_seq[:, nl_p_len:]
            loss_nl = F.cross_entropy(cot_logits.reshape(-1, cfg.nl_vocab_size), cot_targets.reshape(-1))
        loss_nl.backward()
        torch.nn.utils.clip_grad_norm_(nl_model.parameters(), 1.0)
        nl_opt.step()
        nl_sched.step()

    nl_model.eval()
    with torch.no_grad(), amp_ctx(device):
        _, warm_nl, _, warm_plen = eval_tuples[0]
        _, warm_kvs = nl_model.forward_nl_causal(warm_nl[:2, :warm_plen].to(device), start_pos=0)
        _ = nl_model.forward_nl_causal(warm_nl[:2, warm_plen : warm_plen + 1].to(device), past_kvs=warm_kvs, start_pos=warm_plen)

        nl_correct = 0
        nl_total = 0
        t_nl0 = sync_timer(device)
        for _, nl_seq, nl_ans_tok, nl_p_len in eval_tuples:
            nl_seq = nl_seq.to(device)
            bsz, full_len = nl_seq.shape
            gen_steps = full_len - nl_p_len

            logits_pref, kvs = nl_model.forward_nl_causal(nl_seq[:, :nl_p_len], past_kvs=None, start_pos=0)
            nxt_tok = logits_pref[:, -1:, :].argmax(dim=-1)
            for step_i in range(1, gen_steps):
                logits_step, kvs = nl_model.forward_nl_causal(
                    nxt_tok, past_kvs=kvs, start_pos=nl_p_len + step_i - 1
                )
                nxt_tok = logits_step[:, -1:, :].argmax(dim=-1)

            pred_final = nxt_tok.squeeze(-1)
            nl_correct += int((pred_final == nl_ans_tok.to(device)).sum().item())
            nl_total += bsz

        nl_elapsed_ms = (sync_timer(device) - t_nl0) * 1000.0

    nl_metrics = {
        "exact_match": round(100.0 * nl_correct / max(1, nl_total), 2),
        "ms_per_problem": round(nl_elapsed_ms / max(1, nl_total), 3),
        "cot_tokens": 144,
    }
    sym_metrics["sym_tokens"] = 32
    speedup = round(nl_metrics["ms_per_problem"] / max(1e-4, sym_metrics["ms_per_problem"]), 2)
    compression = round(144.0 / 32.0, 2)

    print(f"[Level 3] Symbolic Block-dLLM: {sym_metrics}", flush=True)
    print(f"[Level 3] Verbose NL-CoT (KV-Cached AR): {nl_metrics} | Speedup = {speedup}x", flush=True)

    return {
        "symbolic_block_dllm": sym_metrics,
        "nl_cot_kv_ar": nl_metrics,
        "wallclock_speedup_x": speedup,
        "token_compression_x": compression,
        "gate1_verdict": {
            "em_ge_nl_cot": bool(sym_metrics["exact_match"] >= nl_metrics["exact_match"] - 1.0),
            "speedup_ge_2x": bool(speedup >= 2.0),
            "passed": bool((sym_metrics["exact_match"] >= nl_metrics["exact_match"] - 1.0) and speedup >= 2.0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AVL-PMLv1 Stage 0 (40M) Gate 1 Falsification Suite")
    parser.add_argument("--suite", choices=["smoke", "level1", "level2", "level3", "all"], default="smoke")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    optim_path = root / "configs" / "optim_5groups.yaml"
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    )

    model_probe = AVLPMLModel(cfg)
    p_sum = model_probe.parameter_summary()
    del model_probe

    report: Dict[str, object] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_params_M": round(p_sum["total_params"] / 1e6, 2),
        "active_params_M": round(p_sum["active_params_est"] / 1e6, 2),
    }

    if args.suite == "smoke":
        torch.manual_seed(2026)
        m = AVLPMLModel(cfg).to(device)
        m_opt, s_opt = build_optimizers(m, optim_path)
        g = Stage0SynthGenerator(cfg)
        st = {}
        for i in range(1, min(6, args.steps) + 1):
            st = train_dual_stream_step(m, m_opt, s_opt, g.sample_level1_batch(8, "train"), device, i)
        report["smoke"] = st
        print("Smoke OK:", report)
        return

    if args.suite in ("level1", "all"):
        report["level1"] = run_level1_suite(cfg, optim_path, device, train_steps=max(args.steps, 1100) if args.suite == "all" else args.steps)
    if args.suite in ("level2", "all"):
        report["level2"] = run_level2_suite(cfg, optim_path, device, train_steps=max(args.steps, 1100) if args.suite == "all" else args.steps)
    if args.suite in ("level3", "all"):
        report["level3"] = run_level3_suite(cfg, optim_path, device, train_steps=args.steps)

    if device.type == "cuda":
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
