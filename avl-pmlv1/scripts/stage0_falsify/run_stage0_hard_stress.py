"""Stage 0-Hard Zero-Cheat Stress Test Suite (RTX 5080, 0 RMB, Strict VRAM Isolation < 14.0 GB):

Stress Test 1 — Long-Chain 24-Step x 8-Register Interleaved ALU (mod 16, 304 toks, 1-Step Diff S^RW):
  - Directly tests Track-1 PML Whiteboard (S^RW + S^RO) vs Pure Block-dLLM (No PML) vs Mainstream Causal AR!
  - Reports End-to-End 24-step EM, All-Step Memory Read Acc, Long-Gap (>=8 steps) Read Acc, & K_max=0 drop.

Stress Test 2 — Real 4x4 Sudoku Multi-Branch Backtracking (20% Unseen Solution Boards, 4,096-16,384 space):
  - ZERO explicit logit-subtraction cheat! Tests native Track-2 Pain Imprint (p_neg + M_adv) + 16-Slot Wet-Cement
    vs wo_pain_imprint (identical retry budget!) vs Pure Block-dLLM vs Equal-Wall-Clock Best-of-N Causal AR!

Stress Test 3 — Variable-Length Autonomous <EOT> Halting (L in [2..8] train, [9..10] zero-shot extrap, mod 16):
  - No fixed block count or answer index! Model must stop autonomously at exact L AND get exact mod-16 result.
  - Compares Autonomous Prefix-KV Block-dLLM vs Mainstream Causal KV-AR (+ K_max=0 knock-out & Canary).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F

from avl_pmlv1.config import AVLPMLConfig
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
from gen_synth_tasks import _is_test_instance
from run_gate1_ablation import (
    amp_ctx,
    build_optimizers,
    build_warmup_cosine_scheduler,
    set_u_post_warmup,
    sync_timer,
)
from run_v11_verification import check_vram_guard, free_vram, save_checkpoint


# =============================================================================
# Helper Token Encoders (mod 16 values + mod 8 constants in Escape domain)
# =============================================================================
def val16_tok(vocab: SymbolicVocab, v: int) -> int:
    return vocab.escape_start + 64 + (int(v) % 16)


def const8_tok(vocab: SymbolicVocab, k: int) -> int:
    return vocab.escape_start + 80 + (int(k) % 8)


def apply_op16(op_code: int, a: int, b: int) -> int:
    if op_code == 0:
        return (a + b) % 16
    if op_code == 1:
        return (a - b) % 16
    if op_code == 2:
        return (a ^ b) % 16
    return (3 * a + b + 1) % 16


# =============================================================================
# STRESS TEST 1: 24-Step x 8-Register Interleaved ALU (mod 16, 304 tokens)
# =============================================================================
@dataclass
class HardALU24Batch:
    prompt_ids: torch.Tensor      # [B, 112] (16 init S^RO + 24*4 instructions)
    target_sym_ids: torch.Tensor  # [B, 192] (24 x 8-slot micro-blocks)
    rw_gaps: torch.Tensor         # [B, 24] steps since src_reg was last written
    final_ans_tok: torch.Tensor   # [B] token ID of step-23 new_val (slot 5 of block 23 = idx 189)


def sample_hard_alu24_batch(
    vocab: SymbolicVocab, rng: random.Random, batch_size: int, split: str = "train", num_steps: int = 24
) -> HardALU24Batch:
    want_test = split == "test"
    op_toks = [vocab.token_to_id["+"], vocab.token_to_id["-"], vocab.token_to_id["=="]]
    prompts, syms, gaps, finals = [], [], [], []

    while len(prompts) < batch_size:
        init_regs = [rng.randint(0, 15) for _ in range(8)]
        insts = [(rng.randint(0, 7), rng.randint(0, 7), rng.randint(0, 2), rng.randint(1, 7)) for _ in range(num_steps)]
        key = f"ALU24:{init_regs[:4]}:{insts[:6]}"
        if _is_test_instance(key, test_ratio=0.20) != want_test:
            continue

        p_seq: List[int] = []
        for r_i in range(8):
            p_seq.extend([vocab.in_slot_to_id(r_i), val16_tok(vocab, init_regs[r_i])])
        for dst, src, opc, kv in insts:
            p_seq.extend([vocab.reg_slot_to_id(dst), vocab.reg_slot_to_id(src), op_toks[opc], const8_tok(vocab, kv)])

        regs = list(init_regs)
        last_w = [-1] * 8
        sym_seq: List[int] = []
        step_gaps: List[int] = []
        last_new = 0
        for t, (dst, src, opc, kv) in enumerate(insts):
            old_v = regs[src]
            new_v = apply_op16(opc, old_v, kv)
            step_gaps.append(t - last_w[src])
            regs[dst] = new_v
            last_w[dst] = t
            last_new = new_v
            # 8-slot x 3-beat (1 causal hop per beat):
            # Beat 0 [0..3]: [op_tok, REG_src, const_k, REG_dst]
            # Beat 1 [4]:    [val16(old_v)]  <-- LONG-RANGE INTERFERENCE MEMORY READ!
            # Beat 2 [5..7]: [val16(new_v), BIND, REG_dst]
            sym_seq.extend([
                op_toks[opc], vocab.reg_slot_to_id(src), const8_tok(vocab, kv), vocab.reg_slot_to_id(dst),
                val16_tok(vocab, old_v),
                val16_tok(vocab, new_v), SymbolicVocab.BIND_ID, vocab.reg_slot_to_id(dst),
            ])

        prompts.append(p_seq)
        syms.append(sym_seq)
        gaps.append(step_gaps)
        finals.append(val16_tok(vocab, last_new))

    return HardALU24Batch(
        prompt_ids=torch.tensor(prompts, dtype=torch.long),
        target_sym_ids=torch.tensor(syms, dtype=torch.long),
        rw_gaps=torch.tensor(gaps, dtype=torch.long),
        final_ans_tok=torch.tensor(finals, dtype=torch.long),
    )


def unmask_8slot_3beat_alu(cur_blk: torch.Tensor, logits: torch.Tensor, beat: int) -> torch.Tensor:
    """Exact 3-beat unmasking for 8-slot ALU: Beat 0 -> [0..3], Beat 1 -> [4], Beat 2 -> [5..7]."""
    masked = logits.clone()
    masked[..., SymbolicVocab.MASK_ID] = float("-inf")
    masked[..., SymbolicVocab.PAD_ID] = float("-inf")
    preds = masked.argmax(dim=-1)
    is_m = cur_blk == SymbolicVocab.MASK_ID
    pos = torch.arange(8, device=cur_blk.device).unsqueeze(0)
    if beat == 0:
        open_m = is_m & (pos < 4)
    elif beat == 1:
        open_m = is_m & (pos == 4)
    else:
        open_m = is_m
    return torch.where(open_m, preds, cur_blk)


def train_hard_alu24_dllm_step(
    model: AVLPMLModel,
    opt: torch.optim.AdamW,
    batch: HardALU24Batch,
    device: torch.device,
    step_idx: int,
    total_steps: int,
) -> None:
    model.train()
    set_u_post_warmup(model, step_idx / float(max(1, total_steps)))
    prompt = batch.prompt_ids.to(device)
    sym_clean = batch.target_sym_ids.to(device)
    bsz, p_len = prompt.shape
    s_len = sym_clean.shape[1]
    num_blocks = s_len // 8

    local_pos = (torch.arange(s_len, device=device) % 8).unsqueeze(0).expand(bsz, -1)
    bm = step_idx % 3
    if bm == 0:
        q_rate, mask_sym = 1.0, torch.ones_like(sym_clean, dtype=torch.bool)
    elif bm == 1:
        q_rate, mask_sym = 0.50, local_pos >= 4
    else:
        q_rate, mask_sym = 0.375, local_pos >= 5

    noisy_sym = torch.where(mask_sym, torch.full_like(sym_clean, SymbolicVocab.MASK_ID), sym_clean)
    dual_in = torch.cat([prompt, sym_clean, noisy_sym], dim=1)
    dual_tgt = torch.cat([prompt, sym_clean, sym_clean], dim=1)
    dual_lmask = torch.cat([torch.zeros(bsz, p_len + s_len, device=device, dtype=torch.bool), mask_sym], dim=1)
    attn_mask, rope_pos = build_dual_stream_bd3_mask(p_len, num_blocks, block_size=8, enable_wet_seam=False, device=device)
    rope_pos = rope_pos.expand(bsz, -1)

    opt.zero_grad(set_to_none=True)
    with amp_ctx(device):
        wb = model.init_whiteboard_states(bsz, device)
        if not model.ablation.disable_pml:
            model.populate_ro_from_prompt(prompt, wb)
        out = model.forward_symbolic(
            dual_in, phase=CognitivePhase.FORMAL, positions=rope_pos, attn_mask=attn_mask,
            wb_states=wb, exp_state=None, t_k=q_rate, dual_stream_sym_blocks=num_blocks,
            dual_stream_block_size=8, return_soft_sketch=False,
        )
        loss, _ = model.compute_six_losses(out, dual_tgt, dual_lmask, collect_stats=False)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


@torch.no_grad()
def evaluate_hard_alu24_dllm(
    model: AVLPMLModel, batches: List[HardALU24Batch], device: torch.device
) -> Dict[str, float]:
    """Zero-leakage Prefix-KV 3-beat evaluation over 24-step x 8-register programs."""
    model.eval()
    final_ok = read_ok = long_gap_ok = long_gap_tot = alu_ok = inst_ok = 0
    tot_probs = tot_steps = 0
    act_accum = 0.0
    n_fwds = 0

    t0 = sync_timer(device)
    with amp_ctx(device):
        for batch in batches:
            prompt = batch.prompt_ids.to(device)
            tgt = batch.target_sym_ids.to(device)
            gaps = batch.rw_gaps.to(device)
            bsz, p_len = prompt.shape
            num_blocks = tgt.shape[1] // 8

            wb = model.init_whiteboard_states(bsz, device)
            if not model.ablation.disable_pml:
                model.populate_ro_from_prompt(prompt, wb)
            exp_state = model.init_experience_state(bsz, device)

            m_pref = torch.ones(1, 1, p_len, p_len, device=device, dtype=torch.bool)
            out_p = model.forward_symbolic(
                prompt, phase=CognitivePhase.FORMAL, attn_mask=m_pref, wb_states=wb, exp_state=exp_state,
                t_k=0.0, read_only_pml=True, dual_stream_block_size=8, past_kvs=None, start_pos=0, return_soft_sketch=False,
            )
            h_p = out_p["hidden"].mean(dim=1)  # type: ignore[union-attr]
            kvs = out_p["present_kvs"]  # type: ignore[assignment]

            gen_blocks: List[torch.Tensor] = []
            for b_i in range(num_blocks):
                cur = torch.full((bsz, 8), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
                s_pos = p_len + b_i * 8
                for beat, t_k in [(0, 1.0), (1, 0.50), (2, 0.375)]:
                    out = model.forward_symbolic(
                        cur, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb, exp_state=exp_state,
                        t_k=t_k, read_only_pml=True, dual_stream_block_size=8, past_kvs=kvs, start_pos=s_pos, return_soft_sketch=False,
                    )
                    if b_i == 0 and beat == 0 and not model.ablation.static_top2_baseline:
                        exp_state.delta_b_pre = model.hyper_router(h_p, out["hidden"].mean(dim=1))  # type: ignore[union-attr]
                    moe_aux: List[Dict[str, torch.Tensor]] = out["moe_aux"]  # type: ignore[assignment]
                    if moe_aux:
                        act_accum += float(torch.stack([m["active_count"].mean() for m in moe_aux]).mean().item())
                        n_fwds += 1
                    cur = unmask_8slot_3beat_alu(cur, out["logits"], beat)  # type: ignore[arg-type]

                gen_blocks.append(cur)
                t_blk = tgt[:, b_i * 8 : (b_i + 1) * 8]
                inst_ok += int((cur[:, :4] == t_blk[:, :4]).all(dim=-1).sum().item())
                r_match = cur[:, 4] == t_blk[:, 4]
                read_ok += int(r_match.sum().item())
                is_lg = gaps[:, b_i] >= 8
                long_gap_ok += int((r_match & is_lg).sum().item())
                long_gap_tot += int(is_lg.sum().item())
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

    ms_p = (sync_timer(device) - t0) * 1000.0 / max(1, tot_probs)
    return {
        "final_step24_em": round(100.0 * final_ok / max(1, tot_probs), 2),
        "all_steps_read_acc": round(100.0 * read_ok / max(1, tot_steps), 2),
        "long_gap_ge8_read_acc": round(100.0 * long_gap_ok / max(1, long_gap_tot), 2),
        "all_steps_alu_acc": round(100.0 * alu_ok / max(1, tot_steps), 2),
        "inst_copy_acc": round(100.0 * inst_ok / max(1, tot_steps), 2),
        "avg_active_experts": round(act_accum / max(1, n_fwds), 3),
        "ms_per_problem": round(ms_p, 2),
    }


def run_stress1_alu24(cfg: AVLPMLConfig, optim_path: Path, device: torch.device, steps_dllm: int = 1200, steps_ar: int = 600) -> Dict[str, Any]:
    vocab = SymbolicVocab(cfg)
    rng_ev = random.Random(9101)
    test_batches = [sample_hard_alu24_batch(vocab, rng_ev, 32, "test", 24) for _ in range(5)]  # 160 programs x 24 steps = 3,840 ops!
    res: Dict[str, Any] = {}

    for label, ab in [("v11_full_diff_pml", AblationConfig()), ("pure_block_dllm_no_pml", AblationConfig(disable_pml=True))]:
        torch.manual_seed(42)
        m = AVLPMLModel(cfg, ablation=ab).to(device)
        opt, _ = build_optimizers(m, optim_path, stabilized_v11=True)
        sched = build_warmup_cosine_scheduler(opt, total_steps=steps_dllm, warmup_ratio=0.10, min_lr_ratio=0.03)
        rng_tr = random.Random(4101)
        t0 = sync_timer(device)
        for st in range(1, steps_dllm + 1):
            b = sample_hard_alu24_batch(vocab, rng_tr, 32, "train", 24)
            train_hard_alu24_dllm_step(m, opt, b, device, st, steps_dllm)
            sched.step()
        tr_s = round(sync_timer(device) - t0, 1)
        del opt, sched
        set_u_post_warmup(m, 1.0)
        ev = evaluate_hard_alu24_dllm(m, test_batches, device)
        orig_k = m.cfg.k_max
        m.cfg.k_max = 0
        ev_k0 = evaluate_hard_alu24_dllm(m, test_batches, device)
        m.cfg.k_max = orig_k
        ev["k0_knockout_em"] = ev_k0["final_step24_em"]
        ev["k0_knockout_drop"] = round(ev["final_step24_em"] - ev_k0["final_step24_em"], 2)
        ev["train_sec"] = tr_s
        ev["vram"] = check_vram_guard(f"S1_{label}")
        if not ab.disable_pml:
            ev["learned_alpha_calm_rms"] = round(
                float(torch.stack([l.alpha_calm.detach().float().pow(2).mean().sqrt() for l in m.attn_layers if l.enable_pml]).mean().item()),
                4,
            )
        res[label] = ev
        print(f"[Stress1 ALU24 {label}] EM={ev['final_step24_em']}% | read={ev['all_steps_read_acc']}% | gap>=8={ev['long_gap_ge8_read_acc']}% | alu={ev['all_steps_alu_acc']}% | K0_drop={ev['k0_knockout_drop']}% | {ev['ms_per_problem']}ms | {tr_s}s", flush=True)
        del m
        free_vram()

    # Mainstream Causal AR Baseline (Static Top-2 KV-Cache on EXACT same 304-token sequence, 600 dense steps = 1.5x per-token updates vs 3-beat dLLM!)
    torch.manual_seed(42)
    ar_m = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True, disable_pml=True)).to(device)
    opt_ar, _ = build_optimizers(ar_m, optim_path, stabilized_v11=True)
    sched_ar = build_warmup_cosine_scheduler(opt_ar, total_steps=steps_ar, warmup_ratio=0.10, min_lr_ratio=0.03)
    rng_ar = random.Random(4101)
    ar_m.train()
    t0 = sync_timer(device)
    for _ in range(steps_ar):
        b = sample_hard_alu24_batch(vocab, rng_ar, 32, "train", 24)
        seq = torch.cat([b.prompt_ids, b.target_sym_ids], dim=1).to(device)
        p_len = b.prompt_ids.shape[1]
        opt_ar.zero_grad(set_to_none=True)
        with amp_ctx(device):
            out_ar = ar_m.forward_symbolic(seq[:, :-1], phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, return_soft_sketch=False)
            lg = out_ar["logits"][:, p_len - 1 :, :].float()  # type: ignore[index]
            loss_ar = F.cross_entropy(lg.reshape(-1, cfg.sym_vocab_size), seq[:, p_len:].reshape(-1))
        loss_ar.backward()
        torch.nn.utils.clip_grad_norm_(ar_m.parameters(), 1.0)
        opt_ar.step()
        sched_ar.step()
    tr_ar = round(sync_timer(device) - t0, 1)
    del opt_ar, sched_ar

    ar_m.eval()
    ar_fin = ar_read = ar_lg_ok = ar_lg_tot = ar_alu = ar_probs = ar_steps = 0
    t_ev0 = sync_timer(device)
    with torch.no_grad(), amp_ctx(device):
        for batch in test_batches:
            prompt = batch.prompt_ids.to(device)
            tgt = batch.target_sym_ids.to(device)
            gaps = batch.rw_gaps.to(device)
            bsz, p_len = prompt.shape
            out_p = ar_m.forward_symbolic(prompt, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, past_kvs=None, start_pos=0, return_soft_sketch=False)
            kvs = out_p["present_kvs"]  # type: ignore[assignment]
            nxt = out_p["logits"][:, -1:, :].argmax(dim=-1)  # type: ignore[index]
            gen_list = [nxt]
            for s_i in range(1, tgt.shape[1]):
                out_s = ar_m.forward_symbolic(nxt, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, past_kvs=kvs, start_pos=p_len + s_i - 1, return_soft_sketch=False)
                kvs = out_s["present_kvs"]  # type: ignore[assignment]
                nxt = out_s["logits"][:, -1:, :].argmax(dim=-1)  # type: ignore[index]
                gen_list.append(nxt)
            gen_full = torch.cat(gen_list, dim=1)
            ar_fin += int((gen_full[:, -3] == batch.final_ans_tok.to(device)).sum().item())
            for b_i in range(24):
                r_m = gen_full[:, b_i * 8 + 4] == tgt[:, b_i * 8 + 4]
                ar_read += int(r_m.sum().item())
                is_lg = gaps[:, b_i] >= 8
                ar_lg_ok += int((r_m & is_lg).sum().item())
                ar_lg_tot += int(is_lg.sum().item())
                ar_alu += int((gen_full[:, b_i * 8 + 5] == tgt[:, b_i * 8 + 5]).sum().item())
                ar_steps += bsz
            ar_probs += bsz
    ms_ar = (sync_timer(device) - t_ev0) * 1000.0 / max(1, ar_probs)
    res["mainstream_causal_kv_ar"] = {
        "final_step24_em": round(100.0 * ar_fin / max(1, ar_probs), 2),
        "all_steps_read_acc": round(100.0 * ar_read / max(1, ar_steps), 2),
        "long_gap_ge8_read_acc": round(100.0 * ar_lg_ok / max(1, ar_lg_tot), 2),
        "all_steps_alu_acc": round(100.0 * ar_alu / max(1, ar_steps), 2),
        "ms_per_problem": round(ms_ar, 2),
        "train_sec": tr_ar,
        "vram": check_vram_guard("S1_causal_ar"),
    }
    print(f"[Stress1 ALU24 causal_kv_ar] EM={res['mainstream_causal_kv_ar']['final_step24_em']}% | read={res['mainstream_causal_kv_ar']['all_steps_read_acc']}% | gap>=8={res['mainstream_causal_kv_ar']['long_gap_ge8_read_acc']}% | {ms_ar:.2f}ms | {tr_ar}s", flush=True)
    del ar_m
    free_vram()
    res["pml_gain_over_pure_dllm_em"] = round(res["v11_full_diff_pml"]["final_step24_em"] - res["pure_block_dllm_no_pml"]["final_step24_em"], 2)
    res["pml_gain_long_gap_read"] = round(res["v11_full_diff_pml"]["long_gap_ge8_read_acc"] - res["pure_block_dllm_no_pml"]["long_gap_ge8_read_acc"], 2)
    return res


# =============================================================================
# STRESS TEST 2: Real 4x4 Sudoku (20% Unseen Boards, 6-7 Blanks, ZERO Logit Cheat!)
# =============================================================================
def enumerate_all_4x4_sudoku_boards() -> List[Tuple[int, ...]]:
    boards: List[Tuple[int, ...]] = []
    grid = [0] * 16

    def _bt(idx: int) -> None:
        if idx == 16:
            boards.append(tuple(grid))
            return
        r, c = divmod(idx, 4)
        br, bc = (r // 2) * 2, (c // 2) * 2
        used = set()
        for j in range(4):
            used.add(grid[r * 4 + j])
            used.add(grid[j * 4 + c])
        for dr in range(2):
            for dc in range(2):
                used.add(grid[(br + dr) * 4 + (bc + dc)])
        for d in (1, 2, 3, 4):
            if d not in used:
                grid[idx] = d
                _bt(idx + 1)
                grid[idx] = 0

    _bt(0)
    return boards


def count_4x4_solutions(puzzle: List[int], limit: int = 2) -> int:
    grid = list(puzzle)
    sols = 0

    def _bt(idx: int) -> None:
        nonlocal sols
        if sols >= limit:
            return
        while idx < 16 and grid[idx] != 0:
            idx += 1
        if idx == 16:
            sols += 1
            return
        r, c = divmod(idx, 4)
        br, bc = (r // 2) * 2, (c // 2) * 2
        used = set()
        for j in range(4):
            used.add(grid[r * 4 + j])
            used.add(grid[j * 4 + c])
        for dr in range(2):
            for dc in range(2):
                used.add(grid[(br + dr) * 4 + (bc + dc)])
        for d in (1, 2, 3, 4):
            if d not in used:
                grid[idx] = d
                _bt(idx + 1)
                grid[idx] = 0

    _bt(0)
    return sols


def is_valid_4x4_sudoku(sol: List[int], clues: List[int]) -> bool:
    for i in range(16):
        if sol[i] not in (1, 2, 3, 4):
            return False
        if clues[i] != 0 and sol[i] != clues[i]:
            return False
    for r in range(4):
        if len(set(sol[r * 4 : r * 4 + 4])) != 4:
            return False
    for c in range(4):
        if len({sol[r * 4 + c] for r in range(4)}) != 4:
            return False
    for br in (0, 2):
        for bc in (0, 2):
            if len({sol[(br + dr) * 4 + (bc + dc)] for dr in range(2) for dc in range(2)}) != 4:
                return False
    return True


@dataclass
class Sudoku4x4Batch:
    prompt_ids: torch.Tensor      # [B, 16] clues (digit 1..4 or sep_id for blank)
    target_sym_ids: torch.Tensor  # [B, 32] Block 0: 16-cell board; Block 1: verify frame + EOT/CONTRA
    clues_raw: List[List[int]]
    sol_raw: List[List[int]]


def sample_sudoku4x4_batch(
    vocab: SymbolicVocab,
    boards: List[Tuple[int, ...]],
    rng: random.Random,
    batch_size: int,
    inject_trap_prob: float = 0.0,
) -> Sudoku4x4Batch:
    sep_id = vocab.token_to_id[":"]
    eq_id = vocab.token_to_id["="]
    d_tok = {d: vocab.token_to_id[str(d)] for d in (1, 2, 3, 4)}

    prompts, syms, clues_l, sols_l = [], [], [], []
    while len(prompts) < batch_size:
        board = list(rng.choice(boards))
        n_blanks = rng.choice([6, 7])
        blank_idxs = rng.sample(range(16), n_blanks)
        clues = [0 if i in blank_idxs else board[i] for i in range(16)]
        if count_4x4_solutions(clues, limit=2) != 1:
            continue

        is_trap = rng.random() < inject_trap_prob
        draft_board = list(board)
        if is_trap:
            # Perturb 1-2 blank cells into plausible wrong digits to train genuine conflict detection & wet-cement!
            k_bad = rng.choice([1, 2])
            for bi in rng.sample(blank_idxs, k_bad):
                wrong_choices = [d for d in (1, 2, 3, 4) if d != draft_board[bi]]
                draft_board[bi] = rng.choice(wrong_choices)

        p_seq = [d_tok[c] if c != 0 else sep_id for c in clues]
        blk0 = [d_tok[x] for x in draft_board]
        # Block 1 verifies row/col/box validity and emits EOT vs CONTRA
        blk1 = [SymbolicVocab.VERIFY_ID] + [d_tok[x] for x in board[:12]] + [
            SymbolicVocab.BIND_ID, vocab.reg_slot_to_id(0),
            SymbolicVocab.CONTRA_ID if is_trap else SymbolicVocab.EOT_ID,
        ]
        prompts.append(p_seq)
        syms.append(blk0 + blk1)
        clues_l.append(clues)
        sols_l.append(board)

    return Sudoku4x4Batch(
        prompt_ids=torch.tensor(prompts, dtype=torch.long),
        target_sym_ids=torch.tensor(syms, dtype=torch.long),
        clues_raw=clues_l,
        sol_raw=sols_l,
    )


def train_sudoku_dllm_step(
    model: AVLPMLModel,
    opt: torch.optim.AdamW,
    batch: Sudoku4x4Batch,
    device: torch.device,
    step_idx: int,
    total_steps: int,
) -> None:
    model.train()
    set_u_post_warmup(model, step_idx / float(max(1, total_steps)))
    prompt = batch.prompt_ids.to(device)
    sym_clean = batch.target_sym_ids.to(device)
    bsz, p_len = prompt.shape
    s_len = sym_clean.shape[1]

    # Random continuous mask rate in [0.25, 1.0] so model can denoise from BOTH 100% [MASK] AND partial Wet-Cement remasks!
    bm = step_idx % 4
    if bm == 0:
        q_rate = 1.0
        mask_sym = torch.ones_like(sym_clean, dtype=torch.bool)
    elif bm == 1:
        q_rate = 0.60
        mask_sym = torch.rand(bsz, s_len, device=device) < 0.60
    elif bm == 2:
        q_rate = 0.35
        mask_sym = torch.rand(bsz, s_len, device=device) < 0.35
    else:
        q_rate = 0.25
        mask_sym = torch.rand(bsz, s_len, device=device) < 0.25
        mask_sym[:, -1] = True  # always supervise EOT/CONTRA

    noisy_sym = torch.where(mask_sym, torch.full_like(sym_clean, SymbolicVocab.MASK_ID), sym_clean)
    dual_in = torch.cat([prompt, sym_clean, noisy_sym], dim=1)
    dual_tgt = torch.cat([prompt, sym_clean, sym_clean], dim=1)
    dual_lmask = torch.cat([torch.zeros(bsz, p_len + s_len, device=device, dtype=torch.bool), mask_sym], dim=1)
    wet_on = (not model.ablation.disable_wet_cement) and (step_idx % 3 == 0)
    attn_mask, rope_pos = build_dual_stream_bd3_mask(p_len, 2, block_size=16, enable_wet_seam=wet_on, device=device)
    rope_pos = rope_pos.expand(bsz, -1)

    opt.zero_grad(set_to_none=True)
    with amp_ctx(device):
        wb = model.init_whiteboard_states(bsz, device)
        if not model.ablation.disable_pml:
            model.populate_ro_from_prompt(prompt, wb)
        out = model.forward_symbolic(
            dual_in, phase=CognitivePhase.FORMAL, positions=rope_pos, attn_mask=attn_mask,
            wb_states=wb, exp_state=None, t_k=q_rate, dual_stream_sym_blocks=2,
            dual_stream_block_size=16, return_soft_sketch=False,
        )
        loss, _ = model.compute_six_losses(out, dual_tgt, dual_lmask, collect_stats=False)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


@torch.no_grad()
def evaluate_sudoku_zero_cheat(
    model: AVLPMLModel,
    vocab: SymbolicVocab,
    test_batches: List[Sudoku4x4Batch],
    device: torch.device,
    enable_retries: bool = True,
    max_retries: int = 3,
    gumbel_temp: float = 0.0,
) -> Dict[str, float]:
    """100% ZERO-CHEAT 4x4 Sudoku evaluation (NO explicit logit subtraction!).
    Tests PURE native Track-2 Pain Imprint (p_neg + M_adv) + 16-Slot Wet-Cement remask!
    """
    model.eval()
    d_ids = torch.tensor([vocab.token_to_id[str(d)] for d in (1, 2, 3, 4)], device=device, dtype=torch.long)
    id_to_d = {int(vocab.token_to_id[str(d)]): d for d in (1, 2, 3, 4)}
    m32 = build_block_diffusion_attn_mask([16, 16], device=device)

    draft_ok = final_ok = tot = 0
    retry_attempts = exact_board_relapses = 0
    t0 = sync_timer(device)

    with amp_ctx(device):
        for batch in test_batches:
            prompt = batch.prompt_ids.to(device)
            bsz = prompt.shape[0]
            wb = model.init_whiteboard_states(bsz, device)
            if not model.ablation.disable_pml:
                model.populate_ro_from_prompt(prompt, wb)
            exp_state = model.init_experience_state(bsz, device)

            # Pass 1: Draft from 100% [MASK] over 3 margin-ordered beats (ZERO clue clamping!)
            cur = torch.full((bsz, 16), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
            last_out = None
            margins = torch.zeros(bsz, 16, device=device)
            for beat in range(3):
                out = model.forward_symbolic(
                    torch.cat([prompt, cur], dim=1), phase=CognitivePhase.FORMAL, attn_mask=m32,
                    wb_states=wb, exp_state=exp_state, t_k=1.0 - 0.35 * beat, read_only_pml=True, return_soft_sketch=False,
                )
                last_out = out
                logits_b = out["logits"][:, -16:, :]  # type: ignore[index]
                # Margin-based parallel unmasking (5, 6, 5 tokens across 3 beats)
                d_logits = logits_b.index_select(-1, d_ids).float()
                probs = F.softmax(d_logits, dim=-1)
                top2_p, top2_i = torch.topk(probs, k=2, dim=-1)
                margins = top2_p[..., 0] - top2_p[..., 1]
                pred_tok = d_ids[top2_i[..., 0]]
                is_m = cur == SymbolicVocab.MASK_ID
                if beat == 2:
                    cur = torch.where(is_m, pred_tok, cur)
                else:
                    sc = torch.where(is_m, margins, torch.full_like(margins, -1e9))
                    order = torch.argsort(sc, dim=-1, descending=True)
                    ranks = torch.argsort(order, dim=-1)
                    cur = torch.where(is_m & (ranks < (6 if beat == 0 else 5)), pred_tok, cur)

            def _decode_and_check(blk: torch.Tensor) -> Tuple[List[List[int]], List[bool]]:
                cpu_b = blk.cpu().tolist()
                grids, valids = [], []
                for i in range(bsz):
                    g = [id_to_d.get(tok, 0) for tok in cpu_b[i]]
                    grids.append(g)
                    valids.append(is_valid_4x4_sudoku(g, batch.clues_raw[i]))
                return grids, valids

            prev_grids, valid_flags = _decode_and_check(cur)
            draft_ok += sum(valid_flags)
            best_valid = list(valid_flags)
            cur_margins = margins.clone()
            cur_blk = cur.clone()

            if enable_retries and not all(best_valid):
                m_advs = torch.stack(
                    [f.graph.get_manifolds()[1] for f in model.ffn_layers if isinstance(f, TriStateMoELayer)], dim=0
                )
                for r_step in range(max_retries):
                    failed_mask = torch.tensor([not v for v in best_valid], device=device, dtype=torch.bool)
                    if not bool(failed_mask.any().item()):
                        break
                    retry_attempts += int(failed_mask.sum().item())

                    # Native Track-2 Pain Imprint ONLY (ZERO explicit token logit subtraction!)
                    if not model.ablation.disable_pain_imprint and last_out is not None:
                        moe_aux: List[Dict[str, torch.Tensor]] = last_out["moe_aux"]  # type: ignore[assignment]
                        # Compute lightweight probe signature z from routing layers for culprit pain imprint
                        p_means = torch.stack(
                            [m["probes_z"][:, -16:, :, :].mean(dim=1) for m in moe_aux], dim=1
                        )
                        e_weights = torch.stack([m["block_profile"] for m in moe_aux], dim=1)
                        exp_state.asymmetric_rollback_and_imprint_pain(
                            exp_state.clone_synergy_snapshot(), p_means, e_weights, m_advs
                        )

                    # 16-Slot Wet-Cement selective remasking of lowest-margin slots (or full remask if wet-cement disabled!)
                    if not model.ablation.disable_wet_cement:
                        rem_blk, _ = select_wet_cement_remask_slots(cur_blk, cur_margins, d_bwd=0.80 + 0.08 * r_step)
                    else:
                        rem_blk = torch.full_like(cur_blk, SymbolicVocab.MASK_ID)

                    re_cur = torch.where(failed_mask.unsqueeze(-1), rem_blk, cur_blk)
                    for beat in range(2):
                        out_re = model.forward_symbolic(
                            torch.cat([prompt, re_cur], dim=1),
                            phase=CognitivePhase.POST,
                            attn_mask=m32,
                            wb_states=wb,
                            exp_state=exp_state,
                            t_k=0.35 if beat == 0 else 0.15,
                            read_only_pml=True,
                            force_breaker_div=(not model.ablation.disable_pain_imprint),
                            return_soft_sketch=False,
                        )
                        last_out = out_re
                        d_lg = out_re["logits"][:, -16:, :].index_select(-1, d_ids).float()  # type: ignore[index]
                        if gumbel_temp > 0.0:
                            u = torch.rand_like(d_lg).clamp(1e-6, 1.0 - 1e-6)
                            d_lg = d_lg + gumbel_temp * (-torch.log(-torch.log(u)))
                        probs = F.softmax(d_lg, dim=-1)
                        top2_p, top2_i = torch.topk(probs, k=2, dim=-1)
                        m_re = top2_p[..., 0] - top2_p[..., 1]
                        pred_re = d_ids[top2_i[..., 0]]
                        is_m = re_cur == SymbolicVocab.MASK_ID
                        if beat == 1:
                            re_cur = torch.where(is_m, pred_re, re_cur)
                        else:
                            sc = torch.where(is_m, m_re, torch.full_like(m_re, -1e9))
                            k_op = (is_m.sum(dim=-1, keepdim=True) + 1) // 2
                            ranks = torch.argsort(torch.argsort(sc, dim=-1, descending=True), dim=-1)
                            re_cur = torch.where(is_m & (ranks < k_op), pred_re, re_cur)
                        cur_margins = torch.where(failed_mask.unsqueeze(-1), m_re, cur_margins)

                    new_grids, new_valids = _decode_and_check(re_cur)
                    for i in range(bsz):
                        if not best_valid[i]:
                            if new_grids[i] == prev_grids[i]:
                                exact_board_relapses += 1
                            prev_grids[i] = new_grids[i]
                            if new_valids[i]:
                                best_valid[i] = True
                    cur_blk = torch.where(failed_mask.unsqueeze(-1), re_cur, cur_blk)

            final_ok += sum(best_valid)
            tot += bsz

    ms_p = (sync_timer(device) - t0) * 1000.0 / max(1, tot)
    return {
        "draft_solve_pct": round(100.0 * draft_ok / max(1, tot), 2),
        "final_solve_pct": round(100.0 * final_ok / max(1, tot), 2),
        "retry_gain_pct": round(100.0 * (final_ok - draft_ok) / max(1, tot), 2),
        "exact_board_relapse_pct": round(100.0 * exact_board_relapses / max(1, retry_attempts), 2),
        "ms_per_problem": round(ms_p, 2),
    }


def run_stress2_sudoku(cfg: AVLPMLConfig, optim_path: Path, device: torch.device, steps: int = 1000) -> Dict[str, Any]:
    vocab = SymbolicVocab(cfg)
    all_boards = enumerate_all_4x4_sudoku_boards()  # 288 valid boards
    train_boards = [b for i, b in enumerate(all_boards) if not _is_test_instance(f"SUDOKU4:{b}", 0.20)]
    test_boards = [b for i, b in enumerate(all_boards) if _is_test_instance(f"SUDOKU4:{b}", 0.20)]
    rng_ev = random.Random(9202)
    test_batches = [sample_sudoku4x4_batch(vocab, test_boards, rng_ev, 32, inject_trap_prob=0.0) for _ in range(8)]  # 256 unseen-board puzzles!

    res: Dict[str, Any] = {"num_train_boards": len(train_boards), "num_unseen_test_boards": len(test_boards), "num_test_puzzles": 256}
    variants = [
        ("v11_full_native_pain_wet", AblationConfig(), True),
        ("wo_pain_imprint", AblationConfig(disable_pain_imprint=True), True),
        ("wo_wet_cement", AblationConfig(disable_wet_cement=True), True),
        ("pure_block_dllm", AblationConfig(disable_pml=True, disable_pain_imprint=True, disable_wet_cement=True), False),
    ]
    for name, ab, do_ret in variants:
        torch.manual_seed(42)
        m = AVLPMLModel(cfg, ablation=ab).to(device)
        opt, _ = build_optimizers(m, optim_path, stabilized_v11=True)
        sched = build_warmup_cosine_scheduler(opt, total_steps=steps, warmup_ratio=0.10, min_lr_ratio=0.03)
        rng_tr = random.Random(4202)
        t0 = sync_timer(device)
        for st in range(1, steps + 1):
            b = sample_sudoku4x4_batch(vocab, train_boards, rng_tr, 64, inject_trap_prob=0.30)
            train_sudoku_dllm_step(m, opt, b, device, st, steps)
            sched.step()
        tr_s = round(sync_timer(device) - t0, 1)
        del opt, sched
        set_u_post_warmup(m, 1.0)
        ev_det = evaluate_sudoku_zero_cheat(m, vocab, test_batches, device, enable_retries=do_ret, max_retries=3, gumbel_temp=0.0)
        ev_sto = evaluate_sudoku_zero_cheat(m, vocab, test_batches, device, enable_retries=do_ret, max_retries=3, gumbel_temp=0.30) if do_ret else ev_det
        res[name] = {"deterministic_zero_cheat": ev_det, "mild_gumbel_0_30": ev_sto, "train_sec": tr_s, "vram": check_vram_guard(f"S2_{name}")}
        print(f"[Stress2 Sudoku {name}] det={ev_det['final_solve_pct']}% (draft={ev_det['draft_solve_pct']}%, rel={ev_det['exact_board_relapse_pct']}%) | gumbel={ev_sto['final_solve_pct']}% (rel={ev_sto['exact_board_relapse_pct']}%) | {tr_s}s", flush=True)
        del m
        free_vram()

    # Equal-Wall-Clock Best-of-N Causal AR on EXACT same 256 unseen-board 4x4 Sudoku puzzles!
    torch.manual_seed(42)
    ar_m = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True, disable_pml=True)).to(device)
    opt_ar, _ = build_optimizers(ar_m, optim_path, stabilized_v11=True)
    sched_ar = build_warmup_cosine_scheduler(opt_ar, total_steps=steps, warmup_ratio=0.10, min_lr_ratio=0.03)
    rng_ar = random.Random(4202)
    ar_m.train()
    t0 = sync_timer(device)
    for _ in range(steps):
        b = sample_sudoku4x4_batch(vocab, train_boards, rng_ar, 64, inject_trap_prob=0.0)
        seq = torch.cat([b.prompt_ids, b.target_sym_ids[:, :16]], dim=1).to(device)
        opt_ar.zero_grad(set_to_none=True)
        with amp_ctx(device):
            out_ar = ar_m.forward_symbolic(seq[:, :-1], phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, return_soft_sketch=False)
            loss_ar = F.cross_entropy(out_ar["logits"][:, 15:, :].float().reshape(-1, cfg.sym_vocab_size), seq[:, 16:].reshape(-1))  # type: ignore[index]
        loss_ar.backward()
        torch.nn.utils.clip_grad_norm_(ar_m.parameters(), 1.0)
        opt_ar.step()
        sched_ar.step()
    tr_ar = round(sync_timer(device) - t0, 1)
    del opt_ar, sched_ar

    ar_m.eval()
    d_ids = torch.tensor([vocab.token_to_id[str(d)] for d in (1, 2, 3, 4)], device=device, dtype=torch.long)
    id_to_d = {int(vocab.token_to_id[str(d)]): d for d in (1, 2, 3, 4)}
    ms_avl = float(res["v11_full_native_pain_wet"]["mild_gumbel_0_30"]["ms_per_problem"])

    # Measure 1-sample KV-AR time to set fair equal-wall-clock N
    t_one0 = sync_timer(device)
    with torch.no_grad(), amp_ctx(device):
        p0 = test_batches[0].prompt_ids.to(device)
        o0 = ar_m.forward_symbolic(p0, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, start_pos=0, return_soft_sketch=False)
        kvs = o0["present_kvs"]  # type: ignore[assignment]
        nxt = d_ids[o0["logits"][:, -1:, :].index_select(-1, d_ids).argmax(dim=-1)]  # type: ignore[index]
        for si in range(1, 16):
            os = ar_m.forward_symbolic(nxt, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, past_kvs=kvs, start_pos=16 + si - 1, return_soft_sketch=False)
            kvs = os["present_kvs"]  # type: ignore[assignment]
            nxt = d_ids[os["logits"][:, -1:, :].index_select(-1, d_ids).argmax(dim=-1)]  # type: ignore[index]
    ms_ar1 = (sync_timer(device) - t_one0) * 1000.0 / 32.0
    n_bon = max(1, min(6, int(round(ms_avl / max(0.05, ms_ar1)))))

    ar_p1_ok = ar_bon_ok = ar_tot = 0
    t_bon0 = sync_timer(device)
    with torch.no_grad(), amp_ctx(device):
        for batch in test_batches:
            prompt = batch.prompt_ids.to(device)
            bsz = prompt.shape[0]
            solved = [False] * bsz
            for r_i in range(n_bon):
                o0 = ar_m.forward_symbolic(prompt, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, start_pos=0, return_soft_sketch=False)
                kvs = o0["present_kvs"]  # type: ignore[assignment]
                d_l = o0["logits"][:, -1, :].index_select(-1, d_ids).float()  # type: ignore[index]
                idx0 = d_l.argmax(dim=-1) if r_i == 0 else torch.multinomial(F.softmax(d_l / 0.70, dim=-1), 1).squeeze(-1)
                nxt = d_ids[idx0].unsqueeze(-1)
                seq_g = [nxt]
                for si in range(1, 16):
                    os = ar_m.forward_symbolic(nxt, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, past_kvs=kvs, start_pos=16 + si - 1, return_soft_sketch=False)
                    kvs = os["present_kvs"]  # type: ignore[assignment]
                    d_l = os["logits"][:, -1, :].index_select(-1, d_ids).float()  # type: ignore[index]
                    idx_s = d_l.argmax(dim=-1) if r_i == 0 else torch.multinomial(F.softmax(d_l / 0.70, dim=-1), 1).squeeze(-1)
                    nxt = d_ids[idx_s].unsqueeze(-1)
                    seq_g.append(nxt)
                g_cpu = torch.cat(seq_g, dim=1).cpu().tolist()
                for i in range(bsz):
                    ok = is_valid_4x4_sudoku([id_to_d.get(t, 0) for t in g_cpu[i]], batch.clues_raw[i])
                    if r_i == 0 and ok:
                        ar_p1_ok += 1
                    if ok:
                        solved[i] = True
            ar_bon_ok += sum(solved)
            ar_tot += bsz
    ms_bon = (sync_timer(device) - t_bon0) * 1000.0 / max(1, ar_tot)
    res["equal_wallclock_best_of_n_ar"] = {
        "n_samples": n_bon,
        "pass1_solve_pct": round(100.0 * ar_p1_ok / max(1, ar_tot), 2),
        "best_of_n_solve_pct": round(100.0 * ar_bon_ok / max(1, ar_tot), 2),
        "ms_per_problem": round(ms_bon, 2),
        "train_sec": tr_ar,
        "vram": check_vram_guard("S2_BoN_AR"),
    }
    print(f"[Stress2 Sudoku BoN_AR N={n_bon}] pass1={res['equal_wallclock_best_of_n_ar']['pass1_solve_pct']}% | BoN={res['equal_wallclock_best_of_n_ar']['best_of_n_solve_pct']}% | {ms_bon:.2f}ms", flush=True)
    del ar_m
    free_vram()
    return res


# =============================================================================
# STRESS TEST 3: Variable-Length Autonomous <EOT> Halting (L in 2..8 + 9..10 Extrap)
# =============================================================================
@dataclass
class VarLenBatch:
    prompt_ids: torch.Tensor      # [B, 32] (GOAL, x0, op0, k0, ..., EOT, pad...)
    target_sym_ids: torch.Tensor  # [B, max_blocks*8] (PAD after true_L blocks)
    true_lengths: torch.Tensor    # [B] true L in 2..10
    final_ans_tok: torch.Tensor   # [B] true final val16 token at step L-1


def sample_varlen_batch(
    vocab: SymbolicVocab,
    rng: random.Random,
    batch_size: int,
    split: str = "train",
    min_len: int = 2,
    max_len: int = 8,
    pad_blocks: int = 8,
) -> VarLenBatch:
    want_test = split == "test"
    sep_id = vocab.token_to_id[":"]
    eq_id = vocab.token_to_id["="]
    op_toks = [vocab.token_to_id["+"], vocab.token_to_id["-"], vocab.token_to_id["=="], vocab.token_to_id["*"]]
    prompts, syms, lens, finals = [], [], [], []

    while len(prompts) < batch_size:
        L = rng.randint(min_len, max_len)
        x0 = rng.randint(0, 15)
        ops = [rng.randint(0, 3) for _ in range(L)]
        ks = [rng.randint(1, 7) for _ in range(L)]
        if min_len <= 8:
            key = f"VARLEN:{L}:{ops}:{x0 % 4}"
            if _is_test_instance(key, test_ratio=0.20) != want_test:
                continue

        p_seq = [SymbolicVocab.GOAL_ID, val16_tok(vocab, x0)]
        for opc, kv in zip(ops, ks):
            p_seq.extend([op_toks[opc], const8_tok(vocab, kv)])
        p_seq.append(SymbolicVocab.EOT_ID)
        while len(p_seq) < 32:
            p_seq.append(sep_id)

        cur_v = x0
        sym_seq: List[int] = []
        for b_i in range(L):
            nxt_v = apply_op16(ops[b_i], cur_v, ks[b_i])
            arg1 = val16_tok(vocab, x0) if b_i == 0 else vocab.reg_slot_to_id((b_i - 1) % 2)
            flag = SymbolicVocab.EOT_ID if b_i == L - 1 else eq_id
            # 8-slot x 2-beat: Beat 0 [0..3]: [op_tok, arg1, k, =]; Beat 1 [4..7]: [val16(nxt_v), BIND, REG, flag(EOT/=)]
            sym_seq.extend([
                op_toks[ops[b_i]], arg1, const8_tok(vocab, ks[b_i]), eq_id,
                val16_tok(vocab, nxt_v), SymbolicVocab.BIND_ID, vocab.reg_slot_to_id(b_i % 2), flag,
            ])
            cur_v = nxt_v
        while len(sym_seq) < pad_blocks * 8:
            sym_seq.append(SymbolicVocab.PAD_ID)

        prompts.append(p_seq)
        syms.append(sym_seq)
        lens.append(L)
        finals.append(val16_tok(vocab, cur_v))

    return VarLenBatch(
        prompt_ids=torch.tensor(prompts, dtype=torch.long),
        target_sym_ids=torch.tensor(syms, dtype=torch.long),
        true_lengths=torch.tensor(lens, dtype=torch.long),
        final_ans_tok=torch.tensor(finals, dtype=torch.long),
    )


def train_varlen_dllm_step(
    model: AVLPMLModel, opt: torch.optim.AdamW, batch: VarLenBatch, device: torch.device, step_idx: int, total_steps: int
) -> None:
    model.train()
    set_u_post_warmup(model, step_idx / float(max(1, total_steps)))
    prompt = batch.prompt_ids.to(device)
    sym_clean = batch.target_sym_ids.to(device)
    bsz, p_len = prompt.shape
    s_len = sym_clean.shape[1]
    num_blocks = s_len // 8

    is_non_pad = sym_clean != SymbolicVocab.PAD_ID
    local_pos = (torch.arange(s_len, device=device) % 8).unsqueeze(0).expand(bsz, -1)
    if step_idx % 2 == 0:
        q_rate = 1.0
        mask_sym = is_non_pad
    else:
        q_rate = 0.50
        mask_sym = is_non_pad & (local_pos >= 4)

    noisy_sym = torch.where(mask_sym, torch.full_like(sym_clean, SymbolicVocab.MASK_ID), sym_clean)
    dual_in = torch.cat([prompt, sym_clean, noisy_sym], dim=1)
    dual_tgt = torch.cat([prompt, sym_clean, sym_clean], dim=1)
    dual_lmask = torch.cat([torch.zeros(bsz, p_len + s_len, device=device, dtype=torch.bool), mask_sym], dim=1)
    attn_mask, rope_pos = build_dual_stream_bd3_mask(p_len, num_blocks, block_size=8, enable_wet_seam=False, device=device)
    rope_pos = rope_pos.expand(bsz, -1)

    opt.zero_grad(set_to_none=True)
    with amp_ctx(device):
        wb = model.init_whiteboard_states(bsz, device)
        if not model.ablation.disable_pml:
            model.populate_ro_from_prompt(prompt, wb)
        out = model.forward_symbolic(
            dual_in, phase=CognitivePhase.FORMAL, positions=rope_pos, attn_mask=attn_mask,
            wb_states=wb, exp_state=None, t_k=q_rate, dual_stream_sym_blocks=num_blocks,
            dual_stream_block_size=8, return_soft_sketch=False,
        )
        loss, _ = model.compute_six_losses(out, dual_tgt, dual_lmask, collect_stats=False)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


@torch.no_grad()
def evaluate_varlen_autonomous_dllm(
    model: AVLPMLModel, batches: List[VarLenBatch], device: torch.device, max_gen_blocks: int = 12
) -> Dict[str, float]:
    """TRUE AUTONOMOUS HALTING: stops strictly when slot 7 emits <EOT>, zero length hint!"""
    model.eval()
    halt_ok = em_ok = tot = 0
    gen_blk_sum = 0
    t0 = sync_timer(device)

    with amp_ctx(device):
        for batch in batches:
            prompt = batch.prompt_ids.to(device)
            bsz, p_len = prompt.shape
            true_L = batch.true_lengths.to(device)
            true_ans = batch.final_ans_tok.to(device)

            wb = model.init_whiteboard_states(bsz, device)
            if not model.ablation.disable_pml:
                model.populate_ro_from_prompt(prompt, wb)
            exp_state = model.init_experience_state(bsz, device)

            m_pref = torch.ones(1, 1, p_len, p_len, device=device, dtype=torch.bool)
            out_p = model.forward_symbolic(
                prompt, phase=CognitivePhase.FORMAL, attn_mask=m_pref, wb_states=wb, exp_state=exp_state,
                t_k=0.0, read_only_pml=True, dual_stream_block_size=8, past_kvs=None, start_pos=0, return_soft_sketch=False,
            )
            h_p = out_p["hidden"].mean(dim=1)  # type: ignore[union-attr]
            kvs = out_p["present_kvs"]  # type: ignore[assignment]

            halted = torch.zeros(bsz, device=device, dtype=torch.bool)
            pred_L = torch.full((bsz,), max_gen_blocks, device=device, dtype=torch.long)
            pred_ans = torch.zeros(bsz, device=device, dtype=torch.long)

            for b_i in range(max_gen_blocks):
                cur = torch.full((bsz, 8), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
                s_pos = p_len + b_i * 8
                for beat, t_k in [(0, 1.0), (1, 0.5)]:
                    out = model.forward_symbolic(
                        cur, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb, exp_state=exp_state,
                        t_k=t_k, read_only_pml=True, dual_stream_block_size=8, past_kvs=kvs, start_pos=s_pos, return_soft_sketch=False,
                    )
                    if b_i == 0 and beat == 0 and not model.ablation.static_top2_baseline:
                        exp_state.delta_b_pre = model.hyper_router(h_p, out["hidden"].mean(dim=1))  # type: ignore[union-attr]
                    cur, _ = hierarchical_three_beat_unmask(cur, out["logits"], beat, 2)  # type: ignore[arg-type]

                newly_halted = (~halted) & (cur[:, 7] == SymbolicVocab.EOT_ID)
                pred_L = torch.where(newly_halted, torch.full_like(pred_L, b_i + 1), pred_L)
                pred_ans = torch.where(newly_halted, cur[:, 4], pred_ans)
                halted = halted | newly_halted
                if bool(halted.all().item()):
                    break

                if b_i < max_gen_blocks - 1:
                    out_c = model.forward_symbolic(
                        cur, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb, exp_state=exp_state,
                        t_k=0.0, read_only_pml=model.ablation.disable_pml, dual_stream_block_size=8,
                        past_kvs=kvs, start_pos=s_pos, return_soft_sketch=False,
                    )
                    kvs = out_c["present_kvs"]  # type: ignore[assignment]

            h_m = pred_L == true_L
            halt_ok += int(h_m.sum().item())
            em_ok += int((h_m & (pred_ans == true_ans)).sum().item())
            gen_blk_sum += int(pred_L.sum().item())
            tot += bsz

    ms_p = (sync_timer(device) - t0) * 1000.0 / max(1, tot)
    return {
        "exact_halt_and_ans_em": round(100.0 * em_ok / max(1, tot), 2),
        "exact_halt_step_acc": round(100.0 * halt_ok / max(1, tot), 2),
        "avg_blocks_halted": round(gen_blk_sum / max(1, tot), 2),
        "ms_per_problem": round(ms_p, 2),
    }


def run_stress3_varlen(cfg: AVLPMLConfig, optim_path: Path, device: torch.device, steps_dllm: int = 1100, steps_ar: int = 550) -> Dict[str, Any]:
    vocab = SymbolicVocab(cfg)
    rng_ev = random.Random(9303)
    test_2_8 = [sample_varlen_batch(vocab, rng_ev, 32, "test", 2, 8, 8) for _ in range(8)]     # 256 in-dist varlen
    test_9_10 = [sample_varlen_batch(vocab, rng_ev, 32, "test", 9, 10, 10) for _ in range(4)]  # 128 zero-shot extrap!

    torch.manual_seed(42)
    m = AVLPMLModel(cfg).to(device)
    opt, _ = build_optimizers(m, optim_path, stabilized_v11=True)
    sched = build_warmup_cosine_scheduler(opt, total_steps=steps_dllm, warmup_ratio=0.10, min_lr_ratio=0.03)
    rng_tr = random.Random(4303)
    t0 = sync_timer(device)
    for st in range(1, steps_dllm + 1):
        b = sample_varlen_batch(vocab, rng_tr, 64, "train", 2, 8, 8)
        train_varlen_dllm_step(m, opt, b, device, st, steps_dllm)
        sched.step()
    tr_s = round(sync_timer(device) - t0, 1)
    del opt, sched
    set_u_post_warmup(m, 1.0)

    ev_in = evaluate_varlen_autonomous_dllm(m, test_2_8, device, max_gen_blocks=10)
    ev_ex = evaluate_varlen_autonomous_dllm(m, test_9_10, device, max_gen_blocks=12)
    orig_k = m.cfg.k_max
    m.cfg.k_max = 0
    ev_k0 = evaluate_varlen_autonomous_dllm(m, test_2_8, device, max_gen_blocks=10)
    m.cfg.k_max = orig_k

    # Zero-Leakage Canary (poison target_sym_ids -> -999999)
    pois = VarLenBatch(test_2_8[0].prompt_ids.clone(), torch.full_like(test_2_8[0].target_sym_ids, -999999), test_2_8[0].true_lengths.clone(), test_2_8[0].final_ans_tok.clone())
    c_ok = evaluate_varlen_autonomous_dllm(m, [test_2_8[0]], device)["exact_halt_and_ans_em"] == evaluate_varlen_autonomous_dllm(m, [pois], device)["exact_halt_and_ans_em"]

    v11_res = {
        "in_dist_L2_8": ev_in,
        "extrap_L9_10": ev_ex,
        "k0_knockout_em": ev_k0["exact_halt_and_ans_em"],
        "k0_knockout_drop": round(ev_in["exact_halt_and_ans_em"] - ev_k0["exact_halt_and_ans_em"], 2),
        "zero_leakage_canary": c_ok,
        "train_sec": tr_s,
        "vram": check_vram_guard("S3_v11"),
    }
    print(f"[Stress3 VarLen v11] L2-8 EM={ev_in['exact_halt_and_ans_em']}% (halt={ev_in['exact_halt_step_acc']}%, {ev_in['ms_per_problem']}ms) | L9-10={ev_ex['exact_halt_and_ans_em']}% (halt={ev_ex['exact_halt_step_acc']}%) | K0_drop={v11_res['k0_knockout_drop']}%", flush=True)
    del m
    free_vram()

    # Mainstream Causal AR Baseline (Static Top-2 KV-AR with autonomous <EOT> halting!)
    torch.manual_seed(42)
    ar_m = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True, disable_pml=True)).to(device)
    opt_ar, _ = build_optimizers(ar_m, optim_path, stabilized_v11=True)
    sched_ar = build_warmup_cosine_scheduler(opt_ar, total_steps=steps_ar, warmup_ratio=0.10, min_lr_ratio=0.03)
    rng_ar = random.Random(4303)
    ar_m.train()
    t0 = sync_timer(device)
    for _ in range(steps_ar):
        b = sample_varlen_batch(vocab, rng_ar, 64, "train", 2, 8, 8)
        seq = torch.cat([b.prompt_ids, b.target_sym_ids], dim=1).to(device)
        opt_ar.zero_grad(set_to_none=True)
        with amp_ctx(device):
            out_ar = ar_m.forward_symbolic(seq[:, :-1], phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, return_soft_sketch=False)
            lg = out_ar["logits"][:, 31:, :].float()  # type: ignore[index]
            tgt_s = seq[:, 32:]
            loss_ar = F.cross_entropy(lg.reshape(-1, cfg.sym_vocab_size), tgt_s.reshape(-1), ignore_index=SymbolicVocab.PAD_ID)
        loss_ar.backward()
        torch.nn.utils.clip_grad_norm_(ar_m.parameters(), 1.0)
        opt_ar.step()
        sched_ar.step()
    tr_ar = round(sync_timer(device) - t0, 1)
    del opt_ar, sched_ar

    ar_m.eval()
    def _eval_ar_vl(batches: List[VarLenBatch], max_toks: int) -> Dict[str, float]:
        h_ok = e_ok = tot = 0
        t_a0 = sync_timer(device)
        with torch.no_grad(), amp_ctx(device):
            for batch in batches:
                prompt = batch.prompt_ids.to(device)
                bsz, p_len = prompt.shape
                true_L = batch.true_lengths.to(device)
                true_ans = batch.final_ans_tok.to(device)
                o_p = ar_m.forward_symbolic(prompt, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, start_pos=0, return_soft_sketch=False)
                kvs = o_p["present_kvs"]  # type: ignore[assignment]
                nxt = o_p["logits"][:, -1:, :].argmax(dim=-1)  # type: ignore[index]
                halted = torch.zeros(bsz, device=device, dtype=torch.bool)
                pred_L = torch.full((bsz,), max_toks // 8, device=device, dtype=torch.long)
                pred_ans = torch.zeros(bsz, device=device, dtype=torch.long)
                gen_hist = [nxt]
                for si in range(1, max_toks):
                    o_s = ar_m.forward_symbolic(nxt, phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True, past_kvs=kvs, start_pos=p_len + si - 1, return_soft_sketch=False)
                    kvs = o_s["present_kvs"]  # type: ignore[assignment]
                    nxt = o_s["logits"][:, -1:, :].argmax(dim=-1)  # type: ignore[index]
                    gen_hist.append(nxt)
                    if (si + 1) % 8 == 0:
                        b_idx = si // 8
                        nh = (~halted) & (nxt.squeeze(-1) == SymbolicVocab.EOT_ID)
                        pred_L = torch.where(nh, torch.full_like(pred_L, b_idx + 1), pred_L)
                        pred_ans = torch.where(nh, gen_hist[b_idx * 8 + 4].squeeze(-1), pred_ans)
                        halted = halted | nh
                        if bool(halted.all().item()):
                            break
                hm = pred_L == true_L
                h_ok += int(hm.sum().item())
                e_ok += int((hm & (pred_ans == true_ans)).sum().item())
                tot += bsz
        return {
            "exact_halt_and_ans_em": round(100.0 * e_ok / max(1, tot), 2),
            "exact_halt_step_acc": round(100.0 * h_ok / max(1, tot), 2),
            "ms_per_problem": round((sync_timer(device) - t_a0) * 1000.0 / max(1, tot), 2),
        }

    ar_in = _eval_ar_vl(test_2_8, 80)
    ar_ex = _eval_ar_vl(test_9_10, 96)
    ar_res = {"in_dist_L2_8": ar_in, "extrap_L9_10": ar_ex, "train_sec": tr_ar, "vram": check_vram_guard("S3_ar")}
    print(f"[Stress3 VarLen AR ] L2-8 EM={ar_in['exact_halt_and_ans_em']}% (halt={ar_in['exact_halt_step_acc']}%, {ar_in['ms_per_problem']}ms) | L9-10={ar_ex['exact_halt_and_ans_em']}%", flush=True)
    del ar_m
    free_vram()
    return {"v11_autonomous_prefix_kv": v11_res, "mainstream_causal_kv_ar": ar_res}


# =============================================================================
# Mandatory 5-Step Pre-Flight VRAM & Latency Profiler
# =============================================================================
def run_preflight_profile(cfg: AVLPMLConfig, optim_path: Path, device: torch.device) -> Dict[str, Any]:
    print("=== [STAGE 0-HARD PRE-FLIGHT 5-STEP VRAM & LATENCY PROFILE] ===", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vocab = SymbolicVocab(cfg)
    rng = random.Random(123)
    boards = enumerate_all_4x4_sudoku_boards()
    prof: Dict[str, Any] = {}

    def _prof(name: str, ab: AblationConfig, fn, planned_steps: int, n_runs: int) -> None:
        m = AVLPMLModel(cfg, ablation=ab).to(device)
        opt, _ = build_optimizers(m, optim_path, stabilized_v11=True)
        for w in range(2):
            fn(m, opt, w + 1)
        t0 = sync_timer(device)
        for s in range(5):
            fn(m, opt, s + 3)
        ms = (sync_timer(device) - t0) * 1000.0 / 5.0
        vram = check_vram_guard(f"prof_{name}")
        est = round(ms * planned_steps * n_runs / 1000.0, 1)
        prof[name] = {"ms_per_step": round(ms, 2), "planned": f"{planned_steps}x{n_runs}", "est_sec": est, "vram": vram}
        print(f"  [{name:22s}] {ms:6.2f} ms/step | est={est:5.1f}s | VRAM={vram}", flush=True)
        del m, opt
        free_vram()

    _prof("s1_alu24_full_b32", AblationConfig(), lambda m, o, s: train_hard_alu24_dllm_step(m, o, sample_hard_alu24_batch(vocab, rng, 32, "train", 24), device, s, 100), 1200, 1)
    _prof("s1_alu24_nopml_b32", AblationConfig(disable_pml=True), lambda m, o, s: train_hard_alu24_dllm_step(m, o, sample_hard_alu24_batch(vocab, rng, 32, "train", 24), device, s, 100), 1200, 1)
    _prof("s2_sudoku_full_b64", AblationConfig(), lambda m, o, s: train_sudoku_dllm_step(m, o, sample_sudoku4x4_batch(vocab, boards, rng, 64, 0.3), device, s, 100), 1000, 3)
    _prof("s2_sudoku_pure_b64", AblationConfig(disable_pml=True, disable_pain_imprint=True, disable_wet_cement=True), lambda m, o, s: train_sudoku_dllm_step(m, o, sample_sudoku4x4_batch(vocab, boards, rng, 64, 0.3), device, s, 100), 1000, 1)
    _prof("s3_varlen_full_b64", AblationConfig(), lambda m, o, s: train_varlen_dllm_step(m, o, sample_varlen_batch(vocab, rng, 64, "train", 2, 8, 8), device, s, 100), 1100, 1)

    tot_s = sum(v["est_sec"] for v in prof.values()) + 240.0  # +AR baselines & evals
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
    out_path = root / "outputs" / "stage0_hard_stress.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prof = run_preflight_profile(cfg, optim_path, device)
    if args.profile_only:
        save_checkpoint(out_path, {"status": "profile_only", "profile": prof})
        return

    t_global = time.perf_counter()
    report: Dict[str, Any] = {
        "suite": "Stage 0-Hard Zero-Cheat Stress Tests",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "preflight_profile": prof,
        "stress1_alu24_8reg_mod16": {},
        "stress2_sudoku4x4_zero_cheat": {},
        "stress3_varlen_autonomous_halt": {},
    }
    save_checkpoint(out_path, report)

    report["stress1_alu24_8reg_mod16"] = run_stress1_alu24(cfg, optim_path, device, steps_dllm=1200, steps_ar=600)
    save_checkpoint(out_path, report)

    report["stress2_sudoku4x4_zero_cheat"] = run_stress2_sudoku(cfg, optim_path, device, steps=1000)
    save_checkpoint(out_path, report)

    report["stress3_varlen_autonomous_halt"] = run_stress3_varlen(cfg, optim_path, device, steps_dllm=1100, steps_ar=550)
    report["total_wall_clock_minutes"] = round((time.perf_counter() - t_global) / 60.0, 2)
    report["peak_vram_gb"] = check_vram_guard("hard_final")["peak_alloc_gb"]
    save_checkpoint(out_path, report)
    print(f"\n[DONE] Stage 0-Hard Stress Suite completed in {report['total_wall_clock_minutes']} min! Peak VRAM={report['peak_vram_gb']} GB -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
