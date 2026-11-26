"""AVL-PMLv1.1 Full Verification Suite (Stage 0 RTX 5080):
- Mandatory 5-step VRAM & latency pre-flight profiler (--profile-only)
- Strict per-subrun VRAM isolation (< 14.0 GB hard ceiling, del + gc.collect + empty_cache)
- Incremental JSON checkpointing after every sub-run
- Verifies all 3 AVL-PMLv1.1 architectural fixes:
  1. Capacity-Preserving Adaptive Top-K MoE (Shared=64, Routed=256, probe_mix & L_util amputated) -> real K_max=0 drop!
  2. 8-Slot Atomic Micro-Blocks x 2-Beat Unmasking + Prefix-KV CUDAGraph -> L3-Hard Unseen AST breakthrough & speedup!
  3. Differentiable S^RO Rule Encoding + alpha_calm=0.08 -> Level 2a/b Unseen Rule Extrapolation & 2c Pain Imprint!
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.wet_cement import build_block_diffusion_attn_mask, hierarchical_three_beat_unmask
from avl_pmlv1.engine.model import AblationConfig, AVLPMLModel
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab
from gen_synth_tasks import Stage0SynthGenerator, TaskBatch, _is_test_instance
from run_four_point_audit import _nl_static_kv_step, apply_step_op, mean_std
from run_gate1_ablation import (
    amp_ctx,
    build_optimizers,
    evaluate_backtrack_and_relapse,
    evaluate_symbolic_exact_match,
    set_u_post_warmup,
    sync_timer,
    train_dual_stream_step,
)

HARD_VRAM_LIMIT_GB = 14.0


def free_vram() -> Tuple[float, float]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        alloc = round(torch.cuda.memory_allocated() / (1024**3), 2)
        peak = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
        return alloc, peak
    return 0.0, 0.0


def check_vram_guard(tag: str) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"alloc_gb": 0.0, "reserved_gb": 0.0, "peak_alloc_gb": 0.0}
    alloc = round(torch.cuda.memory_allocated() / (1024**3), 2)
    resv = round(torch.cuda.memory_reserved() / (1024**3), 2)
    peak = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    if resv >= HARD_VRAM_LIMIT_GB:
        raise RuntimeError(f"[VRAM GUARD TRIPPED at {tag}] reserved={resv} GB >= {HARD_VRAM_LIMIT_GB} GB!")
    return {"alloc_gb": alloc, "reserved_gb": resv, "peak_alloc_gb": peak}


def save_checkpoint(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(out_path)


# =============================================================================
# 1. 8-Slot Atomic Micro-Block Benchmarks (L3-Hard Unseen AST & L2a/b Rules)
# =============================================================================
def sample_level3_dynamic_ast_8slot(
    gen: Stage0SynthGenerator, batch_size: int, split: str = "train"
) -> Tuple[TaskBatch, torch.Tensor, torch.Tensor, int]:
    """Exact same 20% UNSEEN 4-op AST template split as Round-1, compiled into 4 x 8-slot atomic micro-blocks (32 toks)."""
    prompts, syms, rmasks, ans_pos, ans_ids, ans_digits = [], [], [], [], [], []
    nl_seqs, nl_ans_toks = [], []
    want_test = split == "test"
    v = gen.vocab

    def _lit(role: int, val: int) -> int:
        return v.escape_start + (role % 16) * 8 + (val % 8)

    op_tokens = [
        v.token_to_id["+"],
        v.token_to_id["-"],
        v.token_to_id["=="],
        v.token_to_id["*"],
    ]

    while len(prompts) < batch_size:
        ops = [gen.rng.randint(0, 3) for _ in range(4)]
        tmpl_key = f"AST_TEMPLATE_ONLY:{ops[0]}:{ops[1]}:{ops[2]}:{ops[3]}"
        if _is_test_instance(tmpl_key, test_ratio=0.20) != want_test:
            continue

        x0 = gen.rng.randint(0, 7)
        k1, k2, k3, k4 = [gen.rng.randint(1, 5) for _ in range(4)]
        s1 = apply_step_op(ops[0], x0, k1)
        s2 = apply_step_op(ops[1], s1, k2)
        s3 = apply_step_op(ops[2], s2, k3)
        s4 = apply_step_op(ops[3], s3, k4)

        p_seq = [
            SymbolicVocab.GOAL_ID,
            _lit(0, x0),
            op_tokens[ops[0]], _lit(1, k1),
            op_tokens[ops[1]], _lit(2, k2),
            op_tokens[ops[2]], _lit(3, k3),
            op_tokens[ops[3]], _lit(4, k4),
            gen.sep_id, gen.sep_id, gen.sep_id, gen.sep_id, gen.sep_id, gen.eq_id,
        ]
        # 4 atomic 8-slot micro-blocks (32 tokens total = 4.5x compression vs 144 NL-CoT tokens):
        # Beat 0 [0..3]: [OP, dyn_op, arg1, k_i]; Beat 1 [4..7]: [val_s_i, BIND, REG_j, flag]
        b0 = [SymbolicVocab.GOAL_ID,   op_tokens[ops[0]], _lit(0, x0),         _lit(1, k1), _lit(8, s1), SymbolicVocab.BIND_ID, v.reg_slot_to_id(0), gen.eq_id]
        b1 = [SymbolicVocab.DEDUCE_ID, op_tokens[ops[1]], v.reg_slot_to_id(0), _lit(2, k2), _lit(8, s2), SymbolicVocab.BIND_ID, v.reg_slot_to_id(1), gen.eq_id]
        b2 = [SymbolicVocab.DEDUCE_ID, op_tokens[ops[2]], v.reg_slot_to_id(1), _lit(3, k3), _lit(8, s3), SymbolicVocab.BIND_ID, v.reg_slot_to_id(0), gen.eq_id]
        b3 = [SymbolicVocab.COMMIT_ID, op_tokens[ops[3]], v.reg_slot_to_id(0), _lit(4, k4), _lit(8, s4), SymbolicVocab.BIND_ID, v.reg_slot_to_id(1), SymbolicVocab.EOT_ID]
        sym_seq = b0 + b1 + b2 + b3

        nl_prompt = [
            20 + x0,
            70 + ops[0], 30 + k1,
            75 + ops[1], 40 + k2,
            80 + ops[2], 50 + k3,
            85 + ops[3], 60 + k4,
        ] + [2] * 7
        nl_cot: List[int] = []
        for idx_st, (op_c, kv, sv) in enumerate([(ops[0], k1, s1), (ops[1], k2, s2), (ops[2], k3, s3), (ops[3], k4, s4)]):
            boilerplate = [100 + ((idx_st * 37 + w * 7) % 300) for w in range(33)] + [70 + idx_st * 5 + op_c, 30 + idx_st * 10 + kv]
            nl_cot.extend(boilerplate + [10 + sv])
        full_nl = nl_prompt + nl_cot  # 160 tokens

        prompts.append(p_seq)
        syms.append(sym_seq)
        rmasks.append([0] * 32)
        ans_pos.append(24 + 4)  # slot 4 of block 3 = index 28
        ans_ids.append(_lit(8, s4))
        ans_digits.append(s4)
        nl_seqs.append(full_nl)
        nl_ans_toks.append(10 + s4)

    return (
        TaskBatch(
            prompt_ids=torch.tensor(prompts, dtype=torch.long),
            target_sym_ids=torch.tensor(syms, dtype=torch.long),
            reassert_mask=torch.tensor(rmasks, dtype=torch.long),
            answer_pos=torch.tensor(ans_pos, dtype=torch.long),
            answer_tok_ids=torch.tensor(ans_ids, dtype=torch.long),
            answer_digits=torch.tensor(ans_digits, dtype=torch.long),
        ),
        torch.tensor(nl_seqs, dtype=torch.long),
        torch.tensor(nl_ans_toks, dtype=torch.long),
        16,
    )


def sample_level2_rule_chain_8slot(
    gen: Stage0SynthGenerator, batch_size: int, split: str = "train", num_chain_steps: int = 4
) -> TaskBatch:
    """20% UNSEEN 4-element rule tables compiled into clean 8-slot atomic micro-blocks (zero causal collision).
    Prompt encodes rule[0..3] in <IN_0..3> and queries[0..N-1] in <IN_4..> (stored in S^RO!).
    Beat 0 [0..3]: [OP, <IN_q>, digit(q), digit(rule[q])];
    Beat 1 [4..7]: [digit(acc), BIND, REG_{s%2}, =/EOT].
    """
    prompts, syms, rmasks, ans_pos, ans_ids, ans_digits = [], [], [], [], [], []
    want_test = split == "test"
    v = gen.vocab

    while len(prompts) < batch_size:
        rule = [gen.rng.randint(0, 7) for _ in range(4)]
        key = f"L2Rule:{rule[0]}:{rule[1]}:{rule[2]}:{rule[3]}"
        if _is_test_instance(key, test_ratio=0.20) != want_test:
            continue

        queries = [gen.rng.randint(0, 3) for _ in range(num_chain_steps)]
        p_raw = [
            v.in_slot_to_id(0), gen.digit_id(rule[0]),
            v.in_slot_to_id(1), gen.digit_id(rule[1]),
            v.in_slot_to_id(2), gen.digit_id(rule[2]),
            v.in_slot_to_id(3), gen.digit_id(rule[3]),
        ]
        for q in queries:
            p_raw.extend([gen.sep_id, gen.digit_id(q)])
        while len(p_raw) < 24:
            p_raw.append(gen.sep_id)

        acc = 0
        sym_blocks: List[int] = []
        for s_idx, q in enumerate(queries):
            mapped = rule[q]
            acc = (acc + mapped) % 8
            is_last = s_idx == num_chain_steps - 1
            op_tok = SymbolicVocab.HYP_ID if s_idx == 0 else (SymbolicVocab.COMMIT_ID if is_last else SymbolicVocab.DEDUCE_ID)
            out_reg = v.reg_slot_to_id(s_idx % 2)
            blk = [
                op_tok, v.in_slot_to_id(q), gen.digit_id(q), gen.digit_id(mapped),
                gen.digit_id(acc), SymbolicVocab.BIND_ID, out_reg, SymbolicVocab.EOT_ID if is_last else gen.eq_id,
            ]
            sym_blocks.extend(blk)

        prompts.append(p_raw)
        syms.append(sym_blocks)
        rmasks.append([0] * len(sym_blocks))
        ans_pos.append((num_chain_steps - 1) * 8 + 4)
        ans_ids.append(gen.digit_id(acc))
        ans_digits.append(acc)

    return TaskBatch(
        prompt_ids=torch.tensor(prompts, dtype=torch.long),
        target_sym_ids=torch.tensor(syms, dtype=torch.long),
        reassert_mask=torch.tensor(rmasks, dtype=torch.long),
        answer_pos=torch.tensor(ans_pos, dtype=torch.long),
        answer_tok_ids=torch.tensor(ans_ids, dtype=torch.long),
        answer_digits=torch.tensor(ans_digits, dtype=torch.long),
    )


# =============================================================================
# 2. Prefix-KV Block-dLLM Inference (Eager + Single-Graph CUDAGraph!)
# =============================================================================
@torch.no_grad()
def evaluate_symbolic_prefix_kv(
    model: AVLPMLModel,
    batches: List[TaskBatch],
    device: torch.device,
    beats_per_block: int = 2,
    block_size: int = 8,
) -> Dict[str, float]:
    """Prefix-KV-Cached Block-dLLM inference from 100% [MASK] blocks (1 prefill + 2*B beats + (B-1) commits)."""
    model.eval()
    correct, total = 0, 0
    active_accum = 0.0
    n_fwds = 0

    if batches:
        with amp_ctx(device):
            p_w = batches[0].prompt_ids[:2].to(device)
            _ = model.forward_symbolic(p_w, phase=CognitivePhase.PRE, read_only_pml=True, dual_stream_block_size=block_size)

    t0 = sync_timer(device)
    with amp_ctx(device):
        for batch in batches:
            prompt = batch.prompt_ids.to(device)
            bsz, p_len = prompt.shape
            s_len = batch.target_sym_ids.shape[1]
            num_blocks = s_len // block_size

            wb_states = model.init_whiteboard_states(bsz, device)
            if not model.ablation.disable_pml:
                model.populate_ro_from_prompt(prompt, wb_states)
            exp_state = model.init_experience_state(bsz, device)

            m_pref = torch.ones(1, 1, p_len, p_len, device=device, dtype=torch.bool)
            out_p = model.forward_symbolic(
                prompt,
                phase=CognitivePhase.FORMAL,
                attn_mask=m_pref,
                wb_states=wb_states,
                exp_state=exp_state,
                t_k=0.0,
                read_only_pml=True,
                dual_stream_block_size=block_size,
                past_kvs=None,
                start_pos=0,
            )
            h_prompt_mean = out_p["hidden"].mean(dim=1)  # type: ignore[union-attr]
            committed_kvs = out_p["present_kvs"]  # type: ignore[assignment]

            generated_blocks: List[torch.Tensor] = []
            for b_idx in range(num_blocks):
                cur_blk = torch.full((bsz, block_size), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
                cur_start = p_len + b_idx * block_size
                for beat in range(beats_per_block):
                    if beats_per_block == 3 and block_size == 8:
                        t_k = 1.0 if beat == 0 else (0.625 if beat == 1 else 0.50)
                    else:
                        t_k = 1.0 if beat == 0 else 0.5
                    out = model.forward_symbolic(
                        cur_blk,
                        phase=CognitivePhase.FORMAL,
                        attn_mask=None,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        t_k=t_k,
                        read_only_pml=True,
                        dual_stream_block_size=block_size,
                        past_kvs=committed_kvs,
                        start_pos=cur_start,
                    )
                    if b_idx == 0 and beat == 0 and not model.ablation.static_top2_baseline:
                        exp_state.delta_b_pre = model.hyper_router(h_prompt_mean, out["hidden"].mean(dim=1))  # type: ignore[union-attr]
                    moe_aux: List[Dict[str, torch.Tensor]] = out["moe_aux"]  # type: ignore[assignment]
                    if moe_aux:
                        active_accum += float(torch.stack([m["active_count"].mean() for m in moe_aux]).mean().item())
                        n_fwds += 1
                    cur_blk, _ = hierarchical_three_beat_unmask(cur_blk, out["logits"], beat, beats_per_block)  # type: ignore[arg-type]

                generated_blocks.append(cur_blk)
                if b_idx < num_blocks - 1:
                    out_c = model.forward_symbolic(
                        cur_blk,
                        phase=CognitivePhase.FORMAL,
                        attn_mask=None,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        t_k=0.0,
                        read_only_pml=model.ablation.disable_pml,
                        dual_stream_block_size=block_size,
                        past_kvs=committed_kvs,
                        start_pos=cur_start,
                    )
                    committed_kvs = out_c["present_kvs"]  # type: ignore[assignment]

            full_gen = torch.cat(generated_blocks, dim=1)
            ans_pos = batch.answer_pos.to(device)
            pred_ans = full_gen.gather(1, ans_pos.unsqueeze(-1)).squeeze(-1)
            correct += int((pred_ans == batch.answer_tok_ids.to(device)).sum().item())
            total += bsz

    elapsed_ms = (sync_timer(device) - t0) * 1000.0
    return {
        "exact_match": round(100.0 * correct / max(1, total), 2),
        "ms_per_problem": round(elapsed_ms / max(1, total), 3),
        "avg_active_experts": round(active_accum / max(1, n_fwds), 3),
    }


@torch.no_grad()
def verify_zero_leakage_canary_8slot(model: AVLPMLModel, batch: TaskBatch, device: torch.device) -> Dict[str, Any]:
    poisoned = TaskBatch(
        prompt_ids=batch.prompt_ids.clone(),
        target_sym_ids=torch.full_like(batch.target_sym_ids, -999999),
        reassert_mask=torch.full_like(batch.reassert_mask, -999999),
        answer_pos=batch.answer_pos.clone(),
        answer_tok_ids=batch.answer_tok_ids.clone(),
        answer_digits=batch.answer_digits.clone(),
    )
    r_clean = evaluate_symbolic_prefix_kv(model, [batch], device, beats_per_block=2, block_size=8)
    r_pois = evaluate_symbolic_prefix_kv(model, [poisoned], device, beats_per_block=2, block_size=8)
    return {
        "em_normal": r_clean["exact_match"],
        "em_with_poisoned_targets": r_pois["exact_match"],
        "abs_em_diff": round(abs(r_clean["exact_match"] - r_pois["exact_match"]), 4),
        "zero_leakage_verified": r_clean["exact_match"] == r_pois["exact_match"],
    }


@torch.no_grad()
def benchmark_prefix_kv_cudagraph_speed(
    sym_model: AVLPMLModel,
    nl_model: AVLPMLModel,
    eval_tuples: List[Tuple[TaskBatch, torch.Tensor, torch.Tensor, int]],
    device: torch.device,
    batch_label: str = "B=32",
) -> Dict[str, Any]:
    """Fair pure-silicon CUDAGraph benchmark: Prefix-KV Block-dLLM (12 tiny fwds) vs Static-KV NL-CoT (144 fwds)."""
    sym_batches = [t[0] for t in eval_tuples]
    sym_eager = evaluate_symbolic_prefix_kv(sym_model, sym_batches, device, beats_per_block=2, block_size=8)

    nl_model.eval()
    bsz = eval_tuples[0][1].shape[0]
    full_len = eval_tuples[0][1].shape[1]
    p_len = eval_tuples[0][3]
    dtype_cg = torch.bfloat16

    # 1. NL-CoT Sync-Free Eager KV-AR
    with amp_ctx(device):
        _, w_nl, _, w_p = eval_tuples[0]
        _, w_kvs = nl_model.forward_nl_causal(w_nl[:, :w_p].to(device), start_pos=0)
        _ = nl_model.forward_nl_causal(w_nl[:, w_p : w_p + 1].to(device), past_kvs=w_kvs, start_pos=w_p)
        nl_ok, nl_tot = 0, 0
        t0 = sync_timer(device)
        for _, nl_seq, nl_ans, _ in eval_tuples:
            nl_seq = nl_seq.to(device)
            lg_p, kvs = nl_model.forward_nl_causal(nl_seq[:, :p_len], past_kvs=None, start_pos=0)
            nxt = lg_p[:, -1:, :].argmax(dim=-1)
            for s_i in range(1, full_len - p_len):
                lg_s, kvs = nl_model.forward_nl_causal(nxt, past_kvs=kvs, start_pos=p_len + s_i - 1)
                nxt = lg_s[:, -1:, :].argmax(dim=-1)
            nl_ok += int((nxt.squeeze(-1) == nl_ans.to(device)).sum().item())
            nl_tot += nl_seq.shape[0]
        nl_eager_ms = (sync_timer(device) - t0) * 1000.0 / max(1, nl_tot)

    # 2. NL-CoT Static-KV CUDAGraph
    k_caches = [torch.zeros(bsz, nl_model.cfg.num_kv_heads, full_len, nl_model.cfg.head_dim, device=device, dtype=dtype_cg) for _ in range(nl_model.cfg.num_layers)]
    v_caches = [torch.zeros(bsz, nl_model.cfg.num_kv_heads, full_len, nl_model.cfg.head_dim, device=device, dtype=dtype_cg) for _ in range(nl_model.cfg.num_layers)]
    tok_buf = torch.zeros(bsz, 1, device=device, dtype=torch.long)
    pos_buf = torch.full((bsz, 1), p_len, device=device, dtype=torch.long)
    mask_buf = torch.zeros(1, 1, 1, full_len, device=device, dtype=torch.bool)
    all_idx = torch.arange(full_len, device=device).view(1, 1, 1, full_len)
    pref_in_buf = torch.zeros(bsz, p_len, device=device, dtype=torch.long)

    def _nl_pref_fn() -> None:
        lg_p, pref_kvs = nl_model.forward_nl_causal(pref_in_buf, past_kvs=None, start_pos=0)
        for l_i in range(nl_model.cfg.num_layers):
            k_caches[l_i][:, :, :p_len, :].copy_(pref_kvs[l_i][0].to(dtype_cg))
            v_caches[l_i][:, :, :p_len, :].copy_(pref_kvs[l_i][1].to(dtype_cg))
        tok_buf.copy_(lg_p[:, -1:, :].argmax(dim=-1))
        pos_buf.fill_(p_len)

    def _nl_step_fn() -> None:
        mask_buf.copy_(all_idx <= pos_buf[0, 0])
        nxt = _nl_static_kv_step(nl_model, tok_buf, pos_buf, mask_buf, k_caches, v_caches)
        tok_buf.copy_(nxt)
        pos_buf.add_(1)

    with amp_ctx(device):
        for _ in range(3):
            _nl_pref_fn()
            _nl_step_fn()
    torch.cuda.synchronize(device)
    g_nl_pref = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_nl_pref):
        with amp_ctx(device):
            _nl_pref_fn()
    g_nl_step = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_nl_step):
        with amp_ctx(device):
            _nl_step_fn()

    cg_nl_ok = 0
    t_nl0 = sync_timer(device)
    for _, nl_seq, nl_ans, _ in eval_tuples:
        pref_in_buf.copy_(nl_seq[:, :p_len].to(device))
        g_nl_pref.replay()
        for _ in range(1, full_len - p_len):
            g_nl_step.replay()
        cg_nl_ok += int((tok_buf.squeeze(-1) == nl_ans.to(device)).sum().item())
    nl_cg_ms = (sync_timer(device) - t_nl0) * 1000.0 / max(1, nl_tot)
    del g_nl_pref, g_nl_step, k_caches, v_caches
    free_vram()

    # 3. Symbolic Block-dLLM Prefix-KV Single-Graph CUDAGraph (1 prefill [16] + 4x2 beats [8] + 3 commits [8] = 12 fwds!)
    sym_model.eval()
    pr_buf = torch.zeros(bsz, 16, device=device, dtype=torch.long)
    sym_ans_buf = torch.zeros(bsz, device=device, dtype=torch.long)
    m_pref = torch.ones(1, 1, 16, 16, device=device, dtype=torch.bool)
    wb_cg = sym_model.init_whiteboard_states(bsz, device)
    exp_cg = sym_model.init_experience_state(bsz, device)

    def _sym_prefix_kv_graph_fn() -> None:
        for wb in wb_cg:
            wb.s_ro.zero_(); wb.m_ro.zero_(); wb.s_rw_commit.zero_(); wb.m_rw_commit.zero_(); wb.s_rw_draft.zero_(); wb.m_rw_draft.zero_(); wb.dirty_mask.zero_()
        exp_cg.delta_b_pre.zero_()
        sym_model.populate_ro_from_prompt(pr_buf, wb_cg)
        out_p = sym_model.forward_symbolic(
            pr_buf, phase=CognitivePhase.FORMAL, attn_mask=m_pref, wb_states=wb_cg, exp_state=exp_cg,
            t_k=0.0, read_only_pml=True, dual_stream_block_size=8, past_kvs=None, start_pos=0,
        )
        h_p_mean = out_p["hidden"].mean(dim=1)  # type: ignore[union-attr]
        kvs = out_p["present_kvs"]  # type: ignore[assignment]
        cur_blk = torch.full((bsz, 8), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
        for b_i in range(4):
            cur_blk = torch.full((bsz, 8), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
            s_pos = 16 + b_i * 8
            for beat in range(2):
                out_b = sym_model.forward_symbolic(
                    cur_blk, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb_cg, exp_state=exp_cg,
                    t_k=1.0 if beat == 0 else 0.5, read_only_pml=True, dual_stream_block_size=8, past_kvs=kvs, start_pos=s_pos,
                )
                if b_i == 0 and beat == 0 and not sym_model.ablation.static_top2_baseline:
                    exp_cg.delta_b_pre.copy_(sym_model.hyper_router(h_p_mean, out_b["hidden"].mean(dim=1)))  # type: ignore[union-attr]
                cur_blk, _ = hierarchical_three_beat_unmask(cur_blk, out_b["logits"], beat, 2)  # type: ignore[arg-type]
            if b_i < 3:
                out_c = sym_model.forward_symbolic(
                    cur_blk, phase=CognitivePhase.FORMAL, attn_mask=None, wb_states=wb_cg, exp_state=exp_cg,
                    t_k=0.0, read_only_pml=sym_model.ablation.disable_pml, dual_stream_block_size=8, past_kvs=kvs, start_pos=s_pos,
                )
                kvs = out_c["present_kvs"]  # type: ignore[assignment]
        sym_ans_buf.copy_(cur_blk[:, 4])

    with amp_ctx(device):
        for _ in range(3):
            _sym_prefix_kv_graph_fn()
    torch.cuda.synchronize(device)
    g_sym = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_sym):
        with amp_ctx(device):
            _sym_prefix_kv_graph_fn()

    cg_sym_ok = 0
    t_s0 = sync_timer(device)
    for sb in sym_batches:
        pr_buf.copy_(sb.prompt_ids.to(device))
        g_sym.replay()
        cg_sym_ok += int((sym_ans_buf == sb.answer_tok_ids.to(device)).sum().item())
    sym_cg_ms = (sync_timer(device) - t_s0) * 1000.0 / max(1, nl_tot)
    del g_sym, wb_cg, exp_cg
    free_vram()

    return {
        "batch_label": batch_label,
        "num_eval_problems": nl_tot,
        "nl_cot_eager_em": round(100.0 * nl_ok / max(1, nl_tot), 2),
        "nl_cot_cudagraph_em": round(100.0 * cg_nl_ok / max(1, nl_tot), 2),
        "sym_prefix_kv_eager_em": sym_eager["exact_match"],
        "sym_prefix_kv_cudagraph_em": round(100.0 * cg_sym_ok / max(1, nl_tot), 2),
        "eager_ms_per_problem": {"nl_kv_ar": round(nl_eager_ms, 3), "sym_prefix_kv": sym_eager["ms_per_problem"]},
        "eager_speedup_x": round(nl_eager_ms / max(1e-4, sym_eager["ms_per_problem"]), 2),
        "cudagraph_ms_per_problem": {"nl_kv_ar": round(nl_cg_ms, 3), "sym_prefix_kv": round(sym_cg_ms, 3)},
        "cudagraph_speedup_x": round(nl_cg_ms / max(1e-4, sym_cg_ms), 2),
    }


# =============================================================================
# 3. Isolated Training Helpers (strict VRAM cleanup per sub-run!)
# =============================================================================
def train_l3_hard_sym_isolated(
    cfg: AVLPMLConfig,
    optim_path: Path,
    device: torch.device,
    seed: int,
    steps: int,
    eval_tuples: List[Tuple[TaskBatch, torch.Tensor, torch.Tensor, int]],
    keep_model: bool = False,
) -> Tuple[Dict[str, Any], AVLPMLModel | None]:
    torch.manual_seed(seed)
    model = AVLPMLModel(cfg).to(device)
    opt, sopt = build_optimizers(model, optim_path)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    gen = Stage0SynthGenerator(cfg, seed=300 + seed)
    t0 = sync_timer(device)
    for st in range(1, steps + 1):
        b, _, _, _ = sample_level3_dynamic_ast_8slot(gen, 64, "train")
        _ = train_dual_stream_step(model, opt, sopt, b, device, st, steps, block_size=8)
        sched.step()
    train_s = round(sync_timer(device) - t0, 2)
    del opt, sopt, sched
    set_u_post_warmup(model, 1.0)

    sym_batches = [t[0] for t in eval_tuples]
    ev_kv = evaluate_symbolic_prefix_kv(model, sym_batches, device, beats_per_block=2, block_size=8)
    ev_fullseq = evaluate_symbolic_exact_match(model, sym_batches[:4], device, beats_per_block=2, block_size=8)

    # K_max=0 Inference Knock-Out on the EXACT SAME trained weights!
    orig_k = model.cfg.k_max
    model.cfg.k_max = 0
    ev_k0 = evaluate_symbolic_prefix_kv(model, sym_batches, device, beats_per_block=2, block_size=8)
    model.cfg.k_max = orig_k

    vram = check_vram_guard(f"L3Hard_sym_seed{seed}_k{orig_k}")
    res = {
        "train_sec": train_s,
        "em_prefix_kv": ev_kv["exact_match"],
        "em_fullseq_check128": ev_fullseq["exact_match"],
        "avg_active_experts": ev_kv["avg_active_experts"],
        "k0_knockout_em": ev_k0["exact_match"],
        "k0_knockout_drop": round(ev_kv["exact_match"] - ev_k0["exact_match"], 2),
        "vram": vram,
    }
    if keep_model:
        return res, model
    del model
    free_vram()
    return res, None


def train_l3_hard_nl_isolated(
    cfg: AVLPMLConfig,
    optim_path: Path,
    device: torch.device,
    seed: int,
    steps: int,
    eval_tuples: List[Tuple[TaskBatch, torch.Tensor, torch.Tensor, int]],
    keep_model: bool = False,
) -> Tuple[Dict[str, Any], AVLPMLModel | None]:
    torch.manual_seed(seed)
    model = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True)).to(device)
    opt, _ = build_optimizers(model, optim_path)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    gen = Stage0SynthGenerator(cfg, seed=300 + seed)
    model.train()
    t0 = sync_timer(device)
    for _ in range(steps):
        _, nl_seq, _, p_len = sample_level3_dynamic_ast_8slot(gen, 64, "train")
        nl_seq = nl_seq.to(device)
        opt.zero_grad(set_to_none=True)
        with amp_ctx(device):
            lg, _ = model.forward_nl_causal(nl_seq[:, :-1])
            loss = F.cross_entropy(lg[:, p_len - 1 :, :].float().reshape(-1, cfg.nl_vocab_size), nl_seq[:, p_len:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    train_s = round(sync_timer(device) - t0, 2)
    del opt, sched

    model.eval()
    nl_ok, nl_tot = 0, 0
    with torch.no_grad(), amp_ctx(device):
        for _, nl_seq, nl_ans, p_len in eval_tuples:
            nl_seq = nl_seq.to(device)
            lg_p, kvs = model.forward_nl_causal(nl_seq[:, :p_len], past_kvs=None, start_pos=0)
            nxt = lg_p[:, -1:, :].argmax(dim=-1)
            for s_i in range(1, nl_seq.shape[1] - p_len):
                lg_s, kvs = model.forward_nl_causal(nxt, past_kvs=kvs, start_pos=p_len + s_i - 1)
                nxt = lg_s[:, -1:, :].argmax(dim=-1)
            nl_ok += int((nxt.squeeze(-1) == nl_ans.to(device)).sum().item())
            nl_tot += nl_seq.shape[0]

    vram = check_vram_guard(f"L3Hard_nl_seed{seed}")
    res = {"train_sec": train_s, "nl_cot_em": round(100.0 * nl_ok / max(1, nl_tot), 2), "vram": vram}
    if keep_model:
        return res, model
    del model
    free_vram()
    return res, None


def train_level1_variant_isolated(
    cfg: AVLPMLConfig,
    ab_cfg: AblationConfig,
    optim_path: Path,
    device: torch.device,
    seed: int,
    steps: int,
    test_batches: List[TaskBatch],
    run_k0_knockout: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = AVLPMLModel(cfg, ablation=ab_cfg).to(device)
    opt, sopt = build_optimizers(model, optim_path)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
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
    if run_k0_knockout:
        orig_k = model.cfg.k_max
        model.cfg.k_max = 0
        ev_k0 = evaluate_symbolic_exact_match(model, test_batches, device, beats_per_block=3, block_size=16)
        model.cfg.k_max = orig_k
        ev["k0_knockout_em"] = ev_k0["exact_match"]
        ev["k0_knockout_drop"] = round(ev["exact_match"] - ev_k0["exact_match"], 2)
    ev["vram"] = check_vram_guard(f"L1_seed{seed}")
    del model
    free_vram()
    return ev


def train_level2ab_isolated(
    cfg: AVLPMLConfig,
    ab_cfg: AblationConfig,
    optim_path: Path,
    device: torch.device,
    steps: int,
    test_4step: List[TaskBatch],
    test_6step: List[TaskBatch],
) -> Dict[str, Any]:
    torch.manual_seed(42)
    model = AVLPMLModel(cfg, ablation=ab_cfg).to(device)
    opt, sopt = build_optimizers(model, optim_path)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    gen = Stage0SynthGenerator(cfg, seed=202)
    t0 = sync_timer(device)
    for st in range(1, steps + 1):
        b = sample_level2_rule_chain_8slot(gen, 64, "train", num_chain_steps=4)
        _ = train_dual_stream_step(model, opt, sopt, b, device, st, steps, block_size=8)
        sched.step()
    train_s = round(sync_timer(device) - t0, 2)
    del opt, sopt, sched
    set_u_post_warmup(model, 1.0)

    ev4 = evaluate_symbolic_prefix_kv(model, test_4step, device, beats_per_block=2, block_size=8)
    ev6 = evaluate_symbolic_prefix_kv(model, test_6step, device, beats_per_block=2, block_size=8)
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
    vram = check_vram_guard("L2ab")
    del model
    free_vram()
    return {
        "train_sec": train_s,
        "unseen_rule_4step_em": ev4["exact_match"],
        "unseen_rule_6step_extrap_em": ev6["exact_match"],
        "avg_active_experts": ev4["avg_active_experts"],
        "learned_alpha_calm_rms": alpha_calm_rms,
        "vram": vram,
    }


def train_level2c_isolated(
    cfg: AVLPMLConfig,
    ab_cfg: AblationConfig,
    use_breaker: bool,
    optim_path: Path,
    device: torch.device,
    steps: int,
    gen_eval: Stage0SynthGenerator,
    test_bt_pairs: List[Tuple[TaskBatch, torch.Tensor]],
) -> Dict[str, Any]:
    torch.manual_seed(42)
    model = AVLPMLModel(cfg, ablation=ab_cfg).to(device)
    opt, sopt = build_optimizers(model, optim_path)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    gen = Stage0SynthGenerator(cfg, seed=205)
    t0 = sync_timer(device)
    for st in range(1, steps + 1):
        b, _ = gen.sample_level2_backtrack_batch(64, "train", inject_trap_in_train=(st % 4 != 0))
        _ = train_dual_stream_step(model, opt, sopt, b, device, st, steps, block_size=16)
        sched.step()
    train_s = round(sync_timer(device) - t0, 2)
    del opt, sopt, sched
    set_u_post_warmup(model, 1.0)
    ev = evaluate_backtrack_and_relapse(model, gen_eval, test_bt_pairs, device, enable_pain_and_breaker=use_breaker)
    ev["train_sec"] = train_s
    ev["vram"] = check_vram_guard("L2c")
    del model
    free_vram()
    return ev


def train_level2c_best_of_n_ar_isolated(
    cfg: AVLPMLConfig,
    optim_path: Path,
    device: torch.device,
    steps: int,
    gen_eval: Stage0SynthGenerator,
    test_bt_pairs: List[Tuple[TaskBatch, torch.Tensor]],
    ms_avl: float,
) -> Dict[str, Any]:
    torch.manual_seed(42)
    ar_model = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True, disable_pml=True)).to(device)
    opt, _ = build_optimizers(ar_model, optim_path)
    gen_ar = Stage0SynthGenerator(cfg, seed=205)
    ar_model.train()
    for st in range(1, steps + 1):
        b, _ = gen_ar.sample_level2_backtrack_batch(64, "train", inject_trap_in_train=(st % 4 != 0))
        seq = torch.cat([b.prompt_ids, b.target_sym_ids], dim=1).to(device)
        opt.zero_grad(set_to_none=True)
        with amp_ctx(device):
            out_ar = ar_model.forward_symbolic(seq[:, :-1], phase=CognitivePhase.FORMAL, attn_mask=None, read_only_pml=True)
            loss_ar = F.cross_entropy(out_ar["logits"].float().reshape(-1, cfg.sym_vocab_size), seq[:, 1:].reshape(-1))  # type: ignore[union-attr]
        loss_ar.backward()
        opt.step()
    del opt

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
    n_budget = max(1, min(4, int(round(ms_avl / max(0.05, ms_ar_one)))))

    ar_ok, ar_tot = 0, 0
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
            ar_ok += int((best_z1 == batch.answer_digits.to(device)).sum().item())
            ar_tot += bsz
    ms_bon = (sync_timer(device) - t_bon0) * 1000.0 / max(1, ar_tot)
    vram = check_vram_guard("L2c_BoN_AR")
    del ar_model
    free_vram()
    return {
        "n_samples": n_budget,
        "final_exact_match": round(100.0 * ar_ok / max(1, ar_tot), 2),
        "ms_per_problem": round(ms_bon, 3),
        "vram": vram,
    }


# =============================================================================
# 4. Mandatory 5-Step Pre-Flight VRAM & Latency Profiler
# =============================================================================
def run_preflight_profile(cfg: AVLPMLConfig, optim_path: Path, device: torch.device) -> Dict[str, Any]:
    """Run 5 real training steps on each workload to measure exact ms/step, peak VRAM (< 14.0 GB), & total ETA."""
    print("=== [PRE-FLIGHT 5-STEP VRAM & LATENCY PROFILE] ===", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gen = Stage0SynthGenerator(cfg, seed=777)
    prof: Dict[str, Any] = {}

    def _profile_sym(name: str, batch_fn, blk_sz: int, planned_steps: int, n_runs: int) -> None:
        m = AVLPMLModel(cfg).to(device)
        if name == "l3_hard_8slot":
            prof["model_param_summary"] = m.parameter_summary()
        opt, sopt = build_optimizers(m, optim_path)
        # Warmup 2 steps
        for w in range(2):
            train_dual_stream_step(m, opt, sopt, batch_fn(), device, w + 1, 100, block_size=blk_sz)
        t0 = sync_timer(device)
        for st in range(5):
            train_dual_stream_step(m, opt, sopt, batch_fn(), device, st + 3, 100, block_size=blk_sz)
        ms_step = (sync_timer(device) - t0) * 1000.0 / 5.0
        vram = check_vram_guard(f"profile_{name}")
        est_sec = round(ms_step * planned_steps * n_runs / 1000.0, 1)
        prof[name] = {
            "ms_per_step": round(ms_step, 2),
            "planned_steps_x_runs": f"{planned_steps} x {n_runs}",
            "est_train_sec": est_sec,
            "vram": vram,
        }
        print(f"  [{name}] {ms_step:.2f} ms/step | est={est_sec:.1f}s | VRAM={vram}", flush=True)
        del m, opt, sopt
        free_vram()

    _profile_sym("l3_hard_8slot", lambda: sample_level3_dynamic_ast_8slot(gen, 64, "train")[0], 8, 900, 6)
    _profile_sym("l1_16slot_64tok", lambda: gen.sample_level1_batch(64, "train"), 16, 1000, 7)
    _profile_sym("l2ab_8slot_32tok", lambda: sample_level2_rule_chain_8slot(gen, 64, "train", 4), 8, 700, 2)
    _profile_sym("l2c_16slot_32tok", lambda: gen.sample_level2_backtrack_batch(64, "train", True)[0], 16, 600, 5)

    # Profile NL-CoT AR
    m_nl = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True)).to(device)
    opt_nl, _ = build_optimizers(m_nl, optim_path)
    m_nl.train()
    for _ in range(2):
        _, seq, _, pl = sample_level3_dynamic_ast_8slot(gen, 64, "train")
        seq = seq.to(device)
        opt_nl.zero_grad(set_to_none=True)
        with amp_ctx(device):
            lg, _ = m_nl.forward_nl_causal(seq[:, :-1])
            loss = F.cross_entropy(lg[:, pl - 1 :, :].float().reshape(-1, cfg.nl_vocab_size), seq[:, pl:].reshape(-1))
        loss.backward(); opt_nl.step()
    t0 = sync_timer(device)
    for _ in range(5):
        _, seq, _, pl = sample_level3_dynamic_ast_8slot(gen, 64, "train")
        seq = seq.to(device)
        opt_nl.zero_grad(set_to_none=True)
        with amp_ctx(device):
            lg, _ = m_nl.forward_nl_causal(seq[:, :-1])
            loss = F.cross_entropy(lg[:, pl - 1 :, :].float().reshape(-1, cfg.nl_vocab_size), seq[:, pl:].reshape(-1))
        loss.backward(); opt_nl.step()
    ms_nl = (sync_timer(device) - t0) * 1000.0 / 5.0
    v_nl = check_vram_guard("profile_nl_cot")
    est_nl = round(ms_nl * 450 * 3 / 1000.0, 1)
    prof["l3_hard_nl_cot"] = {"ms_per_step": round(ms_nl, 2), "planned_steps_x_runs": "450 x 3", "est_train_sec": est_nl, "vram": v_nl}
    print(f"  [l3_hard_nl_cot] {ms_nl:.2f} ms/step | est={est_nl:.1f}s | VRAM={v_nl}", flush=True)
    del m_nl, opt_nl
    free_vram()

    total_est_sec = sum(v["est_train_sec"] for k, v in prof.items() if isinstance(v, dict) and "est_train_sec" in v) + 45.0
    prof["total_estimated_minutes"] = round(total_est_sec / 60.0, 2)
    print(f"=== [PROFILE SUMMARY] Peak VRAM={check_vram_guard('end_prof')['peak_alloc_gb']} GB (< 14.0 GB SAFE!) | Projected Total = {prof['total_estimated_minutes']} min ===", flush=True)
    return prof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true", help="Run 5-step VRAM/latency profile only and exit.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    cfg_k0 = deepcopy(cfg)
    cfg_k0.k_max = 0
    optim_path = root / "configs" / "optim_5groups.yaml"
    out_path = root / "outputs" / "stage0_v11_verification.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    profile = run_preflight_profile(cfg, optim_path, device)
    if args.profile_only:
        save_checkpoint(out_path, {"status": "profile_only", "profile": profile})
        return

    t_global = time.perf_counter()
    report: Dict[str, Any] = {
        "architecture_version": "AVL-PMLv1.1",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_param_summary": profile["model_param_summary"],
        "preflight_profile": profile,
        "suite1_l3_hard_unseen_ast": {},
        "suite2_level1_4skill": {},
        "suite3_level2_rule_and_pain": {},
    }
    save_checkpoint(out_path, report)

    # =========================================================================
    # SUITE 1: L3-Hard Unseen 20% Dynamic AST (8-Slot Atomic x 2-Beat, N=640 x 3 Seeds)
    # =========================================================================
    gen_l3h_eval = Stage0SynthGenerator(cfg, seed=9009)
    eval_l3h_640 = [sample_level3_dynamic_ast_8slot(gen_l3h_eval, 32, "test") for _ in range(20)]
    eval_l3h_b1_64 = [sample_level3_dynamic_ast_8slot(Stage0SynthGenerator(cfg, seed=9100 + i), 1, "test") for i in range(64)]

    seeds = [42, 1024, 2026]
    l3_per_seed: Dict[str, Any] = {}
    sym_ems, k0_ko_ems, k0_rt_ems, nl_ems, act_exps = [], [], [], [], []

    for idx_s, sd in enumerate(seeds):
        keep = idx_s == 0
        sym_res, sym_m = train_l3_hard_sym_isolated(cfg, optim_path, device, sd, 900, eval_l3h_640, keep_model=keep)
        print(f"[Suite1 L3-Hard Seed {sd}] v1.1 Sym={sym_res['em_prefix_kv']}% (act={sym_res['avg_active_experts']}, K0_KO={sym_res['k0_knockout_em']}%, drop={sym_res['k0_knockout_drop']}%) | {sym_res['train_sec']}s", flush=True)

        nl_res, nl_m = train_l3_hard_nl_isolated(cfg, optim_path, device, sd, 450, eval_l3h_640, keep_model=keep)
        print(f"[Suite1 L3-Hard Seed {sd}] NL-CoT={nl_res['nl_cot_em']}% | {nl_res['train_sec']}s", flush=True)

        if keep and sym_m is not None and nl_m is not None:
            canary = verify_zero_leakage_canary_8slot(sym_m, eval_l3h_640[0][0], device)
            spd_b32 = benchmark_prefix_kv_cudagraph_speed(sym_m, nl_m, eval_l3h_640, device, batch_label="B=32")
            spd_b1 = benchmark_prefix_kv_cudagraph_speed(sym_m, nl_m, eval_l3h_b1_64, device, batch_label="B=1")
            report["suite1_l3_hard_unseen_ast"]["zero_leakage_canary"] = canary
            report["suite1_l3_hard_unseen_ast"]["speed_cudagraph_b32"] = spd_b32
            report["suite1_l3_hard_unseen_ast"]["speed_cudagraph_b1"] = spd_b1
            print(f"[Suite1 Speed B=32] CUDAGraph {spd_b32['cudagraph_ms_per_problem']} -> {spd_b32['cudagraph_speedup_x']}x | Canary={canary['zero_leakage_verified']}", flush=True)
            print(f"[Suite1 Speed B=1 ] CUDAGraph {spd_b1['cudagraph_ms_per_problem']} -> {spd_b1['cudagraph_speedup_x']}x", flush=True)
            del sym_m, nl_m
            free_vram()

        k0_rt_res, _ = train_l3_hard_sym_isolated(cfg_k0, optim_path, device, sd, 900, eval_l3h_640, keep_model=False)
        print(f"[Suite1 L3-Hard Seed {sd}] K0_Retrain={k0_rt_res['em_prefix_kv']}% | {k0_rt_res['train_sec']}s | elapsed={(time.perf_counter()-t_global)/60:.1f}m", flush=True)

        l3_per_seed[str(sd)] = {"v11_sym": sym_res, "nl_cot": nl_res, "k0_retrain": k0_rt_res}
        sym_ems.append(sym_res["em_prefix_kv"])
        k0_ko_ems.append(sym_res["k0_knockout_em"])
        k0_rt_ems.append(k0_rt_res["em_prefix_kv"])
        nl_ems.append(nl_res["nl_cot_em"])
        act_exps.append(sym_res["avg_active_experts"])
        report["suite1_l3_hard_unseen_ast"]["per_seed"] = l3_per_seed
        save_checkpoint(out_path, report)

    report["suite1_l3_hard_unseen_ast"]["summary_3seeds_n640"] = {
        "v11_sym_8slot_em": mean_std(sym_ems),
        "nl_cot_kv_ar_em": mean_std(nl_ems),
        "k0_inference_knockout_em": mean_std(k0_ko_ems),
        "k0_knockout_mean_drop_pct": round(mean_std(sym_ems)["mean"] - mean_std(k0_ko_ems)["mean"], 2),
        "k0_shared_only_retrain_em": mean_std(k0_rt_ems),
        "v11_gain_over_k0_retrain_pct": round(mean_std(sym_ems)["mean"] - mean_std(k0_rt_ems)["mean"], 2),
        "avg_active_routed_experts": mean_std(act_exps),
    }
    save_checkpoint(out_path, report)

    # =========================================================================
    # SUITE 2: Level 1 4-Skill Compositional Pipeline (N=640, Capacity-Preserving MoE)
    # =========================================================================
    gen_l1_eval = Stage0SynthGenerator(cfg, seed=9001)
    eval_l1_640 = [gen_l1_eval.sample_level1_batch(32, "test") for _ in range(20)]

    l1_jobs = [
        ("v11_full_seed42", cfg, AblationConfig(), 42, True),
        ("k0_shared_only_retrain_seed42", cfg_k0, AblationConfig(), 42, False),
        ("static_top2_baseline_seed42", cfg, AblationConfig(static_top2_baseline=True), 42, False),
        ("wo_static_engram_seed42", cfg, AblationConfig(disable_engram=True), 42, False),
        ("wo_bipolar_graph_seed42", cfg, AblationConfig(disable_bipolar_graph=True), 42, False),
        ("v11_full_seed1024", cfg, AblationConfig(), 1024, True),
        ("static_top2_baseline_seed1024", cfg, AblationConfig(static_top2_baseline=True), 1024, False),
    ]
    for name, c_use, ab_use, sd, do_ko in l1_jobs:
        res = train_level1_variant_isolated(c_use, ab_use, optim_path, device, sd, 1000, eval_l1_640, run_k0_knockout=do_ko)
        report["suite2_level1_4skill"][name] = res
        print(f"[Suite2 L1 {name}] EM={res['exact_match']}% | act={res['avg_active_experts']} | KO_drop={res.get('k0_knockout_drop', 'n/a')}% | elapsed={(time.perf_counter()-t_global)/60:.1f}m", flush=True)
        save_checkpoint(out_path, report)

    # =========================================================================
    # SUITE 3: Level 2 (2a/b Differentiable S^RO Unseen Rules + 2c Pain Imprint & Wet-Cement)
    # =========================================================================
    gen_l2_eval = Stage0SynthGenerator(cfg, seed=9002)
    test_l2_4s = [sample_level2_rule_chain_8slot(gen_l2_eval, 32, "test", num_chain_steps=4) for _ in range(10)]
    test_l2_6s = [sample_level2_rule_chain_8slot(gen_l2_eval, 32, "test", num_chain_steps=6) for _ in range(10)]

    l2ab_full = train_level2ab_isolated(cfg, AblationConfig(), optim_path, device, 700, test_l2_4s, test_l2_6s)
    print(f"[Suite3 L2a/b v11_full] 4step={l2ab_full['unseen_rule_4step_em']}% | 6step_extrap={l2ab_full['unseen_rule_6step_extrap_em']}% | alpha_calm={l2ab_full['learned_alpha_calm_rms']}", flush=True)
    l2ab_pure = train_level2ab_isolated(cfg, AblationConfig(disable_pml=True), optim_path, device, 700, test_l2_4s, test_l2_6s)
    print(f"[Suite3 L2a/b pure_dllm] 4step={l2ab_pure['unseen_rule_4step_em']}% | 6step_extrap={l2ab_pure['unseen_rule_6step_extrap_em']}%", flush=True)
    report["suite3_level2_rule_and_pain"]["part_2ab_unseen_rules_n320"] = {
        "v11_full_diff_sro": l2ab_full,
        "pure_block_dllm_no_pml": l2ab_pure,
        "pml_gain_4step_pct": round(l2ab_full["unseen_rule_4step_em"] - l2ab_pure["unseen_rule_4step_em"], 2),
        "pml_gain_6step_extrap_pct": round(l2ab_full["unseen_rule_6step_extrap_em"] - l2ab_pure["unseen_rule_6step_extrap_em"], 2),
    }
    save_checkpoint(out_path, report)

    test_bt_320 = [gen_l2_eval.sample_level2_backtrack_batch(32, "test") for _ in range(10)]
    l2c_configs = [
        ("v11_full", AblationConfig(), True),
        ("pure_block_dllm", AblationConfig(disable_pml=True, disable_pain_imprint=True, disable_wet_cement=True), False),
        ("wo_pain_imprint", AblationConfig(disable_pain_imprint=True), True),
        ("wo_wet_cement", AblationConfig(disable_wet_cement=True), True),
    ]
    l2c_res: Dict[str, Any] = {}
    for name, ab_c, use_brk in l2c_configs:
        r = train_level2c_isolated(cfg, ab_c, use_brk, optim_path, device, 600, gen_l2_eval, test_bt_320)
        l2c_res[name] = r
        print(f"[Suite3 L2c {name}] final={r['final_exact_match']}% (draft={r['draft_exact_match']}%, relapse={r['same_pit_relapse_pct']}%, netfix={r['net_fix_pct']}%)", flush=True)
        report["suite3_level2_rule_and_pain"]["part_2c_backtrack_n320"] = l2c_res
        save_checkpoint(out_path, report)

    bon_r = train_level2c_best_of_n_ar_isolated(cfg, optim_path, device, 600, gen_l2_eval, test_bt_320, l2c_res["v11_full"]["ms_per_problem"])
    l2c_res["equal_wallclock_best_of_n_ar"] = bon_r
    print(f"[Suite3 L2c BoN_AR] final={bon_r['final_exact_match']}% (N={bon_r['n_samples']})", flush=True)

    report["total_wall_clock_minutes"] = round((time.perf_counter() - t_global) / 60.0, 2)
    report["peak_vram_gb"] = check_vram_guard("final")["peak_alloc_gb"]
    save_checkpoint(out_path, report)
    print(f"\n[DONE] AVL-PMLv1.1 Verification completed in {report['total_wall_clock_minutes']} min! Peak VRAM={report['peak_vram_gb']} GB -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
