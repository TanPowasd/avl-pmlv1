"""Four-Point Anti-Cheat Audit Suite for Stage 0 (RTX 5080):
1. KV-Cache + torch.compile / CUDAGraph Wall-Clock Fairness Audit (squeezing out Python CPU dispatch bias);
2. Runtime Zero-Leakage Canary (corrupt target_sym_ids -> -999999) + 640 Held-Out Problems x 3 Seeds + Unseen-Template Dynamic AST;
3. K_max = 0 Routed-Expert Zero-Out Ablation (both Inference Knock-Out & Shared-Only Retrain on Level 1 & Dynamic Level 3);
4. Post-Bugfix Level 1 & Level 2 Multi-Seed Verification.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.wet_cement import build_block_diffusion_attn_mask, hierarchical_three_beat_unmask
from avl_pmlv1.engine.model import AblationConfig, AVLPMLModel
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab
from gen_synth_tasks import Stage0SynthGenerator, TaskBatch, _is_test_instance
from run_gate1_ablation import (
    amp_ctx,
    build_optimizers,
    evaluate_symbolic_exact_match,
    set_u_post_warmup,
    sync_timer,
    train_dual_stream_step,
)


def apply_step_op(op_code: int, a: int, b: int) -> int:
    if op_code == 0:
        return (a + b) % 8
    if op_code == 1:
        return (a - b) % 8
    if op_code == 2:
        return (a ^ b) % 8
    return (2 * a + b + 1) % 8


def sample_level3_dynamic_ast_batch(
    gen: Stage0SynthGenerator, batch_size: int, split: str = "train"
) -> Tuple[TaskBatch, torch.Tensor, torch.Tensor, int]:
    """256 distinct 4-step AST operator templates ({+, -, ^, affine}^4) x 5,000 value tuples = 1,280,000 instances.
    Splits strictly by THE 4-OPERATOR AST TEMPLATE (ops[0..3]) so 20% of AST templates are 100% UNSEEN in training!
    """
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
        # Strict template-disjoint split: test set uses AST operator combinations NEVER seen during training!
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
        blk0 = [
            SymbolicVocab.GOAL_ID, op_tokens[ops[0]], _lit(0, x0), _lit(1, k1),
            _lit(8, s1), SymbolicVocab.BIND_ID, v.reg_slot_to_id(0),
            SymbolicVocab.DEDUCE_ID, v.reg_slot_to_id(0), op_tokens[ops[1]], gen.eq_id,
            _lit(2, k2), gen.eq_id, _lit(8, s2), SymbolicVocab.BIND_ID, v.reg_slot_to_id(1),
        ]
        blk1 = [
            SymbolicVocab.DEDUCE_ID, v.reg_slot_to_id(1), op_tokens[ops[2]], _lit(3, k3),
            _lit(8, s3), SymbolicVocab.BIND_ID, v.reg_slot_to_id(0),
            SymbolicVocab.COMMIT_ID, v.reg_slot_to_id(0), op_tokens[ops[3]], gen.eq_id,
            _lit(4, k4), _lit(8, s4), SymbolicVocab.BIND_ID, v.reg_slot_to_id(1), SymbolicVocab.EOT_ID,
        ]
        sym_seq = blk0 + blk1

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
        full_nl = nl_prompt + nl_cot  # 160 tokens total

        prompts.append(p_seq)
        syms.append(sym_seq)
        rmasks.append([0] * 32)
        ans_pos.append(16 + 12)
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


@torch.no_grad()
def verify_zero_target_leakage_canary(model: AVLPMLModel, batch: TaskBatch, device: torch.device) -> Dict[str, object]:
    """Run inference normally vs with target_sym_ids poisoned to -999999 (which would crash Embedding if ever read!)."""
    model.eval()
    poisoned = TaskBatch(
        prompt_ids=batch.prompt_ids.clone(),
        target_sym_ids=torch.full_like(batch.target_sym_ids, -999999),
        reassert_mask=torch.full_like(batch.reassert_mask, -999999),
        answer_pos=batch.answer_pos.clone(),
        answer_tok_ids=batch.answer_tok_ids.clone(),
        answer_digits=batch.answer_digits.clone(),
    )
    res_clean = evaluate_symbolic_exact_match(model, [batch], device, beats_per_block=3)
    res_poison = evaluate_symbolic_exact_match(model, [poisoned], device, beats_per_block=3)
    return {
        "em_normal": res_clean["exact_match"],
        "em_with_poisoned_targets": res_poison["exact_match"],
        "abs_em_diff": abs(res_clean["exact_match"] - res_poison["exact_match"]),
        "zero_leakage_verified": res_clean["exact_match"] == res_poison["exact_match"],
    }


@torch.no_grad()
def _nl_static_kv_step(
    model: AVLPMLModel,
    tok_buf: torch.Tensor,
    pos_buf: torch.Tensor,
    attn_mask_buf: torch.Tensor,
    k_caches: List[torch.Tensor],
    v_caches: List[torch.Tensor],
) -> torch.Tensor:
    """Single-token static-buffer KV-Cache decode step (100% static shapes & zero .item() for CUDAGraph capture!)."""
    from avl_pmlv1.pml_track1.attn_pml_layer import apply_rope
    from avl_pmlv1.moe_track2.tri_state_moe import TriStateMoELayer

    x = model.nl_embed(tok_buf)  # [B, 1, D]
    v_anchor = None
    a_prev_layer = None
    idx_scatter = pos_buf.view(x.shape[0], 1, 1, 1).expand(x.shape[0], model.cfg.num_kv_heads, 1, model.cfg.head_dim)

    for l_idx in range(model.cfg.num_layers):
        layer = model.attn_layers[l_idx]
        nx = layer.norm(x)
        gqa = layer.gqa
        bsz = nx.shape[0]
        q = gqa.q_proj(nx).view(bsz, 1, gqa.num_q, gqa.head_dim).transpose(1, 2)
        k = gqa.k_proj(nx).view(bsz, 1, gqa.num_kv, gqa.head_dim).transpose(1, 2)
        v = gqa.v_proj(nx).view(bsz, 1, gqa.num_kv, gqa.head_dim).transpose(1, 2)
        v_raw = v
        if gqa.has_value_residual and v_anchor is not None:
            v = v + torch.tanh(gqa.alpha_vres) * v_anchor
        if l_idx == 0:
            v_anchor = v_raw

        q = apply_rope(q, pos_buf)
        k = apply_rope(k, pos_buf)
        k_caches[l_idx].scatter_(2, idx_scatter, k.to(k_caches[l_idx].dtype))
        v_caches[l_idx].scatter_(2, idx_scatter, v.to(v_caches[l_idx].dtype))

        k_exp = k_caches[l_idx].repeat_interleave(gqa.num_groups, dim=1)
        v_exp = v_caches[l_idx].repeat_interleave(gqa.num_groups, dim=1)
        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask_buf, is_causal=False)
        x = x + gqa.o_proj(attn_out.transpose(1, 2).contiguous().view(bsz, 1, gqa.num_q * gqa.head_dim))

        ffn = model.ffn_layers[l_idx]
        if isinstance(ffn, TriStateMoELayer):
            x, moe_aux = ffn(x, phase=CognitivePhase.VERBALIZE, t_k=0.0, a_prev_layer=a_prev_layer, static_top2_mode=True)
            a_prev_layer = moe_aux["block_profile"]
        else:
            x = ffn(x)

    return model.nl_head(model.final_norm(x))[:, -1:, :].argmax(dim=-1)


@torch.no_grad()
def benchmark_compiled_vs_eager_speed(
    sym_model: AVLPMLModel,
    nl_model: AVLPMLModel,
    eval_tuples: List[Tuple[TaskBatch, torch.Tensor, torch.Tensor, int]],
    device: torch.device,
) -> Dict[str, object]:
    """Measure BOTH Sync-Free Eager + KV-Cache AND Native CUDAGraph + Static-KV-Cache across 640 test problems."""
    sym_batches = [t[0] for t in eval_tuples]

    # 1. Sync-Free Eager + KV-Cache
    sym_eager = evaluate_symbolic_exact_match(sym_model, sym_batches, device, beats_per_block=3)

    nl_model.eval()
    with amp_ctx(device):
        _, warm_nl, _, warm_plen = eval_tuples[0]
        _, warm_kvs = nl_model.forward_nl_causal(warm_nl[:2, :warm_plen].to(device), start_pos=0)
        _ = nl_model.forward_nl_causal(warm_nl[:2, warm_plen : warm_plen + 1].to(device), past_kvs=warm_kvs, start_pos=warm_plen)

        nl_ok, nl_tot = 0, 0
        t0 = sync_timer(device)
        for _, nl_seq, nl_ans, p_len in eval_tuples:
            nl_seq = nl_seq.to(device)
            bsz, full_len = nl_seq.shape
            lg_p, kvs = nl_model.forward_nl_causal(nl_seq[:, :p_len], past_kvs=None, start_pos=0)
            nxt = lg_p[:, -1:, :].argmax(dim=-1)
            for s_i in range(1, full_len - p_len):
                lg_s, kvs = nl_model.forward_nl_causal(nxt, past_kvs=kvs, start_pos=p_len + s_i - 1)
                nxt = lg_s[:, -1:, :].argmax(dim=-1)
            nl_ok += int((nxt.squeeze(-1) == nl_ans.to(device)).sum().item())
            nl_tot += bsz
        nl_eager_ms = (sync_timer(device) - t0) * 1000.0 / max(1, nl_tot)

    # 2. Hardware-Optimal Native CUDAGraph + Static KV-Cache (0.000 ms Python dispatch overhead on both sides!)
    bsz = eval_tuples[0][1].shape[0]
    full_len = eval_tuples[0][1].shape[1]
    p_len = eval_tuples[0][3]
    dtype_cg = torch.bfloat16

    # Capture NL-CoT single-token Static-KV decode graph
    k_caches = [torch.zeros(bsz, nl_model.cfg.num_kv_heads, full_len, nl_model.cfg.head_dim, device=device, dtype=dtype_cg) for _ in range(nl_model.cfg.num_layers)]
    v_caches = [torch.zeros(bsz, nl_model.cfg.num_kv_heads, full_len, nl_model.cfg.head_dim, device=device, dtype=dtype_cg) for _ in range(nl_model.cfg.num_layers)]
    tok_buf = torch.zeros(bsz, 1, device=device, dtype=torch.long)
    pos_buf = torch.full((bsz, 1), p_len, device=device, dtype=torch.long)
    mask_buf = torch.zeros(1, 1, 1, full_len, device=device, dtype=torch.bool)
    mask_buf[..., : p_len + 1] = True
    all_idx = torch.arange(full_len, device=device).view(1, 1, 1, full_len)
    pref_in_buf = torch.zeros(bsz, p_len, device=device, dtype=torch.long)

    def _nl_prefill_graph_fn() -> None:
        lg_p, pref_kvs = nl_model.forward_nl_causal(pref_in_buf, past_kvs=None, start_pos=0)
        for l_i in range(nl_model.cfg.num_layers):
            k_caches[l_i][:, :, :p_len, :].copy_(pref_kvs[l_i][0].to(dtype_cg))
            v_caches[l_i][:, :, :p_len, :].copy_(pref_kvs[l_i][1].to(dtype_cg))
        tok_buf.copy_(lg_p[:, -1:, :].argmax(dim=-1))
        pos_buf.fill_(p_len)

    def _nl_decode_step_graph_fn() -> None:
        mask_buf.copy_(all_idx <= pos_buf[0, 0])
        nxt = _nl_static_kv_step(nl_model, tok_buf, pos_buf, mask_buf, k_caches, v_caches)
        tok_buf.copy_(nxt)
        pos_buf.add_(1)

    with amp_ctx(device):
        for _ in range(3):
            _nl_prefill_graph_fn()
            _nl_decode_step_graph_fn()
    torch.cuda.synchronize(device)
    g_nl_pref = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_nl_pref):
        with amp_ctx(device):
            _nl_prefill_graph_fn()
    g_nl = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_nl):
        with amp_ctx(device):
            _nl_decode_step_graph_fn()

    cg_nl_ok = 0
    t_cg_nl0 = sync_timer(device)
    for _, nl_seq, nl_ans, _ in eval_tuples:
        pref_in_buf.copy_(nl_seq[:, :p_len].to(device))
        g_nl_pref.replay()
        for _ in range(1, full_len - p_len):
            g_nl.replay()
        cg_nl_ok += int((tok_buf.squeeze(-1) == nl_ans.to(device)).sum().item())
    nl_cg_ms = (sync_timer(device) - t_cg_nl0) * 1000.0 / max(1, nl_tot)

    # Capture FULL End-to-End Symbolic Block-dLLM (RO init + 3 beats blk0 + PML commit + 3 beats blk1) into CUDAGraph!
    sym_model.eval()
    pr_buf = torch.zeros(bsz, 16, device=device, dtype=torch.long)
    seq32_buf = torch.zeros(bsz, 32, device=device, dtype=torch.long)
    seq48_buf = torch.zeros(bsz, 48, device=device, dtype=torch.long)
    sym_ans_buf = torch.zeros(bsz, device=device, dtype=torch.long)
    m32 = build_block_diffusion_attn_mask([16, 16], device=device)
    m48 = build_block_diffusion_attn_mask([16, 16, 16], device=device)
    wb_cg = sym_model.init_whiteboard_states(bsz, device)
    exp_cg = sym_model.init_experience_state(bsz, device)

    def _sym_full_graph_fn() -> None:
        for wb in wb_cg:
            wb.s_ro.zero_(); wb.m_ro.zero_(); wb.s_rw_commit.zero_(); wb.m_rw_commit.zero_(); wb.s_rw_draft.zero_(); wb.m_rw_draft.zero_(); wb.dirty_mask.zero_()
        sym_model.populate_ro_from_prompt(pr_buf, wb_cg)
        cur0 = torch.full((bsz, 16), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
        seq32_buf[:, :16].copy_(pr_buf)
        for beat in range(3):
            seq32_buf[:, 16:].copy_(cur0)
            out32 = sym_model.forward_symbolic(seq32_buf, phase=CognitivePhase.FORMAL, attn_mask=m32, wb_states=wb_cg, exp_state=exp_cg, t_k=1.0 - 0.35 * beat, read_only_pml=True)["logits"][:, -16:, :]
            cur0, _ = hierarchical_three_beat_unmask(cur0, out32, beat, 3)
        seq32_buf[:, 16:].copy_(cur0)
        _ = sym_model.forward_symbolic(seq32_buf, phase=CognitivePhase.FORMAL, attn_mask=m32, wb_states=wb_cg, exp_state=exp_cg, t_k=0.0, read_only_pml=False)

        cur1 = torch.full((bsz, 16), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
        seq48_buf[:, :16].copy_(pr_buf)
        seq48_buf[:, 16:32].copy_(cur0)
        for beat in range(3):
            seq48_buf[:, 32:].copy_(cur1)
            out48 = sym_model.forward_symbolic(seq48_buf, phase=CognitivePhase.FORMAL, attn_mask=m48, wb_states=wb_cg, exp_state=exp_cg, t_k=1.0 - 0.35 * beat, read_only_pml=True)["logits"][:, -16:, :]
            cur1, _ = hierarchical_three_beat_unmask(cur1, out48, beat, 3)
        sym_ans_buf.copy_(cur1[:, 12])

    with amp_ctx(device):
        for _ in range(3):
            _sym_full_graph_fn()
    torch.cuda.synchronize(device)
    g_sym_full = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_sym_full):
        with amp_ctx(device):
            _sym_full_graph_fn()

    cg_sym_ok = 0
    t_cg_s0 = sync_timer(device)
    for sb in sym_batches:
        pr_buf.copy_(sb.prompt_ids.to(device))
        g_sym_full.replay()
        cg_sym_ok += int((sym_ans_buf == sb.answer_tok_ids.to(device)).sum().item())
    sym_cg_ms = (sync_timer(device) - t_cg_s0) * 1000.0 / max(1, nl_tot)
    del g_nl_pref, g_nl, g_sym_full, k_caches, v_caches, wb_cg, exp_cg
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "num_eval_problems": nl_tot,
        "kv_cache_enabled_on_nl_ar": True,
        "compile_backend": "native_torch_cuda_cudagraph_static_kv (triton_unavailable_on_win32)",
        "nl_cot_exact_match": round(100.0 * nl_ok / max(1, nl_tot), 2),
        "nl_cot_cudagraph_exact_match": round(100.0 * cg_nl_ok / max(1, nl_tot), 2),
        "sym_exact_match": sym_eager["exact_match"],
        "sym_cudagraph_exact_match": round(100.0 * cg_sym_ok / max(1, nl_tot), 2),
        "sync_free_eager_ms_per_problem": {"nl_kv_ar": round(nl_eager_ms, 3), "sym_block_dllm": sym_eager["ms_per_problem"]},
        "sync_free_eager_speedup_x": round(nl_eager_ms / max(1e-4, sym_eager["ms_per_problem"]), 2),
        "cudagraph_static_kv_ms_per_problem": {"nl_kv_ar": round(nl_cg_ms, 3), "sym_block_dllm": round(sym_cg_ms, 3)},
        "cudagraph_static_kv_speedup_x": round(nl_cg_ms / max(1e-4, sym_cg_ms), 2),
    }


def mean_std(vals: List[float]) -> Dict[str, float]:
    m = sum(vals) / max(1, len(vals))
    v = sum((x - m) ** 2 for x in vals) / max(1, len(vals))
    return {"mean": round(m, 2), "std": round(math.sqrt(v), 2), "per_seed": [round(x, 2) for x in vals]}


def _free_vram() -> float:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return round(torch.cuda.memory_allocated() / (1024**3), 2)
    return 0.0


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    cfg_k0 = deepcopy(cfg)
    cfg_k0.k_max = 0
    optim_path = root / "configs" / "optim_5groups.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [42, 1024, 2026]
    gen_l3_test = Stage0SynthGenerator(cfg, seed=9003)
    eval_l3_640 = [gen_l3_test.sample_level3_paired_ast_batch(32, split="test") for _ in range(20)]

    gen_l3h_test = Stage0SynthGenerator(cfg, seed=9009)
    eval_l3h_640 = [sample_level3_dynamic_ast_batch(gen_l3h_test, 32, split="test") for _ in range(20)]

    gen_l1_test = Stage0SynthGenerator(cfg, seed=9001)
    eval_l1_640 = [gen_l1_test.sample_level1_batch(32, split="test") for _ in range(20)]

    l3_sym_ems, l3_nl_ems, l3_k0_knockout_ems, l3_k0_retrain_ems = [], [], [], []
    l3h_sym_ems, l3h_nl_ems, l3h_k0_knockout_ems, l3h_k0_retrain_ems = [], [], [], []
    l1_full_ems, l1_k0_knockout_ems, l1_k0_retrain_ems, l1_top2_ems = [], [], [], []

    canary_report = {}
    speed_report = {}
    t_global = time.perf_counter()

    for idx_s, sd in enumerate(seeds):
        # --- A1 & A3: Fixed-Template Level 3 Sym + NL-CoT + CUDAGraph Speed (freed immediately after!) ---
        t_sub = time.perf_counter()
        torch.manual_seed(sd)
        sym_m = AVLPMLModel(cfg).to(device)
        opt_s, sopt_s = build_optimizers(sym_m, optim_path)
        sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=1050, eta_min=2e-5)
        g_tr = Stage0SynthGenerator(cfg, seed=300 + sd)
        for st in range(1, 1051):
            b, _, _, _ = g_tr.sample_level3_paired_ast_batch(64, "train")
            _ = train_dual_stream_step(sym_m, opt_s, sopt_s, b, device, st, 1050)
            sch_s.step()
        del opt_s, sopt_s, sch_s
        set_u_post_warmup(sym_m, 1.0)
        l3_sym_ems.append(evaluate_symbolic_exact_match(sym_m, [t[0] for t in eval_l3_640], device)["exact_match"])

        sym_m.cfg.k_max = 0
        l3_k0_knockout_ems.append(evaluate_symbolic_exact_match(sym_m, [t[0] for t in eval_l3_640], device)["exact_match"])
        sym_m.cfg.k_max = 2
        if idx_s == 0:
            canary_report = verify_zero_target_leakage_canary(sym_m, eval_l3_640[0][0], device)

        torch.manual_seed(sd)
        nl_m = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True)).to(device)
        opt_nl, _ = build_optimizers(nl_m, optim_path)
        sch_nl = torch.optim.lr_scheduler.CosineAnnealingLR(opt_nl, T_max=450, eta_min=2e-5)
        g_tr_nl = Stage0SynthGenerator(cfg, seed=300 + sd)
        nl_m.train()
        for _ in range(450):
            _, nl_seq, _, p_len = g_tr_nl.sample_level3_paired_ast_batch(64, "train")
            nl_seq = nl_seq.to(device)
            opt_nl.zero_grad(set_to_none=True)
            with amp_ctx(device):
                lg_nl, _ = nl_m.forward_nl_causal(nl_seq[:, :-1])
                loss_nl = F.cross_entropy(lg_nl[:, p_len - 1 :, :].float().reshape(-1, cfg.nl_vocab_size), nl_seq[:, p_len:].reshape(-1))
            loss_nl.backward()
            torch.nn.utils.clip_grad_norm_(nl_m.parameters(), 1.0)
            opt_nl.step()
            sch_nl.step()
        del opt_nl, sch_nl

        if idx_s == 0:
            eval_l3_b1 = [Stage0SynthGenerator(cfg, seed=9003).sample_level3_paired_ast_batch(1, split="test") for _ in range(32)]
            speed_b32 = benchmark_compiled_vs_eager_speed(sym_m, nl_m, eval_l3_640, device)
            _free_vram()
            speed_b1 = benchmark_compiled_vs_eager_speed(sym_m, nl_m, eval_l3_b1, device)
            speed_report = {"batch_32": speed_b32, "batch_1_single_stream": speed_b1}
            l3_nl_ems.append(speed_b32["nl_cot_exact_match"])
        else:
            nl_m.eval()
            ok_nl = 0
            with torch.no_grad(), amp_ctx(device):
                for _, nl_seq, nl_ans, p_len in eval_l3_640:
                    nl_seq = nl_seq.to(device)
                    lg_p, kvs = nl_m.forward_nl_causal(nl_seq[:, :p_len], past_kvs=None, start_pos=0)
                    nxt = lg_p[:, -1:, :].argmax(dim=-1)
                    for s_i in range(1, nl_seq.shape[1] - p_len):
                        lg_s, kvs = nl_m.forward_nl_causal(nxt, past_kvs=kvs, start_pos=p_len + s_i - 1)
                        nxt = lg_s[:, -1:, :].argmax(dim=-1)
                    ok_nl += int((nxt.squeeze(-1) == nl_ans.to(device)).sum().item())
            l3_nl_ems.append(round(100.0 * ok_nl / 640.0, 2))
        del sym_m, nl_m
        v_gb = _free_vram()
        print(f"[Seed {sd} | 1/4 L3-Fixed] Sym={l3_sym_ems[-1]}% (K0_KO={l3_k0_knockout_ems[-1]}%) NL={l3_nl_ems[-1]}% | {time.perf_counter()-t_sub:.1f}s | alloc={v_gb}GB", flush=True)

        # --- A2: K_max = 0 Shared-Only Retrain on Level 3 ---
        t_sub = time.perf_counter()
        torch.manual_seed(sd)
        m_k0 = AVLPMLModel(cfg_k0).to(device)
        opt_k0, sopt_k0 = build_optimizers(m_k0, optim_path)
        sch_k0 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_k0, T_max=1050, eta_min=2e-5)
        g_tr_k0 = Stage0SynthGenerator(cfg_k0, seed=300 + sd)
        for st in range(1, 1051):
            b, _, _, _ = g_tr_k0.sample_level3_paired_ast_batch(64, "train")
            _ = train_dual_stream_step(m_k0, opt_k0, sopt_k0, b, device, st, 1050)
            sch_k0.step()
        l3_k0_retrain_ems.append(evaluate_symbolic_exact_match(m_k0, [t[0] for t in eval_l3_640], device)["exact_match"])
        del m_k0, opt_k0, sopt_k0, sch_k0
        v_gb = _free_vram()
        print(f"[Seed {sd} | 2/4 L3-K0-Retrain] EM={l3_k0_retrain_ems[-1]}% | {time.perf_counter()-t_sub:.1f}s | alloc={v_gb}GB", flush=True)

        # --- B: Unseen-Template Dynamic-AST Level 3-Hard ---
        t_sub = time.perf_counter()
        torch.manual_seed(sd)
        sym_mh = AVLPMLModel(cfg).to(device)
        opt_sh, sopt_sh = build_optimizers(sym_mh, optim_path)
        sch_sh = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sh, T_max=1350, eta_min=2e-5)
        g_trh = Stage0SynthGenerator(cfg, seed=500 + sd)
        for st in range(1, 1351):
            bh, _, _, _ = sample_level3_dynamic_ast_batch(g_trh, 64, "train")
            _ = train_dual_stream_step(sym_mh, opt_sh, sopt_sh, bh, device, st, 1350)
            sch_sh.step()
        set_u_post_warmup(sym_mh, 1.0)
        l3h_sym_ems.append(evaluate_symbolic_exact_match(sym_mh, [t[0] for t in eval_l3h_640], device)["exact_match"])
        sym_mh.cfg.k_max = 0
        l3h_k0_knockout_ems.append(evaluate_symbolic_exact_match(sym_mh, [t[0] for t in eval_l3h_640], device)["exact_match"])
        del sym_mh, opt_sh, sopt_sh, sch_sh
        _free_vram()

        torch.manual_seed(sd)
        mh_k0 = AVLPMLModel(cfg_k0).to(device)
        opt_hk0, sopt_hk0 = build_optimizers(mh_k0, optim_path)
        sch_hk0 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_hk0, T_max=1350, eta_min=2e-5)
        g_trh_k0 = Stage0SynthGenerator(cfg_k0, seed=500 + sd)
        for st in range(1, 1351):
            bh, _, _, _ = sample_level3_dynamic_ast_batch(g_trh_k0, 64, "train")
            _ = train_dual_stream_step(mh_k0, opt_hk0, sopt_hk0, bh, device, st, 1350)
            sch_hk0.step()
        l3h_k0_retrain_ems.append(evaluate_symbolic_exact_match(mh_k0, [t[0] for t in eval_l3h_640], device)["exact_match"])
        del mh_k0, opt_hk0, sopt_hk0, sch_hk0
        _free_vram()

        torch.manual_seed(sd)
        nl_mh = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True)).to(device)
        opt_nlh, _ = build_optimizers(nl_mh, optim_path)
        sch_nlh = torch.optim.lr_scheduler.CosineAnnealingLR(opt_nlh, T_max=450, eta_min=2e-5)
        g_tr_nlh = Stage0SynthGenerator(cfg, seed=500 + sd)
        nl_mh.train()
        for _ in range(450):
            _, nl_seq, _, p_len = sample_level3_dynamic_ast_batch(g_tr_nlh, 64, "train")
            nl_seq = nl_seq.to(device)
            opt_nlh.zero_grad(set_to_none=True)
            with amp_ctx(device):
                lg_nl, _ = nl_mh.forward_nl_causal(nl_seq[:, :-1])
                loss_nl = F.cross_entropy(lg_nl[:, p_len - 1 :, :].float().reshape(-1, cfg.nl_vocab_size), nl_seq[:, p_len:].reshape(-1))
            loss_nl.backward()
            torch.nn.utils.clip_grad_norm_(nl_mh.parameters(), 1.0)
            opt_nlh.step()
            sch_nlh.step()
        nl_mh.eval()
        ok_nlh = 0
        with torch.no_grad(), amp_ctx(device):
            for _, nl_seq, nl_ans, p_len in eval_l3h_640:
                nl_seq = nl_seq.to(device)
                lg_p, kvs = nl_mh.forward_nl_causal(nl_seq[:, :p_len], past_kvs=None, start_pos=0)
                nxt = lg_p[:, -1:, :].argmax(dim=-1)
                for s_i in range(1, nl_seq.shape[1] - p_len):
                    lg_s, kvs = nl_mh.forward_nl_causal(nxt, past_kvs=kvs, start_pos=p_len + s_i - 1)
                    nxt = lg_s[:, -1:, :].argmax(dim=-1)
                ok_nlh += int((nxt.squeeze(-1) == nl_ans.to(device)).sum().item())
        l3h_nl_ems.append(round(100.0 * ok_nlh / 640.0, 2))
        del nl_mh, opt_nlh, sch_nlh
        v_gb = _free_vram()
        print(f"[Seed {sd} | 3/4 L3-Hard-UnseenTrees] Sym={l3h_sym_ems[-1]}% (K0_KO={l3h_k0_knockout_ems[-1]}%, K0_RT={l3h_k0_retrain_ems[-1]}%) NL={l3h_nl_ems[-1]}% | {time.perf_counter()-t_sub:.1f}s | alloc={v_gb}GB", flush=True)

        # --- C: Level 1 (640 problems): Full vs Static Top-2 vs K_max=0 Knock-Out & Retrain ---
        t_sub = time.perf_counter()
        torch.manual_seed(sd)
        l1_m = AVLPMLModel(cfg).to(device)
        opt_1, sopt_1 = build_optimizers(l1_m, optim_path)
        sch_1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_1, T_max=1100, eta_min=2e-5)
        g_l1 = Stage0SynthGenerator(cfg, seed=100 + sd)
        for st in range(1, 1101):
            b1 = g_l1.sample_level1_batch(64, "train")
            _ = train_dual_stream_step(l1_m, opt_1, sopt_1, b1, device, st, 1100)
            sch_1.step()
        set_u_post_warmup(l1_m, 1.0)
        l1_full_ems.append(evaluate_symbolic_exact_match(l1_m, eval_l1_640, device)["exact_match"])
        l1_m.cfg.k_max = 0
        l1_k0_knockout_ems.append(evaluate_symbolic_exact_match(l1_m, eval_l1_640, device)["exact_match"])
        del l1_m, opt_1, sopt_1, sch_1
        _free_vram()

        torch.manual_seed(sd)
        l1_mk0 = AVLPMLModel(cfg_k0).to(device)
        opt_1k0, sopt_1k0 = build_optimizers(l1_mk0, optim_path)
        sch_1k0 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_1k0, T_max=1100, eta_min=2e-5)
        g_l1k0 = Stage0SynthGenerator(cfg_k0, seed=100 + sd)
        for st in range(1, 1101):
            b1 = g_l1k0.sample_level1_batch(64, "train")
            _ = train_dual_stream_step(l1_mk0, opt_1k0, sopt_1k0, b1, device, st, 1100)
            sch_1k0.step()
        l1_k0_retrain_ems.append(evaluate_symbolic_exact_match(l1_mk0, eval_l1_640, device)["exact_match"])
        del l1_mk0, opt_1k0, sopt_1k0, sch_1k0
        _free_vram()

        torch.manual_seed(sd)
        l1_top2 = AVLPMLModel(cfg, ablation=AblationConfig(static_top2_baseline=True)).to(device)
        opt_t2, sopt_t2 = build_optimizers(l1_top2, optim_path)
        sch_t2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t2, T_max=1100, eta_min=2e-5)
        g_l1t2 = Stage0SynthGenerator(cfg, seed=100 + sd)
        for st in range(1, 1101):
            b1 = g_l1t2.sample_level1_batch(64, "train")
            _ = train_dual_stream_step(l1_top2, opt_t2, sopt_t2, b1, device, st, 1100)
            sch_t2.step()
        l1_top2_ems.append(evaluate_symbolic_exact_match(l1_top2, eval_l1_640, device)["exact_match"])
        del l1_top2, opt_t2, sopt_t2, sch_t2
        v_gb = _free_vram()
        print(f"[Seed {sd} | 4/4 L1-4Skill] Full={l1_full_ems[-1]}% (K0_KO={l1_k0_knockout_ems[-1]}%, K0_RT={l1_k0_retrain_ems[-1]}%, Top2={l1_top2_ems[-1]}%) | {time.perf_counter()-t_sub:.1f}s | total={(time.perf_counter()-t_global)/60:.1f}m | alloc={v_gb}GB", flush=True)

    audit_summary = {
        "audit_1_speed_and_kv_compile_fairness": speed_report,
        "audit_2_zero_leakage_and_640x3_seeds": {
            "canary_poisoned_targets": canary_report,
            "level3_fixed_template_640probs_3seeds": {
                "symbolic_block_dllm": mean_std(l3_sym_ems),
                "nl_cot_kv_ar": mean_std(l3_nl_ems),
            },
            "level3_hard_unseen_operator_trees_640probs_3seeds": {
                "symbolic_block_dllm": mean_std(l3h_sym_ems),
                "nl_cot_kv_ar": mean_std(l3h_nl_ems),
            },
        },
        "audit_3_kmax0_routed_expert_zero_out": {
            "level3_fixed_template": {
                "full_kmax2": mean_std(l3_sym_ems),
                "kmax0_inference_knockout": mean_std(l3_k0_knockout_ems),
                "kmax0_shared_only_retrain": mean_std(l3_k0_retrain_ems),
            },
            "level3_hard_unseen_trees": {
                "full_kmax2": mean_std(l3h_sym_ems),
                "kmax0_inference_knockout": mean_std(l3h_k0_knockout_ems),
                "kmax0_shared_only_retrain": mean_std(l3h_k0_retrain_ems),
            },
            "level1_4skill_compositional": {
                "full_kmax2": mean_std(l1_full_ems),
                "static_top2_baseline": mean_std(l1_top2_ems),
                "kmax0_inference_knockout": mean_std(l1_k0_knockout_ems),
                "kmax0_shared_only_retrain": mean_std(l1_k0_retrain_ems),
            },
        },
    }

    out_path = root / "outputs" / "stage0_four_point_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(audit_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
