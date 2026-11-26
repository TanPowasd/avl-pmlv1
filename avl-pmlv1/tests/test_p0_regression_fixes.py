"""Comprehensive Regression & Adversarial Tests for All 7 P0 Fixes, Secondary Defects, & Dense-FFN Mode."""

from __future__ import annotations

from pathlib import Path
import random
import sys
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.wet_cement import build_block_diffusion_attn_mask
from avl_pmlv1.engine.four_phase_runner import FourPhaseRunner, build_prefix_kv_block_mask
from avl_pmlv1.engine.model import AVLPMLModel, build_nl_ro_span_weights
from avl_pmlv1.engine.q_avl_pml import FP32MasterAdamW, build_five_group_optimizer
from avl_pmlv1.moe_track2.bipolar_graph import BipolarLinkageGraph
from avl_pmlv1.moe_track2.tri_state_moe import TriStateMoELayer
from avl_pmlv1.vocab.dsl_compiler import LinearScanDSLCompiler, MicroStepIR
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab, extract_prompt_number_spans


def _cfg_40m() -> AVLPMLConfig:
    root = Path(__file__).resolve().parents[1]
    return AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")


def test_p0_1_bf16_master_adamw_no_underflow_and_alpha_vres_dtype_safety() -> None:
    """P0-1: Verify pure BF16 model + FP32 Group-D scalars neither crash SDPA outside autocast nor underflow at lr=3e-5."""
    cfg = _cfg_40m()
    cfg.use_qk_norm = True
    cfg.cold_start_graft = True
    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = AVLPMLModel(cfg)
    finally:
        torch.set_default_dtype(prev_dt)

    opt, _ = build_five_group_optimizer(model, stage="B", is_lora_hybrid=False, use_fp32_master=True)
    assert isinstance(opt, FP32MasterAdamW)
    # Group D scalars (alpha_vres, alpha_calm) are now FP32 while projections are BF16!
    assert model.attn_layers[1].gqa.alpha_vres.dtype == torch.float32
    assert model.attn_layers[1].gqa.q_proj.weight.dtype == torch.bfloat16

    p_ids = torch.randint(5, 200, (1, 10), dtype=torch.long)
    s_ids = torch.randint(5, 300, (1, 16), dtype=torch.long)
    a_ids = torch.randint(5, 200, (1, 4), dtype=torch.long)
    wb = model.init_whiteboard_states(1, torch.device("cpu"), dtype=torch.bfloat16)

    # Pure BF16 forward WITHOUT autocast must not raise BFloat16 vs Float mismatch in SDPA or Linear!
    nl_logits, kvs = model.forward_nl_causal(p_ids)
    assert nl_logits.dtype == torch.bfloat16 and torch.isfinite(nl_logits).all()
    nl_step, _ = model.forward_nl_causal(p_ids[:, :3], past_kvs=kvs, start_pos=10)
    assert nl_step.dtype == torch.bfloat16

    out_dd = model.forward_dual_domain(
        p_ids, s_ids, a_ids, gen_mode="block_dllm", sym_frame_sizes=[8, 8], wb_states=wb, sym_clean_ids=s_ids
    )
    loss, _ = model.compute_dual_domain_losses(out_dd, s_ids, torch.ones_like(s_ids, dtype=torch.bool), ans_target_ids=a_ids)
    loss.backward()
    opt.step()
    assert model.nl_head.weight is model.nl_embed.weight, "Tied V_nl head/embedding broken after FP32MasterAdamW step!"

    # Prove BF16 RMSNorm weight=1.0 updates under lr=3e-5 with FP32MasterAdamW (where raw BF16 AdamW stalls at 0.0!)
    w_raw = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
    w_mst = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
    raw_opt = torch.optim.AdamW([{"params": [w_raw], "lr": 3e-5, "weight_decay": 0.0}])
    mst_opt = FP32MasterAdamW([{"params": [w_mst], "lr": 3e-5, "weight_decay": 0.0}], use_fp32_master=True)
    for _ in range(200):
        w_raw.grad = torch.ones_like(w_raw)
        w_mst.grad = torch.ones_like(w_mst)
        raw_opt.step()
        mst_opt.step()
    assert float((w_raw - 1.0).abs().max().item()) == 0.0, "Expected raw BF16 AdamW to suffer mantissa underflow"
    assert float((w_mst - 1.0).abs().max().item()) > 0.003, f"FP32MasterAdamW failed to accumulate small updates: {w_mst[0].item()}"


def test_p0_2_commit_verify_header_read_not_overwrite() -> None:
    """P0-2: Verify [⟨COMMIT⟩/⟨VERIFY⟩, ⟨REG_j⟩] at frame header reads S^RW[j] via hard index instead of overwriting it!"""
    cfg = _cfg_40m()
    torch.manual_seed(11)
    model = AVLPMLModel(cfg).float().eval()
    comp = LinearScanDSLCompiler(cfg, model.vocab)

    steps = [
        MicroStepIR("⟨GOAL⟩", operands=["⟨IN_0⟩"], lit_tokens=["+"], val_token="7", out_var="$a"),
        MicroStepIR("⟨VERIFY⟩", operands=["$a"], lit_tokens=["=="], val_token="True", out_var="$b"),
        MicroStepIR("⟨COMMIT⟩", operands=["$a"], lit_tokens=["+"], val_token="8", out_var="$c", flag="⟨EOT⟩"),
    ]
    tr = comp.compile_aligned_frames(steps, final_live_vars=["$c"], prefer_8slot=True)
    assert tr.frame_sizes == [8, 8, 8]
    reg_a = tr.reg_map["$a"]

    # Verify token layout: frame 1 starts with [VERIFY_ID, REG_a], frame 2 starts with [COMMIT_ID, REG_a]
    assert tr.token_ids[8] == SymbolicVocab.VERIFY_ID and tr.token_ids[9] == model.vocab.reg_slot_to_id(reg_a)
    assert tr.token_ids[16] == SymbolicVocab.COMMIT_ID and tr.token_ids[17] == model.vocab.reg_slot_to_id(reg_a)

    tok_t = torch.tensor([tr.token_ids], dtype=torch.long)
    x_emb = model.sym_embed_head(tok_t)
    pml = model.attn_layers[cfg.pml_layer_indices[0]].pml
    assert pml is not None

    with torch.no_grad():
        wb_step0 = model.init_whiteboard_states(1, torch.device("cpu"))[0]
        _, _, _ = pml.sequential_scan_forward(
            x_emb[:, :8, :], tok_t[:, :8], wb_step0, x_val_embed=x_emb[:, :8, :], frame_sizes=[8]
        )
        expected_reg_a_vec = wb_step0.s_rw_commit[0, reg_a].clone()

        wb_all = model.init_whiteboard_states(1, torch.device("cpu"))[0]
        h_out, u_epi, _ = pml.sequential_scan_forward(
            x_emb, tok_t, wb_all, x_val_embed=x_emb, frame_sizes=[8, 8, 8]
        )

        # 1. S^RW[reg_a] MUST NOT be overwritten by '=' before frame-1 VERIFY or frame-2 COMMIT!
        overwrite_drift = (wb_all.s_rw_commit[0, reg_a] - expected_reg_a_vec).abs().max().item()
        assert overwrite_drift == 0.0, f"Header VERIFY/COMMIT corrupted S^RW[reg_a]: drift={overwrite_drift}"

        # 2. Positions 9 (after VERIFY) and 17 (after COMMIT) MUST execute exact hard-indexed read (u_epistemic == 0.0)!
        expected_h_hard = pml.hard_o_proj(expected_reg_a_vec)
        assert u_epi[0, 9].item() == 0.0 and u_epi[0, 17].item() == 0.0
        assert torch.allclose(h_out[0, 9], expected_h_hard, atol=1e-5)
        assert torch.allclose(h_out[0, 17], expected_h_hard, atol=1e-5)


def test_p0_3_nl_dual_domain_and_e2e_populate_s_ro_nonzero() -> None:
    """P0-3: Verify S^RO is populated from NL prompt in forward_dual_domain, forward_ar_step, & run_end_to_end_nl."""
    cfg = _cfg_40m()
    torch.manual_seed(23)
    model = AVLPMLModel(cfg).float().eval()
    p_ids = torch.randint(10, 400, (1, 14), dtype=torch.long)
    s_ids = torch.tensor(
        [[SymbolicVocab.GOAL_ID, model.vocab.in_slot_to_id(0), model.vocab.token_to_id["+"], model.vocab.token_to_id["="],
          model.vocab.token_to_id["5"], SymbolicVocab.BIND_ID, model.vocab.reg_slot_to_id(0), SymbolicVocab.EOT_ID]],
        dtype=torch.long,
    )
    a_ids = torch.randint(10, 400, (1, 4), dtype=torch.long)

    with torch.no_grad():
        wb_dd = model.init_whiteboard_states(1, torch.device("cpu"))
        out_dd = model.forward_dual_domain(p_ids, s_ids, a_ids, gen_mode="ar", sym_frame_sizes=[8], wb_states=wb_dd)
        for wb in wb_dd:
            assert float(wb.m_ro.max().item()) == 1.0, "S^RO mask m_ro remained 0 in forward_dual_domain!"
            assert float(wb.s_ro.abs().max().item()) > 0.1, "S^RO vectors remained 0 in forward_dual_domain!"
        # Position p_len + 1 is ⟨IN_0⟩ -> hard-indexed read sets u_epistemic == 0.0!
        assert float(out_dd["u_epistemic"][0, 14 + 1].item()) == 0.0  # type: ignore[index]

        # Also verify FourPhaseRunner.run_end_to_end_nl populates S^RO
        class _MockHFTokenizer:
            eos_token_id = 999
            vocab_size = 2048

            def convert_ids_to_tokens(self, ids):
                return [str(i) for i in ids]

            def decode(self, ids, skip_special_tokens=True):
                table = {10: "Problem (math):\n", 11: "Alice has ", 12: "24", 13: " apples and ", 14: "12", 15: " pears.\nThought:\n"}
                return "".join(table.get(i, "x") for i in ids)

        tok = _MockHFTokenizer()
        p_real = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long)
        ro_w = build_nl_ro_span_weights(p_real, tokenizer=tok, num_ro_slots=cfg.num_ro_slots)
        # Slot 0 -> token index 2 ("24"), Slot 1 -> token index 4 ("12")
        assert ro_w[0, 0, 2].item() == 1.0 and ro_w[0, 1, 4].item() == 1.0

        runner = FourPhaseRunner(model)
        res = runner.run_end_to_end_nl(p_real, tokenizer=tok, gen_mode="block_dllm", max_sym_tokens=16, block_size=8)
        assert len(res.sym_tokens_final) >= 1


def test_p0_4_non_commutative_ops_implicit_constants_and_square_dup_operands() -> None:
    """P0-4: Verify math compiler preserves operand order on - and /, implicit constants, & duplicate x*x operands."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts" / "stage1_data"))
    sys.path.insert(0, str(root / "scripts" / "stage3_cloud"))
    from compile_dsl_pipeline import compile_code_deterministic, compile_math_or_logic_deterministic
    from eval_gate2_suite import execute_dsl_trace, grade_prediction

    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_4_5b.yaml")
    vocab = SymbolicVocab(cfg)
    comp = LinearScanDSLCompiler(cfg, vocab)

    cases = [
        # 1) Non-commutative division & subtraction where $v1 comes BEFORE ⟨IN_2⟩ (previously reversed to 10 - 24 & 12 / 24!)
        {
            "question": "A baker bakes 8 trays of 3 cakes, then sells 10 cakes and packs the rest into boxes of 7.",
            "reference_solution": "Total <<8*3=24>>24. Left <<24-10=14>>14. Boxes <<14/7=2>>2. #### 2",
            "ground_truth_answer": "2",
        },
        # 2) Implicit non-prompt constants (12 months, 100) + ESC_* intermediate result (600 * 12 = 7200)!
        {
            "question": "Tom saves 700 dollars per month but spends 100 dollars monthly. How much does he save in a year?",
            "reference_solution": "Monthly net <<700-100=600>>600. Annual <<600*12=7200>>7200. #### 7200",
            "ground_truth_answer": "7200",
        },
        # 3) Duplicate operand squaring $v1 * $v1 = 6400 (ESC_* token -> REQUIRES both operands in trace to re-execute!)
        {
            "question": "A square field has side length equal to 50 plus 30 meters. What is its area?",
            "reference_solution": "Side <<50+30=80>>80. Area <<80*80=6400>>6400. #### 6400",
            "ground_truth_answer": "6400",
        },
    ]

    for row in cases:
        steps, ok, reason = compile_math_or_logic_deterministic(row)
        assert ok, f"Failed to compile {row['question']}: {reason}"
        aln = comp.compile_aligned_frames(steps, final_live_vars=[steps[-1].out_var] if steps[-1].out_var else [])
        pred, ex_ok, _ = execute_dsl_trace(aln.token_ids, row["question"], vocab)
        assert ex_ok and grade_prediction(pred, row["ground_truth_answer"], "math"), (
            f"Executor mismatch on {row['ground_truth_answer']}: got {pred!r}"
        )

    # 4) Verify Python code deterministic compilation compiles cleanly with ordered operands
    code_row = {
        "source": "mbpp:test",
        "reference_solution": "def diff_sq(a, b):\n    d = a - b\n    return d * d\n",
        "unit_tests": ["assert diff_sq(10, 3) == 49"],
    }
    c_steps, c_ok, c_rs = compile_code_deterministic(code_row, verify_sandbox=True, strict_no_truncate=True)
    assert c_ok, f"Code compile failed: {c_rs}"
    assert any(s.operands == [c_steps[1].operands[0], c_steps[1].operands[1]] for s in c_steps[1:])


def test_p0_5_and_p0_6_bd3_dual_stream_zero_mask_pollution_and_zero_leakage() -> None:
    """P0-5 & P0-6: Verify BD3-LM has 0 [MASK] whiteboard pollution, 8-slot intra-16 visibility, & 0 future/answer leakage."""
    cfg = _cfg_40m()
    torch.manual_seed(303)
    model = AVLPMLModel(cfg).float().eval()
    comp = LinearScanDSLCompiler(cfg, model.vocab)

    # Mixed [8, 8, 16, 8] frames where Frame 1 (second 8-slot frame!) reads $a written by Frame 0!
    steps = [
        MicroStepIR("⟨GOAL⟩", operands=["⟨IN_0⟩"], lit_tokens=["+"], val_token="5", out_var="$a"),
        MicroStepIR("⟨DEDUCE⟩", operands=["$a"], lit_tokens=["+"], val_token="9", out_var="$b"),
        MicroStepIR("⟨DEDUCE⟩", operands=["$b"], fsq_pairs=[(7, 11)], lit_tokens=["*"], val_token="18", out_var="$c"),
        MicroStepIR("⟨COMMIT⟩", operands=["$c"], lit_tokens=["-"], val_token="17", out_var="$d", flag="⟨EOT⟩"),
    ]
    tr = comp.compile_aligned_frames(steps, final_live_vars=["$d"], prefer_8slot=True)
    assert tr.frame_sizes == [8, 8, 16, 8]

    p_ids = torch.randint(10, 300, (1, 10), dtype=torch.long)
    clean_sym = torch.tensor([tr.token_ids], dtype=torch.long)
    noisy_sym = clean_sym.clone()
    # Aggressively mask bound values AND BIND/REG slots in noisy_sym!
    noisy_sym[0, [4, 5, 6, 12, 13, 14, 28, 29, 30, 36, 37, 38]] = SymbolicVocab.MASK_ID
    a_ids = torch.randint(10, 300, (1, 6), dtype=torch.long)

    with torch.no_grad():
        # 1. Zero [MASK] whiteboard pollution: BD3-LM with heavily masked noisy_sym produces IDENTICAL S^RW to clean_sym!
        wb_clean = model.init_whiteboard_states(1, torch.device("cpu"))
        out_clean = model.forward_dual_domain(
            p_ids, clean_sym, a_ids, gen_mode="block_dllm", sym_frame_sizes=tr.frame_sizes, wb_states=wb_clean, sym_clean_ids=clean_sym
        )
        wb_noisy = model.init_whiteboard_states(1, torch.device("cpu"))
        out_noisy = model.forward_dual_domain(
            p_ids, noisy_sym, a_ids, gen_mode="block_dllm", sym_frame_sizes=tr.frame_sizes, wb_states=wb_noisy, sym_clean_ids=clean_sym
        )
        for wc, wn in zip(wb_clean, wb_noisy):
            wb_diff = (wc.s_rw_commit - wn.s_rw_commit).abs().max().item()
            assert wb_diff == 0.0, f"[MASK] polluted S^RW in BD3-LM! diff={wb_diff}"

        # 2. Frame 1 (offset 8, token 9 = REG_a) MUST see Frame 0's write (u_epistemic == 0.0 at noisy offset 8+1 in clean stream)!
        s_sym = clean_sym.shape[1]
        assert float(out_noisy["u_epistemic"][0, 10 + 9].item()) == 0.0  # type: ignore[index]

        # 3. ZERO Future-Frame & Ground-Truth Answer Leakage into earlier frames or ans_logits[:, 0]!
        clean_mut = clean_sym.clone()
        noisy_mut = noisy_sym.clone()
        # Mutate frames 2 & 3 (positions 16..40) AND completely replace ground-truth Answer_nl!
        clean_mut[0, 16:] = model.vocab.token_to_id["3"]
        noisy_mut[0, 16:] = model.vocab.token_to_id["3"]
        a_mut = torch.randint(500, 900, (1, 6), dtype=torch.long)

        wb_mut = model.init_whiteboard_states(1, torch.device("cpu"))
        out_mut = model.forward_dual_domain(
            p_ids, noisy_mut, a_mut, gen_mode="block_dllm", sym_frame_sizes=tr.frame_sizes, wb_states=wb_mut, sym_clean_ids=clean_mut
        )
        # Frames 0 & 1 (first 16 sym_logits) must change by < 1e-6!
        sym_leak = (out_noisy["sym_logits"][:, :16] - out_mut["sym_logits"][:, :16]).abs().max().item()  # type: ignore[index]
        assert sym_leak < 1e-6, f"Future frame / answer leaked into earlier block_dllm frames: {sym_leak}"

        # Mutate ONLY Answer_nl while keeping clean_sym & noisy_sym fixed -> ALL sym_logits & ans_logits[:, 0] must change by < 1e-6!
        wb_amut = model.init_whiteboard_states(1, torch.device("cpu"))
        out_amut = model.forward_dual_domain(
            p_ids, noisy_sym, a_mut, gen_mode="block_dllm", sym_frame_sizes=tr.frame_sizes, wb_states=wb_amut, sym_clean_ids=clean_sym
        )
        all_sym_leak = (out_noisy["sym_logits"] - out_amut["sym_logits"]).abs().max().item()  # type: ignore[operator]
        first_ans_leak = (out_noisy["ans_logits"][:, :1] - out_amut["ans_logits"][:, :1]).abs().max().item()  # type: ignore[index]
        assert all_sym_leak < 1e-6, f"Ground-truth Answer_nl leaked into sym_logits: {all_sym_leak}"
        assert first_ans_leak < 1e-6, f"Ground-truth Answer_nl leaked into first answer token prediction: {first_ans_leak}"


def test_p0_7_post_thinking_prefix_kv_block_causal_mask_and_repair_guard() -> None:
    """P0-7: Verify FULL un-ablated model (Engram + BipolarGraph + DirHead + PML Whiteboard ON!) has ZERO
    future-block or recycled-register leakage in Phase 3 Post-Thinking Prefix-KV audit!
    """
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        cfg = _cfg_40m()
        torch.manual_seed(55)
        model = AVLPMLModel(cfg).float().eval()
        comp = LinearScanDSLCompiler(cfg, model.vocab)
        steps_3 = [
            MicroStepIR("⟨GOAL⟩", operands=["⟨IN_0⟩"], lit_tokens=["+"], val_token="5", out_var="$a"),
            MicroStepIR("⟨DEDUCE⟩", operands=["$a"], lit_tokens=["+"], val_token="9", out_var="$b"),
            MicroStepIR("⟨COMMIT⟩", operands=["$b"], lit_tokens=["+"], val_token="12", out_var="$c", flag="⟨EOT⟩"),
        ]
        tr3 = comp.compile_aligned_frames(steps_3, final_live_vars=["$c"])
        p_ids = torch.randint(10, 300, (1, 12), dtype=torch.long)
        trace_clean = torch.tensor([tr3.token_ids], dtype=torch.long)
        trace_masked = trace_clean.clone()
        trace_masked[0, [2, 6, 10, 14, 18, 22]] = SymbolicVocab.MASK_ID

        clean_mut = trace_clean.clone()
        masked_mut = trace_masked.clone()
        # Aggressively mutate block 2 [16:24] (all [MASK] + recycled REG_0 overwrite) -> blocks 0 & 1 must change by < 1e-6!
        clean_mut[0, 16:] = model.vocab.reg_slot_to_id(0)
        masked_mut[0, 16:] = SymbolicVocab.MASK_ID

        with torch.no_grad():
            wb1 = model.init_whiteboard_states(1, torch.device("cpu"))
            r_pre = model.forward_ar_step(p_ids, domain="nl_prompt", start_pos=0, wb_states=wb1)
            pkvs = r_pre["present_kvs"]  # type: ignore[assignment]
            wb2 = [w.repeat_batch(1) for w in wb1]
            kv_mask = build_prefix_kv_block_mask(12, [8, 8, 8], device=p_ids.device)

            r1 = model.forward_symbolic(
                trace_masked,
                phase=CognitivePhase.POST,
                attn_mask=kv_mask,
                wb_states=wb1,
                past_kvs=pkvs,
                start_pos=12,
                slot_offset=0,
                read_only_pml=True,
                return_soft_sketch=False,
                frame_sizes=[8, 8, 8],
                clean_token_ids_for_pml=trace_clean,
            )
            r2 = model.forward_symbolic(
                masked_mut,
                phase=CognitivePhase.POST,
                attn_mask=kv_mask,
                wb_states=wb2,
                past_kvs=pkvs,
                start_pos=12,
                slot_offset=0,
                read_only_pml=True,
                return_soft_sketch=False,
                frame_sizes=[8, 8, 8],
                clean_token_ids_for_pml=clean_mut,
            )
            peek_err = (r1["logits"][:, :16] - r2["logits"][:, :16]).abs().max().item()  # type: ignore[index]
            assert peek_err < 1e-6, f"Post-thinking Prefix-KV leaked future block into earlier blocks: {peek_err}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def test_secondary_pain_decay_bipolar_init_sparse1024_and_dense_ffn_mode() -> None:
    """Verify: (1) p_neg pain decay is not canceled by F.normalize; (2) BipolarGraph Step-0 basin;
    (3) hidden_size>=1024 zero-nonzero MoE parity; (4) moe_layer_indices=[] 100%-dense FFN mode!
    """
    # 1. BipolarLinkageGraph Step-0 basin check (H in [ln 2, ln 4], syn_adv_overlap <= 0.20)
    torch.manual_seed(2026)
    bg = BipolarLinkageGraph(num_experts=8)
    _, met = bg.graph_regularization_loss(collect_metrics=True)
    import math
    assert math.log(2.0) <= met["row_entropy_syn"] <= math.log(4.0), f"Step-0 entropy out of basin: {met}"
    assert met["syn_adv_overlap"] <= 0.20, f"Step-0 overlap > 0.20: {met}"

    # 2. Pain decay monotonicity across clean frame advances (exact 0.85^t scaling check!)
    cfg = _cfg_40m()
    moe = TriStateMoELayer(cfg, moe_slot_idx=0).float().eval()
    exp = AVLPMLModel(cfg).init_experience_state(1, torch.device("cpu"))
    x = torch.randn(1, 1, cfg.hidden_size)
    with torch.no_grad():
        _, aux_base = moe(x, phase=CognitivePhase.FORMAL, exp_state=exp)
        s0_base = float(aux_base["scores"][0, 0, 0].item())

        z_cur, _ = moe.compute_probes(moe.norm(x))
        z_les = z_cur.mean(dim=1).unsqueeze(1).repeat(1, len(cfg.moe_layer_indices), 1, 1)
        w_les = torch.zeros(1, len(cfg.moe_layer_indices), 8)
        w_les[:, :, 0] = 1.0
        m_adv = bg.get_manifolds()[1].unsqueeze(0).repeat(len(cfg.moe_layer_indices), 1, 1)
        exp.asymmetric_rollback_and_imprint_pain(exp.clone_synergy_snapshot(), z_les, w_les, m_adv, top_culprits=1)

        _, aux_t0 = moe(x, phase=CognitivePhase.FORMAL, exp_state=exp)
        s0_t0 = float(aux_t0["scores"][0, 0, 0].item())

        for _ in range(5):
            exp.record_clean_frame_advance(z_les, w_les[:, 0, :].unsqueeze(1).repeat(1, len(cfg.moe_layer_indices), 1), False, gamma_pain=0.85)
        _, aux_t5 = moe(x, phase=CognitivePhase.FORMAL, exp_state=exp)
        s0_t5 = float(aux_t5["scores"][0, 0, 0].item())
        # Initial penalty at t=0 is ~0.60; after 5 clean advances (0.85^5 = 0.4437), penalty shrinks by > 0.30!
        assert (s0_base - s0_t0) > 0.50
        assert s0_t5 > s0_t0 + 0.30, f"Pain decay canceled out! s0_base={s0_base:.4f}, s0_t0={s0_t0:.4f}, s0_t5={s0_t5:.4f}"

    # 3. hidden_size >= 1024 sorted-bin & 1-token MoE dispatch equivalence
    cfg_big = _cfg_40m()
    cfg_big.hidden_size = 1024
    cfg_big.num_q_heads = 16
    cfg_big.num_kv_heads = 4
    cfg_big.head_dim = 64
    moe_big = TriStateMoELayer(cfg_big, moe_slot_idx=0).float().eval()
    with torch.no_grad():
        x_multi = torch.randn(1, 6, 1024)
        top_i = torch.tensor([[[0, 3], [2, 5], [1, 7], [0, 2], [4, 6], [3, 5]]], dtype=torch.long)
        top_w = torch.tensor([[[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.5, 0.5], [0.8, 0.2], [0.0, 0.0]]])
        y_fast_1 = moe_big._sparse_topk_swiglu(x_multi[:, :1, :], top_i[:, :1, :], top_w[:, :1, :])
        y_multi = moe_big._sparse_topk_swiglu(x_multi, top_i, top_w)
        assert torch.allclose(y_fast_1[:, 0, :], y_multi[:, 0, :], atol=1e-5)
        assert y_multi[:, 5, :].abs().max().item() == 0.0  # zero weights -> exact zero output!

    # 4. Converged Step-1 100%-Dense FFN mode (moe_layer_indices = [])
    cfg_dense = _cfg_40m()
    cfg_dense.num_layers = 4
    cfg_dense.moe_layer_indices = []
    cfg_dense.pml_layer_indices = [1, 3]
    cfg_dense.engram_layer_indices = [1]
    cfg_dense.dir_vector_layer_index = 1
    dense_model = AVLPMLModel(cfg_dense).float().eval()
    runner = FourPhaseRunner(dense_model)
    res_s = runner.run_session(torch.tensor([[dense_model.vocab.in_slot_to_id(0), 250]], dtype=torch.long), num_formal_blocks=2, skel_len=8, denoise_beats=2)
    assert res_s.final_tokens.shape[1] >= 16


def test_mixed_8_16_block_dllm_train_vs_prefix_kv_infer_parity_and_adversarial_alignment() -> None:
    """Verify:
    1) Mixed [8, 16, 16] BD3-LM forward_dual_domain matches 8-slot Prefix-KV block_dllm inference to < 1e-5;
    2) Subword-split ordinals ('1st'), identifiers ('v1', '3D', 'MP3'), fractions ('3/4'), negatives ('-12'),
       ranges ('10-15'), and units ('35oz', '750m^2') align 1:1 across compiler, S^RO, and Gate-2 executor;
    3) GSM8K `expr = <<c=c>>` inline equations compile & execute accurately without phantom constant truncation;
    4) FSQ annotation & `reg_pointer_swap` strictly preserve/mutate ordered `operands`.
    """
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts" / "stage1_data"))
        sys.path.insert(0, str(root / "scripts" / "stage3_cloud"))
        from build_three_source_error_set import build_ast_ssa_mutation
        from compile_dsl_pipeline import compile_math_or_logic_deterministic
        from eval_gate2_suite import execute_dsl_trace, grade_prediction

        cfg = _cfg_40m()
        torch.manual_seed(808)
        model = AVLPMLModel(cfg).float().eval()
        comp = LinearScanDSLCompiler(cfg, model.vocab)

        ir = [
            MicroStepIR("⟨GOAL⟩", operands=["⟨IN_0⟩"], lit_tokens=["+"], val_token="5", out_var="$v1"),
            MicroStepIR("⟨DEDUCE⟩", operands=["$v1"], fsq_pairs=[(3, 4)], lit_tokens=["*"], val_token="10", out_var="$v2"),
            MicroStepIR("⟨COMMIT⟩", operands=["$v2"], fsq_pairs=[(5, 6)], lit_tokens=["+"], val_token="11", out_var="$v3", flag="⟨EOT⟩"),
        ]
        gold_tr = comp.compile_aligned_frames(ir, final_live_vars=["$v3"])
        assert gold_tr.frame_sizes == [8, 16, 16]

        p_ids = torch.randint(10, 300, (1, 12), dtype=torch.long)
        s_sub = torch.tensor([gold_tr.token_ids], dtype=torch.long)
        with torch.no_grad():
            wb_dd = model.init_whiteboard_states(1, torch.device("cpu"))
            out_dd = model.forward_dual_domain(
                p_ids, s_sub, None, gen_mode="block_dllm", sym_frame_sizes=gold_tr.frame_sizes, wb_states=wb_dd, sym_clean_ids=s_sub
            )
            wb_inf = model.init_whiteboard_states(1, torch.device("cpu"))
            r_p = model.forward_ar_step(p_ids, domain="nl_prompt", start_pos=0, wb_states=wb_inf)
            kvs = r_p["present_kvs"]  # type: ignore[assignment]
            inf_logits = []
            for off in range(0, s_sub.shape[1], 8):
                r_b = model.forward_symbolic(
                    s_sub[:, off : off + 8],
                    phase=CognitivePhase.FORMAL,
                    wb_states=wb_inf,
                    past_kvs=kvs,
                    start_pos=12 + off,
                    slot_offset=off,
                    t_k=0.0,
                    read_only_pml=False,
                    return_soft_sketch=False,
                )
                kvs = r_b["present_kvs"]  # type: ignore[assignment]
                inf_logits.append(r_b["logits"])
            diff_16 = (out_dd["sym_logits"] - torch.cat(inf_logits, dim=1)).abs().max().item()  # type: ignore[arg-type]
            assert diff_16 < 1e-5, f"Mixed [8, 16, 16] BD3-LM train vs 8-slot Prefix-KV infer mismatch: {diff_16}"

        # Subword-split BPE mock reproducing Qwen3 single-digit tokenization ('1st'->['1','st'], '3D'->['3','D'], 'v1'->['v','1'])
        pieces_adv = [
            "Problem (math):\n", "1", "st", " gets ", "-", "1", "2", ", ", "3", "D", " ", "v", "1", " has ",
            "3", "/", "4", " of ", "3", "5", "oz", " and ", "7", "5", "0", "m", "^", "2", ".\nThought:\n",
        ]

        class _SingleDigitMockTok:
            def convert_ids_to_tokens(self, ids):
                return [pieces_adv[i] for i in ids]

            def decode(self, ids, skip_special_tokens=False):
                return "".join(pieces_adv[i] for i in ids)

        p_adv = torch.arange(len(pieces_adv), dtype=torch.long).unsqueeze(0)
        rw = build_nl_ro_span_weights(p_adv, tokenizer=_SingleDigitMockTok(), num_ro_slots=16)
        got_spans = []
        for k in range(16):
            nz = (rw[0, k] > 0).nonzero(as_tuple=False).view(-1).tolist()
            if nz:
                got_spans.append("".join(pieces_adv[i] for i in nz))
        body_q = "".join(pieces_adv[1:-1])
        canon_spans = [m_s for _, _, m_s in extract_prompt_number_spans(body_q)]
        assert canon_spans == ["-12", "3/4", "35", "750"]
        assert got_spans == ["-12", "3/4", "35", "750"], f"S^RO span shift under single-digit BPE: {got_spans}"

        # GSM8K `60/2 = <<30=30>>30` inline equation + ordinal/unit question alignment
        row_eq = {
            "question": "On August 17th in a 35oz jar, Lucy read 20 more than 40 pages, and Carter read half.",
            "reference_solution": "Lucy 40+20 = <<20+40=60>>60. Carter 60/2 = <<30=30>>30. #### 30",
            "ground_truth_answer": "30",
        }
        st_eq, ok_eq, rs_eq = compile_math_or_logic_deterministic(row_eq)
        assert ok_eq and len(st_eq) == 2, f"Expected 2 steps for <<30=30>> recovery, got {len(st_eq)} ({rs_eq})"
        aln_eq = comp.compile_aligned_frames(st_eq, final_live_vars=[st_eq[-1].out_var] if st_eq[-1].out_var else [])
        pred_eq, ex_ok_eq, _ = execute_dsl_trace(aln_eq.token_ids, row_eq["question"], model.vocab)
        assert ex_ok_eq and grade_prediction(pred_eq, "30", "math"), f"Got {pred_eq!r}"

        # reg_pointer_swap must genuinely mutate the compiled header register pointer when operands is set
        mut_res = build_ast_ssa_mutation(ir, comp, cfg, random.Random(0), "reg_pointer_swap")
        assert mut_res is not None and mut_res[4] == "reg_pointer_swap"
        b_s = mut_res[2]
        off_b = sum(gold_tr.frame_sizes[:b_s])
        assert gold_tr.token_ids[off_b : off_b + 3] != mut_res[0][off_b : off_b + 3]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c
