"""Dual-Domain Forward (AR & Block-dLLM x PML ON/OFF), Causal Leakage, & Train-vs-KV Parity Tests."""

from __future__ import annotations

from pathlib import Path
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.model import AVLPMLModel, AblationConfig
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab


def _build_sample_seq(model: AVLPMLModel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_ids = torch.randint(10, 500, (1, 12), dtype=torch.long)
    sym_seq = [
        SymbolicVocab.HYP_ID,
        model.vocab.ro_start,
        200,
        SymbolicVocab.BIND_ID,
        model.vocab.rw_start,
        SymbolicVocab.VERIFY_ID,
        201,
        202,
        SymbolicVocab.DEDUCE_ID,
        model.vocab.rw_start,
        203,
        SymbolicVocab.BIND_ID,
        model.vocab.rw_start + 1,
        SymbolicVocab.COMMIT_ID,
        model.vocab.rw_start + 1,
        SymbolicVocab.EOT_ID,
    ]
    s_ids = torch.tensor([sym_seq], dtype=torch.long)
    a_ids = torch.randint(10, 500, (1, 6), dtype=torch.long)
    return p_ids, s_ids, a_ids


def test_ar_dual_domain_zero_future_leakage_and_kv_parity() -> None:
    """Verify AR mode has ZERO future leakage and matches 1-token KV-cached forward_ar_step to < 1e-5."""
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        root = Path(__file__).resolve().parents[1]
        cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
        torch.manual_seed(42)
        model = AVLPMLModel(cfg).float().eval()
        p_ids, s_ids, a_ids = _build_sample_seq(model)

        # 1. Mutate future thinking & answer tokens -> earlier AR predictions must change by EXACTLY 0.0
        s_mut = s_ids.clone()
        s_mut[0, 8:] = 250
        a_mut = torch.randint(10, 500, (1, 6), dtype=torch.long)

        with torch.no_grad():
            wb1 = model.init_whiteboard_states(1, torch.device("cpu"), torch.float32)
            out1 = model.forward_dual_domain(p_ids, s_ids, a_ids, gen_mode="ar", sym_frame_sizes=[8, 8], wb_states=wb1)
            wb2 = model.init_whiteboard_states(1, torch.device("cpu"), torch.float32)
            out2 = model.forward_dual_domain(p_ids, s_mut, a_mut, gen_mode="ar", sym_frame_sizes=[8, 8], wb_states=wb2)
            leak_err = (out1["sym_logits"][:, :9] - out2["sym_logits"][:, :9]).abs().max().item()  # type: ignore[index]
            assert leak_err == 0.0, f"AR causal leak detected: {leak_err}"

            # 2. 1-token incremental KV-cached forward_ar_step must match full-sequence forward_dual_domain (< 1e-5)
            wb_s = model.init_whiteboard_states(1, torch.device("cpu"), torch.float32)
            ec = model.init_engram_causal_caches(1, torch.device("cpu"), torch.float32)
            r_p = model.forward_ar_step(p_ids, domain="nl_prompt", start_pos=0, wb_states=wb_s)
            kvs = r_p["present_kvs"]  # type: ignore[assignment]
            step_sym_logits = [r_p["sym_logits"]]
            step_ans_logits = []
            prev_tok = None
            emb_hist = []
            for t in range(s_ids.shape[1]):
                p2 = emb_hist[-2] if len(emb_hist) >= 2 else None
                r_t = model.forward_ar_step(
                    s_ids[:, t : t + 1],
                    domain="sym",
                    start_pos=p_ids.shape[1] + t,
                    sym_offset=t,
                    past_kvs=kvs,  # type: ignore[arg-type]
                    wb_states=wb_s,
                    engram_caches=ec,
                    prev_sym_token=prev_tok,
                    prev2_sym_embed=p2,
                )
                kvs = r_t["present_kvs"]  # type: ignore[assignment]
                prev_tok = s_ids[:, t]
                emb_hist.append(r_t["x_sym_tok"][:, 0, :])  # type: ignore[index]
                if t < s_ids.shape[1] - 1:
                    step_sym_logits.append(r_t["sym_logits"])
                else:
                    step_ans_logits.append(r_t["nl_logits"])
            for j in range(a_ids.shape[1] - 1):
                r_a = model.forward_ar_step(
                    a_ids[:, j : j + 1],
                    domain="nl_ans",
                    start_pos=p_ids.shape[1] + s_ids.shape[1] + j,
                    past_kvs=kvs,  # type: ignore[arg-type]
                    wb_states=wb_s,
                )
                kvs = r_a["present_kvs"]  # type: ignore[assignment]
                step_ans_logits.append(r_a["nl_logits"])

            cat_sym = torch.cat(step_sym_logits, dim=1)  # type: ignore[arg-type]
            cat_ans = torch.cat(step_ans_logits, dim=1)  # type: ignore[arg-type]
            sym_err = (out1["sym_logits"] - cat_sym).abs().max().item()  # type: ignore[operator]
            ans_err = (out1["ans_logits"] - cat_ans).abs().max().item()  # type: ignore[operator]
            wb_err = max((a.s_rw_commit - b.s_rw_commit).abs().max().item() for a, b in zip(wb1, wb_s))

            assert sym_err < 1e-5, f"AR sym logit parity mismatch: {sym_err}"
            assert ans_err < 1e-5, f"AR ans logit parity mismatch: {ans_err}"
            assert wb_err < 1e-5, f"AR whiteboard state parity mismatch: {wb_err}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def test_four_gate2_arms_forward_and_backward() -> None:
    """Verify all 4 Gate-2 arms ({AR, Block-dLLM} x {PML ON, PML OFF}) run clean forward & backward."""
    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    torch.manual_seed(123)
    for gen_mode in ("ar", "block_dllm"):
        for disable_pml in (False, True):
            model = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=disable_pml)).float().train()
            p_ids, s_ids, a_ids = _build_sample_seq(model)
            wb = model.init_whiteboard_states(1, torch.device("cpu"), torch.float32)
            inp_s = s_ids.clone()
            lmask = torch.ones_like(s_ids, dtype=torch.bool)
            if gen_mode == "block_dllm":
                inp_s[0, [2, 6, 10, 14]] = SymbolicVocab.MASK_ID
                lmask = inp_s == SymbolicVocab.MASK_ID
            out = model.forward_dual_domain(
                p_ids,
                inp_s,
                a_ids,
                gen_mode=gen_mode,
                sym_frame_sizes=[8, 8],
                wb_states=wb,
                t_k=0.25 if gen_mode == "block_dllm" else 0.0,
            )
            loss, stats = model.compute_dual_domain_losses(out, s_ids, lmask, ans_target_ids=a_ids)
            loss.backward()
            assert torch.isfinite(loss), f"Non-finite loss for ({gen_mode}, disable_pml={disable_pml})"
            assert model.sym_embed_head.op_embed.weight.grad is not None
            assert model.nl_embed.weight.grad is not None
            pml_layer = model.attn_layers[cfg.pml_layer_indices[0]].pml
            assert pml_layer is not None
            if disable_pml:
                assert pml_layer.o_proj.weight.grad is None
            else:
                assert pml_layer.o_proj.weight.grad is not None
            assert stats["loss_sym"] > 0.0 and stats["loss_ans"] > 0.0


def test_prefix_kv_block_denoising_and_e2e_nl_runner() -> None:
    """Verify Prefix-KV cached block attention matches uncached block-masked attention (< 1e-5 TF32-off)
    AND verify FourPhaseRunner.run_session(use_prefix_kv=True) + run_end_to_end_nl (AR & Block-dLLM)!
    """
    from avl_pmlv1.engine.four_phase_runner import FourPhaseRunner

    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        root = Path(__file__).resolve().parents[1]
        cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
        torch.manual_seed(77)
        # Disable global sequence-pooling modules for pure Prefix-KV vs full-prefix GQA+MoE parity check
        model = AVLPMLModel(cfg, ablation=AblationConfig(disable_bipolar_graph=True, disable_engram=True)).float().eval()
        p_ids = torch.randint(10, 500, (1, 16), dtype=torch.long)
        blk_ids = torch.full((1, 8), SymbolicVocab.MASK_ID, dtype=torch.long)
        blk_ids[0, 0] = SymbolicVocab.HYP_ID

        with torch.no_grad():
            # Path A: Prefill prefix ONCE -> run ONLY 8-slot block against cached past_kvs!
            wb_a = model.init_whiteboard_states(1, torch.device("cpu"), torch.float32)
            r_pre = model.forward_symbolic(p_ids, phase=CognitivePhase.FORMAL, wb_states=wb_a, read_only_pml=True, return_soft_sketch=False)
            r_blk_kv = model.forward_symbolic(
                blk_ids,
                phase=CognitivePhase.FORMAL,
                wb_states=wb_a,
                read_only_pml=True,
                past_kvs=r_pre["present_kvs"],  # type: ignore[arg-type]
                start_pos=16,
                d_b_override=torch.zeros(1, 4),
                return_soft_sketch=False,
            )

            # Path B: Re-encode full [prefix | block] with exact [causal prefix | bidirectional block] mask
            wb_b = model.init_whiteboard_states(1, torch.device("cpu"), torch.float32)
            full_seq = torch.cat([p_ids, blk_ids], dim=1)
            mask = torch.tril(torch.ones((24, 24), dtype=torch.bool))
            mask[16:, 16:] = True
            r_full = model.forward_symbolic(
                full_seq,
                phase=CognitivePhase.FORMAL,
                attn_mask=mask,
                wb_states=wb_b,
                read_only_pml=True,
                d_b_override=torch.zeros(1, 4),
                return_soft_sketch=False,
            )
            kv_err = (r_blk_kv["logits"] - r_full["logits"][:, 16:, :]).abs().max().item()  # type: ignore[index]
            assert kv_err < 1e-5, f"Prefix-KV block attention mismatch: {kv_err}"

        # End-to-end FourPhaseRunner smoke test with full model (run_session Prefix-KV + run_end_to_end_nl AR & Block-dLLM)
        full_model = AVLPMLModel(cfg).float().eval()
        runner = FourPhaseRunner(full_model)
        prompt_sym = torch.tensor([[SymbolicVocab.HYP_ID, full_model.vocab.ro_start, 200, 201]], dtype=torch.long)
        res_sess = runner.run_session(prompt_sym, num_formal_blocks=2, skel_len=8, denoise_beats=2, use_prefix_kv=True)
        assert res_sess.used_prefix_kv is True
        assert res_sess.final_tokens.shape[0] == 1 and res_sess.final_tokens.shape[1] >= 16

        class _DummyTok:
            eos_token_id = 999

            def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
                return " ".join(str(x) for x in ids)

        prompt_nl = torch.randint(10, 500, (1, 10), dtype=torch.long)
        for gm in ("ar", "block_dllm"):
            res_e2e = runner.run_end_to_end_nl(
                prompt_nl,
                tokenizer=_DummyTok(),
                gen_mode=gm,
                max_sym_tokens=16,
                block_size=8,
                denoise_beats=2,
                max_ans_tokens=6,
                enable_repair=(gm == "block_dllm"),
                audit_draft_for_netfix=(gm == "block_dllm"),
            )
            assert res_e2e.gen_mode == gm
            assert len(res_e2e.sym_tokens_final) >= 1
            assert res_e2e.total_e2e_ms >= 0.0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def test_gate2_deterministic_dsl_executor_and_grader() -> None:
    """Verify Gate 2 Arm 6 deterministic DSL executor handles both ESC re-computation and t-2 literal binds."""
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts" / "stage3_cloud"))
    from eval_gate2_suite import execute_dsl_trace, grade_prediction

    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_4_5b.yaml")
    vocab = SymbolicVocab(cfg)
    tok = vocab.token_to_id

    # Case 1: 35 * 200 = 7000 where 7000 is an opaque ESC token -> recovered deterministically via ⟨IN_1⟩ * ⟨REG_0⟩!
    esc_id = vocab.escape_start + 5
    trace_mul = [
        SymbolicVocab.GOAL_ID, tok["⟨IN_0⟩"], tok["="], tok["200"], SymbolicVocab.BIND_ID, tok["⟨REG_0⟩"],
        SymbolicVocab.COMMIT_ID, tok["⟨IN_1⟩"], tok["⟨REG_0⟩"], tok["*"], tok["="], esc_id, SymbolicVocab.BIND_ID, tok["⟨REG_1⟩"],
        SymbolicVocab.EOT_ID,
    ]
    pred_mul, ok_mul, _ = execute_dsl_trace(trace_mul, "A store has 400 boxes and sells 35 crates.", vocab)
    assert ok_mul and grade_prediction(pred_mul, "7000", "math"), f"Expected 7000, got {pred_mul!r}"

    # Case 2: Logic ⟨COMMIT⟩ ... True ⟨BIND⟩ ⟨REG_0⟩ ⟨EOT⟩ (index-0 ⟨COMMIT⟩ must NOT shadow ⟨BIND⟩!)
    trace_logic = [
        SymbolicVocab.COMMIT_ID, tok["⟨REG_1⟩"], tok["True"], SymbolicVocab.BIND_ID, tok["⟨REG_0⟩"], SymbolicVocab.EOT_ID
    ]
    pred_log, ok_log, _ = execute_dsl_trace(trace_logic, "Is Bob a knight?", vocab)
    assert ok_log and grade_prediction(pred_log, "True", "logic"), f"Expected True, got {pred_log!r}"

    # Case 3: Knights & Knaves boolean vector & ProofWriter last-conclusion logic grading
    assert grade_prediction("Answer: [true, false, true]", "[true, false, true]", "logic")
    assert not grade_prediction("Answer: [true, true, true]", "[true, false, true]", "logic")
    assert grade_prediction("If X is true then Y must be false, so the statement is false.", "False", "logic")


def test_merge_lora_to_bf16_and_pure_bf16_e2e_no_autocast() -> None:
    """Verify Q-AVL-PML LoRA delta merges accurately into pure BF16 dense model AND runs E2E without autocast."""
    from avl_pmlv1.engine.four_phase_runner import FourPhaseRunner
    from avl_pmlv1.engine.q_avl_pml import merge_lora_delta_into_bf16_model, quantize_to_q_avl_pml

    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    cfg.moe_layer_indices = []  # Dense FFN mainline
    cfg.cold_start_graft = True

    torch.manual_seed(314)
    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        base_bf16 = AVLPMLModel(cfg).eval()
        q_model = AVLPMLModel(cfg).eval()
        q_model.load_state_dict(base_bf16.state_dict(), strict=True)
    finally:
        torch.set_default_dtype(prev_dt)

    # Quantize q_model & set non-zero LoRA B weights so merge actually modifies weights
    quantize_to_q_avl_pml(q_model, lora_rank_gqa=4, lora_rank_dense=4, lora_rank_routed=0, group_size=64)
    with torch.no_grad():
        for name, p in q_model.named_parameters():
            if name.endswith(".lora_b"):
                p.fill_(0.02)

    trainable_names = {n for n, p in q_model.named_parameters() if p.requires_grad}
    delta_sd = {k: v.detach().cpu().clone() for k, v in q_model.state_dict().items() if k in trainable_names}

    w_before = base_bf16.attn_layers[0].gqa.q_proj.weight.detach().clone()
    m_info = merge_lora_delta_into_bf16_model(base_bf16, delta_sd, lora_scaling=2.0)
    assert m_info["merged_lora_layers"] > 0
    assert (base_bf16.attn_layers[0].gqa.q_proj.weight - w_before).abs().max().item() > 1e-4
    assert base_bf16.nl_head.weight is base_bf16.nl_embed.weight

    class _DummyTok:
        eos_token_id = 999

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(str(x) for x in ids)

    # Pure BF16 E2E inference WITHOUT autocast must run cleanly on AR, Block-dLLM (+repair), & NL-CoT!
    runner = FourPhaseRunner(base_bf16)
    p_nl = torch.randint(10, 400, (1, 10), dtype=torch.long)
    for gm in ("ar", "block_dllm"):
        res = runner.run_end_to_end_nl(
            p_nl,
            tokenizer=_DummyTok(),
            gen_mode=gm,
            max_sym_tokens=16,
            block_size=8,
            denoise_beats=2,
            max_ans_tokens=4,
            enable_repair=(gm == "block_dllm"),
            audit_draft_for_netfix=(gm == "block_dllm"),
        )
        assert len(res.sym_tokens_final) >= 1
    logits_nl, _ = base_bf16.forward_nl_causal(p_nl)
    assert logits_nl.dtype == torch.bfloat16 and torch.isfinite(logits_nl).all()



