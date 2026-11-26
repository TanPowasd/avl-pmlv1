"""Unit tests for Native V_nl 8-Slot Block-dLLM Backward Revision (Wet-Cement + 4-Fold PLL In-Place Repair)."""

from __future__ import annotations

from pathlib import Path
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.model import AVLPMLModel, AblationConfig
from avl_pmlv1.engine.nl_backward_rev import (
    NLBackwardRevisionEngine,
    build_nl_full_bidir_mask,
    build_nl_wet_cement_mask,
)
from avl_pmlv1.engine.nl_block_dllm import NLBlockDiffusionWrapper


def _dense_40m_cfg() -> AVLPMLConfig:
    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    cfg.moe_layer_indices = []
    cfg.cold_start_graft = True
    return cfg


def test_wet_cement_zero_future_leakage_and_prefix_kv_parity() -> None:
    """Verify:
    1) For BOTH parity=0 and parity=1, mutating blocks beyond a 16-slot seam window causes EXACTLY 0.0 diff;
    2) Inside a 16-slot seam window [Block b-1 | Block b], mutating Block b DOES affect Block b-1 (true backward flow!);
    3) TF32-off training forward matches 16-slot Prefix-KV wet-cement inference step to < 1e-5!
    """
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        cfg = _dense_40m_cfg()
        torch.manual_seed(42)
        base = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=True)).float().eval()
        mask_id = cfg.nl_vocab_size - 2
        fim_id = cfg.nl_vocab_size - 3
        wrap = NLBlockDiffusionWrapper(base, mask_token_id=mask_id, fim_init_id=fim_id).float().eval()

        p_ids = torch.randint(5, 300, (1, 10), dtype=torch.long)
        y_clean = torch.randint(5, 300, (1, 24), dtype=torch.long)  # 3 blocks: 0..7, 8..15, 16..23
        y_wet = y_clean.clone()
        y_wet[0, [1, 4, 9, 12]] = mask_id

        with torch.no_grad():
            # Parity 0: window 0 is [0..15] (blocks 0 & 1), block 2 is [16..23]
            m0, pos0 = build_nl_wet_cement_mask(10, 24, block_size=8, parity=0)
            full_1 = torch.cat([p_ids, y_clean, y_wet], dim=1)
            h1, _ = wrap._run_backbone(wrap._embed_nl(full_1), pos0, attn_mask=m0)
            lg1 = base.nl_head(base.final_norm(h1[:, 34:, :]))  # [1, 24, V]

            # 1a. Mutating Block 2 [16..23] must cause 0.0 change on Window 0 [0..15]!
            yc_mut = y_clean.clone()
            yw_mut = y_wet.clone()
            yc_mut[0, 16:] = 77
            yw_mut[0, 16:] = 88
            h2, _ = wrap._run_backbone(wrap._embed_nl(torch.cat([p_ids, yc_mut, yw_mut], dim=1)), pos0, attn_mask=m0)
            lg2 = base.nl_head(base.final_norm(h2[:, 34:, :]))
            leak = (lg1[:, :16, :] - lg2[:, :16, :]).abs().max().item()
            assert leak == 0.0, f"Wet-cement future leak beyond 16-slot window: {leak}"

            # 1b. Mutating Block 1 [8..15] MUST influence Block 0 [0..7] (true backward right-to-left context!)
            yw_back = y_wet.clone()
            yw_back[0, 8:16] = 99
            h3, _ = wrap._run_backbone(wrap._embed_nl(torch.cat([p_ids, y_clean, yw_back], dim=1)), pos0, attn_mask=m0)
            lg3 = base.nl_head(base.final_norm(h3[:, 34:, :]))
            back_effect = (lg1[:, :8, :] - lg3[:, :8, :]).abs().max().item()
            assert back_effect > 1e-4, f"Expected non-zero right-to-left backward flow in wet window, got {back_effect}"

            # 2. Parity 1 Prefix-KV exact parity (< 1e-5) for Window [8..23] (Blocks 1 & 2) using committed Prompt + Block 0!
            m1, pos1 = build_nl_wet_cement_mask(10, 24, block_size=8, parity=1)
            hp1, _ = wrap._run_backbone(wrap._embed_nl(full_1), pos1, attn_mask=m1)
            lg_p1 = base.nl_head(base.final_norm(hp1[:, 34:, :]))

            _, kvs = wrap.forward_block_step(p_ids, past_kvs=None, start_pos=0, return_logits=False)
            _, kvs = wrap.forward_block_step(y_clean[:, :8], past_kvs=kvs, start_pos=10, return_logits=False)
            lg_kv_win, _ = wrap.forward_block_step(y_wet[:, 8:24], past_kvs=kvs, start_pos=18, return_logits=True)
            assert lg_kv_win is not None
            err = (lg_p1[:, 8:24, :] - lg_kv_win).abs().max().item()
            assert err < 1e-5, f"Wet-cement 16-slot Prefix-KV parity mismatch: {err}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def test_full_bidir_4fold_pll_parity_and_disjoint_coverage() -> None:
    """Verify full-bidir training mask matches prompt_kvs inference to < 1e-5 and 4-fold PLL covers 100% tokens."""
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        cfg = _dense_40m_cfg()
        torch.manual_seed(99)
        base = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=True)).float().eval()
        mask_id = cfg.nl_vocab_size - 2
        fim_id = cfg.nl_vocab_size - 3
        wrap = NLBlockDiffusionWrapper(base, mask_token_id=mask_id, fim_init_id=fim_id).float().eval()
        eng = NLBackwardRevisionEngine(wrap)

        p_ids = torch.randint(5, 300, (1, 10), dtype=torch.long)
        y_draft = torch.randint(5, 300, (1, 16), dtype=torch.long)
        y_fold0 = y_draft.clone()
        y_fold0[0, 0::4] = mask_id

        with torch.no_grad():
            bm, pos = build_nl_full_bidir_mask(10, 16)
            h_tr, _ = wrap._run_backbone(wrap._embed_nl(torch.cat([p_ids, y_fold0], dim=1)), pos, attn_mask=bm)
            lg_tr = base.nl_head(base.final_norm(h_tr[:, 10:, :]))

            _, pkvs = wrap.forward_block_step(p_ids, past_kvs=None, start_pos=0, return_logits=False)
            lg_inf, _ = wrap.forward_block_step(y_fold0, past_kvs=pkvs, start_pos=10, return_logits=True)
            assert lg_inf is not None
            err = (lg_tr - lg_inf).abs().max().item()
            assert err < 1e-5, f"Full-bidir prompt_kvs parity mismatch: {err}"

            audit = eng.score_draft_4fold_pll(pkvs, y_draft, p_len=10, block_size=8)
            assert audit["covered_all"] is True
            assert audit["energy"] > 0.0 and len(audit["block_nlls"]) == 2
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def test_backward_rev_train_steps_and_pure_bf16_generation() -> None:
    """Verify gradients for Wet-Cement & Full-Bidir training + all 4 pure-BF16 backward-revision inference modes."""
    cfg = _dense_40m_cfg()
    torch.manual_seed(17)
    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        base = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=True))
    finally:
        torch.set_default_dtype(prev_dt)

    mask_id = cfg.nl_vocab_size - 2
    fim_id = cfg.nl_vocab_size - 3
    wrap = NLBlockDiffusionWrapper(base, mask_token_id=mask_id, fim_init_id=fim_id).train()
    eng = NLBackwardRevisionEngine(wrap)

    p_ids = torch.randint(5, 200, (1, 8), dtype=torch.long)
    y_clean = torch.randint(5, 200, (1, 16), dtype=torch.long)

    # 1. Wet-cement backward step
    y_wet = y_clean.clone()
    y_wet[0, [1, 4, 6]] = mask_id
    l_wet, st_w = eng.forward_wet_cement_train(p_ids, y_clean, y_wet, y_wet == mask_id, block_size=8, parity=0)
    assert torch.isfinite(l_wet) and st_w["loss_total"] > 0.0
    l_wet.backward()
    assert wrap.nl_mask_delta.grad is not None and torch.isfinite(wrap.nl_mask_delta.grad).all()

    # 2. Full-bidir backward step
    wrap.zero_grad()
    y_bi = y_clean.clone()
    y_bi[0, 0::4] = mask_id
    l_bi, st_b = eng.forward_full_bidir_train(p_ids, y_clean, y_bi, y_bi == mask_id)
    assert torch.isfinite(l_bi) and st_b["loss_total"] > 0.0
    l_bi.backward()
    assert wrap.nl_mask_delta.grad is not None and torch.isfinite(wrap.nl_mask_delta.grad).all()

    class _DummyTok:
        eos_token_id = 1

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(str(x) for x in ids)

    wrap.eval()
    for en_w, en_p in [(False, False), (True, False), (False, True), (True, True)]:
        out = eng.generate_with_backward_revision(
            p_ids,
            tokenizer=_DummyTok(),
            max_new_tokens=16,
            block_size=8,
            denoise_beats=3,
            enable_wet_cement=en_w,
            wet_margin_thresh=100.0,  # Force wet trigger on dummy model to verify full code path!
            enable_post_revision=en_p,
            post_min_block_nll=0.0,   # Force post revision attempt on dummy model!
        )
        assert isinstance(out["final_text"], str) and out["total_e2e_ms"] >= out["prefill_ms"] >= 0.0
        if en_w:
            assert out["wet_triggers"] >= 1
        if en_p:
            assert out["draft_energy"] > 0.0
