"""Unit tests for Native V_nl 8-Slot BD3-LM Dual-Stream Block Diffusion Engine."""

from __future__ import annotations

from pathlib import Path
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.model import AVLPMLModel, AblationConfig
from avl_pmlv1.engine.nl_block_dllm import NLBlockDiffusionWrapper, build_nl_bd3_mask


def _dense_40m_cfg() -> AVLPMLConfig:
    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    cfg.moe_layer_indices = []
    cfg.cold_start_graft = True
    return cfg


def test_nl_bd3_zero_future_leakage_and_prefix_kv_parity() -> None:
    """Verify:
    1) Mutating future block b'>b in Y_clean or Y_noisy causes EXACTLY 0.0 change to block b logits;
    2) TF32-off dual-stream BD3-LM forward matches incremental Prefix-KV block inference to < 1e-5!
    """
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        cfg = _dense_40m_cfg()
        torch.manual_seed(123)
        base = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=True)).float().eval()
        mask_id = cfg.nl_vocab_size - 2
        fim_id = cfg.nl_vocab_size - 3
        wrap = NLBlockDiffusionWrapper(base, mask_token_id=mask_id, fim_init_id=fim_id).float().eval()

        p_ids = torch.randint(5, 300, (1, 10), dtype=torch.long)
        y_clean = torch.randint(5, 300, (1, 16), dtype=torch.long)
        y_noisy = y_clean.clone()
        y_noisy[0, [1, 3, 5, 8, 10, 12, 14]] = mask_id

        with torch.no_grad():
            attn_m, pos = build_nl_bd3_mask(10, 16, block_size=8)
            full_1 = torch.cat([p_ids, y_clean, y_noisy], dim=1)
            h1, _ = wrap._run_backbone(wrap._embed_nl(full_1), pos, attn_mask=attn_m)
            lg1 = base.nl_head(base.final_norm(h1[:, 26:, :]))  # [1, 16, V]

            # 1. Zero future leakage: mutate block 1 (indices 8..15) in BOTH y_clean and y_noisy -> block 0 diff == 0.0!
            y_clean_mut = y_clean.clone()
            y_noisy_mut = y_noisy.clone()
            y_clean_mut[0, 8:] = 42
            y_noisy_mut[0, 8:] = 43
            full_2 = torch.cat([p_ids, y_clean_mut, y_noisy_mut], dim=1)
            h2, _ = wrap._run_backbone(wrap._embed_nl(full_2), pos, attn_mask=attn_m)
            lg2 = base.nl_head(base.final_norm(h2[:, 26:, :]))
            leak = (lg1[:, :8, :] - lg2[:, :8, :]).abs().max().item()
            assert leak == 0.0, f"Future block leakage detected: {leak}"

            # 2. Prefix-KV exact parity (< 1e-5 TF32-off):
            #    Prefill Prompt -> denoise Block 0 -> commit clean Block 0 -> denoise Block 1!
            _, kvs = wrap.forward_block_step(p_ids, past_kvs=None, start_pos=0, return_logits=False)
            lg_b0, _ = wrap.forward_block_step(y_noisy[:, :8], past_kvs=kvs, start_pos=10, return_logits=True)
            _, kvs = wrap.forward_block_step(y_clean[:, :8], past_kvs=kvs, start_pos=10, return_logits=False)
            lg_b1, _ = wrap.forward_block_step(y_noisy[:, 8:16], past_kvs=kvs, start_pos=18, return_logits=True)
            assert lg_b0 is not None and lg_b1 is not None

            err_b0 = (lg1[:, :8, :] - lg_b0).abs().max().item()
            err_b1 = (lg1[:, 8:16, :] - lg_b1).abs().max().item()
            assert err_b0 < 1e-5, f"Block 0 Prefix-KV parity mismatch: {err_b0}"
            assert err_b1 < 1e-5, f"Block 1 Prefix-KV parity mismatch: {err_b1}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def test_nl_bd3_train_backward_and_pure_bf16_generation() -> None:
    """Verify BD3-LM backward updates nl_mask_delta & LoRA/weights, and pure BF16 (no autocast) generation runs cleanly."""
    cfg = _dense_40m_cfg()
    torch.manual_seed(7)
    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        base = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=True))
    finally:
        torch.set_default_dtype(prev_dt)

    mask_id = cfg.nl_vocab_size - 2
    fim_id = cfg.nl_vocab_size - 3
    wrap = NLBlockDiffusionWrapper(base, mask_token_id=mask_id, fim_init_id=fim_id).train()

    p_ids = torch.randint(5, 200, (1, 8), dtype=torch.long)
    y_clean = torch.randint(5, 200, (1, 16), dtype=torch.long)
    y_noisy = y_clean.clone()
    y_noisy[0, [0, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15]] = mask_id
    lmask = y_noisy == mask_id

    loss, st = wrap.forward_bd3_train(p_ids, y_clean, y_noisy, lmask, block_size=8)
    assert torch.isfinite(loss) and st["loss_total"] > 0.0
    loss.backward()
    assert wrap.nl_mask_delta.grad is not None and torch.isfinite(wrap.nl_mask_delta.grad).all()

    class _DummyTok:
        eos_token_id = 1

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(str(x) for x in ids)

    wrap.eval()
    txt, n_toks, pre_ms, tot_ms = wrap.generate_nl_block_dllm(
        p_ids, tokenizer=_DummyTok(), max_new_tokens=16, block_size=8, denoise_beats=3
    )
    assert n_toks >= 0 and tot_ms >= pre_ms >= 0.0 and isinstance(txt, str)
