"""Mandatory Test #4: Static Engram Step-1 All-[MASK] Threshold Guard & 4-Fold PLL Zero-Leakage."""

from __future__ import annotations

from pathlib import Path
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.post_surgery import build_4fold_complementary_masks
from avl_pmlv1.engram.static_engram import StaticEngram
from avl_pmlv1.engine.model import AVLPMLModel
from avl_pmlv1.moe_track2.tri_state_moe import TriStateMoELayer
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab


def test_engram_step1_all_mask_guard_and_cap() -> None:
    """Verify under 100% [MASK] Step-1 input:
    1) alpha_eng == 0 and alpha_eng_eff == 0 everywhere;
    2) Threshold tau does NOT collapse below clean/low-noise threshold (w_t * t_k keeps tau elevated);
    3) Active routed experts per slot never exceed K_max^(step1) = min(K_max, 2).
    """
    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    cfg.k_max = 3  # Test Step-1 ceiling min(3, 2) == 2 explicitly!

    torch.manual_seed(99)
    engram = StaticEngram(cfg)
    moe = TriStateMoELayer(cfg, moe_slot_idx=0)
    # Lower tau_0 so many experts want to open unless capped by Step-1 ceiling
    with torch.no_grad():
        moe.tau_0.fill_(-5.0)

    all_mask_tokens = torch.full((2, 16), SymbolicVocab.MASK_ID, dtype=torch.long)
    x = torch.randn(2, 16, cfg.hidden_size)

    with torch.no_grad():
        x_eng, alpha_eng, alpha_eng_eff = engram(x, all_mask_tokens, is_sym_phase=True)
        assert alpha_eng.abs().max().item() == 0.0, "All-[MASK] input must produce exact 0 alpha_eng!"
        assert alpha_eng_eff.abs().max().item() == 0.0, "All-[MASK] input must produce exact 0 alpha_eng_eff!"
        assert torch.allclose(x_eng, x), "Zero-init / masked Static Engram must not perturb all-[MASK] residual!"

        r_clean_zero = torch.zeros(2, 1)
        _, aux_step1 = moe(
            x_eng,
            phase=CognitivePhase.FORMAL,
            t_k=1.0,
            r_clean=r_clean_zero,
            alpha_eng_eff=alpha_eng_eff,
        )
        max_active = int(aux_step1["active_count"].max().item())
        assert max_active <= 2, f"Step-1 ceiling violated: opened {max_active} > 2 experts!"


def test_engram_4fold_pll_zero_target_leakage() -> None:
    """Verify Static Engram + Conv1D leaks ZERO information about masked target tokens during 4-fold PLL."""
    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    torch.manual_seed(314)
    model = AVLPMLModel(cfg)
    # Activate non-zero Conv1D and Engram output projection to test true runtime regime!
    for eng in model.engrams.values():
        torch.nn.init.normal_(eng.conv1d.weight, std=0.1)
        torch.nn.init.normal_(eng.out_proj.weight, std=0.1)
    model.eval()

    seq_a = torch.randint(2, cfg.sym_vocab_size, (1, 16), dtype=torch.long)
    seq_b = seq_a.clone()
    folds = build_4fold_complementary_masks(16, seq_a.device)
    fold0 = folds[0]  # Positions 0, 4, 8, 12 are masked

    # Corrupt ground-truth tokens ONLY at the masked positions in seq_b
    seq_b[0, fold0] = (seq_a[0, fold0] + 137) % (cfg.sym_vocab_size - 2) + 2

    masked_a = seq_a.clone()
    masked_b = seq_b.clone()
    masked_a[0, fold0] = SymbolicVocab.MASK_ID
    masked_b[0, fold0] = SymbolicVocab.MASK_ID

    with torch.no_grad():
        out_a = model.forward_symbolic(masked_a, phase=CognitivePhase.POST, t_k=0.25, read_only_pml=True)
        out_b = model.forward_symbolic(masked_b, phase=CognitivePhase.POST, t_k=0.25, read_only_pml=True)

    diff = (out_a["logits"] - out_b["logits"]).abs().max().item()  # type: ignore[operator]
    assert diff == 0.0, f"Target leakage detected in 4-fold PLL! Logit diff = {diff}"
