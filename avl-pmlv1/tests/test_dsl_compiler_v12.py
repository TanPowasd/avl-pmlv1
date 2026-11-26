"""Mandatory Stage 1 Test #5: AVL-PMLv1.2 DSL Compiler Check 5 (1-Causal-Hop-Per-Beat) & 8/16-Slot Alignment."""

from __future__ import annotations

from pathlib import Path
import pytest

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.vocab.dsl_compiler import CompilerValidationError, LinearScanDSLCompiler, MicroStepIR
from avl_pmlv1.vocab.sym_vocab import SymbolicVocab


def _compiler() -> LinearScanDSLCompiler:
    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_4_5b.yaml")
    return LinearScanDSLCompiler(cfg, SymbolicVocab(cfg))


def test_check5_rejects_multi_hop_chained_ops() -> None:
    comp = _compiler()
    bad_step = MicroStepIR(
        op="⟨DEDUCE⟩",
        in_refs=["⟨IN_0⟩"],
        lit_tokens=["+", "2", "*", "3"],  # 2 chained binary ops in 1 micro-step -> forbidden!
        out_var="$x",
        val_token="9",
    )
    with pytest.raises(CompilerValidationError, match="Check 5"):
        comp.compile([bad_step], final_live_vars=["$x"])


def test_check5_rejects_missing_bound_value() -> None:
    comp = _compiler()
    bad_step = MicroStepIR(op="⟨DEDUCE⟩", in_refs=["⟨IN_0⟩"], out_var="$x")
    with pytest.raises(CompilerValidationError, match="missing bound value"):
        comp.compile([bad_step], final_live_vars=["$x"])


def test_aligned_8_16_slot_val_at_t_minus_2() -> None:
    comp = _compiler()
    v = comp.vocab
    steps = [
        MicroStepIR("⟨GOAL⟩", in_refs=["⟨IN_0⟩"], lit_tokens=["+"], val_token="5", out_var="$a"),
        MicroStepIR("⟨DEDUCE⟩", var_deps=["$a"], lit_tokens=["*"], fsq_pairs=[(12, 34)], val_token="15", out_var="$b"),
        MicroStepIR("⟨COMMIT⟩", var_deps=["$b"], lit_tokens=["-"], val_token="14", out_var="$c", flag="⟨EOT⟩"),
    ]
    tr = comp.compile_aligned_frames(steps, final_live_vars=["$c"], prefer_8slot=True)
    # Step 0 & 2 fit in 8 slots (7 toks <= 8); Step 1 has FSQ pair + op + val (9 toks) -> auto-promotes to 16 slots!
    assert tr.frame_sizes == [8, 16, 8], f"Expected [8, 16, 8], got {tr.frame_sizes}"
    assert len(tr.token_ids) == 32
    offset = 0
    for f_len in tr.frame_sizes:
        blk = tr.token_ids[offset : offset + f_len]
        offset += f_len
        assert blk[-3] == SymbolicVocab.BIND_ID
        assert v.rw_start <= blk[-2] < v.rw_end
        assert blk[-4] != SymbolicVocab.PAD_ID and blk[-4] != SymbolicVocab.MASK_ID
    assert tr.token_ids[-1] == SymbolicVocab.EOT_ID
