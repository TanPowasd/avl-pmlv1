"""Mandatory Test #1: Lazy Commit Visibility, Wet-Cement Refresh, Recycled BIND Safety & Asymmetric Undo Law."""

from __future__ import annotations

from pathlib import Path
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.wet_cement import select_wet_cement_remask_slots
from avl_pmlv1.engine.model import AVLPMLModel
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab


def _load_40m_model() -> AVLPMLModel:
    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    torch.manual_seed(1234)
    model = AVLPMLModel(cfg)
    model.eval()
    return model


def test_cross_block_draft_visibility_and_lazy_commit() -> None:
    """Verify Block b reads Block b-1's uncommitted tail S^RW_draft seamlessly, and Lazy Commit merges cleanly."""
    model = _load_40m_model()
    wb = model.init_whiteboard_states(batch_size=1, device=torch.device("cpu"))[0]
    pml = model.attn_layers[model.cfg.pml_layer_indices[0]].pml
    assert pml is not None

    # Construct a 16-slot tail block writing to ⟨REG_3⟩
    tail_blk = torch.full((1, 16), SymbolicVocab.DEDUCE_ID, dtype=torch.long)
    tail_blk[0, 14] = SymbolicVocab.BIND_ID
    tail_blk[0, 15] = model.vocab.reg_slot_to_id(3)
    x_tail = torch.randn(1, 16, model.cfg.hidden_size)

    # Write into S^RW_draft ONLY (Lazy Commit deferred!)
    _, _, _ = pml.sequential_scan_forward(x_tail, tail_blk, wb, write_to_draft=True)

    assert bool(wb.dirty_mask[0, 3].item()) is True
    assert wb.s_rw_commit[0, 3].abs().max().item() == 0.0, "S^RW_commit must stay untouched before Lazy Commit!"
    assert wb.s_rw_draft[0, 3].norm().item() > 0.5, "S^RW_draft[3] must hold the newly written state!"
    assert torch.allclose(wb.s_rw_view[0, 3], wb.s_rw_draft[0, 3]), "Unified S^RW_view must expose draft to Block b!"

    # Wet-cement 15%-30% remask on the 16-slot tail refreshes S^RW_draft without touching S^RW_commit
    margins = torch.linspace(0.05, 0.95, 16).unsqueeze(0)
    remasked, remask_bool = select_wet_cement_remask_slots(tail_blk, margins, d_bwd=0.80)
    assert 2 <= int(remask_bool.sum().item()) <= 5
    assert (remasked[remask_bool] == SymbolicVocab.MASK_ID).all()

    wb.discard_draft()
    x_tail_refreshed = x_tail * 1.5 + 0.3
    _, _, _ = pml.sequential_scan_forward(x_tail_refreshed, tail_blk, wb, write_to_draft=True)
    assert wb.s_rw_commit[0, 3].abs().max().item() == 0.0, "Historical S^RW_commit still 100% frozen!"

    # Now execute Lazy Commit once Block b succeeds
    wb.lazy_commit_previous_tail()
    assert not bool(wb.dirty_mask.any().item())
    assert torch.allclose(wb.s_rw_commit[0, 3], wb.s_rw_view[0, 3])
    assert wb.s_rw_commit[0, 3].norm().item() > 0.5


def test_asymmetric_undo_law_and_recycled_bind_no_false_red_light() -> None:
    """Verify:
    1) Recycled register ⟨BIND⟩ with opposite hidden vector does NOT trigger false Red-Light c_wb;
    2) Contradictory ⟨VERIFY⟩ re-assertion DOES trigger Red-Light c_wb;
    3) Asymmetric Undo Law: rollback clears dirty_mask & delta_m_syn, while preserving P_i^- & delta_m_adv!
    """
    model = _load_40m_model()
    wb = model.init_whiteboard_states(batch_size=1, device=torch.device("cpu"))[0]
    exp = model.init_experience_state(batch_size=1, device=torch.device("cpu"))
    pml = model.attn_layers[model.cfg.pml_layer_indices[0]].pml
    assert pml is not None

    # Step A: Commit initial hypothesis into ⟨REG_0⟩
    h_pos = torch.randn(model.cfg.hidden_size)
    _ = pml.apply_single_slot_write(wb.s_rw_commit, wb.m_rw_commit, None, 0, 0, h_pos, is_reassert=False)
    wb.s_rw_draft.copy_(wb.s_rw_commit)
    wb.m_rw_draft.copy_(wb.m_rw_commit)
    committed_snap = wb.clone_committed_snapshot()

    # Step B: Compiler recycles dead ⟨REG_0⟩ for an opposite vector (-h_pos) via routine ⟨BIND⟩ (is_reassert=False)
    c_wb_recycle = pml.apply_single_slot_write(
        wb.s_rw_draft, wb.m_rw_draft, wb.dirty_mask, 0, 0, -h_pos, is_reassert=False
    )
    assert float(c_wb_recycle.item()) == 0.0, "Recycled ⟨BIND⟩ must NEVER trigger false-positive whiteboard Red Light!"

    # Restore and test true contradictory re-assertion via ⟨VERIFY⟩ (is_reassert=True)
    wb.restore_committed_snapshot(committed_snap)
    c_wb_verify = pml.apply_single_slot_write(
        wb.s_rw_draft, wb.m_rw_draft, wb.dirty_mask, 0, 0, -h_pos, is_reassert=True
    )
    assert float(c_wb_verify.item()) > 0.30, "Contradictory ⟨VERIFY⟩ on live slot must trigger positive c_wb Red Light!"

    # Step C: Asymmetric Undo Law on rollback
    syn_snap = exp.clone_synergy_snapshot()
    # Pollute positive synergy in wrong frame
    exp.delta_m_syn.fill_(0.20)
    exp.p_pos.fill_(0.50)

    num_moe = len(model.cfg.moe_layer_indices)
    lesion_z = torch.randn(1, num_moe, model.cfg.num_routed_experts, model.cfg.probe_rank)
    lesion_w = torch.zeros(1, num_moe, model.cfg.num_routed_experts)
    lesion_w[:, :, 2] = 0.90  # Expert 2 is primary culprit
    m_adv_stack = torch.stack(
        [ffn.graph.get_manifolds()[1] for ffn in model.ffn_layers if hasattr(ffn, "graph")], dim=0
    )

    wb.restore_committed_snapshot(committed_snap)
    exp.asymmetric_rollback_and_imprint_pain(syn_snap, lesion_z, lesion_w, m_adv_stack)

    assert not bool(wb.dirty_mask.any().item()), "Dirty draft mask must be wiped on rollback!"
    assert torch.allclose(wb.s_rw_view, committed_snap[0]), "Track-1 whiteboard must roll back to clean snapshot!"
    assert exp.delta_m_syn.abs().max().item() == 0.0, "Wrong frame positive synergy delta_m_syn must be wiped!"
    assert exp.p_neg[0, 0, 2].norm().item() > 0.99, "Culprit pain imprint P_2^- MUST persist across rollback!"
    assert exp.delta_m_adv[0, 0, 2].sum().item() > 0.0, "Nemesis boost Delta_M_adv^+ MUST persist across rollback!"


def test_four_phase_runner_end_to_end() -> None:
    """Verify FourPhaseRunner completes Pre -> Formal (Wet-Cement + Lazy Commit) -> Post (4-fold PLL) cleanly."""
    from avl_pmlv1.engine.four_phase_runner import FourPhaseRunner

    model = _load_40m_model()
    runner = FourPhaseRunner(model)
    prompt_prefix = torch.tensor([[model.vocab.in_slot_to_id(0), model.vocab.in_slot_to_id(1)]], dtype=torch.long)
    res = runner.run_session(prompt_prefix, num_formal_blocks=2, skel_len=16, denoise_beats=3)
    assert res.pre_skeleton.shape == (1, 16)
    assert res.final_tokens.shape[0] == 1 and res.final_tokens.shape[1] >= 32
    assert (res.final_tokens != SymbolicVocab.MASK_ID).all()
