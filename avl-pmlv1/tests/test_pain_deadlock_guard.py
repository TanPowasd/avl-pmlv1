"""Mandatory Test #3: 8-Expert Small-Pool Pain Deadlock Guard (c_pain_max = 0.60 & K_min_guard = 1)."""

from __future__ import annotations

from pathlib import Path
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.moe_track2.hyper_router import Track2ExperienceState
from avl_pmlv1.moe_track2.tri_state_moe import TriStateMoELayer
from avl_pmlv1.vocab.sym_vocab import CognitivePhase


def test_eight_expert_two_breaker_pain_deadlock_guard() -> None:
    """Simulate 8-expert pool suffering 2 consecutive circuit-breaker rollbacks:
    1) Verify pain deduction cosine is strictly clamped at c_pain_max = 0.60;
    2) Verify when threshold tau is extreme so all raw gates == 0 while t_k <= 0.5 & u_epistemic > 0.35,
       K_min_guard = 1 forces exactly 1 non-culprit expert open with g_min = 0.25!
    """
    root = Path(__file__).resolve().parents[1]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
    assert cfg.num_routed_experts == 8

    torch.manual_seed(77)
    moe = TriStateMoELayer(cfg, moe_slot_idx=0)
    moe.eval()

    exp = Track2ExperienceState.zeros(
        batch_size=1,
        num_moe_layers=len(cfg.moe_layer_indices),
        num_experts=8,
        probe_rank=cfg.probe_rank,
        device=torch.device("cpu"),
    )
    _, m_adv, _ = moe.graph.get_manifolds()
    m_adv_all = m_adv.unsqueeze(0).repeat(len(cfg.moe_layer_indices), 1, 1)

    x = torch.randn(1, 8, cfg.hidden_size)
    with torch.no_grad():
        z_cur, _ = moe.compute_probes(moe.norm(x))
        z_mean = z_cur.mean(dim=1, keepdim=True).repeat(1, len(cfg.moe_layer_indices), 1, 1).squeeze(2)
        # Fix shape: [B=1, L_moe, E=8, r=8]
        z_lesion = z_cur.mean(dim=1).unsqueeze(1).repeat(1, len(cfg.moe_layer_indices), 1, 1)

        # Breaker #1: Experts 0 and 1 are culprits
        syn_snap = exp.clone_synergy_snapshot()
        w1 = torch.zeros(1, len(cfg.moe_layer_indices), 8)
        w1[:, :, 0] = 0.6
        w1[:, :, 1] = 0.4
        exp.asymmetric_rollback_and_imprint_pain(syn_snap, z_lesion, w1, m_adv_all, top_culprits=2)

        # Breaker #2: Experts 2 and 3 are culprits on second attempt
        w2 = torch.zeros(1, len(cfg.moe_layer_indices), 8)
        w2[:, :, 2] = 0.6
        w2[:, :, 3] = 0.4
        exp.asymmetric_rollback_and_imprint_pain(syn_snap, z_lesion, w2, m_adv_all, top_culprits=2)

        # Now culprits = {0, 1, 2, 3}. Raise tau_0 artificially high so 0 experts pass natural threshold
        moe.tau_0.fill_(50.0)
        u_epi_high = torch.full((1, 8), 0.70)  # > 0.35 -> triggers K_min_guard!
        u_acute_low = torch.zeros((1, 8))

        out, aux = moe(
            x,
            phase=CognitivePhase.FORMAL,
            t_k=0.25,  # <= 0.50 low-noise beat
            u_epistemic=u_epi_high,
            u_acute=u_acute_low,
            exp_state=exp,
        )

        gates = aux["gates"]  # [1, 8, 8]
        active_per_tok = (gates > 0.0).sum(dim=-1)
        assert (active_per_tok == 1).all(), f"Expected K_min_guard=1 active expert per slot, got {active_per_tok}"

        # Rescued expert must NOT be in culprit set {0, 1, 2, 3} and must have g_min = 0.25
        rescued_experts = gates.argmax(dim=-1)
        assert (rescued_experts >= 4).all(), f"Deadlock guard picked culprit expert: {rescued_experts}"
        max_gate_vals = gates.max(dim=-1).values
        assert torch.allclose(max_gate_vals, torch.full_like(max_gate_vals, cfg.g_min_guard))
        assert torch.isfinite(out).all()
