"""Track-2 Pure-Forward Hyper-Router & Bipolar Experience State (Asymmetric Undo Law)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AVLPMLConfig


@dataclass
class Track2ExperienceState:
    """Per-request pure-forward experience package (< 200 KB).
    Asymmetric Undo Law:
      - delta_m_syn rolls back / clears on lesion rollback;
      - p_neg (pain imprint) and delta_m_adv (nemesis gain) PERSIST across rollbacks!
    """

    delta_b_pre: torch.Tensor    # [B, L_moe, E]
    p_pos: torch.Tensor          # [B, L_moe, E, r]
    p_neg: torch.Tensor          # [B, L_moe, E, r]  <-- PERSISTENT ACROSS ROLLBACKS
    delta_m_syn: torch.Tensor    # [B, L_moe, E, E]
    delta_m_adv: torch.Tensor    # [B, L_moe, E, E]  <-- PERSISTENT ACROSS ROLLBACKS
    culprit_mask: torch.Tensor   # [B, L_moe, E] bool

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_moe_layers: int,
        num_experts: int,
        probe_rank: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "Track2ExperienceState":
        return cls(
            delta_b_pre=torch.zeros(batch_size, num_moe_layers, num_experts, device=device, dtype=dtype),
            p_pos=torch.zeros(batch_size, num_moe_layers, num_experts, probe_rank, device=device, dtype=dtype),
            p_neg=torch.zeros(batch_size, num_moe_layers, num_experts, probe_rank, device=device, dtype=dtype),
            delta_m_syn=torch.zeros(batch_size, num_moe_layers, num_experts, num_experts, device=device, dtype=dtype),
            delta_m_adv=torch.zeros(batch_size, num_moe_layers, num_experts, num_experts, device=device, dtype=dtype),
            culprit_mask=torch.zeros(batch_size, num_moe_layers, num_experts, device=device, dtype=torch.bool),
        )

    def clone_synergy_snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.p_pos.clone(), self.delta_m_syn.clone()

    def asymmetric_rollback_and_imprint_pain(
        self,
        synergy_snapshot: Tuple[torch.Tensor, torch.Tensor],
        lesion_probe_means: torch.Tensor,
        lesion_expert_weights: torch.Tensor,
        m_adv_manifolds: torch.Tensor,
        gamma_pain: float = 0.85,
        top_culprits: int = 2,
    ) -> None:
        """Erase wrong draft's positive synergy, but permanently imprint P_i^- and nemesis gain!
        Args:
            synergy_snapshot: (p_pos_snap, delta_m_syn_snap) at b* - 1
            lesion_probe_means: [B, L_moe, E, r] mean probe vectors z_i on lesion frame b*
            lesion_expert_weights: [B, L_moe, E] activation weights on lesion frame b*
            m_adv_manifolds: [L_moe, E, E] adversarial graphs
        """
        # 1. Roll back positive bias from polluted frames
        self.p_pos = synergy_snapshot[0].clone()
        self.delta_m_syn = synergy_snapshot[1].clone()

        # 2. Identify top culprit experts on lesion frame b*
        k_cul = min(top_culprits, lesion_expert_weights.shape[-1])
        _, top_idx = torch.topk(lesion_expert_weights, k=k_cul, dim=-1)
        culprit = torch.zeros_like(self.culprit_mask)
        culprit.scatter_(-1, top_idx, True)
        self.culprit_mask = self.culprit_mask | culprit

        # 3. Imprint pain direction P_i^- (persists across rollbacks!)
        z_norm = F.normalize(lesion_probe_means.float(), dim=-1, eps=1e-6).to(self.p_neg.dtype)
        updated_neg = F.normalize(
            gamma_pain * self.p_neg.float() + (1.0 - gamma_pain) * z_norm.float(),
            dim=-1,
            eps=1e-6,
        ).to(self.p_neg.dtype)
        self.p_neg = torch.where(culprit.unsqueeze(-1), updated_neg, self.p_neg)

        # 4. Boost nemesis edges Delta_M_adv^+ from culprits
        cul_f = culprit.to(self.delta_m_adv.dtype).unsqueeze(-1)  # [B, L, E, 1]
        nemesis_boost = 0.10 * cul_f * m_adv_manifolds.unsqueeze(0).to(self.delta_m_adv.dtype)
        self.delta_m_adv = (self.delta_m_adv + nemesis_boost).clamp(0.0, 0.25)

    def record_clean_frame_advance(
        self,
        probe_means: torch.Tensor,
        clean_profile: torch.Tensor,
        converged_hard_frame: bool,
        gamma_pain: float = 0.85,
    ) -> None:
        """Decay old pain by gamma_pain on clean progress; reinforce P_i^+ & Delta_M_syn^+ if hard-converged."""
        self.p_neg = self.p_neg * gamma_pain
        self.culprit_mask.zero_()

        if converged_hard_frame:
            z_norm = F.normalize(probe_means.float(), dim=-1, eps=1e-6).to(self.p_pos.dtype)
            active_mask = clean_profile > (1.0 / float(clean_profile.shape[-1]))
            has_prior = self.p_pos.norm(dim=-1, keepdim=True) > 1e-5
            blended = torch.where(
                has_prior,
                F.normalize(0.8 * self.p_pos.float() + 0.2 * z_norm.float(), dim=-1, eps=1e-6).to(self.p_pos.dtype),
                z_norm,
            )
            self.p_pos = torch.where(active_mask.unsqueeze(-1), blended, self.p_pos)

            co_act = torch.matmul(clean_profile.unsqueeze(-1), clean_profile.unsqueeze(-2))
            eye = torch.eye(clean_profile.shape[-1], device=co_act.device, dtype=torch.bool)
            co_act = co_act.masked_fill(eye, 0.0)
            self.delta_m_syn = (0.90 * self.delta_m_syn + 0.08 * co_act).clamp(0.0, 0.25)


class HyperRouter(nn.Module):
    """Pure-forward Pre-thinking router initializer producing Delta_b_pre in one pass."""

    def __init__(self, cfg: AVLPMLConfig) -> None:
        super().__init__()
        self.num_moe_layers = len(cfg.moe_layer_indices)
        self.num_experts = cfg.num_routed_experts
        hidden_mid = max(64, cfg.hidden_size // 4)
        if self.num_moe_layers > 0:
            self.w_hr1: nn.Linear | None = nn.Linear(cfg.hidden_size * 2, hidden_mid, bias=True)
            self.w_hr2: nn.Linear | None = nn.Linear(hidden_mid, self.num_moe_layers * self.num_experts, bias=False)
            nn.init.zeros_(self.w_hr2.weight)
        else:
            self.w_hr1 = None
            self.w_hr2 = None

    def forward(self, h_prompt_pool: torch.Tensor, h_skel_pool: torch.Tensor) -> torch.Tensor:
        if self.num_moe_layers == 0 or self.w_hr1 is None or self.w_hr2 is None:
            return torch.zeros(
                h_prompt_pool.shape[0], 0, self.num_experts, device=h_prompt_pool.device, dtype=h_prompt_pool.dtype
            )
        ctx = torch.cat([h_prompt_pool, h_skel_pool], dim=-1)
        out = self.w_hr2(F.silu(self.w_hr1(ctx)))
        out = out.view(ctx.shape[0], self.num_moe_layers, self.num_experts)
        return out - out.mean(dim=-1, keepdim=True)
