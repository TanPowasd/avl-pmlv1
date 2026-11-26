"""FP32 Log-Domain 6-Step Sinkhorn-Knopp Zero-Diagonal Bipolar Graphs & Relay Graph."""

from __future__ import annotations

import math
from typing import Tuple
import torch
import torch.nn as nn


def sinkhorn_log_domain_batched(
    logits: torch.Tensor,
    zero_diag_mask: torch.Tensor,
    num_iters: int = 6,
) -> torch.Tensor:
    """Batched FP32 log-domain Sinkhorn over [N, E, E] with per-slice zero-diagonal boolean mask [N, E, E]."""
    log_m = logits.float().masked_fill(zero_diag_mask, -1e9)
    for _ in range(num_iters):
        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)
        log_m = log_m.masked_fill(zero_diag_mask, -1e9)
        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)
        log_m = log_m.masked_fill(zero_diag_mask, -1e9)
    m = torch.exp(log_m).masked_fill(zero_diag_mask, 0.0)
    return m / m.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def sinkhorn_log_domain(
    logits: torch.Tensor,
    zero_diagonal: bool = True,
    num_iters: int = 6,
) -> torch.Tensor:
    """Project E x E parameter matrix onto (zero-diagonal) doubly-stochastic manifold in FP32."""
    e_dim = logits.shape[-1]
    log_m = logits.float()
    if zero_diagonal:
        eye = torch.eye(e_dim, device=logits.device, dtype=torch.bool)
        log_m = log_m.masked_fill(eye, -1e9)

    for _ in range(num_iters):
        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)
        if zero_diagonal:
            log_m = log_m.masked_fill(eye, -1e9)
        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)
        if zero_diagonal:
            log_m = log_m.masked_fill(eye, -1e9)

    m = torch.exp(log_m)
    if zero_diagonal:
        m = m.masked_fill(eye, 0.0)
    # Final exact row normalization to eliminate residual 1e-6 FP32 drift
    m = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return m


def row_normalize_overlay(base_m: torch.Tensor, delta_pos: torch.Tensor, clip_max: float = 0.25) -> torch.Tensor:
    """Compute T_r(M + clip(Delta_M^+, 0, clip_max)) preserving row-stochastic l_1 norm == 1."""
    eye = torch.eye(base_m.shape[-1], device=base_m.device, dtype=torch.bool)
    overlay = delta_pos.float().clamp(0.0, clip_max).masked_fill(eye, 0.0)
    combined = base_m.float() + overlay
    return combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-8)


class BipolarLinkageGraph(nn.Module):
    """Maintains M_syn, M_adv (zero-diagonal doubly-stochastic) and M_inter (Birkhoff)."""

    def __init__(self, num_experts: int, num_iters: int = 6) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_iters = num_iters
        # Disjoint circulant symmetry-breaking initialization strictly inside healthy [ln 2, ln 4] & overlap <= 0.20 basin
        init_syn = torch.randn(num_experts, num_experts, dtype=torch.float32) * 0.08
        init_adv = torch.randn(num_experts, num_experts, dtype=torch.float32) * 0.08
        half = max(2, num_experts // 2)
        for i in range(num_experts):
            init_syn[i, (i + 1) % num_experts] += 2.60
            init_syn[i, (i + 2) % num_experts] += 2.30
            init_adv[i, (i + half) % num_experts] += 2.60
            init_adv[i, (i + half + 1) % num_experts] += 2.30
        init_inter = torch.randn(num_experts, num_experts, dtype=torch.float32) * 0.08
        for i in range(num_experts):
            init_inter[i, i] += 2.40
            init_inter[i, (i + 1) % num_experts] += 1.65
        self.theta_syn = nn.Parameter(init_syn)
        self.theta_adv = nn.Parameter(init_adv)
        self.theta_inter = nn.Parameter(init_inter)

    def get_manifolds(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m_syn = sinkhorn_log_domain(self.theta_syn, zero_diagonal=True, num_iters=self.num_iters)
        m_adv = sinkhorn_log_domain(self.theta_adv, zero_diagonal=True, num_iters=self.num_iters)
        m_inter = sinkhorn_log_domain(self.theta_inter, zero_diagonal=False, num_iters=self.num_iters)
        return m_syn, m_adv, m_inter

    def graph_regularization_loss(
        self,
        manifolds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        collect_metrics: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        m_syn, m_adv, m_inter = self.get_manifolds() if manifolds is None else manifolds
        h_min = math.log(2.0)
        h_max = math.log(min(4.0, float(max(2, self.num_experts - 1))))

        def _bandpass_entropy(m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            p = m.clamp(min=1e-8)
            ent = -(p * p.log()).sum(dim=-1)
            loss_e = (torch.relu(h_min - ent).pow(2) + torch.relu(ent - h_max).pow(2)).mean()
            return loss_e, ent.mean()

        l_e_syn, ent_syn = _bandpass_entropy(m_syn)
        l_e_adv, _ = _bandpass_entropy(m_adv)
        l_e_inter, _ = _bandpass_entropy(m_inter)

        trace_sq = torch.diagonal(torch.matmul(m_syn, m_syn)).mean()
        l_cycle = torch.relu(trace_sq - 0.18).pow(2)

        cos_orth = (m_syn * m_adv).sum() / (m_syn.norm() * m_adv.norm() + 1e-8)
        total = l_e_syn + l_e_adv + l_e_inter + l_cycle + cos_orth
        if collect_metrics:
            metrics = {
                "row_entropy_syn": float(ent_syn.detach().item()),
                "syn_adv_overlap": float(cos_orth.detach().item()),
            }
        else:
            metrics = {}
        return total, metrics
