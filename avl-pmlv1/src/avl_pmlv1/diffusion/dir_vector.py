"""Four-Dimensional Multi-Directional Diffusion Vector d_b = [d_fwd, d_bwd, d_hier, d_div]."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AVLPMLConfig


class DirectionVectorHead(nn.Module):
    """Lightweight 2-layer MLP (< 0.1M params) producing d_b in (0, 1)^4."""

    def __init__(self, cfg: AVLPMLConfig, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.hidden_size + 4  # [h_pool; u_epistemic; u_acute; t_k; r_clean]
        self.fc1 = nn.Linear(in_dim, bottleneck_dim, bias=True)
        self.fc2 = nn.Linear(bottleneck_dim, 4, bias=True)
        nn.init.zeros_(self.fc2.weight)
        # Default neutral operating point: moderate forward/hier, low backward/div
        nn.init.constant_(self.fc2.bias, 0.0)
        with torch.no_grad():
            self.fc2.bias.copy_(torch.tensor([0.2, -0.4, 0.4, -0.8]))

    def forward(
        self,
        h_block: torch.Tensor,
        valid_mask: torch.Tensor,
        u_epistemic_mean: torch.Tensor,
        u_acute_mean: torch.Tensor,
        t_k: float | torch.Tensor,
        r_clean: float | torch.Tensor,
        force_breaker_div: bool = False,
    ) -> torch.Tensor:
        w = valid_mask.to(h_block.dtype).unsqueeze(-1)
        h_pool = (h_block * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        bsz = h_block.shape[0]
        if isinstance(t_k, torch.Tensor):
            tk_t = t_k.reshape(bsz, 1).to(h_block.dtype)
        else:
            tk_t = torch.full((bsz, 1), float(t_k), device=h_block.device, dtype=h_block.dtype)
        if isinstance(r_clean, torch.Tensor):
            rc_t = r_clean.view(bsz, 1).to(h_block.dtype)
        else:
            rc_t = torch.full((bsz, 1), float(r_clean), device=h_block.device, dtype=h_block.dtype)
        ctx = torch.cat(
            [
                h_pool,
                u_epistemic_mean.view(bsz, 1).to(h_block.dtype),
                u_acute_mean.view(bsz, 1).to(h_block.dtype),
                tk_t,
                rc_t,
            ],
            dim=-1,
        )
        d_b = torch.sigmoid(self.fc2(F.silu(self.fc1(ctx))))
        if force_breaker_div:
            d_div = d_b[:, 3:4].clamp(min=0.80)
            d_b = torch.cat([d_b[:, :3], d_div], dim=-1)
        return d_b

    @staticmethod
    def decide_next_block_size(
        d_b: torch.Tensor,
        u_epistemic: float,
        has_long_frame_op: bool = False,
        has_short_frame_op: bool = False,
    ) -> int:
        if has_short_frame_op:
            return 16
        d_fwd = float(d_b[:, 0].mean().item())
        d_div = float(d_b[:, 3].mean().item())
        if has_long_frame_op or (d_fwd > 0.75 and u_epistemic < 0.25):
            return 64
        if u_epistemic > 0.55 or d_div > 0.65:
            return 16
        return 32
