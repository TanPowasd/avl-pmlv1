"""Equal-width out-of-bounds-free Dual-Layer 5D FSQ & Factored Hypercube Head."""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AVLPMLConfig


class DualLayerFSQCodec(nn.Module):
    """Two-tier equal-width 5D Finite Scalar Quantizer with exact [0, prod(L)-1] bijection."""

    def __init__(self, levels: List[int]) -> None:
        super().__init__()
        self.levels = list(levels)
        self.dim = len(self.levels)
        levels_t = torch.tensor(self.levels, dtype=torch.long)
        strides = torch.ones_like(levels_t)
        for d in range(1, self.dim):
            strides[d] = strides[d - 1] * levels_t[d - 1]
        self.register_buffer("levels_t", levels_t, persistent=False)
        self.register_buffer("strides_t", strides, persistent=False)
        self.codebook_size = int(torch.prod(levels_t).item())

    def quantize_single_tier(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize u (..., 5) -> (code_ids, integer_coords, ste_continuous_codes)."""
        if u.shape[-1] != self.dim:
            raise ValueError(f"Expected latent dim {self.dim}, got {u.shape[-1]}")
        levels_f = self.levels_t.to(device=u.device, dtype=u.dtype)
        v = 0.5 * (torch.tanh(u) + 1.0)
        coords = torch.floor(levels_f * v).to(torch.long)
        max_coords = (self.levels_t.to(u.device) - 1).view(*([1] * (u.ndim - 1)), self.dim)
        coords = torch.minimum(coords.clamp(min=0), max_coords)

        q_hat = (2.0 * coords.to(u.dtype) + 1.0) / levels_f - 1.0
        tanh_u = torch.tanh(u)
        q_ste = tanh_u + (q_hat - tanh_u).detach()

        strides = self.strides_t.to(u.device)
        code_ids = (coords * strides).sum(dim=-1)
        return code_ids, coords, q_ste

    def index_to_coords(self, code_ids: torch.Tensor) -> torch.Tensor:
        """Bijection inverse: map discrete code_ids (...) -> coords (..., 5)."""
        rem = code_ids.clamp(0, self.codebook_size - 1)
        coords = []
        for d in range(self.dim):
            lvl = int(self.levels[d])
            coords.append(rem % lvl)
            rem = rem // lvl
        return torch.stack(coords, dim=-1)


class FactoredSymbolicEmbedding(nn.Module):
    """5D coordinate-factored hypercube embedding + low-rank residual for V_sym."""

    def __init__(self, cfg: AVLPMLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.codec = DualLayerFSQCodec(cfg.fsq_levels)
        self.fsq_start = cfg.op_vocab_size
        self.fsq_mid = self.fsq_start + cfg.single_fsq_size
        self.fsq_end = self.fsq_mid + cfg.single_fsq_size

        # Non-FSQ static symbols: V_op + V_lit (primitives + escape + registers)
        self.op_embed = nn.Embedding(cfg.op_vocab_size, cfg.hidden_size)
        non_fsq_tail = cfg.sym_vocab_size - self.fsq_end
        self.tail_embed = nn.Embedding(non_fsq_tail, cfg.hidden_size)

        # Tier 1 (Coarse) & Tier 2 (Fine) 5D axis embeddings + low-rank specificity residual
        self.coarse_axes = nn.ModuleList(
            [nn.Embedding(lvl, cfg.hidden_size) for lvl in cfg.fsq_levels]
        )
        self.fine_axes = nn.ModuleList(
            [nn.Embedding(lvl, cfg.hidden_size) for lvl in cfg.fsq_levels]
        )
        self.coarse_res_code = nn.Embedding(cfg.single_fsq_size, cfg.fsq_residual_rank)
        self.coarse_res_up = nn.Linear(cfg.fsq_residual_rank, cfg.hidden_size, bias=False)
        self.fine_res_code = nn.Embedding(cfg.single_fsq_size, cfg.fsq_residual_rank)
        self.fine_res_up = nn.Linear(cfg.fsq_residual_rank, cfg.hidden_size, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self._eval_weight_cache: tuple[tuple[object, ...], torch.Tensor] | None = None

        self._reset_parameters()

    def train(self, mode: bool = True) -> "FactoredSymbolicEmbedding":
        if mode:
            self._eval_weight_cache = None
        return super().train(mode)

    def _cache_key(self, dev: torch.device) -> tuple[object, ...]:
        vers = [
            self.op_embed.weight._version,
            self.tail_embed.weight._version,
            self.coarse_res_code.weight._version,
            self.coarse_res_up.weight._version,
            self.fine_res_code.weight._version,
            self.fine_res_up.weight._version,
        ]
        vers.extend(ax.weight._version for ax in self.coarse_axes)
        vers.extend(ax.weight._version for ax in self.fine_axes)
        return (dev, self.op_embed.weight.dtype, *vers)

    def _reset_parameters(self) -> None:
        std = 1.0 / (self.cfg.hidden_size ** 0.5)
        nn.init.normal_(self.op_embed.weight, std=std)
        nn.init.normal_(self.tail_embed.weight, std=std)
        for emb in list(self.coarse_axes) + list(self.fine_axes):
            nn.init.normal_(emb.weight, std=0.45 * std / (len(self.cfg.fsq_levels) ** 0.5))
        # Balanced factor variance prevents 350x asymmetric gradient explosion on res_up in 4.5B/8.9B models!
        code_std = 0.15 if self.cfg.cold_start_graft else 0.50
        up_std = (0.5 * std / (self.cfg.fsq_residual_rank ** 0.5)) / max(code_std, 1e-4)
        nn.init.normal_(self.coarse_res_code.weight, std=code_std)
        nn.init.normal_(self.coarse_res_up.weight, std=min(up_std, 0.05))
        nn.init.normal_(self.fine_res_code.weight, std=code_std)
        nn.init.normal_(self.fine_res_up.weight, std=min(up_std, 0.05))
        if self.cfg.cold_start_graft:
            with torch.no_grad():
                self.logit_scale.fill_(0.5)

    def _tier_table(self, axes: nn.ModuleList, res_code: nn.Embedding, res_up: nn.Linear, device: torch.device) -> torch.Tensor:
        ids = torch.arange(self.cfg.single_fsq_size, device=device)
        coords = self.codec.index_to_coords(ids)  # [N_code, 5]
        acc = torch.zeros(self.cfg.single_fsq_size, self.cfg.hidden_size, device=device, dtype=res_up.weight.dtype)
        for d, axis_emb in enumerate(axes):
            acc = acc + axis_emb(coords[:, d])
        acc = acc + res_up(res_code(ids))
        return acc

    def _lookup_tier(
        self, code_ids: torch.Tensor, axes: nn.ModuleList, res_code: nn.Embedding, res_up: nn.Linear
    ) -> torch.Tensor:
        coords = self.codec.index_to_coords(code_ids)
        acc = res_up(res_code(code_ids))
        for d, axis_emb in enumerate(axes):
            acc = acc + axis_emb(coords[..., d])
        return acc

    def full_weight_matrix(self, device: torch.device | None = None) -> torch.Tensor:
        dev = device if device is not None else self.op_embed.weight.device
        use_cache = (not self.training) and (not torch.is_grad_enabled())
        if use_cache:
            key = self._cache_key(dev)
            if self._eval_weight_cache is not None and self._eval_weight_cache[0] == key:
                return self._eval_weight_cache[1]
        coarse_w = self._tier_table(self.coarse_axes, self.coarse_res_code, self.coarse_res_up, dev)
        fine_w = self._tier_table(self.fine_axes, self.fine_res_code, self.fine_res_up, dev)
        full_w = torch.cat([self.op_embed.weight, coarse_w, fine_w, self.tail_embed.weight], dim=0)
        if use_cache:
            self._eval_weight_cache = (self._cache_key(dev), full_w)
        return full_w

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.numel() <= 512:
            # Fast O(B*S) direct factored lookup (avoids building 99MB [19328, D] table on 1-tok AR & 8-tok dLLM!)
            out = torch.empty(
                (*token_ids.shape, self.cfg.hidden_size),
                device=token_ids.device,
                dtype=self.op_embed.weight.dtype,
            )
            m_op = token_ids < self.fsq_start
            m_c = (token_ids >= self.fsq_start) & (token_ids < self.fsq_mid)
            m_f = (token_ids >= self.fsq_mid) & (token_ids < self.fsq_end)
            m_tail = token_ids >= self.fsq_end
            if m_op.any():
                out[m_op] = self.op_embed(token_ids[m_op])
            if m_c.any():
                out[m_c] = self._lookup_tier(
                    token_ids[m_c] - self.fsq_start, self.coarse_axes, self.coarse_res_code, self.coarse_res_up
                )
            if m_f.any():
                out[m_f] = self._lookup_tier(
                    token_ids[m_f] - self.fsq_mid, self.fine_axes, self.fine_res_code, self.fine_res_up
                )
            if m_tail.any():
                out[m_tail] = self.tail_embed(token_ids[m_tail] - self.fsq_end)
            return out
        weight = self.full_weight_matrix(token_ids.device)
        return F.embedding(token_ids, weight)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.full_weight_matrix(hidden.device)
        return F.linear(hidden, weight) * self.logit_scale.to(hidden.dtype)
