"""Attn-PML Hybrid Layer: Left-brain GQA (+ shallow value residual) || Right-brain Track-1 PML."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AVLPMLConfig
from ..vocab.sym_vocab import SymbolicVocab
from .whiteboard import RMSNorm, Track1PMLModule, WhiteboardState


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply rotary position embeddings to x [B, H, S, d_h] using integer positions [B, S]."""
    d_h = x.shape[-1]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, d_h, 2, device=x.device, dtype=torch.float32) / float(d_h))
    )
    freqs = torch.einsum("bs,d->bsd", positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(1)  # [B, 1, S, d_h]
    cos = emb.cos().to(x.dtype)
    sin = emb.sin().to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class LeftBrainGQA(nn.Module):
    """Grouped-Query Attention with zero-initialized shallow value residual."""

    def __init__(self, cfg: AVLPMLConfig, has_value_residual: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_q = cfg.num_q_heads
        self.num_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.num_groups = self.num_q // self.num_kv

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_q * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q * self.head_dim, cfg.hidden_size, bias=False)

        self.use_qk_norm = cfg.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.has_value_residual = has_value_residual
        if has_value_residual:
            self.alpha_vres = nn.Parameter(torch.zeros(1, self.num_kv, 1, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        v_anchor: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_q, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv, self.head_dim)
        if self.use_qk_norm and self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv, self.head_dim).transpose(1, 2)
        v_raw = v

        if self.has_value_residual and v_anchor is not None and v_anchor.shape == v.shape:
            v = v + torch.tanh(self.alpha_vres).to(v.dtype) * v_anchor

        q = apply_rope(q, positions, base=self.cfg.rope_theta)
        k = apply_rope(k, positions, base=self.cfg.rope_theta)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present_kv = (k, v)

        if self.num_groups > 1:
            k_exp = k.repeat_interleave(self.num_groups, dim=1)
            v_exp = v.repeat_interleave(self.num_groups, dim=1)
        else:
            k_exp, v_exp = k, v

        out = F.scaled_dot_product_attention(
            q,
            k_exp,
            v_exp,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None and past_kv is None),
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_q * self.head_dim)
        return self.o_proj(out), v_raw, present_kv


class AttnPMLHybridLayer(nn.Module):
    """Left-Brain Full GQA || Right-Brain Track-1 PML with zero-initialized calm gate + 4D FiLM."""

    def __init__(
        self,
        cfg: AVLPMLConfig,
        vocab: SymbolicVocab,
        enable_pml: bool = True,
        is_first_layer: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.enable_pml = enable_pml
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.gqa = LeftBrainGQA(cfg, has_value_residual=(not is_first_layer))

        if enable_pml:
            self.pml = Track1PMLModule(cfg, vocab)
            init_calm = 0.08 if cfg.cold_start_graft else 0.45
            self.alpha_calm = nn.Parameter(torch.full((cfg.hidden_size,), init_calm))
            self.film_proj = nn.Linear(4, cfg.hidden_size, bias=True)
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)
        else:
            self.pml = None

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        wb_state: Optional[WhiteboardState] = None,
        d_b: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        v_anchor: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reassert_flags: Optional[torch.Tensor] = None,
        write_to_draft: bool = False,
        read_only_pml: bool = False,
        dual_stream_sym_blocks: int = 0,
        dual_stream_block_size: int = 0,
        precomputed_write_cols: Optional[list[int]] = None,
        x_val_embed: Optional[torch.Tensor] = None,
        frame_sizes: Optional[list[int]] = None,
        clean_token_ids_for_pml: Optional[torch.Tensor] = None,
        clean_val_embed_for_pml: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        norm_x = self.norm(x)
        h_gqa, v_raw, present_kv = self.gqa(
            norm_x, positions=positions, attn_mask=attn_mask, v_anchor=v_anchor, past_kv=past_kv
        )

        if self.enable_pml and self.pml is not None and wb_state is not None:
            if read_only_pml:
                if (
                    clean_token_ids_for_pml is not None
                    and clean_val_embed_for_pml is not None
                    and frame_sizes is not None
                    and len(frame_sizes) > 0
                ):
                    h_pml, u_epi = self.pml.read_masked_blocks_with_clean_trace(
                        norm_x,
                        masked_token_ids=token_ids,
                        base_state=wb_state,
                        clean_token_ids=clean_token_ids_for_pml,
                        clean_val_embed=clean_val_embed_for_pml,
                        frame_sizes=frame_sizes,
                    )
                else:
                    h_pml, u_epi = self.pml.read_step_or_block(norm_x, wb_state, token_ids=token_ids)
                u_acute = (token_ids == SymbolicVocab.CONTRA_ID).to(x.dtype)
            else:
                h_pml, u_epi, u_acute = self.pml.sequential_scan_forward(
                    norm_x,
                    token_ids=token_ids,
                    state=wb_state,
                    reassert_flags=reassert_flags,
                    write_to_draft=write_to_draft,
                    dual_stream_sym_blocks=dual_stream_sym_blocks,
                    dual_stream_block_size=dual_stream_block_size,
                    precomputed_write_cols=precomputed_write_cols,
                    x_val_embed=x_val_embed,
                    frame_sizes=frame_sizes,
                )
            if d_b is not None and d_b.dim() == 3:
                gamma_film: float | torch.Tensor = self.film_proj(d_b.to(x.dtype))
            elif d_b is not None:
                gamma_film = self.film_proj(d_b.to(x.dtype)).unsqueeze(1)
            else:
                gamma_film = 0.0
            gate = torch.tanh(self.alpha_calm.view(1, 1, -1) * (1.0 + gamma_film)).to(h_gqa.dtype)
            h_fused = h_gqa + gate * h_pml
        else:
            h_fused = h_gqa
            u_epi = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
            u_acute = (token_ids == SymbolicVocab.CONTRA_ID).to(x.dtype)

        return x + h_fused, v_raw, u_epi, u_acute, present_kv
