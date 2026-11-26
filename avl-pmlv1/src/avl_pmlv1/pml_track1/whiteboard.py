"""Track-1 PML Whiteboard: S^RO / S^RW_commit / S^RW_draft + E_slot + Householder + Dual Lights."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AVLPMLConfig
from ..vocab.sym_vocab import SymbolicVocab


@dataclass
class WhiteboardState:
    """Three-view Track-1 memory state with Lazy Commit & <64KB draft isolation."""

    s_ro: torch.Tensor          # [B, N_ro, d_m]
    m_ro: torch.Tensor          # [B, N_ro] in [0, 1]
    s_rw_commit: torch.Tensor   # [B, N_rw, d_m]
    m_rw_commit: torch.Tensor   # [B, N_rw] in [0, 1]
    s_rw_draft: torch.Tensor    # [B, N_rw, d_m]
    m_rw_draft: torch.Tensor    # [B, N_rw] in [0, 1]
    dirty_mask: torch.Tensor    # [B, N_rw] bool

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_ro: int,
        num_rw: int,
        d_m: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "WhiteboardState":
        return cls(
            s_ro=torch.zeros(batch_size, num_ro, d_m, device=device, dtype=dtype),
            m_ro=torch.zeros(batch_size, num_ro, device=device, dtype=dtype),
            s_rw_commit=torch.zeros(batch_size, num_rw, d_m, device=device, dtype=dtype),
            m_rw_commit=torch.zeros(batch_size, num_rw, device=device, dtype=dtype),
            s_rw_draft=torch.zeros(batch_size, num_rw, d_m, device=device, dtype=dtype),
            m_rw_draft=torch.zeros(batch_size, num_rw, device=device, dtype=dtype),
            dirty_mask=torch.zeros(batch_size, num_rw, device=device, dtype=torch.bool),
        )

    @property
    def s_rw_view(self) -> torch.Tensor:
        return torch.where(self.dirty_mask.unsqueeze(-1), self.s_rw_draft, self.s_rw_commit)

    @property
    def m_rw_view(self) -> torch.Tensor:
        return torch.where(self.dirty_mask, self.m_rw_draft, self.m_rw_commit)

    def clone_committed_snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.s_rw_commit.clone(), self.m_rw_commit.clone()

    def lazy_commit_previous_tail(self) -> None:
        """Atomically merge S^RW_draft into S^RW_commit and clear dirty_mask."""
        self.s_rw_commit = self.s_rw_view.clone()
        self.m_rw_commit = self.m_rw_view.clone()
        self.s_rw_draft = self.s_rw_commit.clone()
        self.m_rw_draft = self.m_rw_commit.clone()
        self.dirty_mask = torch.zeros_like(self.dirty_mask)

    def discard_draft(self) -> None:
        """Discard uncommitted wet-cement tail writes without touching S^RW_commit."""
        self.s_rw_draft = self.s_rw_commit.clone()
        self.m_rw_draft = self.m_rw_commit.clone()
        self.dirty_mask = torch.zeros_like(self.dirty_mask)

    def restore_committed_snapshot(self, snapshot: Tuple[torch.Tensor, torch.Tensor]) -> None:
        s_com, m_com = snapshot
        self.s_rw_commit = s_com.clone()
        self.m_rw_commit = m_com.clone()
        self.discard_draft()

    def reset_rw(self) -> None:
        """Reset all read-write registers (commit + draft) to zero while preserving read-only S^RO / M^RO."""
        self.s_rw_commit = torch.zeros_like(self.s_rw_commit)
        self.m_rw_commit = torch.zeros_like(self.m_rw_commit)
        self.s_rw_draft = torch.zeros_like(self.s_rw_draft)
        self.m_rw_draft = torch.zeros_like(self.m_rw_draft)
        self.dirty_mask = torch.zeros_like(self.dirty_mask)

    def repeat_batch(self, rep_factor: int) -> "WhiteboardState":
        """Repeat WhiteboardState along batch dimension for batched 4-fold PLL Post-Thinking audit."""
        if rep_factor <= 1:
            return WhiteboardState(
                s_ro=self.s_ro.clone(),
                m_ro=self.m_ro.clone(),
                s_rw_commit=self.s_rw_commit.clone(),
                m_rw_commit=self.m_rw_commit.clone(),
                s_rw_draft=self.s_rw_draft.clone(),
                m_rw_draft=self.m_rw_draft.clone(),
                dirty_mask=self.dirty_mask.clone(),
            )
        return WhiteboardState(
            s_ro=self.s_ro.repeat(rep_factor, 1, 1),
            m_ro=self.m_ro.repeat(rep_factor, 1),
            s_rw_commit=self.s_rw_commit.repeat(rep_factor, 1, 1),
            m_rw_commit=self.m_rw_commit.repeat(rep_factor, 1),
            s_rw_draft=self.s_rw_draft.repeat(rep_factor, 1, 1),
            m_rw_draft=self.m_rw_draft.repeat(rep_factor, 1),
            dirty_mask=self.dirty_mask.repeat(rep_factor, 1),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f * rms * self.weight.float()).to(orig_dtype)


class Track1PMLModule(nn.Module):
    """Right-brain symbolic state rail with E_slot hybrid addressing & decoupled Dual Lights."""

    def __init__(self, cfg: AVLPMLConfig, vocab: SymbolicVocab) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.d_m = cfg.pml_state_dim
        self.n_ro = cfg.num_ro_slots
        self.n_rw = cfg.num_rw_slots
        self.n_slots = cfg.total_slots

        # Orthogonal slot address embeddings E_slot in R^(N_slot x d_m) (FP32 QR safe for BF16 construction!)
        self.e_slot = nn.Parameter(torch.empty(self.n_slots, self.d_m))
        with torch.no_grad():
            self.e_slot.copy_(nn.init.orthogonal_(torch.empty(self.n_slots, self.d_m, dtype=torch.float32)).to(self.e_slot.dtype))

        self.q_proj = nn.Linear(cfg.hidden_size, self.d_m, bias=False)
        self.q_norm = RMSNorm(self.d_m, cfg.rms_norm_eps)
        self.k_proj = nn.Linear(self.d_m, self.d_m, bias=False)
        self.v_proj = nn.Linear(self.d_m, self.d_m, bias=False)
        self.o_proj = nn.Linear(self.d_m, cfg.hidden_size, bias=False)

        # Orthogonal encoder-decoder pair for exact hard-indexed S^RO / S^RW register read/write
        self.ro_init_proj = nn.Linear(cfg.hidden_size, self.d_m, bias=False)
        self.hard_o_proj = nn.Linear(self.d_m, cfg.hidden_size, bias=False)
        with torch.no_grad():
            ro_f32 = nn.init.orthogonal_(torch.empty(self.d_m, cfg.hidden_size, dtype=torch.float32))
            self.ro_init_proj.weight.copy_(ro_f32.to(self.ro_init_proj.weight.dtype))
            if cfg.cold_start_graft:
                # Master Manual M3 cold-start: tiny W_O scale (std=0.01) avoids Step-0 disturbance to Qwen3-4B!
                nn.init.normal_(self.o_proj.weight, std=0.01)
                self.hard_o_proj.weight.copy_((0.01 * ro_f32.t()).to(self.hard_o_proj.weight.dtype))
            else:
                self.hard_o_proj.weight.copy_(ro_f32.t().to(self.hard_o_proj.weight.dtype))
        self.slot_norm = RMSNorm(self.d_m, cfg.rms_norm_eps)

        # Householder + Erase-gate overwrite parameters on postfix ⟨REG_j⟩
        self.w_k_write = nn.Linear(cfg.hidden_size, self.d_m, bias=False)
        self.w_v_write = nn.Linear(cfg.hidden_size, self.d_m, bias=False)
        self.w_gamma_keep = nn.Linear(cfg.hidden_size, 1, bias=True)
        self.w_beta = nn.Linear(cfg.hidden_size, 1, bias=True)
        self.w_eta = nn.Linear(cfg.hidden_size, 1, bias=True)

        # Epistemic yellow light predictor W_pred: x_t -> r_t
        self.w_pred = nn.Linear(cfg.hidden_size, self.d_m, bias=False)
        self._last_pred_loss: Optional[torch.Tensor] = None

    def pop_pred_loss(self) -> Optional[torch.Tensor]:
        loss = self._last_pred_loss
        self._last_pred_loss = None
        return loss

    def init_ro_slot(self, state: WhiteboardState, slot_idx: int, span_hidden: torch.Tensor) -> None:
        """Write pooled Prompt entity span hidden [B, D] into S^RO[:, slot_idx]."""
        val = self.slot_norm(self.ro_init_proj(span_hidden)).to(state.s_ro.dtype)
        state.s_ro[:, slot_idx, :] = val
        state.m_ro[:, slot_idx] = 1.0

    def read_step_or_block(
        self,
        x: torch.Tensor,
        state: WhiteboardState,
        use_committed_only: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        prev_tokens: Optional[torch.Tensor] = None,
        prev2_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from M_t = [S^RO; S^RW] with:
        1) Hard-Indexed Exact Shadow Register Read on pointer tokens ⟨REG_j⟩ (non-write) & ⟨IN_k⟩ (zero Softmax blur!);
        2) Hybrid soft-attention read + epistemic yellow light on non-pointer tokens.
        Returns (h_pml [B, S, D], u_epistemic [B, S]).
        """
        s_rw = state.s_rw_commit if use_committed_only else state.s_rw_view
        m_rw = state.m_rw_commit if use_committed_only else state.m_rw_view

        m_bank = torch.cat([state.s_ro, s_rw], dim=1).to(x.dtype)        # [B, N_slot, d_m]
        conf_mask = torch.cat([state.m_ro, m_rw], dim=1).to(x.dtype)     # [B, N_slot]

        q = self.q_norm(self.q_proj(x))                                  # [B, S, d_m]
        k = self.e_slot.unsqueeze(0).to(x.dtype) + self.k_proj(m_bank)   # [B, N_slot, d_m]
        log_m = torch.log(conf_mask.clamp(min=1e-4)).unsqueeze(1)        # [B, 1, N_slot]
        v = self.v_proj(m_bank)                                          # [B, N_slot, d_m]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_m) + log_m
        p_attn = F.softmax(scores.float(), dim=-1).to(x.dtype)           # [B, S, N_slot]
        r_t = torch.matmul(p_attn, v)                                    # [B, S, d_m]
        h_pml = self.o_proj(r_t)

        # Yellow light: epistemic surprise in [0, 1] (detach x into w_pred + record self-supervised MSE!)
        p_f = p_attn.float().clamp(min=1e-8)
        entropy_norm = -(p_f * p_f.log()).sum(dim=-1) / math.log(float(self.n_slots))
        pred_r = self.w_pred(x.detach()).float()
        r_f = r_t.float()
        if self.training and torch.is_grad_enabled():
            p_mse = F.mse_loss(pred_r, r_f.detach())
            self._last_pred_loss = p_mse if self._last_pred_loss is None else (0.5 * self._last_pred_loss + 0.5 * p_mse)
        rel_err = ((pred_r.detach() - r_f).norm(dim=-1) / (r_f.norm(dim=-1) + 1e-6)).clamp(0.0, 1.0)
        u_epistemic = (0.5 * entropy_norm + 0.5 * rel_err).clamp(0.0, 1.0).to(x.dtype)

        # Hard-Indexed Exact Register Dereference on explicit pointer tokens ⟨REG_j⟩ / ⟨IN_k⟩ (zero Softmax blur!)
        if token_ids is not None:
            bsz, slen = token_ids.shape
            if prev_tokens is None:
                prev_tokens = torch.cat(
                    [torch.zeros(bsz, 1, dtype=token_ids.dtype, device=token_ids.device), token_ids[:, :-1]], dim=1
                )
            if prev2_tokens is None:
                prev2_tokens = torch.cat(
                    [torch.zeros(bsz, 1, dtype=token_ids.dtype, device=token_ids.device), prev_tokens[:, :-1]], dim=1
                )
            is_write_dst = self.vocab.is_postfix_write_dst(token_ids, prev_tokens, prev2_tokens)
            is_rw_ptr = self.vocab.is_rw_register_tensor(token_ids) & (~is_write_dst)
            is_ro_ptr = (token_ids >= self.vocab.ro_start) & (token_ids < self.vocab.ro_end)

            # Shadow register semantics: REG_j reads S^RW[j] if written (m_rw > 0.05), else falls back to S^RO[j]!
            n_sh = min(self.n_rw, self.n_ro)
            s_reg_sh = s_rw.clone()
            m_reg_sh = m_rw.clone()
            has_rw_sh = m_rw[:, :n_sh] > 0.05
            s_reg_sh[:, :n_sh, :] = torch.where(has_rw_sh.unsqueeze(-1), s_rw[:, :n_sh, :], state.s_ro[:, :n_sh, :])
            m_reg_sh[:, :n_sh] = torch.where(has_rw_sh, m_rw[:, :n_sh], state.m_ro[:, :n_sh])

            rw_idx = (token_ids - self.vocab.rw_start).clamp(0, self.n_rw - 1)
            ro_idx = (token_ids - self.vocab.ro_start).clamp(0, self.n_ro - 1)
            r_rw = s_reg_sh.gather(1, rw_idx.unsqueeze(-1).expand(-1, -1, self.d_m))
            m_rw_g = m_reg_sh.gather(1, rw_idx)
            r_ro = state.s_ro.gather(1, ro_idx.unsqueeze(-1).expand(-1, -1, self.d_m))
            m_ro_g = state.m_ro.gather(1, ro_idx)

            r_hard = torch.where(is_rw_ptr.unsqueeze(-1), r_rw, r_ro)
            m_hard = torch.where(is_rw_ptr, m_rw_g, m_ro_g)
            valid_ptr = (is_rw_ptr | is_ro_ptr) & (m_hard > 0.05)

            h_hard = self.hard_o_proj(r_hard.to(x.dtype))
            h_pml = torch.where(valid_ptr.unsqueeze(-1), h_hard, h_pml)
            u_epistemic = torch.where(valid_ptr, torch.zeros_like(u_epistemic), u_epistemic)

        return h_pml, u_epistemic

    def read_masked_blocks_with_clean_trace(
        self,
        norm_x: torch.Tensor,
        masked_token_ids: torch.Tensor,
        base_state: WhiteboardState,
        clean_token_ids: torch.Tensor,
        clean_val_embed: torch.Tensor,
        frame_sizes: list[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strictly block-causal Post-Thinking PML audit:
        Reads each block b of `masked_token_ids` against the clean pre-block whiteboard snapshot accumulated over
        blocks 0 .. b-1 of `clean_token_ids` (zero [MASK] whiteboard pollution, zero future-register time-travel!).
        """
        scan_state = base_state.repeat_batch(1)
        clean_x_proxy = torch.where(
            (masked_token_ids == SymbolicVocab.MASK_ID).unsqueeze(-1),
            clean_val_embed.to(norm_x.dtype),
            norm_x,
        )
        h_parts: list[torch.Tensor] = []
        u_parts: list[torch.Tensor] = []
        seq_len = norm_x.shape[1]
        off = 0
        for flen in frame_sizes:
            if flen <= 0:
                continue
            end = min(seq_len, off + flen)
            if end <= off:
                break
            h_b, u_b = self.read_step_or_block(
                norm_x[:, off:end, :],
                scan_state,
                use_committed_only=True,
                token_ids=masked_token_ids[:, off:end],
            )
            h_parts.append(h_b)
            u_parts.append(u_b)
            if end < seq_len:
                _ = self.sequential_scan_forward(
                    clean_x_proxy[:, off:end, :],
                    token_ids=clean_token_ids[:, off:end],
                    state=scan_state,
                    write_to_draft=False,
                    x_val_embed=clean_val_embed[:, off:end, :],
                    frame_sizes=[end - off],
                )
            off = end
        return torch.cat(h_parts, dim=1), torch.cat(u_parts, dim=1)

    def apply_single_slot_write(
        self,
        s_rw: torch.Tensor,
        m_rw: torch.Tensor,
        dirty_mask: Optional[torch.Tensor],
        batch_idx: int,
        slot_j: int,
        h_reg: torch.Tensor,
        is_reassert: bool,
    ) -> torch.Tensor:
        """Single postfix ⟨REG_j⟩ Householder + erase-gate overwrite; returns scalar c_wb conflict."""
        h_f = h_reg.float()
        s_old = s_rw[batch_idx, slot_j].float()
        m_old = m_rw[batch_idx, slot_j].float()

        k_j = F.normalize(self.w_k_write(h_reg).float(), dim=-1, eps=1e-6)
        v_j = self.w_v_write(h_reg).float()
        gamma_keep = torch.sigmoid(self.w_gamma_keep(h_reg).float()).squeeze(-1)
        if not is_reassert:
            # Recycled/fresh ⟨BIND⟩ wipes stale dead dimensions cleanly
            gamma_keep = 0.05 * gamma_keep
        beta_j = torch.sigmoid(self.w_beta(h_reg).float()).squeeze(-1)
        eta_j = torch.sigmoid(self.w_eta(h_reg).float()).squeeze(-1)

        # Decoupled whiteboard conflict c_wb: ONLY active on re-assertion (⟨VERIFY⟩ / ⟨COMMIT⟩)
        if is_reassert:
            cos_sim = F.cosine_similarity(s_old.unsqueeze(0), v_j.unsqueeze(0), dim=-1, eps=1e-6).squeeze(0)
            c_wb = F.relu(-cos_sim) * m_old
        else:
            c_wb = torch.zeros((), device=h_f.device, dtype=torch.float32)

        proj = torch.dot(s_old, k_j) * k_j
        s_raw = gamma_keep * (s_old - beta_j * proj) + eta_j * v_j
        s_new = self.slot_norm(s_raw.to(h_reg.dtype)).to(s_rw.dtype)

        s_rw[batch_idx, slot_j] = s_new
        base_m = m_old if is_reassert else torch.zeros_like(m_old)
        m_rw[batch_idx, slot_j] = torch.clamp(0.5 * gamma_keep * base_m + 0.5 * eta_j + 0.35, max=1.0).to(m_rw.dtype)
        if dirty_mask is not None:
            dirty_mask[batch_idx, slot_j] = True
        return c_wb

    def sequential_scan_forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        state: WhiteboardState,
        reassert_flags: Optional[torch.Tensor] = None,
        write_to_draft: bool = False,
        contra_probs: Optional[torch.Tensor] = None,
        sandbox_error: float = 0.0,
        prev_last_token: Optional[torch.Tensor] = None,
        dual_stream_sym_blocks: int = 0,
        dual_stream_block_size: int = 0,
        precomputed_write_cols: Optional[list[int]] = None,
        x_val_embed: Optional[torch.Tensor] = None,
        prev2_val_embed: Optional[torch.Tensor] = None,
        prev2_last_token: Optional[torch.Tensor] = None,
        prev2_ctx_hidden: Optional[torch.Tensor] = None,
        frame_sizes: Optional[list[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared left-to-right sequential scan kernel for exact train-inference equivalence (< 1e-5).
        v1.2 Hard-Indexed Register Memory:
        - Pre-projects value embeddings ONCE via shared ro_init_proj so S^RO & S^RW live in exact aligned geometry;
        - Fresh ⟨BIND⟩ cleanly overwrites S^RW[j] (gamma_keep = 0.0), eliminating 24-step stale ghost drift;
        - Supports mixed 8/16-slot physical frame_sizes AND dual-stream BD3-LM without splitting frames or polluting S^RW;
        - Batched read_step_or_block performs exact hard-indexed shadow register lookup on ⟨REG_j⟩ / ⟨IN_k⟩!
        """
        bsz, seq_len, dim = x.shape
        chunk = dual_stream_block_size if dual_stream_block_size > 0 else self.cfg.atomic_chunk_size
        u_acute_steps = torch.zeros(bsz, seq_len, device=x.device, dtype=torch.float32)

        # Precompute exact t-1 and t-2 tokens for every position (preserving chunk-boundary parity & postfix write guard!)
        first_p = (
            prev_last_token.unsqueeze(1)
            if prev_last_token is not None
            else torch.zeros(bsz, 1, dtype=token_ids.dtype, device=token_ids.device)
        )
        prev_tokens_all = torch.cat([first_p, token_ids[:, :-1]], dim=1) if seq_len > 1 else first_p
        first_p2 = (
            prev2_last_token.unsqueeze(1)
            if prev2_last_token is not None
            else torch.zeros(bsz, 1, dtype=token_ids.dtype, device=token_ids.device)
        )
        prev2_tokens_all = torch.cat([first_p2, prev_tokens_all[:, :-1]], dim=1) if seq_len > 1 else first_p2

        # Build exact scan intervals: either variable frame_sizes (8/16-slot exact boundaries) or uniform chunk width
        scan_spans: list[Tuple[int, int]] = []
        noisy_spans: list[Tuple[int, int]] = []
        clean_start_chunk = 10**9
        noisy_start_chunk = 10**9

        if frame_sizes is not None and len(frame_sizes) > 0:
            s_frames = sum(frame_sizes)
            if dual_stream_sym_blocks > 0 and len(frame_sizes) == dual_stream_sym_blocks and seq_len >= 2 * s_frames:
                p_pre = seq_len - 2 * s_frames
                for st in range(0, p_pre, chunk):
                    scan_spans.append((st, min(st + chunk, p_pre)))
                clean_start_chunk = len(scan_spans)
                cur_off = p_pre
                for flen in frame_sizes:
                    scan_spans.append((cur_off, cur_off + flen))
                    cur_off += flen
                noisy_start_chunk = len(scan_spans)
                for flen in frame_sizes:
                    noisy_spans.append((cur_off, cur_off + flen))
                    cur_off += flen
                prev_tokens_all[:, p_pre + s_frames] = 0
                prev2_tokens_all[:, p_pre + s_frames] = 0
                if s_frames > 1:
                    prev2_tokens_all[:, p_pre + s_frames + 1] = 0
            elif dual_stream_sym_blocks == 0 and s_frames == seq_len:
                cur_off = 0
                for flen in frame_sizes:
                    if cur_off > 0 and cur_off < seq_len:
                        prev_tokens_all[:, cur_off] = 0
                        prev2_tokens_all[:, cur_off] = 0
                        if cur_off + 1 < seq_len:
                            prev2_tokens_all[:, cur_off + 1] = 0
                    scan_spans.append((cur_off, cur_off + flen))
                    cur_off += flen

        if not scan_spans and not noisy_spans:
            total_chunks = (seq_len + chunk - 1) // chunk
            if dual_stream_sym_blocks > 0:
                noisy_start_chunk = total_chunks - dual_stream_sym_blocks
                clean_start_chunk = noisy_start_chunk - dual_stream_sym_blocks
            for c_i, st in enumerate(range(0, seq_len, chunk)):
                en = min(st + chunk, seq_len)
                if c_i < noisy_start_chunk:
                    scan_spans.append((st, en))
                else:
                    noisy_spans.append((st, en))

        all_spans = scan_spans + noisy_spans
        batch_unit = 8 if (frame_sizes is not None and all((e - s) % 8 == 0 and (e - s) > 0 for s, e in all_spans)) else chunk
        can_batch_all = (seq_len > 0) and (seq_len % batch_unit == 0) and all((e - s) % batch_unit == 0 for s, e in all_spans)

        boundary_snaps: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        s_rw_per_unit: list[torch.Tensor] = []
        m_rw_per_unit: list[torch.Tensor] = []
        h_pml_chunks: list[torch.Tensor] = []
        u_epi_chunks: list[torch.Tensor] = []

        # Pre-project clean value embeddings ONCE per layer for direct aligned S^RW writes (zero per-step GEMM loop!)
        v_emb_proj_all = self.ro_init_proj(x_val_embed).float() if x_val_embed is not None else None

        for c_idx, (start, end) in enumerate(scan_spans):
            span_len = end - start
            if clean_start_chunk <= c_idx < noisy_start_chunk:
                b_rel = c_idx - clean_start_chunk
                boundary_snaps[b_rel] = state.clone_committed_snapshot()

            if can_batch_all:
                s_snap_u = state.s_rw_view.clone()
                m_snap_u = state.m_rw_view.clone()
                for _ in range(span_len // batch_unit):
                    s_rw_per_unit.append(s_snap_u)
                    m_rw_per_unit.append(m_snap_u)
            else:
                h_c, u_epi_c = self.read_step_or_block(
                    x[:, start:end, :],
                    state,
                    token_ids=token_ids[:, start:end],
                    prev_tokens=prev_tokens_all[:, start:end],
                    prev2_tokens=prev2_tokens_all[:, start:end],
                )
                h_pml_chunks.append(h_c)
                u_epi_chunks.append(u_epi_c)

            if precomputed_write_cols is not None:
                t_iter = [t for t in precomputed_write_cols if start <= t < end]
            elif torch.is_grad_enabled():
                c_toks = token_ids[:, start:end]
                p_toks = prev_tokens_all[:, start:end]
                p2_toks = prev2_tokens_all[:, start:end]
                w_any = self.vocab.is_postfix_write_dst(c_toks, p_toks, p2_toks)
                if start == 0 and prev_last_token is None:
                    w_any[:, 0] = False
                active_offsets = w_any.any(dim=0).nonzero(as_tuple=False).view(-1).tolist()
                t_iter = [start + off for off in active_offsets]
            else:
                t_iter = list(range(start, end))

            if not t_iter:
                continue

            # Batched left-to-right deterministic write scan inside chunk:
            # - Training: 1-step truncated gradient (detach s_old/m_old while training ro_init_proj & w_v_write!)
            # - Inference (torch.no_grad): 100% pure-forward zero-backprop, in-place CUDAGraph-safe!
            next_s = state.s_rw_view.clone() if write_to_draft else state.s_rw_commit.clone()
            next_m = state.m_rw_view.clone() if write_to_draft else state.m_rw_commit.clone()
            next_dirty = state.dirty_mask.clone() if write_to_draft else None
            grad_on = torch.is_grad_enabled()

            for t in t_iter:
                if t == 0 and prev_last_token is None:
                    continue
                prev_tok = prev_tokens_all[:, t]
                prev2_tok = prev2_tokens_all[:, t]
                cur_tok = token_ids[:, t]
                is_write = self.vocab.is_postfix_write_dst(cur_tok, prev_tok, prev2_tok)

                slot_j = (cur_tok - self.vocab.rw_start).clamp(0, self.n_rw - 1)  # [B]
                idx_s = slot_j.view(bsz, 1, 1).expand(-1, 1, self.d_m)
                idx_m = slot_j.view(bsz, 1)

                # Truncate recurrence at s_old/m_old (O(1) memory, zero cross-block BPTT!)
                s_old = next_s.gather(1, idx_s).squeeze(1).detach().float()   # [B, d_m]
                m_old = next_m.gather(1, idx_m).squeeze(1).detach().float()   # [B]
                h_reg = x[:, t, :]                                            # [B, D]

                if reassert_flags is not None:
                    is_reassert = reassert_flags[:, t] > 0
                else:
                    is_reassert = (prev_tok == SymbolicVocab.VERIFY_ID) | (prev_tok == SymbolicVocab.COMMIT_ID)

                k_j = F.normalize(self.w_k_write(h_reg).float(), dim=-1, eps=1e-6)
                v_ctx = self.w_v_write(h_reg).float()
                if v_emb_proj_all is not None and (t - start) >= 2:
                    v_disc = v_emb_proj_all[:, t - 2, :] + 0.15 * v_ctx
                    if self.cfg.continuous_esc_whiteboard:
                        is_esc_t2 = self.vocab.is_escape_anchor_tensor(token_ids[:, t - 2]).unsqueeze(-1)
                        v_cont = self.ro_init_proj(x[:, t - 2, :]).float() + 0.35 * v_ctx
                        v_j = torch.where(is_esc_t2, v_cont, v_disc)
                    else:
                        v_j = v_disc
                elif v_emb_proj_all is not None and t >= 2 and dual_stream_block_size == 1:
                    v_disc = v_emb_proj_all[:, t - 2, :] + 0.15 * v_ctx
                    if self.cfg.continuous_esc_whiteboard:
                        is_esc_t2 = self.vocab.is_escape_anchor_tensor(token_ids[:, t - 2]).unsqueeze(-1)
                        v_cont = self.ro_init_proj(x[:, t - 2, :]).float() + 0.35 * v_ctx
                        v_j = torch.where(is_esc_t2, v_cont, v_disc)
                    else:
                        v_j = v_disc
                elif prev2_val_embed is not None and seq_len == 1:
                    v_disc = self.ro_init_proj(prev2_val_embed).float() + 0.15 * v_ctx
                    if self.cfg.continuous_esc_whiteboard and prev2_last_token is not None and prev2_ctx_hidden is not None:
                        is_esc_t2 = self.vocab.is_escape_anchor_tensor(prev2_last_token).unsqueeze(-1)
                        v_cont = self.ro_init_proj(prev2_ctx_hidden).float() + 0.35 * v_ctx
                        v_j = torch.where(is_esc_t2, v_cont, v_disc)
                    else:
                        v_j = v_disc
                else:
                    v_j = v_ctx

                gamma_re = torch.sigmoid(self.w_gamma_keep(h_reg).float()).squeeze(-1)
                # Fresh <BIND> performs exact clean overwrite (gamma_keep = 0.0 -> zero ghost residue across 24+ steps!)
                gamma_keep = torch.where(is_reassert, gamma_re, torch.zeros_like(gamma_re))
                beta_j = torch.sigmoid(self.w_beta(h_reg).float()).squeeze(-1)
                eta_re = torch.sigmoid(self.w_eta(h_reg).float()).squeeze(-1)
                eta_j = torch.where(is_reassert, eta_re, torch.ones_like(eta_re))

                with torch.no_grad():
                    cos_sim = F.cosine_similarity(s_old, v_j.detach(), dim=-1, eps=1e-6)
                    c_wb = torch.where(is_reassert & is_write, F.relu(-cos_sim) * m_old, torch.zeros_like(m_old))
                    u_acute_steps[:, t] = torch.maximum(u_acute_steps[:, t], c_wb)

                proj = (s_old * k_j).sum(dim=-1, keepdim=True) * k_j
                s_raw = gamma_keep.unsqueeze(-1) * (s_old - beta_j.unsqueeze(-1) * proj) + eta_j.unsqueeze(-1) * v_j
                s_new = self.slot_norm(s_raw.to(h_reg.dtype)).to(next_s.dtype)

                base_m = torch.where(is_reassert, m_old, torch.zeros_like(m_old))
                m_new = torch.clamp(0.5 * gamma_keep * base_m + 0.5 * eta_j + 0.50, max=1.0).to(next_m.dtype)

                cur_slot_s = next_s.gather(1, idx_s).squeeze(1)
                cur_slot_m = next_m.gather(1, idx_m).squeeze(1)
                upd_s = torch.where(is_write.unsqueeze(-1), s_new, cur_slot_s)
                upd_m = torch.where(is_write, m_new, cur_slot_m)
                if grad_on:
                    next_s = next_s.scatter(1, idx_s, upd_s.unsqueeze(1))
                    next_m = next_m.scatter(1, idx_m, upd_m.unsqueeze(1))
                    if next_dirty is not None:
                        cur_d = next_dirty.gather(1, idx_m).squeeze(1)
                        next_dirty = next_dirty.scatter(1, idx_m, (cur_d | is_write).unsqueeze(1))
                else:
                    next_s.scatter_(1, idx_s, upd_s.unsqueeze(1))
                    next_m.scatter_(1, idx_m, upd_m.unsqueeze(1))
                    if next_dirty is not None:
                        cur_d = next_dirty.gather(1, idx_m).squeeze(1)
                        next_dirty.scatter_(1, idx_m, (cur_d | is_write).unsqueeze(1))

            if write_to_draft:
                if torch.is_grad_enabled():
                    state.s_rw_draft = next_s
                    state.m_rw_draft = next_m
                    assert next_dirty is not None
                    state.dirty_mask = next_dirty
                else:
                    state.s_rw_draft.copy_(next_s)
                    state.m_rw_draft.copy_(next_m)
                    assert next_dirty is not None
                    state.dirty_mask.copy_(next_dirty)
            else:
                if torch.is_grad_enabled():
                    state.s_rw_commit = next_s
                    state.m_rw_commit = next_m
                    state.s_rw_draft = next_s.clone()
                    state.m_rw_draft = next_m.clone()
                    state.dirty_mask = torch.zeros_like(state.dirty_mask)
                else:
                    state.s_rw_commit.copy_(next_s)
                    state.m_rw_commit.copy_(next_m)
                    state.s_rw_draft.copy_(next_s)
                    state.m_rw_draft.copy_(next_m)
                    state.dirty_mask.zero_()

        if can_batch_all:
            if noisy_spans:
                for b_rel, (ns, ne) in enumerate(noisy_spans):
                    s_snap_n, m_snap_n = boundary_snaps[b_rel]
                    for _ in range((ne - ns) // batch_unit):
                        s_rw_per_unit.append(s_snap_n)
                        m_rw_per_unit.append(m_snap_n)
            n_tot = len(s_rw_per_unit)
            x_bat = x.reshape(bsz * n_tot, batch_unit, dim)
            tok_bat = token_ids.reshape(bsz * n_tot, batch_unit)
            prev_bat = prev_tokens_all.reshape(bsz * n_tot, batch_unit)
            prev2_bat = prev2_tokens_all.reshape(bsz * n_tot, batch_unit)
            s_ro_exp = state.s_ro.unsqueeze(1).expand(-1, n_tot, -1, -1).reshape(bsz * n_tot, self.n_ro, self.d_m)
            m_ro_exp = state.m_ro.unsqueeze(1).expand(-1, n_tot, -1).reshape(bsz * n_tot, self.n_ro)
            s_rw_bat = torch.stack(s_rw_per_unit, dim=1).reshape(bsz * n_tot, self.n_rw, self.d_m)
            m_rw_bat = torch.stack(m_rw_per_unit, dim=1).reshape(bsz * n_tot, self.n_rw)
            dummy_d = torch.zeros(bsz * n_tot, self.n_rw, device=x.device, dtype=torch.bool)
            bat_state = WhiteboardState(s_ro_exp, m_ro_exp, s_rw_bat, m_rw_bat, s_rw_bat, m_rw_bat, dummy_d)
            h_all, u_all = self.read_step_or_block(
                x_bat,
                bat_state,
                use_committed_only=True,
                token_ids=tok_bat,
                prev_tokens=prev_bat,
                prev2_tokens=prev2_bat,
            )
            h_out = h_all.reshape(bsz, seq_len, dim)
            u_epi_out = u_all.reshape(bsz, seq_len)
        else:
            if noisy_spans:
                for b_rel, (ns, ne) in enumerate(noisy_spans):
                    s_snap_n, m_snap_n = boundary_snaps[b_rel]
                    dummy_d = torch.zeros(bsz, self.n_rw, device=x.device, dtype=torch.bool)
                    n_state = WhiteboardState(state.s_ro, state.m_ro, s_snap_n, m_snap_n, s_snap_n, m_snap_n, dummy_d)
                    h_n, u_n = self.read_step_or_block(
                        x[:, ns:ne, :],
                        n_state,
                        use_committed_only=True,
                        token_ids=token_ids[:, ns:ne],
                        prev_tokens=prev_tokens_all[:, ns:ne],
                        prev2_tokens=prev2_tokens_all[:, ns:ne],
                    )
                    h_pml_chunks.append(h_n)
                    u_epi_chunks.append(u_n)
            h_out = torch.cat(h_pml_chunks, dim=1)
            u_epi_out = torch.cat(u_epi_chunks, dim=1)

        is_contra_tok = (token_ids == SymbolicVocab.CONTRA_ID).float()
        p_contra = contra_probs.float() if contra_probs is not None else is_contra_tok
        p_contra = torch.maximum(p_contra, is_contra_tok)
        sb_tensor = torch.full_like(u_acute_steps, float(sandbox_error))
        u_acute = torch.maximum(torch.maximum(p_contra, u_acute_steps), sb_tensor).clamp(0.0, 1.0)

        return h_out, u_epi_out, u_acute.to(x.dtype)
