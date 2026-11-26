"""Static Engram: Fully vectorized Three-way anti-[MASK] prime-bucket hashing & Step-1 guard."""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AVLPMLConfig
from ..vocab.sym_vocab import SymbolicVocab


class StaticEngram(nn.Module):
    """Sparse symbolic shortcut table mounted on select layers (e.g. l in {4, 16})."""

    PRIMES_A = (1315423911, 2654435761, 805306457, 1610612741, 402653189, 201326611)
    PRIMES_B = (374761393, 668265263, 1274126177, 999999937, 198491317, 776531419)

    def __init__(self, cfg: AVLPMLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.engram_num_heads
        self.num_buckets = cfg.engram_num_buckets
        self.engram_dim = cfg.engram_dim

        self.tables = nn.ModuleList(
            [nn.Embedding(self.num_buckets, self.engram_dim) for _ in range(self.num_heads)]
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.engram_dim,
            out_channels=self.engram_dim,
            kernel_size=3,
            padding=0,
            groups=self.engram_dim,
            bias=True,
        )
        self.gate_proj = nn.Linear(cfg.hidden_size + self.engram_dim, 1, bias=True)
        self.out_proj = nn.Linear(self.engram_dim, cfg.hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for tbl in self.tables:
            nn.init.normal_(tbl.weight, std=0.04)
        nn.init.zeros_(self.conv1d.weight)
        nn.init.zeros_(self.conv1d.bias)
        if self.cfg.cold_start_graft:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, -0.5)

    def _compute_three_way_hashes(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        """Vectorized anti-[MASK] bucket indices (zero Python time loops, zero target leakage)."""
        bsz, seq_len = token_ids.shape
        device = token_ids.device
        is_valid = (token_ids != SymbolicVocab.MASK_ID) & (token_ids != SymbolicVocab.PAD_ID)
        safe_tok = torch.where(is_valid, token_ids, torch.zeros_like(token_ids))

        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        # 1. Vectorized last-seen operator via cummax on valid operator positions
        is_op = is_valid & (token_ids >= 2) & (token_ids < self.cfg.op_vocab_size)
        op_pos = torch.where(is_op, pos_idx, torch.zeros_like(pos_idx))
        last_op_pos = torch.cummax(op_pos, dim=1).values
        has_seen_op = torch.cummax(is_op.int(), dim=1).values > 0
        op_carry = torch.where(has_seen_op, safe_tok.gather(1, last_op_pos), torch.zeros_like(safe_tok))

        # 2. Adjacent left token (zero if masked)
        left_tok = torch.cat([torch.zeros(bsz, 1, dtype=torch.long, device=device), safe_tok[:, :-1]], dim=1)

        # 3. Vectorized skip-[MASK] nearest preceding valid token via cummax on shifted valid positions
        prev_valid_pos_raw = torch.where(is_valid, pos_idx + 1, torch.zeros_like(pos_idx))
        shifted_valid_pos = torch.cat(
            [torch.zeros(bsz, 1, dtype=torch.long, device=device), prev_valid_pos_raw[:, :-1]], dim=1
        )
        nearest_prev_plus1 = torch.cummax(shifted_valid_pos, dim=1).values
        has_prev = nearest_prev_plus1 > 0
        gather_prev = (nearest_prev_plus1 - 1).clamp(min=0)
        skip_bigram_prev = torch.where(has_prev, safe_tok.gather(1, gather_prev), torch.zeros_like(safe_tok))

        hashes = []
        for h in range(self.num_heads):
            pa = self.PRIMES_A[h % len(self.PRIMES_A)]
            pb = self.PRIMES_B[h % len(self.PRIMES_B)]
            mode = h % 3
            if mode == 0:
                raw_h = (op_carry * pa + safe_tok * pb + (h + 1) * 97) % self.num_buckets
            elif mode == 1:
                raw_h = (left_tok * pa + safe_tok * pb + (h + 1) * 193) % self.num_buckets
            else:
                raw_h = (skip_bigram_prev * pa + safe_tok * pb + (h + 1) * 389) % self.num_buckets
            hashes.append(raw_h.clamp(0, self.num_buckets - 1))
        return hashes

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        is_sym_phase: bool = True,
        dual_stream_sym_len: int = 0,
        dual_stream_block_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        if not is_sym_phase:
            zeros = torch.zeros(bsz, seq_len, device=x.device, dtype=x.dtype)
            return x, zeros, zeros

        if dual_stream_sym_len > 0 and seq_len > dual_stream_sym_len:
            # Isolate [Prompt, Y_clean] and run each noisy block b against its visible prefix
            # (or strictly block-locally for 8-slot atomic micro-blocks so Prefix-KV inference matches 100% bit-for-bit!)
            p_len = seq_len - 2 * dual_stream_sym_len
            blk_sz = dual_stream_block_size if dual_stream_block_size > 0 else self.cfg.atomic_chunk_size
            num_blks = dual_stream_sym_len // blk_sz
            if blk_sz == 8:
                x_p, a_p, ae_p = self.forward(x[:, :p_len], token_ids[:, :p_len], True, 0)
                n_sym_blks = 2 * num_blks
                x_blks = x[:, p_len:].reshape(bsz * n_sym_blks, blk_sz, -1)
                t_blks = token_ids[:, p_len:].reshape(bsz * n_sym_blks, blk_sz)
                xb, ab, aeb = self.forward(x_blks, t_blks, True, 0)
                return (
                    torch.cat([x_p, xb.reshape(bsz, 2 * dual_stream_sym_len, -1)], dim=1),
                    torch.cat([a_p, ab.reshape(bsz, 2 * dual_stream_sym_len)], dim=1),
                    torch.cat([ae_p, aeb.reshape(bsz, 2 * dual_stream_sym_len)], dim=1),
                )

            x_pc, a_pc, ae_pc = self.forward(
                x[:, : p_len + dual_stream_sym_len], token_ids[:, : p_len + dual_stream_sym_len], True, 0
            )
            x_n_list, a_n_list, ae_n_list = [], [], []
            for b_i in range(num_blks):
                c_end = p_len + b_i * blk_sz
                n_start = p_len + dual_stream_sym_len + b_i * blk_sz
                n_end = n_start + blk_sz
                sub_x = torch.cat([x[:, :c_end], x[:, n_start:n_end]], dim=1)
                sub_t = torch.cat([token_ids[:, :c_end], token_ids[:, n_start:n_end]], dim=1)
                sx, sa, sae = self.forward(sub_x, sub_t, True, 0)
                x_n_list.append(sx[:, -blk_sz:])
                a_n_list.append(sa[:, -blk_sz:])
                ae_n_list.append(sae[:, -blk_sz:])
            return (
                torch.cat([x_pc] + x_n_list, dim=1),
                torch.cat([a_pc] + a_n_list, dim=1),
                torch.cat([ae_pc] + ae_n_list, dim=1),
            )

        is_valid = (token_ids != SymbolicVocab.MASK_ID) & (token_ids != SymbolicVocab.PAD_ID)
        non_pad = (token_ids != SymbolicVocab.PAD_ID).to(x.dtype)
        r_clean = is_valid.to(x.dtype).sum(dim=-1, keepdim=True) / non_pad.sum(dim=-1, keepdim=True).clamp(min=1.0)

        hashes = self._compute_three_way_hashes(token_ids)
        e_sum = torch.zeros(bsz, seq_len, self.engram_dim, device=x.device, dtype=x.dtype)
        for h, tbl in enumerate(self.tables):
            e_sum = e_sum + tbl(hashes[h]).to(x.dtype)
        e_raw = (e_sum / float(self.num_heads)) * is_valid.unsqueeze(-1).to(x.dtype)

        e_padded = F.pad(e_raw.transpose(1, 2), (2, 0))
        e_conv = e_raw + self.conv1d(e_padded).transpose(1, 2)
        e_conv = e_conv * non_pad.unsqueeze(-1)

        gate_in = torch.cat([x, e_conv], dim=-1)
        alpha_raw = torch.sigmoid(self.gate_proj(gate_in)).squeeze(-1)
        # Gate is zero on [MASK]/<PAD> so all-[MASK] Step-1 produces exact 0 alpha_eng
        alpha_eng = alpha_raw * is_valid.to(x.dtype)
        alpha_eng_eff = r_clean * alpha_eng.detach()

        # Neighbor context from e_conv can still assist masked slots when neighboring anchor tokens are clean!
        has_any_clean = (is_valid.any(dim=-1, keepdim=True)).to(x.dtype)
        conv_gate = torch.where(is_valid, alpha_eng, 0.5 * has_any_clean * alpha_raw) * non_pad
        delta_x = conv_gate.unsqueeze(-1) * self.out_proj(e_conv)
        return x + delta_x, alpha_eng, alpha_eng_eff

    def init_causal_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """Minimal O(1) causal state for exact 1-token AR StaticEngram parity."""
        return {
            "last_op": torch.zeros(batch_size, dtype=torch.long, device=device),
            "left_tok": torch.zeros(batch_size, dtype=torch.long, device=device),
            "prev_valid": torch.zeros(batch_size, dtype=torch.long, device=device),
            "e_hist": torch.zeros(batch_size, 2, self.engram_dim, dtype=dtype, device=device),
        }

    def forward_causal_step(
        self,
        x_step: torch.Tensor,
        tok_step: torch.Tensor,
        cache: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact single-token causal StaticEngram step (matches full-sequence AR forward to < 1e-6)."""
        bsz = x_step.shape[0]
        t_cur = tok_step[:, 0]  # [B]
        is_valid = (t_cur != SymbolicVocab.MASK_ID) & (t_cur != SymbolicVocab.PAD_ID)
        non_pad = (t_cur != SymbolicVocab.PAD_ID).to(x_step.dtype).unsqueeze(1)
        safe_tok = torch.where(is_valid, t_cur, torch.zeros_like(t_cur))

        is_op = is_valid & (t_cur >= 2) & (t_cur < self.cfg.op_vocab_size)
        op_carry = torch.where(is_op, safe_tok, cache["last_op"])
        left_tok = cache["left_tok"]
        skip_prev = cache["prev_valid"]

        e_sum = torch.zeros(bsz, self.engram_dim, device=x_step.device, dtype=x_step.dtype)
        for h, tbl in enumerate(self.tables):
            pa = self.PRIMES_A[h % len(self.PRIMES_A)]
            pb = self.PRIMES_B[h % len(self.PRIMES_B)]
            mode = h % 3
            if mode == 0:
                raw_h = (op_carry * pa + safe_tok * pb + (h + 1) * 97) % self.num_buckets
            elif mode == 1:
                raw_h = (left_tok * pa + safe_tok * pb + (h + 1) * 193) % self.num_buckets
            else:
                raw_h = (skip_prev * pa + safe_tok * pb + (h + 1) * 389) % self.num_buckets
            e_sum = e_sum + tbl(raw_h.clamp(0, self.num_buckets - 1)).to(x_step.dtype)
        e_raw = (e_sum / float(self.num_heads)) * is_valid.unsqueeze(-1).to(x_step.dtype)  # [B, E_dim]

        e_win = torch.cat([cache["e_hist"], e_raw.unsqueeze(1)], dim=1)  # [B, 3, E_dim]
        e_conv = (e_raw.unsqueeze(1) + self.conv1d(e_win.transpose(1, 2)).transpose(1, 2)) * non_pad.unsqueeze(-1)

        # Update causal cache in-place
        cache["last_op"] = op_carry
        cache["left_tok"] = safe_tok
        cache["prev_valid"] = torch.where(is_valid, safe_tok, skip_prev)
        cache["e_hist"] = e_win[:, 1:, :].detach()

        gate_in = torch.cat([x_step, e_conv], dim=-1)
        alpha_raw = torch.sigmoid(self.gate_proj(gate_in)).squeeze(-1)  # [B, 1]
        is_v_2d = is_valid.unsqueeze(1)
        alpha_eng = alpha_raw * is_v_2d.to(x_step.dtype)
        # In causal AR over clean tokens r_clean == 1.0 when valid
        alpha_eng_eff = alpha_eng.detach()
        delta_x = (alpha_eng * non_pad).unsqueeze(-1) * self.out_proj(e_conv)
        return x_step + delta_x, alpha_eng, alpha_eng_eff
