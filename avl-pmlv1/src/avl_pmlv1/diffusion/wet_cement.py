"""16-Slot Wet-Cement Backward Micro-Diffusion, Block Masks, & 3-Beat Hierarchical Unmasking."""

from __future__ import annotations

import math
from typing import List, Tuple
import torch
import torch.nn.functional as F

from ..vocab.sym_vocab import SymbolicVocab


def select_wet_cement_remask_slots(
    tail_tokens: torch.Tensor,
    tail_margins: torch.Tensor,
    d_bwd: float,
    threshold: float = 0.55,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remask 15%-30% lowest-confidence slots in previous block's 16-slot wet-cement tail when d_bwd > 0.55.
    Returns (remasked_tail_tokens [B, 16], remask_bool [B, 16]).
    """
    bsz, tail_len = tail_tokens.shape
    if d_bwd <= threshold:
        return tail_tokens.clone(), torch.zeros_like(tail_tokens, dtype=torch.bool)

    p_remask = 0.15 + 0.15 * min(1.0, max(0.0, (d_bwd - threshold) / (1.0 - threshold)))
    k_remask = max(1, min(tail_len, int(round(p_remask * tail_len))))

    # Protect <PAD> from remasking
    eligible = tail_tokens != SymbolicVocab.PAD_ID
    adj_margins = torch.where(eligible, tail_margins, torch.full_like(tail_margins, 1e9))
    _, low_idx = torch.topk(adj_margins, k=k_remask, dim=-1, largest=False)

    remask_bool = torch.zeros(bsz, tail_len, device=tail_tokens.device, dtype=torch.bool)
    remask_bool.scatter_(-1, low_idx, True)
    remask_bool = remask_bool & eligible

    out_tokens = torch.where(remask_bool, torch.full_like(tail_tokens, SymbolicVocab.MASK_ID), tail_tokens)
    return out_tokens, remask_bool


def build_block_diffusion_attn_mask(
    block_lengths: List[int],
    wet_seam_flags: List[bool] | None = None,
    device: torch.device | None = None,
    wet_slots: int = 16,
) -> torch.Tensor:
    """Build boolean SDPA attention mask [1, 1, S, S] (True = allowed to attend).
    - Block-causal across blocks; bidirectional within each block;
    - When wet_seam_flags[b] is True for block b >= 1, block b and the last 16 slots of block b-1
      form a bidirectional local wet-cement seam window [16 + B_b] while keeping < (end_{b-1} - 16) frozen!
    """
    total_len = sum(block_lengths)
    dev = device if device is not None else torch.device("cpu")
    mask = torch.zeros(total_len, total_len, device=dev, dtype=torch.bool)

    starts: List[int] = []
    cur = 0
    for blen in block_lengths:
        starts.append(cur)
        cur += blen

    for b, blen in enumerate(block_lengths):
        s_b = starts[b]
        e_b = s_b + blen
        # Causal access to all preceding blocks + full bidirectional within block b
        mask[s_b:e_b, :e_b] = True

        if wet_seam_flags is not None and b > 0 and wet_seam_flags[b]:
            tail_start = max(starts[b - 1], s_b - wet_slots)
            mask[tail_start:e_b, tail_start:e_b] = True

    return mask.unsqueeze(0).unsqueeze(0)


def compute_top1_top2_margin(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pred_ids [B, S], margin [B, S] = P_(1) - P_(2))."""
    probs = F.softmax(logits.float(), dim=-1)
    top2_p, top2_idx = torch.topk(probs, k=2, dim=-1)
    margin = top2_p[..., 0] - top2_p[..., 1]
    return top2_idx[..., 0], margin


def hierarchical_three_beat_unmask(
    cur_tokens: torch.Tensor,
    logits: torch.Tensor,
    beat_idx: int,
    total_beats: int,
    d_hier: float = 0.6,
    gumbel_temp: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Three-beat postfix frame unmasking:
      Beat 0: lock header slots (first 2 masked positions);
      Intermediate beats: unmask highest-Margin body slots;
      Final beat: unmask all remaining slots (including postfix ⟨BIND⟩ ⟨REG_j⟩).
    """
    masked_logits = logits.clone()
    masked_logits[..., SymbolicVocab.MASK_ID] = float("-inf")
    masked_logits[..., SymbolicVocab.PAD_ID] = float("-inf")

    pred_ids, margins = compute_top1_top2_margin(masked_logits)
    is_mask = cur_tokens == SymbolicVocab.MASK_ID
    if beat_idx >= total_beats - 1:
        return torch.where(is_mask, pred_ids, cur_tokens), margins

    bsz, seq_len = cur_tokens.shape
    scores = margins.clone()
    if gumbel_temp > 0.0:
        u = torch.rand_like(scores).clamp(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        scores = scores + gumbel_temp * gumbel

    # Structural tier schedule for single or multi-frame packed 16-slot blocks:
    # Beat 0: lock high-confidence syntax/header skeleton while holding value & write-back slots;
    # Beat 1: lock first half (Sub-frame 1 value & BIND, pos < seq_len // 2) + remaining syntax so Sub-frame 2 sees Sub-frame 1!
    # Final beat: unmask Sub-frame 2 / block-tail value & postfix write-back cleanly!
    # Support BOTH 8-slot 2-beat atomic micro-blocks (Beat 0: [0..3] op+args, Beat 1: [4..7] val+BIND+REG)
    # AND 16-slot 3-beat frames (Beat 0: header/slot 3, Beat 1: [0..11], Beat 2: [12..15])!
    pos = torch.arange(seq_len, device=cur_tokens.device).unsqueeze(0).expand(bsz, -1)
    half = seq_len // 2
    if total_beats == 2 and seq_len == 8:
        # 8-slot atomic micro-block (2-beat): Beat 0 strictly opens [0..3] (OP, dyn_op, arg1, k_i), holding [4..7] for Beat 1!
        scores = scores + (pos < 4).float() * 4.0 - (pos >= 4).float() * 4.0
        scores = torch.where(is_mask, scores, torch.full_like(scores, -1e9))
        rem_counts = is_mask.sum(dim=-1, keepdim=True)
        k_open = torch.minimum(rem_counts, torch.full_like(rem_counts, 4))
        order = torch.argsort(scores, dim=-1, descending=True)
        ranks = torch.argsort(order, dim=-1)
        return torch.where(is_mask & (ranks < k_open), pred_ids, cur_tokens), margins

    if total_beats == 3 and seq_len == 8:
        # 8-slot atomic micro-block (3-beat 2-hop dereferencing):
        # Beat 0: opens [0, 1, 2] ([OP, <IN_q>, q]);
        # Beat 1: opens [3] (mapped = S^RO[q] conditioned on clean q!);
        # Beat 2: opens [4..7] ([acc, BIND, REG, flag] conditioned on clean mapped!).
        if beat_idx == 0:
            scores = scores + (pos < 3).float() * 6.0 - (pos >= 3).float() * 6.0
            k_target = 3
        else:
            scores = scores + (pos == 3).float() * 6.0 - (pos >= 4).float() * 6.0
            k_target = 1
        scores = torch.where(is_mask, scores, torch.full_like(scores, -1e9))
        rem_counts = is_mask.sum(dim=-1, keepdim=True)
        k_open = torch.minimum(rem_counts, torch.full_like(rem_counts, k_target))
        order = torch.argsort(scores, dim=-1, descending=True)
        ranks = torch.argsort(order, dim=-1)
        return torch.where(is_mask & (ranks < k_open), pred_ids, cur_tokens), margins

    if beat_idx == 0:
        # Hold Sub-frame 1 value (slot 4) & Sub-frame 2 operand/tail (slots >= 11) so header & slot 3 lock first
        is_val_slot = (pos == 4) | (pos >= 11)
        scores = scores + (pos <= 3).float() * 1.5 - is_val_slot.float() * 3.0
    else:
        # Beat 1 completes all slots [0..11] (including Sub-frame 1 value & BIND) while holding [12..15] for Beat 2
        is_late_tail = pos >= 12
        scores = scores + (pos < half).float() * 3.0 + float(d_hier) * 0.2 - is_late_tail.float() * 3.0

    scores = torch.where(is_mask, scores, torch.full_like(scores, -1e9))
    rem_counts = is_mask.sum(dim=-1, keepdim=True)  # [B, 1]
    rem_beats = max(1, total_beats - beat_idx)

    # Fully vectorized sync-free rank unmasking (zero .item() CPU stalls, CUDAGraph-friendly!)
    k_base = (rem_counts + rem_beats - 1) // rem_beats
    if beat_idx == 0:
        k_open = torch.minimum(rem_counts, torch.clamp(k_base, min=6))
    elif beat_idx == 1:
        k_open = torch.minimum(rem_counts, torch.clamp(rem_counts - 4, min=1))
    else:
        k_open = torch.minimum(rem_counts, torch.clamp(k_base, min=1))

    order = torch.argsort(scores, dim=-1, descending=True)
    ranks = torch.argsort(order, dim=-1)  # 0 = highest score in row
    do_open = is_mask & (ranks < k_open)
    new_tokens = torch.where(do_open, pred_ids, cur_tokens)
    return new_tokens, margins


def build_dual_stream_bd3_mask(
    prompt_len: int,
    num_blocks: int,
    block_size: int = 16,
    enable_wet_seam: bool = False,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact BD3-LM Dual-Stream [Prompt, Y_clean, Y_noisy] attention mask & shared RoPE positions.
    Total length = P + S + S (where S = num_blocks * block_size).
    - Prompt [0 : P] attends to Prompt;
    - Y_clean block b attends to Prompt + Y_clean[0 .. b];
    - Y_noisy block b attends to Prompt + Y_clean[0 .. b-1] + Y_noisy[b]
      (plus last 16 slots of Y_noisy[b-1] when enable_wet_seam is True!).
    Returns (attn_mask [1, 1, P+2S, P+2S], rope_positions [1, P+2S]).
    """
    dev = device if device is not None else torch.device("cpu")
    s_len = num_blocks * block_size
    tot = prompt_len + 2 * s_len
    mask = torch.zeros(tot, tot, device=dev, dtype=torch.bool)

    # 1. Prompt attends to Prompt
    mask[:prompt_len, :prompt_len] = True

    # 2. Y_clean [P : P + S] block-causal
    c_base = prompt_len
    n_base = prompt_len + s_len
    for b in range(num_blocks):
        cs = c_base + b * block_size
        ce = cs + block_size
        mask[cs:ce, :prompt_len] = True
        mask[cs:ce, c_base:ce] = True

        # 3. Y_noisy [P + S : P + 2S] sees Prompt + clean history (< b) + own noisy block b
        ns = n_base + b * block_size
        ne = ns + block_size
        mask[ns:ne, :prompt_len] = True
        if b > 0:
            mask[ns:ne, c_base:cs] = True
        mask[ns:ne, ns:ne] = True
        if enable_wet_seam and b > 0:
            prev_tail_s = ns - min(16, block_size)
            # Allow current noisy block b to attend to noisy tail of b-1 WITHOUT leaking future block b into b-1!
            mask[ns:ne, prev_tail_s:ne] = True

    pos_p = torch.arange(0, prompt_len, device=dev)
    pos_s = torch.arange(prompt_len, prompt_len + s_len, device=dev)
    rope_pos = torch.cat([pos_p, pos_s, pos_s], dim=0).unsqueeze(0)
    return mask.unsqueeze(0).unsqueeze(0), rope_pos
