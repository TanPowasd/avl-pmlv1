"""Post-Thinking 4-Fold Complementary Pseudo-Log-Likelihood (PLL) + CVaR Energy Guard."""

from __future__ import annotations

import inspect
from typing import Callable, Tuple
import torch
import torch.nn.functional as F

from ..vocab.sym_vocab import SymbolicVocab


def build_4fold_complementary_masks(seq_len: int, device: torch.device) -> torch.Tensor:
    """Construct 4 deterministic complementary 25% masks [4, seq_len] covering 100% of positions once."""
    idx = torch.arange(seq_len, device=device)
    masks = torch.stack([(idx % 4) == r for r in range(4)], dim=0)
    return masks


class PostThinkingSurgeryGuard:
    """Paired 4-fold deterministic PLL + Top-2 CVaR + length-tax accept/reject gate (NetFix >= 0)."""

    def __init__(
        self,
        atomic_chunk_size: int = 16,
        guard_margin: float = 0.08,
        lambda_pll: float = 1.0,
        lambda_epi: float = 0.4,
        lambda_contra: float = 2.0,
        lambda_len: float = 0.015,
    ) -> None:
        self.chunk_size = atomic_chunk_size
        self.guard_margin = guard_margin
        self.lambda_pll = lambda_pll
        self.lambda_epi = lambda_epi
        self.lambda_contra = lambda_contra
        self.lambda_len = lambda_len

    @torch.no_grad()
    def compute_trace_energy(
        self,
        tokens: torch.Tensor,
        audit_forward_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute composite audit energy E(Y) [B] via a single 4x batched forward pass.
        audit_forward_fn(masked_tokens [4*B, S]) -> (logits [4*B, S, V], u_epistemic [4*B, S]).
        """
        bsz, seq_len = tokens.shape
        folds = build_4fold_complementary_masks(seq_len, tokens.device)  # [4, S]
        valid = tokens != SymbolicVocab.PAD_ID  # [B, S]

        rep_tokens = tokens.unsqueeze(0).expand(4, -1, -1).clone()  # [4, B, S]
        fold_mask = folds.unsqueeze(1).expand(4, bsz, seq_len) & valid.unsqueeze(0)
        rep_tokens[fold_mask] = SymbolicVocab.MASK_ID

        flat_in = rep_tokens.reshape(4 * bsz, seq_len)
        clean_rep = tokens.unsqueeze(0).expand(4, -1, -1).reshape(4 * bsz, seq_len)
        n_params = len(inspect.signature(audit_forward_fn).parameters)
        if n_params >= 2:
            logits, u_epi = audit_forward_fn(flat_in, clean_rep)  # type: ignore[call-arg]
        else:
            logits, u_epi = audit_forward_fn(flat_in)
        log_probs = F.log_softmax(logits.float(), dim=-1).view(4, bsz, seq_len, -1)
        u_epi_4 = u_epi.float().view(4, bsz, seq_len)

        target = tokens.unsqueeze(0).expand(4, -1, -1).unsqueeze(-1)
        tok_nll = -log_probs.gather(-1, target).squeeze(-1)  # [4, B, S]

        # Each position belongs to exactly one of the 4 complementary masks
        pll_per_tok = (tok_nll * fold_mask.float()).sum(dim=0)        # [B, S]
        epi_per_tok = u_epi_4.mean(dim=0) * valid.float()             # [B, S]

        # Aggregate per frame (chunk_size = 16)
        num_frames = max(1, (seq_len + self.chunk_size - 1) // self.chunk_size)
        frame_energies = []
        active_frame_counts = torch.zeros(bsz, device=tokens.device, dtype=torch.float32)

        for f_idx in range(num_frames):
            s = f_idx * self.chunk_size
            e = min(seq_len, s + self.chunk_size)
            v_f = valid[:, s:e].float()
            denom = v_f.sum(dim=-1).clamp(min=1.0)
            has_f = (v_f.sum(dim=-1) > 0).float()
            active_frame_counts = active_frame_counts + has_f

            pll_b = pll_per_tok[:, s:e].sum(dim=-1) / denom
            epi_b = epi_per_tok[:, s:e].sum(dim=-1) / denom
            e_b = (self.lambda_pll * pll_b + self.lambda_epi * epi_b) * has_f
            frame_energies.append(e_b)

        e_mat = torch.stack(frame_energies, dim=-1)  # [B, N_f]
        mean_e = e_mat.sum(dim=-1) / active_frame_counts.clamp(min=1.0)
        k_cvar = min(2, e_mat.shape[-1])
        top2_vals, _ = torch.topk(e_mat, k=k_cvar, dim=-1)
        cvar_top2 = top2_vals.mean(dim=-1)

        unresolved_contra = (tokens == SymbolicVocab.CONTRA_ID).float().sum(dim=-1)
        total_energy = (
            0.5 * mean_e
            + 0.5 * cvar_top2
            + self.lambda_contra * unresolved_contra
            + self.lambda_len * active_frame_counts
        )
        return total_energy

    def accept_or_reject(
        self,
        y_old: torch.Tensor,
        y_new: torch.Tensor,
        audit_forward_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        sandbox_passed: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (y_accepted, accepted_bool [B], e_old [B], e_new [B])."""
        e_old = self.compute_trace_energy(y_old, audit_forward_fn)
        e_new = self.compute_trace_energy(y_new, audit_forward_fn)
        energy_ok = e_new <= (e_old - self.guard_margin)
        accepted = sandbox_passed if sandbox_passed is not None else energy_ok
        y_final = torch.where(accepted.unsqueeze(-1), y_new, y_old)
        return y_final, accepted, e_old, e_new
