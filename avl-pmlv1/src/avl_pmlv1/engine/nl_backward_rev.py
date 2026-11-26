"""Native V_nl 8-Slot Block-dLLM Backward Revision Engine ('向前修改': Wet-Cement + 4-Fold PLL In-Place Repair).

Implements two physically grounded, zero-leakage backward-revision mechanisms over NLBlockDiffusionWrapper:
  1) Online 16-Slot Wet-Cement Seam Back-Refinement (湿水泥向后微扩散, C_wet = 2 * 8 = 16):
     - After Block b (8 slots) is denoised, if Block b-1 had low-margin uncertain tokens, re-mask those
       uncertain tokens in Block b-1 and run 1 bidirectional 16-slot pass over [Block b-1_noisy | Block b_clean]
       using past_kvs strictly before Block b-1.
     - Accept edits strictly under a confidence-margin gain guard and refresh the 16-slot KV cache.
  2) Post-Draft 4-Fold Complementary Bidirectional PLL Audit & Surgical Backward Overwrite (后思考原地回修):
     - Reuse Prompt KV cache to run 4 exact complementary 25% mask folds over the full draft Y_draft
       (every token masked in exactly 1 fold -> zero self-copy leakage!).
     - Compute exact Energy E(Y) = mean_NLL + 0.35 * worst_block_NLL, locate the worst 8-slot block b*,
       revise its disagreed/low-PLL slots bidirectionally (and roll out clean suffix blocks b*+1..Answer if
       b* precedes Answer), and accept strictly if E(Y_new) <= E(Y_old) - delta_accept (NetFix >= 0 guard!).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .nl_block_dllm import NLBlockDiffusionWrapper


def _sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def build_nl_wet_cement_mask(
    p_len: int,
    s_len: int,
    block_size: int = 8,
    parity: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact 16-slot Wet-Cement dual-stream mask for [Prompt(P) | Y_clean(S) | Y_wet_noisy(S)].
    - Prompt [0..P-1]: causal over 0..P-1.
    - Y_clean block k: sees Prompt + Y_clean blocks 0..k (8-slot block-causal, matching committed Prefix-KV).
    - Y_wet_noisy 16-slot seam window w = [w_s .. w_e) covering (Block b-1, Block b):
        sees Prompt + strictly earlier Y_clean[0..w_s) + Y_wet_noisy[w_s..w_e) bidirectionally!
        ZERO attention leakage from any block > b!
      (If parity == 1, leading orphan Block 0 [0..B) attends only to Prompt + itself.)
    """
    tot = p_len + 2 * s_len
    mask = torch.zeros((tot, tot), dtype=torch.bool, device=device)

    p_idx = torch.arange(p_len, device=device)
    mask[:p_len, :p_len] = p_idx.unsqueeze(1) >= p_idx.unsqueeze(0)

    # 1. Y_clean: standard 8-slot block-causal
    off = 0
    while off < s_len:
        end = min(off + block_size, s_len)
        c_s, c_e = p_len + off, p_len + end
        mask[c_s:c_e, :p_len] = True
        mask[c_s:c_e, p_len : p_len + end] = True
        off = end

    # 2. Y_wet_noisy: non-overlapping 2*B seam windows shifted by parity * B
    win_start = (parity % 2) * block_size
    if win_start > 0 and win_start <= s_len:
        # Leading orphan block [0..win_start): attends only to Prompt + itself
        n_s, n_e = p_len + s_len, p_len + s_len + min(win_start, s_len)
        mask[n_s:n_e, :p_len] = True
        mask[n_s:n_e, n_s:n_e] = True

    w_off = win_start
    while w_off < s_len:
        w_end = min(w_off + 2 * block_size, s_len)
        n_s, n_e = p_len + s_len + w_off, p_len + s_len + w_end
        mask[n_s:n_e, :p_len] = True
        if w_off > 0:
            mask[n_s:n_e, p_len : p_len + w_off] = True
        mask[n_s:n_e, n_s:n_e] = True
        w_off = w_end

    pos_p = torch.arange(0, p_len, device=device, dtype=torch.long)
    pos_y = torch.arange(p_len, p_len + s_len, device=device, dtype=torch.long)
    positions = torch.cat([pos_p, pos_y, pos_y], dim=0).unsqueeze(0)
    return mask, positions


def build_nl_full_bidir_mask(
    p_len: int,
    s_len: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact [Prompt(P, causal) | Y_masked(S, full bidirectional over Y + attends to Prompt)] mask."""
    tot = p_len + s_len
    mask = torch.zeros((tot, tot), dtype=torch.bool, device=device)
    p_idx = torch.arange(p_len, device=device)
    mask[:p_len, :p_len] = p_idx.unsqueeze(1) >= p_idx.unsqueeze(0)
    mask[p_len:, :p_len] = True
    mask[p_len:, p_len:] = True
    positions = torch.arange(0, tot, device=device, dtype=torch.long).unsqueeze(0)
    return mask, positions


class NLBackwardRevisionEngine:
    """Backward-revision trainer & pure-BF16 inference controller over an NLBlockDiffusionWrapper."""

    def __init__(self, wrapper: NLBlockDiffusionWrapper) -> None:
        self.wrapper = wrapper

    def forward_wet_cement_train(
        self,
        prompt_nl_ids: torch.Tensor,
        y_clean_ids: torch.Tensor,
        y_wet_noisy_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        block_size: int = 8,
        parity: int = 0,
        ans_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train 16-slot Wet-Cement backward seam refinement (Block b-1 denoised using clean Block b!)."""
        bsz, p_len = prompt_nl_ids.shape
        s_len = y_clean_ids.shape[1]
        device = prompt_nl_ids.device

        full_ids = torch.cat([prompt_nl_ids, y_clean_ids, y_wet_noisy_ids], dim=1)
        x = self.wrapper._embed_nl(full_ids)
        attn_mask, base_pos = build_nl_wet_cement_mask(p_len, s_len, block_size=block_size, parity=parity, device=device)
        positions = base_pos.expand(bsz, -1)

        h_all, _ = self.wrapper._run_backbone(x, positions=positions, attn_mask=attn_mask, past_kvs=None, return_kvs=False)
        h_noisy = self.wrapper.model.final_norm(h_all[:, p_len + s_len :, :])

        return self._masked_ce_loss(h_noisy, y_clean_ids, loss_mask, ans_mask)

    def forward_full_bidir_train(
        self,
        prompt_nl_ids: torch.Tensor,
        y_clean_ids: torch.Tensor,
        y_masked_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        ans_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train full-draft bidirectional 4-fold PLL & in-place error repair over [Prompt | Y_masked]."""
        bsz, p_len = prompt_nl_ids.shape
        s_len = y_clean_ids.shape[1]
        device = prompt_nl_ids.device

        full_ids = torch.cat([prompt_nl_ids, y_masked_ids], dim=1)
        x = self.wrapper._embed_nl(full_ids)
        attn_mask, base_pos = build_nl_full_bidir_mask(p_len, s_len, device=device)
        positions = base_pos.expand(bsz, -1)

        h_all, _ = self.wrapper._run_backbone(x, positions=positions, attn_mask=attn_mask, past_kvs=None, return_kvs=False)
        h_y = self.wrapper.model.final_norm(h_all[:, p_len:, :])

        return self._masked_ce_loss(h_y, y_clean_ids, loss_mask, ans_mask)

    def _masked_ce_loss(
        self,
        h_seq: torch.Tensor,
        y_clean_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        ans_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        flat_mask = loss_mask.reshape(-1)
        if not flat_mask.any():
            zero = h_seq.sum() * 0.0
            return zero, {"loss_total": 0.0, "tok_acc": 0.0, "loss_ans": 0.0, "ans_acc": 0.0}

        h_sel = h_seq.reshape(-1, h_seq.shape[-1])[flat_mask]
        tgt_sel = y_clean_ids.reshape(-1)[flat_mask]
        logits_sel = self.wrapper.model.nl_head(h_sel)
        loss = F.cross_entropy(logits_sel.float(), tgt_sel)

        with torch.no_grad():
            preds_sel = logits_sel.argmax(dim=-1)
            tok_acc = float((preds_sel == tgt_sel).float().mean().item()) * 100.0
            loss_ans_val = 0.0
            ans_acc_val = 0.0
            if ans_mask is not None:
                a_sel_mask = (ans_mask & loss_mask).reshape(-1)[flat_mask]
                if a_sel_mask.any():
                    loss_ans_val = float(F.cross_entropy(logits_sel[a_sel_mask].float(), tgt_sel[a_sel_mask]).item())
                    ans_acc_val = float((preds_sel[a_sel_mask] == tgt_sel[a_sel_mask]).float().mean().item()) * 100.0

        return loss, {
            "loss_total": float(loss.item()),
            "tok_acc": round(tok_acc, 2),
            "loss_ans": round(loss_ans_val, 4),
            "ans_acc": round(ans_acc_val, 2),
        }

    @torch.no_grad()
    def score_draft_4fold_pll(
        self,
        prompt_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        draft_ids: torch.Tensor,
        p_len: int,
        block_size: int = 8,
    ) -> Dict[str, Any]:
        """Exact 4-fold complementary 25% masked Pseudo-Log-Likelihood (zero self-copy leakage!).
        Reuses prompt_kvs so Prompt is NEVER recomputed.
        Returns Energy = mean_NLL + 0.35 * max_block_NLL, per-token NLL, per-token argmax predictions & margins.
        """
        device = draft_ids.device
        s_len = int(draft_ids.shape[1])
        mask_id = self.wrapper.mask_token_id

        tok_nll = torch.zeros(s_len, dtype=torch.float32, device=device)
        tok_preds = torch.zeros(s_len, dtype=torch.long, device=device)
        tok_margins = torch.zeros(s_len, dtype=torch.float32, device=device)
        covered = torch.zeros(s_len, dtype=torch.bool, device=device)

        for fold in range(4):
            fold_pos = torch.arange(fold, s_len, 4, device=device, dtype=torch.long)
            if fold_pos.numel() == 0:
                continue
            y_fold = draft_ids.clone()
            y_fold[0, fold_pos] = mask_id
            logits, _ = self.wrapper.forward_block_step(y_fold, past_kvs=prompt_kvs, start_pos=p_len, return_logits=True)
            assert logits is not None
            lg_sel = logits[0, fold_pos, :].float()
            lg_sel[:, mask_id] = float("-inf")
            log_p = F.log_softmax(lg_sel, dim=-1)
            tgt = draft_ids[0, fold_pos]
            nll_f = -log_p.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)

            top2_v, top2_i = torch.topk(lg_sel, k=2, dim=-1)
            tok_nll[fold_pos] = nll_f
            tok_preds[fold_pos] = top2_i[:, 0]
            tok_margins[fold_pos] = top2_v[:, 0] - top2_v[:, 1]
            covered[fold_pos] = True

        n_blocks = max(1, (s_len + block_size - 1) // block_size)
        block_nlls: List[float] = []
        for b in range(n_blocks):
            b_s = b * block_size
            b_e = min(b_s + block_size, s_len)
            block_nlls.append(float(tok_nll[b_s:b_e].mean().item()))

        mean_nll = float(tok_nll.mean().item())
        top2_blk = sorted(block_nlls, reverse=True)[: min(2, len(block_nlls))]
        cvar_nll = sum(top2_blk) / max(1, len(top2_blk))
        energy = mean_nll + 0.35 * cvar_nll

        # Pick worst block that has at least 1 disagreed token (argmax != draft)
        worst_blk = -1
        worst_score = -1.0
        for b in range(n_blocks):
            b_s = b * block_size
            b_e = min(b_s + block_size, s_len)
            has_disagree = bool((tok_preds[b_s:b_e] != draft_ids[0, b_s:b_e]).any().item())
            if has_disagree and block_nlls[b] > worst_score:
                worst_score = block_nlls[b]
                worst_blk = b

        return {
            "energy": round(energy, 5),
            "mean_nll": round(mean_nll, 5),
            "cvar_nll": round(cvar_nll, 5),
            "tok_nll": tok_nll,
            "tok_preds": tok_preds,
            "tok_margins": tok_margins,
            "block_nlls": block_nlls,
            "worst_block_idx": worst_blk,
            "covered_all": bool(covered.all().item()),
        }

    @torch.no_grad()
    def _denoise_single_block(
        self,
        cur_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        cur_pos: int,
        cur_blk_len: int,
        denoise_beats: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denoise one 8-slot block and return (blk_toks [1, B], final_margins [B])."""
        mask_id = self.wrapper.mask_token_id
        blk_toks = torch.full((1, cur_blk_len), mask_id, dtype=torch.long, device=device)
        blk_margins = torch.zeros(cur_blk_len, dtype=torch.float32, device=device)

        for beat in range(denoise_beats):
            logits_b, _ = self.wrapper.forward_block_step(blk_toks, past_kvs=cur_kvs, start_pos=cur_pos, return_logits=True)
            assert logits_b is not None
            lg = logits_b[0].float()
            lg[:, mask_id] = float("-inf")
            top2_vals, top2_idx = torch.topk(lg, k=2, dim=-1)
            preds = top2_idx[:, 0]
            margins = top2_vals[:, 0] - top2_vals[:, 1]

            masked_pos = (blk_toks[0] == mask_id).nonzero(as_tuple=False).view(-1)
            n_rem = int(masked_pos.numel())
            if n_rem == 0:
                break
            if beat == denoise_beats - 1:
                blk_toks[0, masked_pos] = preds[masked_pos]
                blk_margins[masked_pos] = margins[masked_pos]
            else:
                beats_left = denoise_beats - beat
                k_unmask = max(1, (n_rem + beats_left - 1) // beats_left)
                _, pick_rel = torch.topk(margins[masked_pos], k=min(k_unmask, n_rem))
                chosen = masked_pos[pick_rel]
                blk_toks[0, chosen] = preds[chosen]
                blk_margins[chosen] = margins[chosen]

        return blk_toks, blk_margins

    @staticmethod
    def _decode_clean(gen_ids: List[int], tokenizer: Any) -> str:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        cut = gen_ids
        if eos_id is not None and eos_id in gen_ids:
            cut = gen_ids[: gen_ids.index(eos_id)]
        text = tokenizer.decode(cut, skip_special_tokens=True).strip()
        for stop_seq in ("\nProblem (", "\nQuestion:", "\n\nProblem"):
            if stop_seq in text:
                text = text.split(stop_seq, 1)[0].strip()
        return text

    @staticmethod
    def _should_stop(gen_ids: List[int], tokenizer: Any, hit_eos: bool) -> bool:
        if hit_eos:
            return True
        cur_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if any(s in cur_txt for s in ("\nProblem (", "\nQuestion:", "\n\nProblem")):
            return True
        if re.search(r"(?:Answer|Final Answer|####)\s*:?\s*[^\n]+\n", cur_txt, re.IGNORECASE) is not None:
            return True
        return False

    @torch.no_grad()
    def _rollout_from_prefix(
        self,
        prefix_ids: List[int],
        cur_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        p_len: int,
        tokenizer: Any,
        max_new_tokens: int,
        block_size: int,
        denoise_beats: int,
        device: torch.device,
    ) -> List[int]:
        """Continue 8-slot Prefix-KV block diffusion from an already-committed prefix."""
        eos_id = getattr(tokenizer, "eos_token_id", None)
        gen_ids = list(prefix_ids)
        cur_pos = p_len + len(gen_ids)
        max_blocks = max(0, (max_new_tokens - len(gen_ids) + block_size - 1) // block_size)

        for _ in range(max_blocks):
            cur_blk_len = min(block_size, max_new_tokens - len(gen_ids))
            if cur_blk_len <= 0:
                break
            blk_toks, _ = self._denoise_single_block(cur_kvs, cur_pos, cur_blk_len, denoise_beats, device)
            blk_list = blk_toks[0].tolist()
            hit_eos = False
            if eos_id is not None and eos_id in blk_list:
                gen_ids.extend(blk_list[: blk_list.index(eos_id)])
                hit_eos = True
            else:
                gen_ids.extend(blk_list)

            if self._should_stop(gen_ids, tokenizer, hit_eos) or len(gen_ids) >= max_new_tokens:
                break
            _, cur_kvs = self.wrapper.forward_block_step(blk_toks, past_kvs=cur_kvs, start_pos=cur_pos, return_logits=False)
            cur_pos += cur_blk_len

        return gen_ids

    @torch.no_grad()
    def generate_with_backward_revision(
        self,
        prompt_nl_ids: torch.Tensor,
        tokenizer: Any,
        max_new_tokens: int = 256,
        block_size: int = 8,
        denoise_beats: int = 3,
        enable_wet_cement: bool = True,
        wet_margin_thresh: float = 2.5,
        wet_max_remask: int = 2,
        wet_accept_gain: float = 0.5,
        enable_post_revision: bool = True,
        post_energy_delta: float = 0.05,
        post_min_block_nll: float = 0.85,
        post_max_rounds: int = 1,
    ) -> Dict[str, Any]:
        """Run native V_nl 8-slot Block-dLLM with optional Online 16-Slot Wet-Cement & Post-Draft 4-Fold PLL Revision."""
        device = prompt_nl_ids.device
        eos_id = getattr(tokenizer, "eos_token_id", None)
        mask_id = self.wrapper.mask_token_id
        p_len = int(prompt_nl_ids.shape[1])

        t0 = _sync_time(device)
        _, prompt_kvs = self.wrapper.forward_block_step(prompt_nl_ids, past_kvs=None, start_pos=0, return_logits=False)
        t1 = _sync_time(device)
        prefill_ms = (t1 - t0) * 1000.0

        cur_kvs = prompt_kvs
        kvs_before_prev: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        prev_blk_toks: Optional[torch.Tensor] = None
        prev_blk_margins: Optional[torch.Tensor] = None
        prev_blk_start_idx: int = 0

        gen_ids: List[int] = []
        max_blocks = max(1, (max_new_tokens + block_size - 1) // block_size)
        cur_pos = p_len

        wet_triggers = 0
        wet_accepted_edits = 0
        wet_ms = 0.0

        for _ in range(max_blocks):
            cur_blk_len = min(block_size, max_new_tokens - len(gen_ids))
            if cur_blk_len <= 0:
                break

            blk_toks, blk_margins = self._denoise_single_block(cur_kvs, cur_pos, cur_blk_len, denoise_beats, device)
            cur_blk_start_idx = len(gen_ids)
            blk_list = blk_toks[0].tolist()
            hit_eos = False
            if eos_id is not None and eos_id in blk_list:
                gen_ids.extend(blk_list[: blk_list.index(eos_id)])
                hit_eos = True
            else:
                gen_ids.extend(blk_list)

            # --- Online 16-Slot Wet-Cement Backward Seam Refinement on Block b-1 using clean Block b! ---
            revised_prev = False
            if (
                enable_wet_cement
                and prev_blk_toks is not None
                and prev_blk_margins is not None
                and kvs_before_prev is not None
            ):
                low_idx = (prev_blk_margins < wet_margin_thresh).nonzero(as_tuple=False).view(-1)
                if low_idx.numel() > 0:
                    wet_triggers += 1
                    tw0 = _sync_time(device)
                    # Pick up to wet_max_remask lowest-margin slots in Block b-1
                    k_w = min(wet_max_remask, int(low_idx.numel()))
                    _, worst_rel = torch.topk(-prev_blk_margins[low_idx], k=k_w)
                    remask_slots = low_idx[worst_rel]

                    prev_len = int(prev_blk_toks.shape[1])
                    prev_noisy = prev_blk_toks.clone()
                    prev_noisy[0, remask_slots] = mask_id
                    wet_16 = torch.cat([prev_noisy, blk_toks], dim=1)
                    wet_start_pos = p_len + prev_blk_start_idx

                    lg_16, _ = self.wrapper.forward_block_step(
                        wet_16, past_kvs=kvs_before_prev, start_pos=wet_start_pos, return_logits=True
                    )
                    assert lg_16 is not None
                    lg_prev = lg_16[0, :prev_len, :].float()
                    lg_prev[:, mask_id] = float("-inf")
                    top2_v, top2_i = torch.topk(lg_prev, k=2, dim=-1)
                    new_preds = top2_i[:, 0]
                    new_margins = top2_v[:, 0] - top2_v[:, 1]

                    new_prev_toks = prev_blk_toks.clone()
                    for s_i in remask_slots.tolist():
                        old_t = int(prev_blk_toks[0, s_i].item())
                        new_t = int(new_preds[s_i].item())
                        if new_t != old_t and float(new_margins[s_i].item()) > float(prev_blk_margins[s_i].item()) + wet_accept_gain:
                            # Never inject EOS mid-stream before current block!
                            if eos_id is not None and new_t == eos_id:
                                continue
                            new_prev_toks[0, s_i] = new_t
                            gen_ids[prev_blk_start_idx + s_i] = new_t
                            prev_blk_margins[s_i] = new_margins[s_i]
                            wet_accepted_edits += 1
                            revised_prev = True
                    if revised_prev:
                        prev_blk_toks = new_prev_toks
                    tw1 = _sync_time(device)
                    wet_ms += (tw1 - tw0) * 1000.0

            stop_now = self._should_stop(gen_ids, tokenizer, hit_eos) or len(gen_ids) >= max_new_tokens
            if stop_now:
                break

            if revised_prev and prev_blk_toks is not None and kvs_before_prev is not None:
                # Re-commit both revised Block b-1 and clean Block b cleanly in 1 single 16-slot commit pass!
                tw2 = _sync_time(device)
                wet_clean_16 = torch.cat([prev_blk_toks, blk_toks], dim=1)
                _, cur_kvs = self.wrapper.forward_block_step(
                    wet_clean_16, past_kvs=kvs_before_prev, start_pos=p_len + prev_blk_start_idx, return_logits=False
                )
                # Build exact kvs_before_curr (slice off the last cur_blk_len KV entries!)
                kvs_before_prev = [(k[:, :, :-cur_blk_len, :], v[:, :, :-cur_blk_len, :]) for (k, v) in cur_kvs]
                tw3 = _sync_time(device)
                wet_ms += (tw3 - tw2) * 1000.0
            else:
                kvs_before_prev = cur_kvs
                _, cur_kvs = self.wrapper.forward_block_step(blk_toks, past_kvs=cur_kvs, start_pos=cur_pos, return_logits=False)

            prev_blk_toks = blk_toks
            prev_blk_margins = blk_margins
            prev_blk_start_idx = cur_blk_start_idx
            cur_pos += cur_blk_len

        t_draft_end = _sync_time(device)
        draft_text = self._decode_clean(gen_ids, tokenizer)
        final_ids = list(gen_ids)
        final_text = draft_text

        post_ms = 0.0
        post_attempted = False
        post_accepted = False
        post_edited_block = -1
        draft_energy = 0.0
        final_energy = 0.0

        # --- Post-Draft 4-Fold Bidirectional PLL Audit & Surgical In-Place Backward Revision ---
        if enable_post_revision and len(final_ids) >= block_size:
            tp0 = _sync_time(device)
            for _round in range(post_max_rounds):
                draft_t = torch.tensor([final_ids], dtype=torch.long, device=device)
                audit = self.score_draft_4fold_pll(prompt_kvs, draft_t, p_len=p_len, block_size=block_size)
                draft_energy = audit["energy"]
                final_energy = draft_energy
                b_star = int(audit["worst_block_idx"])
                if b_star < 0 or audit["block_nlls"][b_star] < post_min_block_nll:
                    break

                post_attempted = True
                post_edited_block = b_star
                b_s = b_star * block_size
                b_e = min(b_s + block_size, len(final_ids))

                # Step 1: Bidirectional surgical overwrite of disagreed slots inside worst block b*
                cand_block_t = draft_t.clone()
                disagree_mask = audit["tok_preds"][b_s:b_e] != draft_t[0, b_s:b_e]
                bad_rel = disagree_mask.nonzero(as_tuple=False).view(-1)
                if bad_rel.numel() == 0:
                    break
                # Re-mask the worst disagreed slots (up to 4 slots inside b*) and run 1 bidirectional full-draft repair pass!
                k_rep = min(4, int(bad_rel.numel()))
                _, top_bad = torch.topk(audit["tok_nll"][b_s:b_e][bad_rel], k=k_rep)
                edit_abs = b_s + bad_rel[top_bad]

                y_rep_in = draft_t.clone()
                y_rep_in[0, edit_abs] = mask_id
                lg_rep, _ = self.wrapper.forward_block_step(y_rep_in, past_kvs=prompt_kvs, start_pos=p_len, return_logits=True)
                assert lg_rep is not None
                lg_sel = lg_rep[0, edit_abs, :].float()
                lg_sel[:, mask_id] = float("-inf")
                n_total_blocks = (len(final_ids) + block_size - 1) // block_size
                if b_star < n_total_blocks - 1 and eos_id is not None:
                    lg_sel[:, eos_id] = float("-inf")
                new_toks = lg_sel.argmax(dim=-1)
                cand_block_t[0, edit_abs] = new_toks

                # Step 2: If b* is NOT the final block, roll out downstream blocks b*+1..Answer causally
                # so downstream steps & final Answer stay 100% consistent with the backward edit!
                if b_star < n_total_blocks - 1:
                    prefix_revised = cand_block_t[0, :b_e].tolist()
                    # Commit prefix blocks 0..b* in 8-slot chunks to match exact Prefix-KV geometry
                    kvs_re = prompt_kvs
                    off_r = 0
                    while off_r < b_e:
                        end_r = min(off_r + block_size, b_e)
                        _, kvs_re = self.wrapper.forward_block_step(
                            cand_block_t[:, off_r:end_r], past_kvs=kvs_re, start_pos=p_len + off_r, return_logits=False
                        )
                        off_r = end_r
                    cand_ids = self._rollout_from_prefix(
                        prefix_revised,
                        cur_kvs=kvs_re,
                        p_len=p_len,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        block_size=block_size,
                        denoise_beats=denoise_beats,
                        device=device,
                    )
                else:
                    cand_ids = cand_block_t[0].tolist()

                if not cand_ids or cand_ids == final_ids:
                    break

                # Step 3: Strict NetFix >= 0 Energy Acceptance Gate via identical 4-fold PLL!
                cand_t = torch.tensor([cand_ids], dtype=torch.long, device=device)
                cand_audit = self.score_draft_4fold_pll(prompt_kvs, cand_t, p_len=p_len, block_size=block_size)
                if cand_audit["energy"] <= draft_energy - post_energy_delta:
                    final_ids = cand_ids
                    final_text = self._decode_clean(final_ids, tokenizer)
                    final_energy = cand_audit["energy"]
                    post_accepted = True
                else:
                    break

            tp1 = _sync_time(device)
            post_ms = (tp1 - tp0) * 1000.0

        t_end = _sync_time(device)
        total_e2e_ms = (t_end - t0) * 1000.0
        think_ms = max(0.0, (t_draft_end - t1) * 1000.0 - wet_ms)

        return {
            "draft_text": draft_text,
            "final_text": final_text,
            "n_tokens": len(final_ids),
            "prefill_ms": round(prefill_ms, 2),
            "think_ms": round(think_ms, 2),
            "wet_ms": round(wet_ms, 2),
            "post_ms": round(post_ms, 2),
            "repair_ms": round(wet_ms + post_ms, 2),
            "total_e2e_ms": round(total_e2e_ms, 2),
            "wet_triggers": wet_triggers,
            "wet_accepted_edits": wet_accepted_edits,
            "post_attempted": post_attempted,
            "post_accepted": post_accepted,
            "post_edited_block": post_edited_block,
            "draft_energy": draft_energy,
            "final_energy": final_energy,
        }
