"""Native V_nl 8-Slot BD3-LM Dual-Stream Block Diffusion Engine over Pretrained Qwen3-4B-Base.

Preserves 100% of Qwen3-4B-Base's pretrained 151,936-token natural-language vocabulary (zero cold-start
V_sym tax!) while applying exact BD3-LM dual-stream 8-slot block diffusion + Prefix-KV parallel decoding:
  - Training layout: [Prompt_nl (causal) | Y_clean (block-causal 8-slot) | Y_noisy (attends to Prompt + Y_clean[<b] + Y_noisy[b])]
  - Inference layout: Prefill Prompt ONCE -> for each 8-slot block: 3-beat confidence-margin parallel unmasking
    against Prefix-KV -> 1 clean block commit pass appending 8 clean KV slots -> exact <1e-5 TF32-off parity!
  - Dedicated mask token NL_MASK_ID=151662 (<|fim_pad|>) uses a tiny trainable 1-vector delta (2,560 params)
    initialized to <|fim_middle|> (151660) so frozen 389M nl_embed stays untouched and intact.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import AVLPMLModel

NL_MASK_ID = 151662  # Qwen3 <|fim_pad|>: unused in math/logic text, strictly inside 151,936 nl_embed
NL_FIM_MID_ID = 151660  # Qwen3 <|fim_middle|>: pretrained infilling representation for Step-0 init


def _sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def build_nl_bd3_mask(
    p_len: int,
    s_len: int,
    block_size: int = 8,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact BD3-LM dual-stream attention mask & RoPE positions for [Prompt(P) | Y_clean(S) | Y_noisy(S)].
    - Prompt [0..P-1]: causal over 0..P-1; RoPE 0..P-1.
    - Y_clean block b: sees Prompt + Y_clean blocks 0..b (bidirectional inside block b); RoPE P+b*B .. P+(b+1)*B-1.
    - Y_noisy block b: sees Prompt + strictly earlier Y_clean blocks 0..b-1 + Y_noisy block b (bidirectional);
      RoPE P+b*B .. P+(b+1)*B-1 (matches Prefix-KV inference bit-for-bit!).
    """
    tot = p_len + 2 * s_len
    mask = torch.zeros((tot, tot), dtype=torch.bool, device=device)

    # 1. Causal Prompt
    p_idx = torch.arange(p_len, device=device)
    mask[:p_len, :p_len] = p_idx.unsqueeze(1) >= p_idx.unsqueeze(0)

    # 2. Y_clean & Y_noisy blocks
    off = 0
    while off < s_len:
        end = min(off + block_size, s_len)
        c_s, c_e = p_len + off, p_len + end
        n_s, n_e = p_len + s_len + off, p_len + s_len + end

        # Y_clean block b sees Prompt + Y_clean[0..end]
        mask[c_s:c_e, :p_len] = True
        mask[c_s:c_e, p_len : p_len + end] = True

        # Y_noisy block b sees Prompt + Y_clean[0..off] (strictly < b) + Y_noisy[off..end] (itself)
        mask[n_s:n_e, :p_len] = True
        if off > 0:
            mask[n_s:n_e, p_len : p_len + off] = True
        mask[n_s:n_e, n_s:n_e] = True

        off = end

    pos_p = torch.arange(0, p_len, device=device, dtype=torch.long)
    pos_y = torch.arange(p_len, p_len + s_len, device=device, dtype=torch.long)
    positions = torch.cat([pos_p, pos_y, pos_y], dim=0).unsqueeze(0)  # [1, tot]
    return mask, positions


class NLBlockDiffusionWrapper(nn.Module):
    """Thin wrapper over AVLPMLModel (dense FFN, PML disabled) adding 8-slot native V_nl BD3-LM diffusion."""

    def __init__(
        self,
        model: AVLPMLModel,
        mask_token_id: int = NL_MASK_ID,
        fim_init_id: int = NL_FIM_MID_ID,
    ) -> None:
        super().__init__()
        self.model = model
        self.mask_token_id = int(mask_token_id)
        # Tiny 1-vector learnable mask embedding delta so 389M nl_embed stays 100% frozen & tied with nl_head!
        with torch.no_grad():
            vocab_sz = model.nl_embed.weight.shape[0]
            if 0 <= fim_init_id < vocab_sz and 0 <= self.mask_token_id < vocab_sz:
                init_delta = (
                    model.nl_embed.weight[fim_init_id] - model.nl_embed.weight[self.mask_token_id]
                ).detach().clone()
            else:
                init_delta = torch.zeros(model.cfg.hidden_size, dtype=model.nl_embed.weight.dtype)
        self.nl_mask_delta = nn.Parameter(init_delta.view(1, 1, -1))

    def _embed_nl(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.nl_embed(token_ids)
        is_m = token_ids == self.mask_token_id
        if is_m.any():
            x = x + is_m.unsqueeze(-1).to(x.dtype) * self.nl_mask_delta.to(x.dtype)
        return x

    def _run_backbone(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_kvs: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run all 36 Left-Brain GQA + DenseSwiGLU layers (zero PML/Engram overhead)."""
        v_anchor: Optional[torch.Tensor] = None
        present_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        use_ckpt = bool(self.model.gradient_checkpointing and self.model.training and torch.is_grad_enabled())

        for l_idx in range(self.model.cfg.num_layers):
            attn_l = self.model.attn_layers[l_idx]
            ffn = self.model.ffn_layers[l_idx]
            pkv_in = past_kvs[l_idx] if past_kvs is not None else None
            if use_ckpt and l_idx > 0 and pkv_in is None and not return_kvs:
                def _layer_fn(
                    h_in: torch.Tensor,
                    v_anc: Optional[torch.Tensor],
                    _attn=attn_l,
                    _ffn=ffn,
                ) -> torch.Tensor:
                    h_gqa_c, _, _ = _attn.gqa(
                        _attn.norm(h_in), positions=positions, attn_mask=attn_mask, v_anchor=v_anc, past_kv=None
                    )
                    return _ffn(h_in + h_gqa_c)

                x = torch.utils.checkpoint.checkpoint(_layer_fn, x, v_anchor, use_reentrant=False)
                continue

            norm_x = attn_l.norm(x)
            h_gqa, v_raw, pkv_out = attn_l.gqa(
                norm_x, positions=positions, attn_mask=attn_mask, v_anchor=v_anchor, past_kv=pkv_in
            )
            x = x + h_gqa
            if return_kvs:
                present_kvs.append(pkv_out)
            if l_idx == 0:
                v_anchor = v_raw

            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(ffn, x, use_reentrant=False)
            else:
                x = ffn(x)

        return x, present_kvs

    def forward_bd3_train(
        self,
        prompt_nl_ids: torch.Tensor,
        y_clean_ids: torch.Tensor,
        y_noisy_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        block_size: int = 8,
        ans_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Dual-stream BD3-LM forward & masked-only CE loss over [Prompt | Y_clean | Y_noisy]."""
        bsz, p_len = prompt_nl_ids.shape
        s_len = y_clean_ids.shape[1]
        device = prompt_nl_ids.device

        full_ids = torch.cat([prompt_nl_ids, y_clean_ids, y_noisy_ids], dim=1)
        x = self._embed_nl(full_ids)
        attn_mask, base_pos = build_nl_bd3_mask(p_len, s_len, block_size=block_size, device=device)
        positions = base_pos.expand(bsz, -1)

        h_all, _ = self._run_backbone(x, positions=positions, attn_mask=attn_mask, past_kvs=None, return_kvs=False)
        h_noisy = self.model.final_norm(h_all[:, p_len + s_len :, :])  # [B, S, D]

        # Memory-efficient masked-only projection (projects ONLY masked slots into 151,936 V_nl!)
        flat_mask = loss_mask.reshape(-1)
        if not flat_mask.any():
            zero = h_noisy.sum() * 0.0
            return zero, {"loss_total": 0.0, "tok_acc": 0.0, "loss_ans": 0.0, "ans_acc": 0.0}

        h_sel = h_noisy.reshape(-1, h_noisy.shape[-1])[flat_mask]
        tgt_sel = y_clean_ids.reshape(-1)[flat_mask]
        logits_sel = self.model.nl_head(h_sel)
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

        stats = {
            "loss_total": float(loss.item()),
            "tok_acc": round(tok_acc, 2),
            "loss_ans": round(loss_ans_val, 4),
            "ans_acc": round(ans_acc_val, 2),
        }
        return loss, stats

    @torch.no_grad()
    def forward_block_step(
        self,
        token_ids: torch.Tensor,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        return_logits: bool = True,
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Causal Prompt prefill (past_kvs=None) OR bidirectional 8-slot Prefix-KV block step (past_kvs!=None).
        Matches forward_bd3_train to < 1e-5 (TF32-off)!
        """
        bsz, seq_len = token_ids.shape
        device = token_ids.device
        positions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        x = self._embed_nl(token_ids)
        h_out, present_kvs = self._run_backbone(x, positions=positions, attn_mask=None, past_kvs=past_kvs, return_kvs=True)
        if not return_logits:
            return None, present_kvs
        logits = self.model.nl_head(self.model.final_norm(h_out))
        return logits, present_kvs

    @torch.no_grad()
    def generate_nl_block_dllm(
        self,
        prompt_nl_ids: torch.Tensor,
        tokenizer: Any,
        max_new_tokens: int = 256,
        block_size: int = 8,
        denoise_beats: int = 3,
    ) -> Tuple[str, int, float, float]:
        """Prefix-KV 8-slot confidence-margin parallel block diffusion decoding over native V_nl."""
        device = prompt_nl_ids.device
        eos_id = getattr(tokenizer, "eos_token_id", None)
        p_len = int(prompt_nl_ids.shape[1])

        t0 = _sync_time(device)
        _, cur_kvs = self.forward_block_step(prompt_nl_ids, past_kvs=None, start_pos=0, return_logits=False)
        t1 = _sync_time(device)
        prefill_ms = (t1 - t0) * 1000.0

        gen_ids: List[int] = []
        max_blocks = max(1, (max_new_tokens + block_size - 1) // block_size)
        cur_pos = p_len

        for _ in range(max_blocks):
            cur_blk_len = min(block_size, max_new_tokens - len(gen_ids))
            if cur_blk_len <= 0:
                break
            blk_toks = torch.full((1, cur_blk_len), self.mask_token_id, dtype=torch.long, device=device)

            # Confidence-margin iterative unmasking over the 8-slot block (3 beats: e.g. 3 + 3 + 2 tokens!)
            for beat in range(denoise_beats):
                logits_b, _ = self.forward_block_step(blk_toks, past_kvs=cur_kvs, start_pos=cur_pos, return_logits=True)
                assert logits_b is not None
                # Forbid predicting the mask token itself
                logits_b = logits_b.clone()
                logits_b[:, :, self.mask_token_id] = float("-inf")

                top2_vals, top2_idx = torch.topk(logits_b[0], k=2, dim=-1)  # [B_blk, 2]
                preds = top2_idx[:, 0]
                margins = (top2_vals[:, 0] - top2_vals[:, 1]).float()

                masked_pos = (blk_toks[0] == self.mask_token_id).nonzero(as_tuple=False).view(-1)
                n_rem = int(masked_pos.numel())
                if n_rem == 0:
                    break
                if beat == denoise_beats - 1:
                    blk_toks[0, masked_pos] = preds[masked_pos]
                else:
                    beats_left = denoise_beats - beat
                    k_unmask = max(1, (n_rem + beats_left - 1) // beats_left)
                    rem_margins = margins[masked_pos]
                    _, pick_rel = torch.topk(rem_margins, k=min(k_unmask, n_rem))
                    chosen = masked_pos[pick_rel]
                    blk_toks[0, chosen] = preds[chosen]

            blk_list = blk_toks[0].tolist()
            hit_eos = False
            if eos_id is not None and eos_id in blk_list:
                eos_idx = blk_list.index(eos_id)
                gen_ids.extend(blk_list[:eos_idx])
                hit_eos = True
            else:
                gen_ids.extend(blk_list)

            # Check early stop BEFORE wasting a commit pass if we already hit EOS or finished Answer!
            cur_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            hit_stop = hit_eos or any(s in cur_txt for s in ("\nProblem (", "\nQuestion:", "\n\nProblem"))
            if not hit_stop:
                if re.search(r"(?:Answer|Final Answer|####)\s*:?\s*[^\n]+\n", cur_txt, re.IGNORECASE) is not None:
                    hit_stop = True

            if hit_stop or len(gen_ids) >= max_new_tokens:
                break

            # Single clean commit pass appends finalized 8-slot block to Prefix-KV cache!
            _, cur_kvs = self.forward_block_step(blk_toks, past_kvs=cur_kvs, start_pos=cur_pos, return_logits=False)
            cur_pos += cur_blk_len

        t2 = _sync_time(device)
        total_ms = (t2 - t0) * 1000.0
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        for stop_seq in ("\nProblem (", "\nQuestion:", "\n\nProblem"):
            if stop_seq in text:
                text = text.split(stop_seq, 1)[0].strip()
        return text, len(gen_ids), round(prefill_ms, 2), round(total_ms, 2)
