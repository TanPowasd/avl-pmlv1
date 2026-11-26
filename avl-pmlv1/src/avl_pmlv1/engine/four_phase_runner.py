"""Four-Phase Cognitive Lifecycle Runner with Prefix-KV Caching, Wet-Cement, Lazy Commit, & E2E NL Runner."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from ..diffusion.dir_vector import DirectionVectorHead
from ..diffusion.post_surgery import PostThinkingSurgeryGuard
from ..diffusion.wet_cement import (
    build_block_diffusion_attn_mask,
    compute_top1_top2_margin,
    hierarchical_three_beat_unmask,
    select_wet_cement_remask_slots,
)
from ..moe_track2.hyper_router import Track2ExperienceState
from ..moe_track2.tri_state_moe import TriStateMoELayer
from ..pml_track1.whiteboard import WhiteboardState
from ..vocab.sym_vocab import CognitivePhase, SymbolicVocab
from .model import AVLPMLModel, build_nl_ro_span_weights


KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


def slice_kv_cache(kvs: KVCache, length: int) -> KVCache:
    """Slice Prefix-KV cache cleanly to [0 : length] across all layers (used on Wet-Cement & Breaker rollback)."""
    return [(k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous()) for k, v in kvs]


def repeat_kv_cache(kvs: KVCache, rep_factor: int) -> KVCache:
    """Repeat Prefix-KV cache along batch dimension for batched 4-fold PLL+CVaR Post-Thinking audit."""
    if rep_factor <= 1:
        return kvs
    return [(k.repeat(rep_factor, 1, 1, 1), v.repeat(rep_factor, 1, 1, 1)) for k, v in kvs]


def repeat_exp_state(exp: Track2ExperienceState, rep_factor: int) -> Track2ExperienceState:
    if rep_factor <= 1:
        return exp
    return Track2ExperienceState(
        delta_b_pre=exp.delta_b_pre.repeat(rep_factor, 1, 1),
        p_pos=exp.p_pos.repeat(rep_factor, 1, 1, 1),
        p_neg=exp.p_neg.repeat(rep_factor, 1, 1, 1),
        delta_m_syn=exp.delta_m_syn.repeat(rep_factor, 1, 1, 1),
        delta_m_adv=exp.delta_m_adv.repeat(rep_factor, 1, 1, 1),
        culprit_mask=exp.culprit_mask.repeat(rep_factor, 1, 1),
    )


def build_prefix_kv_block_mask(past_len: int, block_sizes: List[int], device: torch.device) -> torch.Tensor:
    """Build strict block-causal SDPA mask [1, 1, S_trace, past_len + S_trace] when past_kv is cached!
    Prevents SDPA (is_causal=False when past_kv is not None) from opening unmasked bidirectional future peeking!
    """
    s_trace = sum(block_sizes)
    left = torch.ones((s_trace, past_len), dtype=torch.bool, device=device)
    right = build_block_diffusion_attn_mask(block_sizes, device=device).squeeze(0).squeeze(0)
    return torch.cat([left, right], dim=1).unsqueeze(0).unsqueeze(0)


def _sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@dataclass
class SparseCheckpoint:
    frame_idx: int
    wb_snapshots: List[Tuple[torch.Tensor, torch.Tensor]]
    synergy_snapshot: Tuple[torch.Tensor, torch.Tensor]
    committed_tokens: torch.Tensor


@dataclass
class InferenceSessionResult:
    pre_skeleton: torch.Tensor
    formal_tokens: torch.Tensor
    final_tokens: torch.Tensor
    post_accepted: torch.Tensor
    breaker_triggers: int
    wet_cement_triggers: int
    exp_state: Track2ExperienceState
    wb_states: List[WhiteboardState]
    used_prefix_kv: bool = True


@dataclass
class EndToEndNLResult:
    gen_mode: str
    pml_enabled: bool
    prompt_tokens: int
    sym_tokens_draft: List[int]
    sym_tokens_final: List[int]
    ans_tokens: List[int]
    draft_answer_text: str
    final_answer_text: str
    repair_enabled: bool
    repair_accepted: bool
    prefill_ms: float
    think_ms: float
    repair_ms: float
    verbalize_ms: float
    total_e2e_ms: float


class FourPhaseRunner:
    """Executes Pre -> Formal (Prefix-KV + Wet-Cement + Breaker + Lazy Commit) -> Post (PLL+CVaR) -> Verbalize."""

    def __init__(self, model: AVLPMLModel) -> None:
        self.model = model
        self.cfg = model.cfg
        self.surgery_guard = PostThinkingSurgeryGuard(
            atomic_chunk_size=self.cfg.atomic_chunk_size,
            guard_margin=self.cfg.pll_guard_margin,
        )

    def _get_adv_manifolds(self, device: torch.device) -> torch.Tensor:
        advs = []
        for ffn in self.model.ffn_layers:
            if isinstance(ffn, TriStateMoELayer):
                _, m_adv, _ = ffn.graph.get_manifolds()
                advs.append(m_adv)
        if not advs:
            e = self.cfg.num_routed_experts
            return torch.zeros((0, e, e), device=device, dtype=torch.float32)
        return torch.stack(advs, dim=0)

    @torch.no_grad()
    def run_session(
        self,
        prompt_sym_prefix: torch.Tensor,
        num_formal_blocks: int = 3,
        skel_len: int = 16,
        denoise_beats: int = 3,
        use_prefix_kv: bool = True,
    ) -> InferenceSessionResult:
        """Symbolic-only 4-phase session.
        When use_prefix_kv=True (default), prefixes are cached ONCE in KV memory so denoising beats
        only compute attention over the current 8/16-slot block instead of re-encoding the entire prefix!
        """
        bsz, p_len = prompt_sym_prefix.shape
        device = prompt_sym_prefix.device
        dtype = next(self.model.parameters()).dtype
        wb_states = self.model.init_whiteboard_states(bsz, device, dtype=dtype)
        if not self.model.ablation.disable_pml:
            self.model.populate_ro_from_prompt(prompt_sym_prefix, wb_states)
        exp_state = self.model.init_experience_state(bsz, device, dtype=torch.float32)

        # Phase 1: Pre-Thinking (K_max = 0, 2-beat skeleton denoising + Hyper-Router)
        skel_tokens = torch.full((bsz, skel_len), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
        prompt_kvs: Optional[KVCache] = None
        h_prompt_pool: Optional[torch.Tensor] = None

        if use_prefix_kv:
            # Prefill prompt_sym_prefix ONCE into Prefix-KV cache!
            out_p = self.model.forward_symbolic(
                prompt_sym_prefix,
                phase=CognitivePhase.PRE,
                wb_states=wb_states,
                exp_state=exp_state,
                t_k=0.0,
                read_only_pml=True,
                start_pos=0,
                return_soft_sketch=False,
            )
            prompt_kvs = out_p["present_kvs"]  # type: ignore[assignment]
            h_prompt_pool = out_p["hidden"].mean(dim=1)  # type: ignore[union-attr]

        sketch: Optional[torch.Tensor] = None
        for beat in range(2):
            t_k = 1.0 - 0.5 * beat
            if use_prefix_kv:
                out_pre = self.model.forward_symbolic(
                    skel_tokens,
                    phase=CognitivePhase.PRE,
                    wb_states=wb_states,
                    exp_state=exp_state,
                    soft_sketch=sketch,
                    t_k=t_k,
                    read_only_pml=True,
                    past_kvs=prompt_kvs,
                    start_pos=p_len,
                    return_soft_sketch=(beat < 1),
                )
                sketch = out_pre["soft_sketch"]  # type: ignore[assignment]
                skel_logits = out_pre["logits"]  # type: ignore[assignment]
            else:
                pre_seq = torch.cat([prompt_sym_prefix, skel_tokens], dim=1)
                out_pre = self.model.forward_symbolic(
                    pre_seq,
                    phase=CognitivePhase.PRE,
                    wb_states=wb_states,
                    exp_state=exp_state,
                    soft_sketch=sketch,
                    t_k=t_k,
                    read_only_pml=True,
                )
                sketch = out_pre["soft_sketch"]  # type: ignore[assignment]
                skel_logits = out_pre["logits"][:, -skel_len:, :]  # type: ignore[index]
            skel_tokens, _ = hierarchical_three_beat_unmask(skel_tokens, skel_logits, beat, 2)

        # Finalize [prompt, skel] KV cache & Hyper-Router Delta_b_pre
        base_len = p_len + skel_len
        base_kvs: Optional[KVCache] = None
        prefix_kvs: Optional[KVCache] = None
        if use_prefix_kv:
            out_skel_clean = self.model.forward_symbolic(
                skel_tokens,
                phase=CognitivePhase.PRE,
                wb_states=wb_states,
                exp_state=exp_state,
                t_k=0.0,
                read_only_pml=True,
                past_kvs=prompt_kvs,
                start_pos=p_len,
                return_soft_sketch=False,
            )
            base_kvs = out_skel_clean["present_kvs"]  # type: ignore[assignment]
            prefix_kvs = base_kvs
            h_skel_pool = out_skel_clean["hidden"].mean(dim=1)  # type: ignore[union-attr]
            assert h_prompt_pool is not None
            exp_state.delta_b_pre = self.model.hyper_router(h_prompt_pool, h_skel_pool)
        else:
            h_all: torch.Tensor = out_pre["hidden"]  # type: ignore[assignment]
            h_prompt_pool = h_all[:, :p_len, :].mean(dim=1)
            h_skel_pool = h_all[:, -skel_len:, :].mean(dim=1)
            exp_state.delta_b_pre = self.model.hyper_router(h_prompt_pool, h_skel_pool)

        # Phase 2: Formal Thinking with Prefix-KV + 16-slot Wet-Cement + Lazy Commit + Circuit Breaker
        committed_blocks: List[torch.Tensor] = []
        block_margins: List[torch.Tensor] = []
        checkpoints: Dict[int, SparseCheckpoint] = {
            0: SparseCheckpoint(
                frame_idx=0,
                wb_snapshots=[wb.clone_committed_snapshot() for wb in wb_states],
                synergy_snapshot=exp_state.clone_synergy_snapshot(),
                committed_tokens=torch.empty((bsz, 0), device=device, dtype=torch.long),
            )
        }

        breaker_count = 0
        wet_triggers = 0
        prev_high_acute = False
        b_idx = 0
        force_div = False
        cur_block_size = self.cfg.default_block_size

        while b_idx < num_formal_blocks:
            block_toks = torch.full((bsz, cur_block_size), SymbolicVocab.MASK_ID, device=device, dtype=torch.long)
            prefix_len = base_len + sum(blk.shape[1] for blk in committed_blocks)

            # Denoise current block (read-only against S^RW_view so wet-cement draft is visible!)
            blk_sketch: Optional[torch.Tensor] = None
            u_epi_step1 = 0.0
            for beat in range(denoise_beats):
                t_k = float((block_toks == SymbolicVocab.MASK_ID).float().mean().item())
                if use_prefix_kv:
                    # ONLY run the current cur_block_size block against cached prefix_kvs (ZERO prefix recompute!)
                    out_blk = self.model.forward_symbolic(
                        block_toks,
                        phase=CognitivePhase.FORMAL,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        soft_sketch=blk_sketch,
                        t_k=t_k,
                        read_only_pml=True,
                        force_breaker_div=force_div,
                        past_kvs=prefix_kvs,
                        start_pos=prefix_len,
                        return_soft_sketch=(beat < denoise_beats - 1),
                    )
                    if beat == 0:
                        u_epi_step1 = float(out_blk["u_epistemic"].mean().item())  # type: ignore[union-attr]
                    blk_logits = out_blk["logits"]  # type: ignore[assignment]
                    blk_sketch = out_blk["soft_sketch"]  # type: ignore[assignment]
                else:
                    prefix_fixed = torch.cat([prompt_sym_prefix, skel_tokens] + committed_blocks, dim=1)
                    full_in = torch.cat([prefix_fixed, block_toks], dim=1)
                    out_blk = self.model.forward_symbolic(
                        full_in,
                        phase=CognitivePhase.FORMAL,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        soft_sketch=None if blk_sketch is None else torch.cat(
                            [torch.zeros(bsz, prefix_fixed.shape[1], self.cfg.hidden_size, device=device, dtype=dtype), blk_sketch],
                            dim=1,
                        ),
                        t_k=t_k,
                        read_only_pml=True,
                        force_breaker_div=force_div,
                    )
                    if beat == 0:
                        u_epi_step1 = float(out_blk["u_epistemic"][:, -cur_block_size:].mean().item())  # type: ignore[index]
                    blk_logits = out_blk["logits"][:, -cur_block_size:, :]  # type: ignore[index]
                    blk_sketch = out_blk["soft_sketch"][:, -cur_block_size:, :]  # type: ignore[index]

                d_b_cur: torch.Tensor = out_blk["d_b"]  # type: ignore[assignment]
                d_hier = float(d_b_cur[:, 2].mean().item())
                block_toks, margins = hierarchical_three_beat_unmask(
                    block_toks, blk_logits, beat, denoise_beats, d_hier=d_hier
                )

            # Optional 16-slot Wet-Cement backward micro-diffusion on previous block's tail
            d_bwd = float(d_b_cur[:, 1].mean().item())
            if b_idx > 0 and d_bwd > 0.55 and committed_blocks[-1].shape[1] >= self.cfg.wet_cement_slots:
                wet_triggers += 1
                prev_blk = committed_blocks[-1]
                prev_m = block_margins[-1]
                w_slots = self.cfg.wet_cement_slots
                tail_toks = prev_blk[:, -w_slots:]
                tail_m = prev_m[:, -w_slots:]
                remasked_tail, _ = select_wet_cement_remask_slots(tail_toks, tail_m, d_bwd)

                if use_prefix_kv:
                    assert prefix_kvs is not None
                    frozen_len = prefix_len - w_slots
                    frozen_kvs = slice_kv_cache(prefix_kvs, frozen_len)
                    seam_toks = torch.cat([remasked_tail, block_toks], dim=1)
                    out_seam = self.model.forward_symbolic(
                        seam_toks,
                        phase=CognitivePhase.FORMAL,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        t_k=0.20,
                        read_only_pml=True,
                        past_kvs=frozen_kvs,
                        start_pos=frozen_len,
                        return_soft_sketch=False,
                    )
                    seam_logits = out_seam["logits"]  # type: ignore[assignment]
                else:
                    frozen_head = torch.cat(
                        [prompt_sym_prefix, skel_tokens] + committed_blocks[:-1] + [prev_blk[:, :-w_slots]],
                        dim=1,
                    )
                    seam_in = torch.cat([frozen_head, remasked_tail, block_toks], dim=1)
                    out_seam = self.model.forward_symbolic(
                        seam_in,
                        phase=CognitivePhase.FORMAL,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        t_k=0.20,
                        read_only_pml=True,
                        return_soft_sketch=False,
                    )
                    seam_logits = out_seam["logits"][:, -(w_slots + cur_block_size) :, :]  # type: ignore[index]

                seam_preds, seam_margins = compute_top1_top2_margin(self.model.vocab.apply_domain_mask(seam_logits))
                new_tail = torch.where(remasked_tail == SymbolicVocab.MASK_ID, seam_preds[:, :w_slots], remasked_tail)
                committed_blocks[-1] = torch.cat([prev_blk[:, :-w_slots], new_tail], dim=1)
                block_margins[-1] = torch.cat([prev_m[:, :-w_slots], seam_margins[:, :w_slots]], dim=1)
                # Replay refreshed 16-slot tail onto S^RW_draft & update prefix_kvs tail cleanly!
                for wb in wb_states:
                    wb.discard_draft()
                out_tail_ref = self.model.forward_symbolic(
                    new_tail,
                    phase=CognitivePhase.FORMAL,
                    wb_states=wb_states,
                    exp_state=exp_state,
                    t_k=0.0,
                    write_to_draft=True,
                    read_only_pml=False,
                    past_kvs=frozen_kvs if use_prefix_kv else None,
                    start_pos=(prefix_len - w_slots) if use_prefix_kv else 0,
                    return_soft_sketch=False,
                )
                if use_prefix_kv:
                    prefix_kvs = out_tail_ref["present_kvs"]  # type: ignore[assignment]

            # Evaluate candidate block writes onto temporary draft to check Red Light before Lazy Commit
            cand_out = self.model.forward_symbolic(
                block_toks,
                phase=CognitivePhase.FORMAL,
                wb_states=wb_states,
                exp_state=exp_state,
                t_k=0.0,
                write_to_draft=True,
                read_only_pml=False,
                past_kvs=prefix_kvs if use_prefix_kv else None,
                start_pos=prefix_len if use_prefix_kv else 0,
                return_soft_sketch=False,
            )
            u_acute_max = float(cand_out["u_acute"].max().item())  # type: ignore[union-attr]
            trigger_breaker = (
                breaker_count < self.cfg.max_circuit_breakers
                and (u_acute_max >= 0.65 or (prev_high_acute and u_acute_max >= 0.45))
            )

            if trigger_breaker:
                breaker_count += 1
                target_frame = max(0, b_idx - 1)
                snap_key = (target_frame // self.cfg.sparse_snapshot_interval) * self.cfg.sparse_snapshot_interval
                snap = checkpoints[snap_key]
                for wb, s_tup in zip(wb_states, snap.wb_snapshots):
                    wb.restore_committed_snapshot(s_tup)

                moe_aux: List[Dict[str, torch.Tensor]] = cand_out["moe_aux"]  # type: ignore[assignment]
                if moe_aux:
                    p_means = torch.stack([m["probes_z"].mean(dim=1) for m in moe_aux], dim=1)
                    e_weights = torch.stack([m["block_profile"] for m in moe_aux], dim=1)
                    exp_state.asymmetric_rollback_and_imprint_pain(
                        synergy_snapshot=snap.synergy_snapshot,
                        lesion_probe_means=p_means,
                        lesion_expert_weights=e_weights,
                        m_adv_manifolds=self._get_adv_manifolds(device),
                        gamma_pain=self.cfg.gamma_pain,
                    )
                committed_blocks = committed_blocks[:target_frame]
                block_margins = block_margins[:target_frame]
                if use_prefix_kv:
                    assert prefix_kvs is not None
                    rollback_len = base_len + sum(blk.shape[1] for blk in committed_blocks)
                    prefix_kvs = slice_kv_cache(prefix_kvs, rollback_len)
                if target_frame > snap_key and committed_blocks[snap_key:target_frame]:
                    replay_seq = torch.cat(committed_blocks[snap_key:target_frame], dim=1)
                    _ = self.model.forward_symbolic(
                        replay_seq,
                        phase=CognitivePhase.FORMAL,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        t_k=0.0,
                        return_soft_sketch=False,
                    )
                b_idx = target_frame
                force_div = True
                prev_high_acute = False
                continue

            # Lazy Commit: reconstruct exact committed prefix + merge previous tail before writing block b_idx
            if b_idx == 0:
                for wb, s_tup in zip(wb_states, checkpoints[0].wb_snapshots):
                    wb.restore_committed_snapshot(s_tup)
            else:
                snap_key = (b_idx // self.cfg.sparse_snapshot_interval) * self.cfg.sparse_snapshot_interval
                if snap_key in checkpoints:
                    for wb, s_tup in zip(wb_states, checkpoints[snap_key].wb_snapshots):
                        wb.restore_committed_snapshot(s_tup)
                    if b_idx > snap_key:
                        rep = torch.cat(committed_blocks[snap_key:b_idx], dim=1)
                        _ = self.model.forward_symbolic(
                            rep,
                            phase=CognitivePhase.FORMAL,
                            wb_states=wb_states,
                            exp_state=exp_state,
                            t_k=0.0,
                            return_soft_sketch=False,
                        )

            if cur_block_size > self.cfg.wet_cement_slots:
                head_part = block_toks[:, : -self.cfg.wet_cement_slots]
                tail_part = block_toks[:, -self.cfg.wet_cement_slots :]
                _ = self.model.forward_symbolic(
                    head_part,
                    phase=CognitivePhase.FORMAL,
                    wb_states=wb_states,
                    exp_state=exp_state,
                    t_k=0.0,
                    return_soft_sketch=False,
                )
            else:
                tail_part = block_toks
            _ = self.model.forward_symbolic(
                tail_part,
                phase=CognitivePhase.FORMAL,
                wb_states=wb_states,
                exp_state=exp_state,
                t_k=0.0,
                write_to_draft=True,
                return_soft_sketch=False,
            )

            # Advance Prefix-KV cache by committed block_toks
            if use_prefix_kv:
                prefix_kvs = cand_out["present_kvs"]  # type: ignore[assignment]

            moe_aux = cand_out["moe_aux"]  # type: ignore[assignment]
            if moe_aux:
                p_means = torch.stack([m["probes_z"].mean(dim=1) for m in moe_aux], dim=1)
                e_weights = torch.stack([m["block_profile"] for m in moe_aux], dim=1)
                hard_conv = u_epi_step1 > 0.35 and float(margins.mean().item()) > 0.65 and u_acute_max < 0.30
                exp_state.record_clean_frame_advance(p_means, e_weights, hard_conv, gamma_pain=self.cfg.gamma_pain)

            committed_blocks.append(block_toks)
            block_margins.append(margins)
            prev_high_acute = u_acute_max >= 0.45
            force_div = False
            b_idx += 1

            if b_idx % self.cfg.sparse_snapshot_interval == 0:
                for wb in wb_states:
                    wb.lazy_commit_previous_tail()
                checkpoints[b_idx] = SparseCheckpoint(
                    frame_idx=b_idx,
                    wb_snapshots=[wb.clone_committed_snapshot() for wb in wb_states],
                    synergy_snapshot=exp_state.clone_synergy_snapshot(),
                    committed_tokens=torch.cat(committed_blocks, dim=1),
                )

            u_epi_cur = float(cand_out["u_epistemic"].mean().item())  # type: ignore[union-attr]
            has_long = bool((block_toks == SymbolicVocab.LONG_FRAME_ID).any().item())
            has_short = bool((block_toks == SymbolicVocab.SHORT_FRAME_ID).any().item())
            cur_block_size = DirectionVectorHead.decide_next_block_size(d_b_cur, u_epi_cur, has_long, has_short)

        for wb in wb_states:
            wb.lazy_commit_previous_tail()

        formal_tokens = torch.cat(committed_blocks, dim=1)

        # Phase 3: Post-Thinking Surgical Audit + 4-Fold PLL+CVaR Energy Guard (strictly block-causal!)
        all_margins = torch.cat(block_margins, dim=1)
        formal_blk_sizes = [blk.shape[1] for blk in committed_blocks]
        wb_base_post = [wb.repeat_batch(1) for wb in wb_states]
        for wb_p, s_tup in zip(wb_base_post, checkpoints[0].wb_snapshots):
            wb_p.restore_committed_snapshot(s_tup)

        cand_remask, _ = select_wet_cement_remask_slots(formal_tokens, all_margins, d_bwd=0.75)
        if use_prefix_kv:
            post_kv_mask = build_prefix_kv_block_mask(base_len, formal_blk_sizes, device=device)
            post_out = self.model.forward_symbolic(
                cand_remask,
                phase=CognitivePhase.POST,
                attn_mask=post_kv_mask,
                wb_states=wb_base_post,
                exp_state=exp_state,
                t_k=0.25,
                read_only_pml=True,
                past_kvs=base_kvs,
                start_pos=base_len,
                return_soft_sketch=False,
                frame_sizes=formal_blk_sizes,
                clean_token_ids_for_pml=formal_tokens,
            )
            post_logits = post_out["logits"]  # type: ignore[assignment]
        else:
            full_blk_sizes = [p_len, skel_len] + formal_blk_sizes
            post_full_mask = build_block_diffusion_attn_mask(full_blk_sizes, device=device)
            post_out = self.model.forward_symbolic(
                torch.cat([prompt_sym_prefix, skel_tokens, cand_remask], dim=1),
                phase=CognitivePhase.POST,
                attn_mask=post_full_mask,
                wb_states=wb_base_post,
                exp_state=exp_state,
                t_k=0.25,
                read_only_pml=True,
                return_soft_sketch=False,
                frame_sizes=full_blk_sizes,
                clean_token_ids_for_pml=torch.cat([prompt_sym_prefix, skel_tokens, formal_tokens], dim=1),
            )
            post_logits = post_out["logits"][:, -formal_tokens.shape[1] :, :]  # type: ignore[index]

        post_preds, _ = compute_top1_top2_margin(self.model.vocab.apply_domain_mask(post_logits))
        y_candidate = torch.where(cand_remask == SymbolicVocab.MASK_ID, post_preds, formal_tokens)

        def _audit_fn(masked_y: torch.Tensor, clean_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            rep_factor = masked_y.shape[0] // bsz
            wb_rep = [wb.repeat_batch(rep_factor) for wb in wb_base_post]
            exp_rep = repeat_exp_state(exp_state, rep_factor)
            if use_prefix_kv and base_kvs is not None:
                res = self.model.forward_symbolic(
                    masked_y,
                    phase=CognitivePhase.POST,
                    attn_mask=build_prefix_kv_block_mask(base_len, formal_blk_sizes, device=device),
                    wb_states=wb_rep,
                    exp_state=exp_rep,
                    t_k=0.25,
                    read_only_pml=True,
                    past_kvs=repeat_kv_cache(base_kvs, rep_factor),
                    start_pos=base_len,
                    return_soft_sketch=False,
                    frame_sizes=formal_blk_sizes,
                    clean_token_ids_for_pml=clean_y,
                )
                return res["logits"], res["u_epistemic"]  # type: ignore[return-value]
            pref = torch.cat([prompt_sym_prefix, skel_tokens], dim=1).repeat(rep_factor, 1)
            all_blks = [p_len, skel_len] + formal_blk_sizes
            res = self.model.forward_symbolic(
                torch.cat([pref, masked_y], dim=1),
                phase=CognitivePhase.POST,
                attn_mask=build_block_diffusion_attn_mask(all_blks, device=device),
                wb_states=wb_rep,
                exp_state=exp_rep,
                t_k=0.25,
                read_only_pml=True,
                return_soft_sketch=False,
                frame_sizes=all_blks,
                clean_token_ids_for_pml=torch.cat([pref, clean_y], dim=1),
            )
            s_len = masked_y.shape[1]
            return res["logits"][:, -s_len:, :], res["u_epistemic"][:, -s_len:]  # type: ignore[index]

        final_tokens, accepted, _, _ = self.surgery_guard.accept_or_reject(formal_tokens, y_candidate, _audit_fn)

        return InferenceSessionResult(
            pre_skeleton=skel_tokens,
            formal_tokens=formal_tokens,
            final_tokens=final_tokens,
            post_accepted=accepted,
            breaker_triggers=breaker_count,
            wet_cement_triggers=wet_triggers,
            exp_state=exp_state,
            wb_states=wb_states,
            used_prefix_kv=use_prefix_kv,
        )

    @torch.no_grad()
    def _verbalize_from_kv(
        self,
        first_nl_logits: torch.Tensor,
        kvs: KVCache,
        start_pos: int,
        wb_states: List[WhiteboardState],
        tokenizer: Any,
        max_ans_tokens: int = 32,
        prompt_hidden_for_copy: Optional[torch.Tensor] = None,
        prompt_ids_for_copy: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], str]:
        """Greedy KV-cached V_nl answer decoding directly conditioned on [Prompt_nl | Thinking_sym] + committed PML."""
        eos_id = getattr(tokenizer, "eos_token_id", None)
        cur_logits = first_nl_logits
        cur_kvs = kvs
        cur_pos = start_pos
        ans_ids: List[int] = []
        device = first_nl_logits.device

        for _ in range(max_ans_tokens):
            next_tok = int(cur_logits[0, -1, :].argmax(dim=-1).item())
            if eos_id is not None and next_tok == eos_id:
                break
            ans_ids.append(next_tok)
            if len(ans_ids) >= max_ans_tokens:
                break
            tok_t = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            r_a = self.model.forward_ar_step(
                tok_t,
                domain="nl_ans",
                start_pos=cur_pos,
                past_kvs=cur_kvs,
                wb_states=wb_states,
                prompt_hidden_for_copy=prompt_hidden_for_copy,
                prompt_ids_for_copy=prompt_ids_for_copy,
            )
            cur_kvs = r_a["present_kvs"]  # type: ignore[assignment]
            cur_logits = r_a["nl_logits"]  # type: ignore[assignment]
            cur_pos += 1

        text = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()
        return ans_ids, text

    @torch.no_grad()
    def run_end_to_end_nl(
        self,
        prompt_nl_ids: torch.Tensor,
        tokenizer: Any,
        gen_mode: str = "ar",
        max_sym_tokens: int = 128,
        block_size: int = 8,
        denoise_beats: int = 3,
        max_ans_tokens: int = 32,
        enable_repair: bool = False,
        audit_draft_for_netfix: bool = False,
    ) -> EndToEndNLResult:
        """True End-to-End NL Prompt -> V_sym Thinking (AR or Prefix-KV Block-dLLM) -> Repair -> V_nl Answer.
        Measures synchronized wall-clock latency per stage for Gate 2 Section 5.5 reporting.
        """
        device = prompt_nl_ids.device
        dtype = next(self.model.parameters()).dtype
        p_len = int(prompt_nl_ids.shape[1])
        wb_states = self.model.init_whiteboard_states(1, device, dtype=dtype)
        exp_state = self.model.init_experience_state(1, device, dtype=torch.float32)

        # Stage 1: NL Prompt Prefill (ONCE! Also populates S^RO / M^RO from prompt numeric/entity spans!)
        t0 = _sync_time(device)
        ro_w = build_nl_ro_span_weights(prompt_nl_ids, tokenizer=tokenizer, num_ro_slots=self.cfg.num_ro_slots)
        r_prefill = self.model.forward_ar_step(
            prompt_nl_ids, domain="nl_prompt", start_pos=0, wb_states=wb_states, ro_span_weights=ro_w
        )
        prompt_kvs: KVCache = r_prefill["present_kvs"]  # type: ignore[assignment]
        prompt_h_copy: Optional[torch.Tensor] = (
            self.model.final_norm(r_prefill["hidden"]) if self.cfg.enable_copy_gate else None  # type: ignore[arg-type]
        )
        t1 = _sync_time(device)
        prefill_ms = (t1 - t0) * 1000.0

        # Stage 2: Symbolic Thinking (AR 1-tok KV decode OR Prefix-KV 8-slot Block-dLLM denoising)
        t_think_start = _sync_time(device)
        draft_sym_ids: List[int] = []
        all_block_margins: List[torch.Tensor] = []
        draft_kvs: KVCache = prompt_kvs
        draft_first_ans_logits: Optional[torch.Tensor] = None

        if gen_mode == "ar":
            engram_caches = self.model.init_engram_causal_caches(1, device, dtype=dtype)
            cur_kvs = prompt_kvs
            cur_logits = self.model.vocab.apply_domain_mask(r_prefill["sym_logits"])  # type: ignore[arg-type]
            prev_tok: Optional[torch.Tensor] = None
            prev2_tok: Optional[torch.Tensor] = None
            emb_hist: List[torch.Tensor] = []
            norm_hist: List[Dict[int, torch.Tensor]] = []

            for t in range(max_sym_tokens):
                next_sym = int(cur_logits[0, -1, :].argmax(dim=-1).item())
                draft_sym_ids.append(next_sym)
                tok_t = torch.tensor([[next_sym]], dtype=torch.long, device=device)
                p2 = emb_hist[-2] if len(emb_hist) >= 2 else None
                p2_nx = norm_hist[-2] if len(norm_hist) >= 2 else None
                r_s = self.model.forward_ar_step(
                    tok_t,
                    domain="sym",
                    start_pos=p_len + t,
                    sym_offset=t,
                    past_kvs=cur_kvs,
                    wb_states=wb_states,
                    engram_caches=engram_caches,
                    exp_state=exp_state,
                    prev_sym_token=prev_tok,
                    prev2_sym_embed=p2,
                    prev2_sym_token=prev2_tok,
                    prev2_pml_norm_x=p2_nx,
                    prompt_hidden_for_copy=prompt_h_copy,
                    prompt_ids_for_copy=prompt_nl_ids if self.cfg.enable_copy_gate else None,
                )
                cur_kvs = r_s["present_kvs"]  # type: ignore[assignment]
                prev2_tok = prev_tok
                prev_tok = tok_t[:, 0]
                emb_hist.append(r_s["x_sym_tok"][:, 0, :])  # type: ignore[index]
                norm_hist.append(r_s["pml_norm_x"])  # type: ignore[arg-type]
                if next_sym == SymbolicVocab.EOT_ID or t == max_sym_tokens - 1:
                    draft_first_ans_logits = r_s["nl_logits"]  # type: ignore[assignment]
                    break
                cur_logits = self.model.vocab.apply_domain_mask(r_s["sym_logits"])  # type: ignore[arg-type]
            draft_kvs = cur_kvs
        else:
            # Prefix-KV Block-dLLM over [Prompt_nl (cached) | V_sym blocks]
            cur_kvs = prompt_kvs
            sym_offset = 0
            max_blocks = max(1, (max_sym_tokens + block_size - 1) // block_size)
            last_hidden_block: Optional[torch.Tensor] = None
            use_sketch = bool(self.model.sc_proj.weight.detach().abs().max().item() > 0.0)

            for _ in range(max_blocks):
                cur_blk_len = min(block_size, max_sym_tokens - sym_offset)
                if cur_blk_len <= 0:
                    break
                blk_toks = torch.full((1, cur_blk_len), SymbolicVocab.MASK_ID, dtype=torch.long, device=device)
                blk_sketch: Optional[torch.Tensor] = None
                margins = torch.ones((1, cur_blk_len), device=device, dtype=torch.float32)

                for beat in range(denoise_beats):
                    t_k = float((blk_toks == SymbolicVocab.MASK_ID).float().mean().item())
                    out_b = self.model.forward_symbolic(
                        blk_toks,
                        phase=CognitivePhase.FORMAL,
                        wb_states=wb_states,
                        exp_state=exp_state,
                        soft_sketch=blk_sketch,
                        t_k=t_k,
                        read_only_pml=True,
                        past_kvs=cur_kvs,
                        start_pos=p_len + sym_offset,
                        slot_offset=sym_offset,
                        return_soft_sketch=(use_sketch and beat < denoise_beats - 1),
                    )
                    d_b_cur: torch.Tensor = out_b["d_b"]  # type: ignore[assignment]
                    d_hier = float(d_b_cur[:, 2].mean().item())
                    blk_toks, margins = hierarchical_three_beat_unmask(
                        blk_toks, out_b["logits"], beat, denoise_beats, d_hier=d_hier  # type: ignore[arg-type]
                    )
                    blk_sketch = out_b["soft_sketch"]  # type: ignore[assignment]

                # Single clean commit pass updates PML + appends block KV to cur_kvs!
                out_commit = self.model.forward_symbolic(
                    blk_toks,
                    phase=CognitivePhase.FORMAL,
                    wb_states=wb_states,
                    exp_state=exp_state,
                    t_k=0.0,
                    write_to_draft=False,
                    read_only_pml=False,
                    past_kvs=cur_kvs,
                    start_pos=p_len + sym_offset,
                    slot_offset=sym_offset,
                    return_soft_sketch=False,
                )
                cur_kvs = out_commit["present_kvs"]  # type: ignore[assignment]
                last_hidden_block = out_commit["hidden_raw"]  # type: ignore[assignment]

                blk_list = blk_toks[0].tolist()
                if SymbolicVocab.EOT_ID in blk_list:
                    eot_idx = blk_list.index(SymbolicVocab.EOT_ID)
                    draft_sym_ids.extend(blk_list[: eot_idx + 1])
                    all_block_margins.append(margins[:, : eot_idx + 1])
                    cur_kvs = slice_kv_cache(cur_kvs, p_len + len(draft_sym_ids))
                    eot_h = self.model.final_norm(last_hidden_block[:, eot_idx : eot_idx + 1, :])
                    draft_first_ans_logits = self.model.nl_head(eot_h)
                    if self.cfg.enable_copy_gate and prompt_h_copy is not None:
                        draft_first_ans_logits = self.model.copy_gate(
                            eot_h, draft_first_ans_logits, prompt_h_copy, prompt_nl_ids
                        )
                    break

                draft_sym_ids.extend(blk_list)
                all_block_margins.append(margins)
                sym_offset += cur_blk_len

            if draft_first_ans_logits is None:
                assert last_hidden_block is not None
                last_h = self.model.final_norm(last_hidden_block[:, -1:, :])
                draft_first_ans_logits = self.model.nl_head(last_h)
                if self.cfg.enable_copy_gate and prompt_h_copy is not None:
                    draft_first_ans_logits = self.model.copy_gate(
                        last_h, draft_first_ans_logits, prompt_h_copy, prompt_nl_ids
                    )
            draft_kvs = cur_kvs

        t_think_end = _sync_time(device)
        think_ms = (t_think_end - t_think_start) * 1000.0

        # Optional pre-repair draft verbalization (timed OUTSIDE E2E latency for pure NetFix audit!)
        draft_ans_text = ""
        if enable_repair and audit_draft_for_netfix:
            assert draft_first_ans_logits is not None
            _, draft_ans_text = self._verbalize_from_kv(
                draft_first_ans_logits,
                draft_kvs,
                p_len + len(draft_sym_ids),
                wb_states,
                tokenizer,
                max_ans_tokens,
                prompt_hidden_for_copy=prompt_h_copy,
                prompt_ids_for_copy=prompt_nl_ids if self.cfg.enable_copy_gate else None,
            )

        # Stage 3: Optional Post-Thinking Surgical Repair + 4-Fold PLL+CVaR Guard (strictly block-causal + PML-aware!)
        final_sym_ids = list(draft_sym_ids)
        final_kvs = draft_kvs
        final_first_ans_logits = draft_first_ans_logits
        repair_accepted = False
        repair_ms = 0.0

        if enable_repair and gen_mode != "ar" and len(draft_sym_ids) >= 8:
            t_rep_start = _sync_time(device)
            formal_t = torch.tensor([draft_sym_ids], dtype=torch.long, device=device)
            if all_block_margins:
                m_all = torch.cat(all_block_margins, dim=1)
                trace_blk_sizes = [int(m.shape[1]) for m in all_block_margins if m.shape[1] > 0]
            else:
                m_all = torch.full((1, len(draft_sym_ids)), 0.5, device=device, dtype=torch.float32)
                trace_blk_sizes = []
                rem_t = len(draft_sym_ids)
                while rem_t > 0:
                    step_b = min(block_size, rem_t)
                    trace_blk_sizes.append(step_b)
                    rem_t -= step_b

            wb_ro_only = [wb.repeat_batch(1) for wb in wb_states]
            for wb_r in wb_ro_only:
                wb_r.reset_rw()

            post_mask = build_prefix_kv_block_mask(p_len, trace_blk_sizes, device=device)
            cand_remask, _ = select_wet_cement_remask_slots(formal_t, m_all, d_bwd=0.75)
            post_out = self.model.forward_symbolic(
                cand_remask,
                phase=CognitivePhase.POST,
                attn_mask=post_mask,
                wb_states=wb_ro_only,
                exp_state=exp_state,
                t_k=0.25,
                read_only_pml=True,
                past_kvs=prompt_kvs,
                start_pos=p_len,
                slot_offset=0,
                return_soft_sketch=False,
                frame_sizes=trace_blk_sizes,
                clean_token_ids_for_pml=formal_t,
            )
            post_preds, _ = compute_top1_top2_margin(self.model.vocab.apply_domain_mask(post_out["logits"]))  # type: ignore[arg-type]
            y_cand = torch.where(cand_remask == SymbolicVocab.MASK_ID, post_preds, formal_t)

            def _nl_audit_fn(masked_y: torch.Tensor, clean_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                rep_f = masked_y.shape[0]
                res = self.model.forward_symbolic(
                    masked_y,
                    phase=CognitivePhase.POST,
                    attn_mask=post_mask,
                    wb_states=[wb.repeat_batch(rep_f) for wb in wb_ro_only],
                    exp_state=repeat_exp_state(exp_state, rep_f),
                    t_k=0.25,
                    read_only_pml=True,
                    past_kvs=repeat_kv_cache(prompt_kvs, rep_f),
                    start_pos=p_len,
                    slot_offset=0,
                    return_soft_sketch=False,
                    frame_sizes=trace_blk_sizes,
                    clean_token_ids_for_pml=clean_y,
                )
                return res["logits"], res["u_epistemic"]  # type: ignore[return-value]

            repaired_t, acc_t, _, _ = self.surgery_guard.accept_or_reject(formal_t, y_cand, _nl_audit_fn)
            repair_accepted = bool(acc_t[0].item() and not torch.equal(repaired_t, formal_t))
            if repair_accepted:
                rep_list = repaired_t[0].tolist()
                if SymbolicVocab.EOT_ID in rep_list:
                    rep_list = rep_list[: rep_list.index(SymbolicVocab.EOT_ID) + 1]
                final_sym_ids = rep_list
                rep_t_clean = torch.tensor([final_sym_ids], dtype=torch.long, device=device)
                rep_blks: List[int] = []
                rem_c = len(final_sym_ids)
                while rem_c > 0:
                    sb = min(block_size, rem_c)
                    rep_blks.append(sb)
                    rem_c -= sb
                rep_commit_mask = build_prefix_kv_block_mask(p_len, rep_blks, device=device)
                # Reset RW registers while strictly preserving S^RO / M^RO populated during prompt prefill!
                for wb in wb_states:
                    wb.reset_rw()
                out_rep_commit = self.model.forward_symbolic(
                    rep_t_clean,
                    phase=CognitivePhase.FORMAL,
                    attn_mask=rep_commit_mask,
                    wb_states=wb_states,
                    exp_state=exp_state,
                    t_k=0.0,
                    write_to_draft=False,
                    read_only_pml=False,
                    past_kvs=prompt_kvs,
                    start_pos=p_len,
                    slot_offset=0,
                    return_soft_sketch=False,
                    frame_sizes=rep_blks,
                )
                final_kvs = out_rep_commit["present_kvs"]  # type: ignore[assignment]
                rep_last_h = self.model.final_norm(out_rep_commit["hidden_raw"][:, -1:, :])  # type: ignore[index]
                final_first_ans_logits = self.model.nl_head(rep_last_h)
                if self.cfg.enable_copy_gate and prompt_h_copy is not None:
                    final_first_ans_logits = self.model.copy_gate(
                        rep_last_h, final_first_ans_logits, prompt_h_copy, prompt_nl_ids
                    )
            t_rep_end = _sync_time(device)
            repair_ms = (t_rep_end - t_rep_start) * 1000.0

        # Stage 4: Verbalize Final NL Answer (KV-cached greedy V_nl decode!)
        t_verb_start = _sync_time(device)
        assert final_first_ans_logits is not None
        ans_ids, final_ans_text = self._verbalize_from_kv(
            final_first_ans_logits,
            final_kvs,
            p_len + len(final_sym_ids),
            wb_states,
            tokenizer,
            max_ans_tokens,
            prompt_hidden_for_copy=prompt_h_copy,
            prompt_ids_for_copy=prompt_nl_ids if self.cfg.enable_copy_gate else None,
        )
        t_verb_end = _sync_time(device)
        verbalize_ms = (t_verb_end - t_verb_start) * 1000.0

        if not draft_ans_text:
            draft_ans_text = final_ans_text

        total_e2e_ms = prefill_ms + think_ms + repair_ms + verbalize_ms
        return EndToEndNLResult(
            gen_mode=gen_mode,
            pml_enabled=not self.model.ablation.disable_pml,
            prompt_tokens=p_len,
            sym_tokens_draft=draft_sym_ids,
            sym_tokens_final=final_sym_ids,
            ans_tokens=ans_ids,
            draft_answer_text=draft_ans_text,
            final_answer_text=final_ans_text,
            repair_enabled=enable_repair,
            repair_accepted=repair_accepted,
            prefill_ms=round(prefill_ms, 2),
            think_ms=round(think_ms, 2),
            repair_ms=round(repair_ms, 2),
            verbalize_ms=round(verbalize_ms, 2),
            total_e2e_ms=round(total_e2e_ms, 2),
        )
