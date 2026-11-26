"""AVL-PMLv1.1 Capacity-Preserving Adaptive Top-K MoE with Bipolar Linkage Graph & Dual Crash Guards.

Stage 0 Gate 1 Refactor Summary:
- Amputated residual-stealing dense `probe_mix` & noisy `L_util` delayed scorer (which caused K_max=0 zero-drop & Kendall tau=0.06).
- Shrunk shared micro-expert (64-dim) and expanded routed experts (256-dim) so routed experts carry 80%+ of FFN capacity.
- End-to-end differentiable router + unit-sum adaptive Top-K gate (k in {1, 2}) + Bipolar Sinkhorn Graph + Track-2 Pain Imprint
  signature `probe_u` + L2-cache-resident sync-free `_sparse_topk_swiglu` (CUDAGraph-safe & ~3x faster training).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AVLPMLConfig
from ..pml_track1.whiteboard import RMSNorm
from ..vocab.sym_vocab import CognitivePhase
from .bipolar_graph import BipolarLinkageGraph, row_normalize_overlay
from .hyper_router import Track2ExperienceState


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DenseSwiGLU(nn.Module):
    def __init__(self, cfg: AVLPMLConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.hidden_size, cfg.dense_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class TriStateMoELayer(nn.Module):
    """AVL-PMLv1.1 Micro-Shared + E Full-Rank Routed Experts + Unit-Sum Adaptive Top-K + Bipolar Graph."""

    def __init__(self, cfg: AVLPMLConfig, moe_slot_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.moe_slot_idx = moe_slot_idx
        self.num_experts = cfg.num_routed_experts
        self.probe_rank = cfg.probe_rank
        self.hidden_size = cfg.hidden_size

        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.shared_expert = SwiGLU(cfg.hidden_size, cfg.shared_expert_intermediate_size)
        self.routed_experts = nn.ModuleList(
            [SwiGLU(cfg.hidden_size, cfg.routed_expert_intermediate_size) for _ in range(self.num_experts)]
        )

        # Lightweight r-rank signature projection ONLY for Track-2 bipolar pain/synergy prototype matching
        # (NO dense probe_v / probe_mix stealing gradients from routed full-rank experts!)
        u_init = torch.empty(self.num_experts, cfg.hidden_size, self.probe_rank)
        with torch.no_grad():
            if cfg.hidden_size >= 1024:
                # Fast orthogonal init for large 4.5B/8.9B models; exact GPU SVD of real upcycled weights runs in surgery!
                for i in range(self.num_experts):
                    u_init[i].copy_(
                        nn.init.orthogonal_(torch.empty(cfg.hidden_size, self.probe_rank, dtype=torch.float32)).to(u_init.dtype)
                    )
            else:
                for i, exp in enumerate(self.routed_experts):
                    u_svd, _, _ = torch.linalg.svd(exp.w1.weight.t().float(), full_matrices=False)
                    u_init[i] = u_svd[:, : self.probe_rank].contiguous()
        self.probe_u = nn.Parameter(u_init)

        # End-to-end differentiable router projection
        self.static_router = nn.Linear(cfg.hidden_size, self.num_experts, bias=False)

        # Group D manifold & adaptive depth threshold scalars (FP32, weight_decay = 0.0)
        self.b_stat = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        self.tau_0 = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.w_t = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.w_e = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.w_u = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.lambda_pos = nn.Parameter(torch.tensor(0.40, dtype=torch.float32))
        self.lambda_neg = nn.Parameter(torch.tensor(1.00, dtype=torch.float32))

        # Token-level linkage gate balancing local routing vs Bipolar Sinkhorn Graph
        self.w_lk = nn.Linear(cfg.hidden_size, 1, bias=True)
        nn.init.zeros_(self.w_lk.weight)
        nn.init.constant_(self.w_lk.bias, -0.5)

        self.film_tau = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(self.film_tau.weight)

        self.graph = BipolarLinkageGraph(self.num_experts)
        self.register_buffer("alpha_upcycle", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("u_post_warmup", torch.tensor(1.0, dtype=torch.float32))

    def compute_probes(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return lightweight Track-2 experience signature z [B, S, E, r] (no dense probe_mix)."""
        z = F.silu(torch.einsum("bsd,edr->bser", x_norm, self.probe_u))
        return z, None

    def _sparse_topk_swiglu(
        self, x_norm: torch.Tensor, top_idx: torch.Tensor, top_weights: torch.Tensor
    ) -> torch.Tensor:
        """Sparse SwiGLU over selected K experts:
        - Small models (D < 1024): L2-cache-resident sync-free stacked einsum (CUDAGraph-safe for Stage 0 40M!).
        - Large models (D >= 1024): True token-dispatched sparse calls to routed_experts[e](x_e), avoiding 4.8 GB
          weight stacking overhead and directly supporting 4-bit Q-AVL-PML quantized Linear layers!
        """
        if self.hidden_size >= 1024:
            bsz, seq_len, dim = x_norm.shape
            k_sel = top_idx.shape[-1]
            n_tok = bsz * seq_len
            flat_x = x_norm.reshape(n_tok, dim)
            flat_idx = top_idx.reshape(n_tok, k_sel)
            flat_w = top_weights.to(x_norm.dtype).reshape(n_tok, k_sel)
            out_flat = torch.zeros_like(flat_x)
            if n_tok == 1:
                # Zero-nonzero single-token AR fast-path (eliminates 8x GPU-CPU stream syncs per layer!)
                idx_row = flat_idx[0].tolist()
                w_row = flat_w[0].float().tolist()
                for k_pos, (e_id, w_val) in enumerate(zip(idx_row, w_row)):
                    if w_val > 0.0:
                        out_flat = out_flat + self.routed_experts[e_id](flat_x).to(out_flat.dtype) * flat_w[:, k_pos : k_pos + 1]
                return out_flat.view(bsz, seq_len, dim)

            # Single-sync sorted bin dispatch across all E experts (zero per-expert torch.nonzero stalls!)
            bins = torch.where(flat_w > 0.0, flat_idx, torch.full_like(flat_idx, self.num_experts)).reshape(-1)
            counts = torch.bincount(bins, minlength=self.num_experts + 1)[: self.num_experts].tolist()
            perm = torch.argsort(bins)
            tok_ids_all = torch.div(perm, k_sel, rounding_mode="floor")
            flat_w_1d = flat_w.reshape(-1)
            offset = 0
            for e_id, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                sel_perm = perm[offset : offset + cnt]
                sel_tok = tok_ids_all[offset : offset + cnt]
                offset += cnt
                x_e = flat_x.index_select(0, sel_tok)
                w_e = flat_w_1d.index_select(0, sel_perm).unsqueeze(-1)
                y_e = self.routed_experts[e_id](x_e).to(out_flat.dtype) * w_e
                out_flat.index_add_(0, sel_tok, y_e)
            return out_flat.view(bsz, seq_len, dim)

        w1_all = torch.stack([e.w1.weight for e in self.routed_experts], dim=0)  # [E, H, D]
        w3_all = torch.stack([e.w3.weight for e in self.routed_experts], dim=0)  # [E, H, D]
        w2_all = torch.stack([e.w2.weight for e in self.routed_experts], dim=0)  # [E, D, H]

        idx_h = top_idx.unsqueeze(-1).expand(-1, -1, -1, w1_all.shape[1])        # [B, S, K, H]
        h1_sel = torch.gather(torch.einsum("bsd,ehd->bseh", x_norm, w1_all), 2, idx_h)
        h3_sel = torch.gather(torch.einsum("bsd,ehd->bseh", x_norm, w3_all), 2, idx_h)
        act_k = F.silu(h1_sel) * h3_sel                                          # [B, S, K, H]

        act_e = torch.zeros(
            x_norm.shape[0],
            x_norm.shape[1],
            self.num_experts,
            w1_all.shape[1],
            device=x_norm.device,
            dtype=x_norm.dtype,
        )
        act_e.scatter_add_(2, idx_h, act_k * top_weights.to(x_norm.dtype).unsqueeze(-1))
        return torch.einsum("bseh,edh->bsd", act_e, w2_all)

    def forward(
        self,
        x: torch.Tensor,
        phase: CognitivePhase = CognitivePhase.FORMAL,
        t_k: float = 0.0,
        r_clean: Optional[torch.Tensor] = None,
        alpha_eng_eff: Optional[torch.Tensor] = None,
        u_epistemic: Optional[torch.Tensor] = None,
        u_acute: Optional[torch.Tensor] = None,
        d_b: Optional[torch.Tensor] = None,
        exp_state: Optional[Track2ExperienceState] = None,
        a_prev_step: Optional[torch.Tensor] = None,
        a_prev_layer: Optional[torch.Tensor] = None,
        prior_dropout_prob: float = 0.0,
        static_top2_mode: bool = False,
        disable_bipolar_graph: bool = False,
        disable_pain_imprint: bool = False,
        cache_utility: bool = False,
        utility_mask: Optional[torch.Tensor] = None,
        precomputed_manifolds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        full_sum_eval: bool = False,
        causal_routing: bool = False,
        segment_specs: Optional[List[Tuple[int, bool]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len, _ = x.shape
        x_norm = self.norm(x)
        y_shared = self.shared_expert(x_norm)

        # Exact full-partition conservation check (512 shared + sum_{e=0..7} 1152 routed == exact 9728 Qwen3 MLP!)
        if full_sum_eval:
            y_moe = y_shared
            for exp in self.routed_experts:
                y_moe = y_moe + exp(x_norm)
            w_unif = torch.full((bsz, seq_len, self.num_experts), 1.0 / self.num_experts, device=x.device, dtype=x.dtype)
            aux_full = {
                "scores": torch.zeros(bsz, seq_len, self.num_experts, device=x.device),
                "gates": torch.ones(bsz, seq_len, self.num_experts, device=x.device),
                "w_route": w_unif,
                "probes_z": None,
                "block_profile": w_unif.float().mean(dim=1),
                "active_count": torch.full((bsz, seq_len), float(self.num_experts), device=x.device),
                "graph_manifolds": None,
            }
            return x + y_moe, aux_full

        # Equal-parameter Static Top-2 MoE Baseline mode
        if static_top2_mode:
            router_logits = self.static_router(x_norm).float() + (self.b_stat - self.b_stat.mean()).view(1, 1, -1)
            k_sel = 0 if (phase == CognitivePhase.PRE or self.cfg.k_max <= 0) else min(2, self.num_experts)
            w_route = F.softmax(router_logits, dim=-1).to(x.dtype)
            if k_sel == 0:
                y_moe = y_shared
                gates = torch.zeros(bsz, seq_len, self.num_experts, device=x.device)
            else:
                topv, topi = torch.topk(router_logits, k=k_sel, dim=-1)
                topw = F.softmax(topv, dim=-1).to(x.dtype)
                gates = torch.zeros(bsz, seq_len, self.num_experts, device=x.device, dtype=x.dtype)
                gates.scatter_(-1, topi, topw)
                routed_sum = self._sparse_topk_swiglu(x_norm, topi, topw)
                y_moe = y_shared + self.alpha_upcycle.to(x.dtype) * routed_sum
            aux_static = {
                "scores": router_logits,
                "gates": gates.float(),
                "w_route": w_route,
                "probes_z": torch.zeros(bsz, seq_len, self.num_experts, self.probe_rank, device=x.device),
                "block_profile": w_route.float().mean(dim=1),
                "active_count": (gates > 0.0).float().sum(dim=-1),
                "y_moe_norm": y_moe.detach().float().norm(dim=-1).mean(),
            }
            return x + y_moe, aux_static

        # Lightweight Track-2 experience signature z (only computed when exp_state is active or needed!)
        z: Optional[torch.Tensor] = None

        # End-to-end differentiable router score (NO x_norm.detach() & NO noisy L_util score!)
        b_zero_mean = self.b_stat - self.b_stat.mean()
        g_prior = self.static_router(x_norm).float() + b_zero_mean.view(1, 1, -1)
        delta_b_pre = (
            exp_state.delta_b_pre[:, self.moe_slot_idx, :].unsqueeze(1).float()
            if exp_state is not None
            else 0.0
        )
        s_base = g_prior + delta_b_pre

        # Track-2 bipolar prototype score s_mem (with c_pain_max = 0.60 cap)
        if exp_state is not None:
            z, _ = self.compute_probes(x_norm.detach())
            p_pos = exp_state.p_pos[:, self.moe_slot_idx].unsqueeze(1).float()
            p_neg = exp_state.p_neg[:, self.moe_slot_idx].unsqueeze(1).float()
            z_f = F.normalize(z.float(), dim=-1, eps=1e-6)
            pos_has = (p_pos.norm(dim=-1) > 1e-5).float()
            cos_pos = F.relu((z_f * F.normalize(p_pos, dim=-1, eps=1e-6)).sum(dim=-1)) * pos_has
            if disable_pain_imprint:
                s_mem = self.lambda_pos.abs() * cos_pos
            else:
                p_norm = p_neg.norm(dim=-1)
                neg_has = (p_norm > 1e-5).float()
                # Preserve exact gamma_pain^t norm decay instead of letting F.normalize cancel it to 1.0!
                p_strength = p_norm.clamp(0.0, 1.0)
                cos_neg = F.relu((z_f * F.normalize(p_neg, dim=-1, eps=1e-6)).sum(dim=-1)) * neg_has
                cos_neg_capped = torch.clamp(cos_neg, min=0.0, max=self.cfg.c_pain_max) * p_strength
                s_mem = self.lambda_pos.abs() * cos_pos - self.lambda_neg.abs() * cos_neg_capped
        else:
            s_mem = 0.0

        # Proven Bipolar Sinkhorn Linkage Graph single-hop damped modulation (0.40 intra + 0.45 inter < 1)
        grouped_segs: Optional[List[Tuple[int, bool, int]]] = None
        if segment_specs is not None:
            grouped_segs = []
            for seg_len, seg_causal in segment_specs:
                if seg_len <= 0:
                    continue
                if (not seg_causal) and grouped_segs and (not grouped_segs[-1][1]) and grouped_segs[-1][0] == seg_len:
                    prev_l, _, prev_cnt = grouped_segs[-1]
                    grouped_segs[-1] = (prev_l, False, prev_cnt + 1)
                else:
                    grouped_segs.append((seg_len, seg_causal, 1))

        cached_manifolds = None
        if disable_bipolar_graph:
            scores = s_base + s_mem
        else:
            m_syn, m_adv, m_inter = (
                precomputed_manifolds if precomputed_manifolds is not None else self.graph.get_manifolds()
            )
            cached_manifolds = (m_syn, m_adv, m_inter)
            if exp_state is not None:
                m_eff_adv = row_normalize_overlay(m_adv.unsqueeze(0), exp_state.delta_m_adv[:, self.moe_slot_idx])
                m_eff_syn = row_normalize_overlay(m_syn.unsqueeze(0), exp_state.delta_m_syn[:, self.moe_slot_idx])
            else:
                m_eff_adv = m_adv.unsqueeze(0).expand(bsz, -1, -1)
                m_eff_syn = m_syn.unsqueeze(0).expand(bsz, -1, -1)

            if phase == CognitivePhase.POST:
                adv_w_3d: Optional[torch.Tensor] = None
                m_eff = m_eff_adv
            elif d_b is not None and d_b.dim() == 3:
                adv_w_3d = (d_b[:, :, 3].detach() >= 0.75).float().unsqueeze(-1)  # [B, S, 1]
                m_eff = m_eff_syn
            elif d_b is not None:
                adv_w_3d = None
                adv_w_b = (d_b[:, 3].detach() >= 0.75).float().view(bsz, 1, 1)
                m_eff = adv_w_b * m_eff_adv + (1.0 - adv_w_b) * m_eff_syn
            else:
                adv_w_3d = None
                m_eff = m_eff_syn

            if grouped_segs is not None:
                # Strict per-segment isolation (vectorized across consecutive equal-width blocks!):
                c_intra_parts: List[torch.Tensor] = []
                c_inter_parts: List[torch.Tensor] = []
                off_s = 0
                for seg_len, seg_causal, seg_cnt in grouped_segs:
                    tot_seg = seg_len * seg_cnt
                    end_s = off_s + tot_seg
                    sb_seg = s_base[:, off_s:end_s, :]
                    if seg_causal:
                        a_tok = F.softmax(sb_seg, dim=-1)
                        ci_syn = torch.einsum("bij,bsi->bsj", m_eff_syn, a_tok)
                        if adv_w_3d is not None:
                            ci_adv = torch.einsum("bij,bsi->bsj", m_eff_adv, a_tok)
                            ci_seg = adv_w_3d[:, off_s:end_s, :] * ci_adv + (1.0 - adv_w_3d[:, off_s:end_s, :]) * ci_syn
                        else:
                            ci_seg = torch.einsum("bij,bsi->bsj", m_eff, a_tok)
                        ce_seg = (
                            torch.einsum("ij,bsi->bsj", m_inter, a_prev_layer[:, off_s:end_s, :].float())
                            if (a_prev_layer is not None and a_prev_layer.dim() == 3)
                            else torch.zeros_like(ci_seg)
                        )
                    else:
                        sb_grp = sb_seg.reshape(bsz, seg_cnt, seg_len, self.num_experts)
                        a_frm = F.softmax(sb_grp.mean(dim=2), dim=-1)  # [B, K, E] strictly frame-local!
                        if adv_w_3d is not None:
                            ci_syn_f = (
                                torch.einsum("bij,bki->bkj", m_eff_syn, a_frm)
                                .unsqueeze(2)
                                .expand(-1, -1, seg_len, -1)
                                .reshape(bsz, tot_seg, self.num_experts)
                            )
                            ci_adv_f = (
                                torch.einsum("bij,bki->bkj", m_eff_adv, a_frm)
                                .unsqueeze(2)
                                .expand(-1, -1, seg_len, -1)
                                .reshape(bsz, tot_seg, self.num_experts)
                            )
                            ci_seg = adv_w_3d[:, off_s:end_s, :] * ci_adv_f + (1.0 - adv_w_3d[:, off_s:end_s, :]) * ci_syn_f
                        else:
                            ci_seg = (
                                torch.einsum("bij,bki->bkj", m_eff, a_frm)
                                .unsqueeze(2)
                                .expand(-1, -1, seg_len, -1)
                                .reshape(bsz, tot_seg, self.num_experts)
                            )
                        ce_seg = (
                            torch.einsum("ij,bki->bkj", m_inter, a_prev_layer[:, off_s:end_s:seg_len, :].float())
                            .unsqueeze(2)
                            .expand(-1, -1, seg_len, -1)
                            .reshape(bsz, tot_seg, self.num_experts)
                            if (a_prev_layer is not None and a_prev_layer.dim() == 3)
                            else torch.zeros_like(ci_seg)
                        )
                    c_intra_parts.append(ci_seg)
                    c_inter_parts.append(ce_seg)
                    off_s = end_s
                c_intra = torch.cat(c_intra_parts, dim=1)
                c_inter = torch.cat(c_inter_parts, dim=1)
            elif causal_routing:
                # Strictly token-local causal routing: zero future-token pooling in AR mode & exact 1-tok KV parity!
                a_src_tok = F.softmax(s_base, dim=-1)  # [B, S, E]
                c_intra = torch.einsum("bij,bsi->bsj", m_eff, a_src_tok)
                c_inter = (
                    torch.einsum("ij,bsi->bsj", m_inter, a_prev_layer.float())
                    if (a_prev_layer is not None and a_prev_layer.dim() == 3)
                    else torch.zeros_like(c_intra)
                )
            else:
                a_src = F.softmax(s_base.mean(dim=1), dim=-1) if a_prev_step is None else a_prev_step.float()
                c_intra = torch.einsum("bij,bi->bj", m_eff, a_src).unsqueeze(1)
                c_inter = (
                    torch.einsum("ij,bi->bj", m_inter, a_prev_layer.float()).unsqueeze(1)
                    if a_prev_layer is not None
                    else torch.zeros_like(c_intra)
                )
            lk_gate = torch.sigmoid(self.w_lk(x_norm).float())
            scores = s_base + s_mem + lk_gate * (0.40 * c_intra + 0.45 * c_inter)

        # Guardrail 1: Three-factor adaptive depth threshold + Step-1 Engram ceiling
        u_epi = u_epistemic.float() if u_epistemic is not None else torch.zeros(bsz, seq_len, device=x.device)
        a_eng = alpha_eng_eff.float() if alpha_eng_eff is not None else torch.zeros(bsz, seq_len, device=x.device)
        if d_b is not None and d_b.dim() == 3:
            delta_film: float | torch.Tensor = self.film_tau(d_b.to(x.dtype)).float().squeeze(-1)
        elif d_b is not None:
            delta_film = self.film_tau(d_b.to(x.dtype)).float().squeeze(-1).unsqueeze(1)
        else:
            delta_film = 0.0

        t_k_term = t_k.float() if isinstance(t_k, torch.Tensor) else float(t_k)
        tau = (
            self.tau_0
            + self.w_t.abs() * t_k_term
            + self.w_e.abs() * a_eng
            - self.w_u.abs() * u_epi
            + delta_film
        ).unsqueeze(-1)

        if phase == CognitivePhase.PRE or self.cfg.k_max <= 0:
            k_cap = 0
            tau_eff = torch.full_like(tau, float("inf"))
        else:
            k_full = min(self.cfg.k_max, self.num_experts)
            k_step1 = min(self.cfg.k_max, 2, self.num_experts)
            k_cap = k_full
            sorted_s, _ = torch.sort(scores, dim=-1, descending=True)
            tau_full = tau if k_full >= self.num_experts else torch.maximum(tau, sorted_s[..., k_full : k_full + 1])
            tau_step1 = tau if k_step1 >= self.num_experts else torch.maximum(tau, sorted_s[..., k_step1 : k_step1 + 1])
            if r_clean is not None:
                rc_3d = r_clean.float().unsqueeze(-1) if r_clean.dim() == 2 else r_clean.float()
                is_s1 = rc_3d < 0.25
                tau_eff = torch.where(is_s1, tau_step1, tau_full)
            elif (not isinstance(t_k, torch.Tensor)) and (1.0 - float(t_k)) < 0.25:
                tau_eff = tau_step1
            else:
                tau_eff = tau_full

        margin_pos = F.relu(scores - tau_eff)
        kappa_eff = min(self.cfg.kappa_gate, 0.20)
        gates = margin_pos / (margin_pos + kappa_eff)

        # Guardrail 2: 8-Expert Small-Pool Pain Deadlock Protection (K_min_guard = 1)
        tk_low_mask: Optional[torch.Tensor] = (
            (t_k.float() <= 0.50) if isinstance(t_k, torch.Tensor) else None
        )
        if k_cap > 0 and phase in (CognitivePhase.FORMAL, CognitivePhase.POST) and (
            tk_low_mask is not None or float(t_k) <= 0.50  # type: ignore[arg-type]
        ):
            zero_active = (gates > 0.0).sum(dim=-1) == 0
            need_rescue = zero_active & (u_epi > 0.35)
            if tk_low_mask is not None:
                need_rescue = need_rescue & tk_low_mask
            if exp_state is not None and not disable_pain_imprint:
                cul = exp_state.culprit_mask[:, self.moe_slot_idx, :].unsqueeze(1)
                all_cul = cul.all(dim=-1, keepdim=True)
                masked_s = torch.where(cul & (~all_cul), torch.full_like(scores, -1e9), scores)
            else:
                masked_s = scores
            rescue_idx = masked_s.argmax(dim=-1, keepdim=True)
            rescue_gate = torch.zeros_like(gates).scatter_(-1, rescue_idx, self.cfg.g_min_guard)
            gates = torch.where(need_rescue.unsqueeze(-1), rescue_gate, gates)

        w_route = F.softmax(scores / self.cfg.router_temp, dim=-1).to(x.dtype)

        # Capacity-preserving non-attenuating gate normalization:
        # - At k=1 (g_sum <= 1): g_unit = g_1 (preserves direct L_diff gradient into router with zero attenuation!)
        # - At k=2 (g_sum > 1):  g_unit = g / g_sum (strictly unit-sum == 1.0, preventing residual magnitude explosion!)
        g_sum = gates.sum(dim=-1, keepdim=True)
        g_unit = gates / g_sum.clamp(min=1.0)

        if phase == CognitivePhase.POST:
            rho_guide = (1.0 + 0.25 * torch.sigmoid(4.0 * (u_epi - 0.50))).clamp(1.0, 1.25).to(x.dtype).unsqueeze(-1)
        else:
            rho_guide = (1.0 + 0.25 * (u_epi > 0.50).float() * torch.sigmoid(4.0 * (u_epi - 0.50))).clamp(1.0, 1.25).to(x.dtype).unsqueeze(-1)

        if k_cap > 0:
            k_eval = min(k_cap, self.num_experts)
            top_g, top_i = torch.topk(g_unit, k=k_eval, dim=-1)
            eff_w = top_g.to(x.dtype) * rho_guide
            routed_out = self._sparse_topk_swiglu(x_norm, top_i, eff_w)
            y_moe = y_shared + self.alpha_upcycle.to(x.dtype) * routed_out
        else:
            y_moe = y_shared

        out = x + y_moe
        if grouped_segs is not None:
            seg_profiles: List[torch.Tensor] = []
            off_p = 0
            for seg_len, seg_causal, seg_cnt in grouped_segs:
                tot_seg = seg_len * seg_cnt
                end_p = off_p + tot_seg
                if seg_causal:
                    seg_profiles.append(w_route[:, off_p:end_p, :].float())
                else:
                    pi_f = F.softmax(u_epi[:, off_p:end_p].reshape(bsz, seg_cnt, seg_len), dim=-1)
                    wr_grp = w_route[:, off_p:end_p, :].float().reshape(bsz, seg_cnt, seg_len, self.num_experts)
                    bp_f = torch.einsum("bkl,bkle->bke", pi_f.float(), wr_grp)
                    seg_profiles.append(
                        bp_f.unsqueeze(2).expand(-1, -1, seg_len, -1).reshape(bsz, tot_seg, self.num_experts)
                    )
                off_p = end_p
            seg_relay = torch.cat(seg_profiles, dim=1)  # [B, S, E]
            block_profile = seg_relay.mean(dim=1)
        else:
            pi_w = F.softmax(u_epi, dim=-1)
            block_profile = torch.einsum("bs,bse->be", pi_w.float(), w_route.float())
            seg_relay = None

        aux = {
            "scores": scores,
            "gates": gates,
            "w_route": w_route,
            "probes_z": z,
            "block_profile": block_profile,
            "seg_relay": seg_relay,
            "active_count": (gates > 0.0).float().sum(dim=-1),
            "graph_manifolds": cached_manifolds,
        }
        return out, aux

    def load_balance_loss(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Switch/DeepSeek-style auxiliary load-balancing loss so all E experts specialize cleanly."""
        w_prob = aux["w_route"].float().mean(dim=(0, 1))                # [E]
        active_frac = (aux["gates"].float() > 0.0).float().mean(dim=(0, 1))
        active_norm = active_frac / active_frac.sum().clamp(min=1e-6)
        return self.num_experts * (w_prob * active_norm).sum()
