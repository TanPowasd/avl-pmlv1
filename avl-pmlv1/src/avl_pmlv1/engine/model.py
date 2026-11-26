"""Complete AVL-PMLv1 Backbone Assembly, Ablation Switches, & Delayed Scorer Micro-Step."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AVLPMLConfig
from ..diffusion.dir_vector import DirectionVectorHead
from ..engram.static_engram import StaticEngram
from ..moe_track2.bipolar_graph import sinkhorn_log_domain_batched
from ..moe_track2.hyper_router import HyperRouter, Track2ExperienceState
from ..moe_track2.tri_state_moe import DenseSwiGLU, TriStateMoELayer
from ..pml_track1.attn_pml_layer import AttnPMLHybridLayer
from ..pml_track1.whiteboard import RMSNorm, Track1PMLModule, WhiteboardState
from ..vocab.fsq_codec import FactoredSymbolicEmbedding
from ..vocab.sym_vocab import CognitivePhase, SymbolicVocab, extract_prompt_number_spans

_HEADER_STOP_WORDS = {
    "problem", "math", "logic", "code", "thought", "the", "a", "an", "if", "in", "on", "at",
    "when", "what", "how", "who", "which", "is", "are", "suppose", "assume", "given", "find",
}


def build_nl_ro_span_weights(
    prompt_nl_ids: torch.Tensor,
    tokenizer: Any = None,
    num_ro_slots: int = 64,
) -> torch.Tensor:
    """Build [B, N_ro, p_len] normalized pooling weights mapping prompt numbers & entities into S^RO slots:
    - Priority 1: Contiguous numeric token spans in order of appearance (1:1 aligned with ⟨IN_0..k⟩ in Math/TIR);
    - Priority 2: Capitalized entity / content spans for Logic problems when < 2 numeric spans exist;
    - Fallback: Deterministic token/window partition when tokenizer is None or dummy (preserves <1e-5 KV parity!).
    """
    bsz, p_len = prompt_nl_ids.shape
    device = prompt_nl_ids.device
    weights = torch.zeros((bsz, num_ro_slots, p_len), dtype=torch.float32, device=device)
    if p_len == 0 or num_ro_slots == 0:
        return weights

    has_real_tok = tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens")
    ids_cpu = prompt_nl_ids.detach().cpu().tolist()

    for b in range(bsz):
        row_ids = ids_cpu[b]
        spans: List[List[int]] = []
        if has_real_tok:
            try:
                pieces = [tokenizer.decode([tid], skip_special_tokens=False) for tid in row_ids]
            except Exception:
                pieces = []
            if len(pieces) == p_len:
                # Exclude "Problem (<dom>):\n" prefix and "\nThought:\n" suffix from span extraction
                full_so_far = ""
                body_start = 0
                body_end = p_len
                acc_lens: List[Tuple[int, int]] = []
                for idx_p, pc in enumerate(pieces):
                    s_c = len(full_so_far)
                    full_so_far += pc
                    acc_lens.append((s_c, len(full_so_far)))
                m_hdr = re.match(r"^Problem\s*\([^)]*\)\s*:\s*\n?", full_so_far)
                hdr_end_char = m_hdr.end() if m_hdr else 0
                if m_hdr:
                    for idx_p, (_, e_c) in enumerate(acc_lens):
                        if e_c > hdr_end_char:
                            body_start = idx_p
                            break
                m_tail = re.search(r"\n?Thought\s*:\s*$", full_so_far)
                tail_start_char = m_tail.start() if m_tail else len(full_so_far)
                if m_tail:
                    for idx_p, (s_c, _) in enumerate(acc_lens):
                        if s_c >= tail_start_char:
                            body_end = idx_p
                            break

                # 1. Exact character-span numeric matching (1:1 aligned with compiler ⟨IN_k⟩ & Gate-2 executor!)
                if tail_start_char > hdr_end_char:
                    body_text = full_so_far[hdr_end_char:tail_start_char]
                    for rel_s, rel_e, _ in extract_prompt_number_spans(body_text):
                        abs_s = hdr_end_char + rel_s
                        abs_e = hdr_end_char + rel_e
                        num_toks = [
                            i
                            for i in range(body_start, body_end)
                            if acc_lens[i][1] > abs_s and acc_lens[i][0] < abs_e
                        ]
                        if num_toks:
                            spans.append(num_toks)
                            if len(spans) >= num_ro_slots:
                                break

                # 2. Capitalized / entity word spans for Logic problems (e.g., Knights & Knaves / ProofWriter)
                if len(spans) < 2:
                    used_pos = {p_i for sp in spans for p_i in sp}
                    cur_ent: List[int] = []
                    for i in range(body_start, body_end):
                        if i in used_pos:
                            if cur_ent:
                                spans.append(cur_ent)
                                cur_ent = []
                            continue
                        raw_p = pieces[i]
                        clean_w = re.sub(r"[^A-Za-z]", "", raw_p)
                        starts_new_word = raw_p.startswith((" ", "\n", "\t")) or i == body_start
                        if clean_w:
                            if starts_new_word:
                                if cur_ent:
                                    w_txt = "".join(re.sub(r"[^A-Za-z]", "", pieces[k]) for k in cur_ent)
                                    if len(w_txt) >= 2 and w_txt.lower() not in _HEADER_STOP_WORDS:
                                        spans.append(cur_ent)
                                        if len(spans) >= num_ro_slots:
                                            cur_ent = []
                                            break
                                    cur_ent = []
                                if clean_w[0].isupper() or len(clean_w) >= 4:
                                    cur_ent = [i]
                            elif cur_ent:
                                cur_ent.append(i)
                        else:
                            if cur_ent:
                                w_txt = "".join(re.sub(r"[^A-Za-z]", "", pieces[k]) for k in cur_ent)
                                if len(w_txt) >= 2 and w_txt.lower() not in _HEADER_STOP_WORDS:
                                    spans.append(cur_ent)
                                    if len(spans) >= num_ro_slots:
                                        cur_ent = []
                                        break
                                cur_ent = []
                    if cur_ent and len(spans) < num_ro_slots:
                        w_txt = "".join(re.sub(r"[^A-Za-z]", "", pieces[k]) for k in cur_ent)
                        if len(w_txt) >= 2 and w_txt.lower() not in _HEADER_STOP_WORDS:
                            spans.append(cur_ent)

        if not spans:
            # Deterministic fallback: partition [0..p_len) into <= num_ro_slots contiguous spans
            n_fill = min(p_len, num_ro_slots)
            for k in range(n_fill):
                s_i = (k * p_len) // n_fill
                e_i = max(s_i + 1, ((k + 1) * p_len) // n_fill)
                spans.append(list(range(s_i, min(p_len, e_i))))

        for slot_k, pos_list in enumerate(spans[:num_ro_slots]):
            if not pos_list:
                continue
            inv = 1.0 / float(len(pos_list))
            for p_i in pos_list:
                weights[b, slot_k, p_i] = inv

    return weights


def build_dual_domain_mask(
    p_len: int,
    sym_frame_sizes: List[int],
    ans_len: int,
    gen_mode: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build exact attention mask for single-stream [Prompt(V_nl) | Thinking(V_sym) | Answer(V_nl)]:
      - gen_mode == 'ar'        : returns None (uses fused causal SDPA across the entire sequence!).
      - gen_mode == 'block_dllm': causal within Prompt_nl, block-bidirectional within each V_sym frame
                                  (attending to full Prompt_nl + all earlier V_sym frames), and causal
                                  within Answer_nl (attending to full Prompt_nl + all V_sym frames).
    """
    if gen_mode == "ar":
        return None
    s_sym = sum(sym_frame_sizes)
    tot = p_len + s_sym + ans_len
    mask = torch.tril(torch.ones((tot, tot), dtype=torch.bool, device=device))
    off = p_len
    for flen in sym_frame_sizes:
        mask[off : off + flen, off : off + flen] = True
        off += flen
    return mask


def build_dual_domain_bd3_mask(
    p_len: int,
    sym_frame_sizes: List[int],
    ans_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact BD3-LM Dual-Stream [Prompt_nl (P) | Y_clean (S) | Y_noisy (S) | Answer_nl (A)] mask & RoPE positions:
      - Prompt_nl [0 : P]           : strictly causal;
      - Y_clean frame m [cs : ce]   : attends to Prompt_nl + Y_clean[0 .. m] (bidirectional inside clean frame m);
      - Y_noisy frame m [ns : ne]   : attends to Prompt_nl + Y_clean[0 .. m-1] + Y_noisy[m] ONLY (zero clean self-leak,
                                      zero future-frame leak, zero Answer_nl leak!);
      - Answer_nl [P+2S : P+2S+A]   : attends to Prompt_nl + full Y_clean + causal Answer_nl (zero Y_noisy leak!).
    """
    s_sym = sum(sym_frame_sizes)
    tot = p_len + 2 * s_sym + ans_len
    mask = torch.zeros((tot, tot), dtype=torch.bool, device=device)

    # 1. Causal Prompt_nl
    mask[:p_len, :p_len] = torch.tril(torch.ones((p_len, p_len), dtype=torch.bool, device=device))

    # 2. Y_clean & Y_noisy per exact physical frame (8 or 16 slots)
    c_base = p_len
    n_base = p_len + s_sym
    off = 0
    for flen in sym_frame_sizes:
        cs, ce = c_base + off, c_base + off + flen
        ns, ne = n_base + off, n_base + off + flen
        mask[cs:ce, :p_len] = True
        mask[cs:ce, c_base:ce] = True

        mask[ns:ne, :p_len] = True
        if off > 0:
            mask[ns:ne, c_base:cs] = True
        mask[ns:ne, ns:ne] = True
        off += flen

    # 3. Causal Answer_nl conditioned strictly on [Prompt_nl | Y_clean]
    if ans_len > 0:
        a_base = p_len + 2 * s_sym
        mask[a_base:, :p_len] = True
        mask[a_base:, c_base : c_base + s_sym] = True
        mask[a_base:, a_base:] = torch.tril(torch.ones((ans_len, ans_len), dtype=torch.bool, device=device))

    pos_p = torch.arange(0, p_len, device=device)
    pos_s = torch.arange(p_len, p_len + s_sym, device=device)
    pos_a = torch.arange(p_len + s_sym, p_len + s_sym + ans_len, device=device)
    rope_pos = torch.cat([pos_p, pos_s, pos_s, pos_a], dim=0).unsqueeze(0)
    return mask, rope_pos


@dataclass
class AblationConfig:
    static_top2_baseline: bool = False
    disable_engram: bool = False
    disable_pml: bool = False
    disable_bipolar_graph: bool = False
    disable_pain_imprint: bool = False
    disable_wet_cement: bool = False


class DualSourceCopyGate(nn.Module):
    """Verbalize phase pointer copy-gate over Prompt spans + S^RO/S^RW slots for lossless literals."""

    def __init__(self, cfg: AVLPMLConfig) -> None:
        super().__init__()
        self.q_copy = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_copy = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.gate = nn.Linear(cfg.hidden_size, 1, bias=True)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self,
        h_verb: torch.Tensor,
        vocab_logits: torch.Tensor,
        prompt_hidden: Optional[torch.Tensor] = None,
        prompt_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prompt_hidden is None or prompt_token_ids is None:
            return vocab_logits
        g_cp = torch.sigmoid(self.gate(h_verb))
        scores = torch.matmul(self.q_copy(h_verb), self.k_copy(prompt_hidden).transpose(-1, -2)) / (
            h_verb.shape[-1] ** 0.5
        )
        p_copy = F.softmax(scores.float(), dim=-1).to(vocab_logits.dtype)
        p_gen = F.softmax(vocab_logits.float(), dim=-1).to(vocab_logits.dtype)

        blended = (1.0 - g_cp) * p_gen
        idx = prompt_token_ids.unsqueeze(1).expand(-1, h_verb.shape[1], -1)
        blended = blended.scatter_add(-1, idx, g_cp * p_copy)
        return torch.log(blended.clamp(min=1e-9))


class AVLPMLModel(nn.Module):
    """Unified AVL-PMLv1 pure-forward architecture (40M / 4.5B / 8.9B)."""

    def __init__(self, cfg: AVLPMLConfig, ablation: Optional[AblationConfig] = None) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.ablation = ablation or AblationConfig()
        self.vocab = SymbolicVocab(cfg)

        self.sym_embed_head = FactoredSymbolicEmbedding(cfg)
        self.nl_embed = nn.Embedding(cfg.nl_vocab_size, cfg.hidden_size)
        nn.init.normal_(self.nl_embed.weight, std=1.0 / (cfg.hidden_size ** 0.5))
        self.nl_head = nn.Linear(cfg.hidden_size, cfg.nl_vocab_size, bias=False)
        self.nl_head.weight = self.nl_embed.weight
        self.copy_gate = DualSourceCopyGate(cfg)

        self.in_span_tag = nn.Embedding(cfg.num_ro_slots, cfg.hidden_size)
        nn.init.normal_(self.in_span_tag.weight, std=0.35 / (cfg.hidden_size ** 0.5))
        self.alpha_span_tag = nn.Parameter(torch.tensor([0.5]))

        # Explicit intra-chunk slot (0..15) embedding breaks all-[MASK] symmetry at Layer 0.
        # Stage 2+ (use_block_idx_embed=False) removes absolute block_idx_embed -> pure relative RoPE infinite extrapolation!
        self.chunk_slot_embed = nn.Embedding(cfg.atomic_chunk_size, cfg.hidden_size)
        slot_scale = 0.15 if cfg.cold_start_graft else 0.50
        nn.init.normal_(self.chunk_slot_embed.weight, std=slot_scale / (cfg.hidden_size ** 0.5))
        self.use_block_idx_embed = cfg.use_block_idx_embed
        if self.use_block_idx_embed:
            self.block_idx_embed: Optional[nn.Embedding] = nn.Embedding(64, cfg.hidden_size)
            nn.init.normal_(self.block_idx_embed.weight, std=0.5 / (cfg.hidden_size ** 0.5))
        else:
            self.block_idx_embed = None
        self.gradient_checkpointing: bool = False

        self.sc_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        nn.init.zeros_(self.sc_proj.weight)

        self.engram_map: Dict[int, str] = {}
        self.engrams = nn.ModuleDict()
        for l_idx in cfg.engram_layer_indices:
            key = str(l_idx)
            self.engram_map[l_idx] = key
            self.engrams[key] = StaticEngram(cfg)

        self.pml_layer_set = set(cfg.pml_layer_indices)
        self.pml_slot_of_layer: Dict[int, int] = {
            l_idx: slot for slot, l_idx in enumerate(cfg.pml_layer_indices)
        }
        self.attn_layers = nn.ModuleList(
            [
                AttnPMLHybridLayer(
                    cfg,
                    self.vocab,
                    enable_pml=(l_idx in self.pml_layer_set),
                    is_first_layer=(l_idx == 0),
                )
                for l_idx in range(cfg.num_layers)
            ]
        )

        self.moe_slot_of_layer: Dict[int, int] = {
            l_idx: slot for slot, l_idx in enumerate(cfg.moe_layer_indices)
        }
        self.ffn_layers = nn.ModuleList(
            [
                TriStateMoELayer(cfg, self.moe_slot_of_layer[l_idx])
                if l_idx in self.moe_slot_of_layer
                else DenseSwiGLU(cfg)
                for l_idx in range(cfg.num_layers)
            ]
        )

        self.dir_head = DirectionVectorHead(cfg)
        self.hyper_router = HyperRouter(cfg)
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Dedicated unit-scale output norm for V_sym (isolates symbolic head from Qwen3's large final_norm.weight ~2.8!)
        self.sym_final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps) if cfg.cold_start_graft else None

    def init_whiteboard_states(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> List[WhiteboardState]:
        return [
            WhiteboardState.zeros(
                batch_size, self.cfg.num_ro_slots, self.cfg.num_rw_slots, self.cfg.pml_state_dim, device, dtype
            )
            for _ in self.cfg.pml_layer_indices
        ]

    def init_experience_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Track2ExperienceState:
        return Track2ExperienceState.zeros(
            batch_size,
            len(self.cfg.moe_layer_indices),
            self.cfg.num_routed_experts,
            self.cfg.probe_rank,
            device,
            dtype,
        )

    def populate_ro_from_prompt(
        self,
        prompt_token_ids: torch.Tensor,
        wb_states: List[WhiteboardState],
    ) -> None:
        """Vectorized zero-sync scan of prompt tokens for ⟨IN_k⟩ + value spans into S^RO."""
        bsz, plen = prompt_token_ids.shape
        if plen < 2:
            return
        x_emb = self.sym_embed_head(prompt_token_ids)  # [B, P, D]
        span_vecs = x_emb[:, 1:, :]  # Pure literal value embedding aligned 1:1 with S^RW value writes!
        tok_left = prompt_token_ids[:, :-1]
        is_ro = (tok_left >= self.vocab.ro_start) & (tok_left < self.vocab.ro_end)  # [B, P-1]
        slot_k = (tok_left - self.vocab.ro_start).clamp(0, self.cfg.num_ro_slots - 1)  # [B, P-1]

        for slot_idx, l_idx in enumerate(self.cfg.pml_layer_indices):
            pml = self.attn_layers[l_idx].pml
            if pml is None:
                continue
            proj = pml.slot_norm(pml.ro_init_proj(span_vecs)).to(wb_states[slot_idx].s_ro.dtype)  # [B, P-1, d_m]
            idx_s = slot_k.unsqueeze(-1).expand(-1, -1, self.cfg.pml_state_dim)
            cur_s = wb_states[slot_idx].s_ro.gather(1, idx_s)
            upd_s = torch.where(is_ro.unsqueeze(-1), proj, cur_s)
            cur_m = wb_states[slot_idx].m_ro.gather(1, slot_k)
            upd_m = torch.where(is_ro, torch.ones_like(cur_m), cur_m)
            if torch.is_grad_enabled():
                wb_states[slot_idx].s_ro = wb_states[slot_idx].s_ro.scatter(1, idx_s, upd_s)
                wb_states[slot_idx].m_ro = wb_states[slot_idx].m_ro.scatter(1, slot_k, upd_m)
            else:
                wb_states[slot_idx].s_ro.scatter_(1, idx_s, upd_s)
                wb_states[slot_idx].m_ro.scatter_(1, slot_k, upd_m)

    @staticmethod
    def _populate_nl_ro_layer(
        pml: Track1PMLModule,
        wb_l: WhiteboardState,
        prompt_ctx_hidden: torch.Tensor,
        ro_span_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Populate S^RO / M^RO at a Track-1 PML layer from Left-Brain contextual prompt hidden states [B, P, D].
        Uses zero-sync tensor masking so pre-populated symbolic S^RO slots remain untouched while NL dual-domain
        and 1-step AR prefill populate identical (< 1e-6) non-zero S^RO registers!
        """
        bsz, p_len, _ = prompt_ctx_hidden.shape
        n_ro = wb_l.s_ro.shape[1]
        if p_len == 0 or n_ro == 0:
            return
        device = prompt_ctx_hidden.device
        if ro_span_weights is None:
            # Deterministic zero-sync fallback mapping prompt positions into S^RO slots
            ro_w = torch.zeros((bsz, n_ro, p_len), device=device, dtype=prompt_ctx_hidden.dtype)
            n_fill = min(p_len, n_ro)
            for k in range(n_fill):
                s_i = (k * p_len) // n_fill
                e_i = max(s_i + 1, ((k + 1) * p_len) // n_fill)
                ro_w[:, k, s_i:e_i] = 1.0 / float(e_i - s_i)
        else:
            ro_w = ro_span_weights.to(device=device, dtype=prompt_ctx_hidden.dtype)

        active_slot = ro_w.float().sum(dim=-1) > 1e-6  # [B, N_ro]
        pooled_h = torch.einsum("bkp,bpd->bkd", ro_w, prompt_ctx_hidden)
        proj_ro = pml.slot_norm(pml.ro_init_proj(pooled_h)).to(wb_l.s_ro.dtype)
        cand_s = torch.where(active_slot.unsqueeze(-1), proj_ro, torch.zeros_like(wb_l.s_ro))
        cand_m = active_slot.to(wb_l.m_ro.dtype)

        has_existing = (wb_l.m_ro > 0.05).any(dim=-1, keepdim=True)  # [B, 1]
        upd_s = torch.where(has_existing.unsqueeze(-1), wb_l.s_ro, cand_s)
        upd_m = torch.where(has_existing, wb_l.m_ro, cand_m)
        if torch.is_grad_enabled():
            wb_l.s_ro = upd_s
            wb_l.m_ro = upd_m
        else:
            wb_l.s_ro.copy_(upd_s)
            wb_l.m_ro.copy_(upd_m)

    def _precompute_moe_manifolds(
        self, device: torch.device, full_sum_eval: bool = False
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batch ALL MoE layers' 3 Sinkhorn graphs into ONE [3*L_moe, E, E] pass (eliminates 612+ tiny kernels/tok!)."""
        precomputed_graph_map: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if full_sum_eval or self.ablation.static_top2_baseline or self.ablation.disable_bipolar_graph:
            return precomputed_graph_map
        moe_indices = self.cfg.moe_layer_indices
        if not moe_indices:
            return precomputed_graph_map
        e_dim = self.cfg.num_routed_experts
        stacked_logits = torch.stack(
            [
                t
                for l_i in moe_indices
                for t in (
                    self.ffn_layers[l_i].graph.theta_syn,  # type: ignore[union-attr]
                    self.ffn_layers[l_i].graph.theta_adv,  # type: ignore[union-attr]
                    self.ffn_layers[l_i].graph.theta_inter,  # type: ignore[union-attr]
                )
            ],
            dim=0,
        )
        eye = torch.eye(e_dim, device=device, dtype=torch.bool)
        zeros_e = torch.zeros_like(eye)
        z_mask = torch.stack([eye, eye, zeros_e], dim=0).repeat(len(moe_indices), 1, 1)
        n_it = self.ffn_layers[moe_indices[0]].graph.num_iters  # type: ignore[union-attr]
        all_m = sinkhorn_log_domain_batched(stacked_logits, z_mask, num_iters=n_it)
        for idx_m, l_i in enumerate(moe_indices):
            base_m = idx_m * 3
            precomputed_graph_map[l_i] = (all_m[base_m], all_m[base_m + 1], all_m[base_m + 2])
        return precomputed_graph_map

    def _collect_pml_pred_loss(self, fallback_ref: torch.Tensor) -> torch.Tensor:
        terms: List[torch.Tensor] = []
        for l_idx in self.cfg.pml_layer_indices:
            pml = self.attn_layers[l_idx].pml
            if pml is not None:
                pl = pml.pop_pred_loss()
                if pl is not None:
                    terms.append(pl)
        return torch.stack(terms).mean() if terms else torch.zeros_like(fallback_ref)

    def parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        routed = sum(
            p.numel()
            for layer in self.ffn_layers
            if isinstance(layer, TriStateMoELayer)
            for exp in layer.routed_experts
            for p in exp.parameters()
        )
        engram_tables = sum(
            tbl.weight.numel()
            for eng in self.engrams.values()
            for tbl in eng.tables
        )
        nl_vocab_params = self.nl_embed.weight.numel()
        per_expert_fraction = self.cfg.k_max / float(max(1, self.cfg.num_routed_experts))
        active_est = int(total - routed * (1.0 - per_expert_fraction) - engram_tables - nl_vocab_params)
        return {"total_params": total, "routed_expert_params": routed, "active_params_est": active_est}

    def forward_symbolic(
        self,
        token_ids: torch.Tensor,
        phase: CognitivePhase = CognitivePhase.FORMAL,
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        wb_states: Optional[List[WhiteboardState]] = None,
        exp_state: Optional[Track2ExperienceState] = None,
        soft_sketch: Optional[torch.Tensor] = None,
        t_k: float = 0.0,
        d_b_override: Optional[torch.Tensor] = None,
        reassert_flags: Optional[torch.Tensor] = None,
        write_to_draft: bool = False,
        read_only_pml: bool = False,
        force_breaker_div: bool = False,
        prior_dropout_prob: float = 0.0,
        cache_utility: bool = False,
        utility_mask: Optional[torch.Tensor] = None,
        dual_stream_sym_blocks: int = 0,
        dual_stream_block_size: int = 0,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        slot_offset: Optional[int] = None,
        return_soft_sketch: bool = True,
        frame_sizes: Optional[List[int]] = None,
        clean_token_ids_for_pml: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        bsz, seq_len = token_ids.shape
        device = token_ids.device
        if positions is None:
            positions = torch.arange(start_pos, start_pos + seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        blk_sz_eff = dual_stream_block_size if dual_stream_block_size > 0 else self.cfg.atomic_chunk_size
        slot_pos = (
            torch.arange(slot_offset, slot_offset + seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            if slot_offset is not None
            else positions
        )

        x_tok_emb = self.sym_embed_head(token_ids)
        clean_tok_emb = (
            self.sym_embed_head(clean_token_ids_for_pml) if clean_token_ids_for_pml is not None else None
        )
        x = x_tok_emb + self.chunk_slot_embed(slot_pos % blk_sz_eff)
        if self.use_block_idx_embed and self.block_idx_embed is not None:
            x = x + self.block_idx_embed((slot_pos // blk_sz_eff).clamp(0, 63))
        if seq_len > 1 and phase == CognitivePhase.PRE and past_kvs is None:
            tok_prev = token_ids[:, :-1]
            tok_curr = token_ids[:, 1:]
            is_in_prev = (tok_prev >= self.vocab.ro_start) & (tok_prev < self.vocab.ro_end)
            eq_tok = self.vocab.token_to_id.get("=", -1)
            sep_tok = self.vocab.token_to_id.get(":", -1)
            is_val_curr = (
                (tok_curr >= self.cfg.op_vocab_size)
                & (tok_curr < self.vocab.ro_start)
                & (tok_curr != eq_tok)
                & (tok_curr != sep_tok)
            )
            span_mask = (is_in_prev & is_val_curr).unsqueeze(-1).to(x.dtype)
            slot_ids = (tok_prev - self.vocab.ro_start).clamp(0, self.cfg.num_ro_slots - 1)
            tag_delta = span_mask * self.alpha_span_tag.to(x.dtype) * self.in_span_tag(slot_ids)
            x = torch.cat([x[:, :1, :], x[:, 1:, :] + tag_delta], dim=1)
        if soft_sketch is not None:
            is_mask = (token_ids == SymbolicVocab.MASK_ID).unsqueeze(-1).to(x.dtype)
            x = x + is_mask * self.sc_proj(soft_sketch)

        multi_frame_iso = (
            frame_sizes is not None
            and len(frame_sizes) > 1
            and sum(frame_sizes) == seq_len
            and dual_stream_sym_blocks == 0
        )
        segment_specs: Optional[List[Tuple[int, bool]]] = None
        if multi_frame_iso:
            assert frame_sizes is not None
            segment_specs = [(fl, False) for fl in frame_sizes]
            rc_parts: List[torch.Tensor] = []
            tk_parts: List[torch.Tensor] = []
            off_f = 0
            for fl in frame_sizes:
                tf = token_ids[:, off_f : off_f + fl]
                non_p = (tf != SymbolicVocab.PAD_ID).float()
                is_v = ((tf != SymbolicVocab.MASK_ID) & (tf != SymbolicVocab.PAD_ID)).float()
                rc_f = (is_v.sum(dim=-1, keepdim=True) / non_p.sum(dim=-1, keepdim=True).clamp(min=1.0)).expand(-1, fl)
                rc_parts.append(rc_f)
                tk_parts.append(1.0 - rc_f)
                off_f += fl
            r_clean_t = torch.cat(rc_parts, dim=1).to(x.dtype)
            t_k_eff: float | torch.Tensor = torch.cat(tk_parts, dim=1)
        else:
            is_valid = (token_ids != SymbolicVocab.MASK_ID) & (token_ids != SymbolicVocab.PAD_ID)
            non_pad = (token_ids != SymbolicVocab.PAD_ID).float()
            r_clean_t = is_valid.float().sum(dim=-1, keepdim=True) / non_pad.sum(dim=-1, keepdim=True).clamp(min=1.0)
            t_k_eff = t_k

        v_anchor: Optional[torch.Tensor] = None
        alpha_eng_broadcast: Optional[torch.Tensor] = None
        alpha_eng_raw_list: List[torch.Tensor] = []
        u_epi_latest = torch.zeros(bsz, seq_len, device=device, dtype=x.dtype)
        u_acute_latest = (token_ids == SymbolicVocab.CONTRA_ID).to(x.dtype)
        d_b = d_b_override
        a_prev_layer: Optional[torch.Tensor] = None
        moe_aux_list: List[Dict[str, torch.Tensor]] = []
        present_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        use_engram = (not self.ablation.disable_engram) and (not self.ablation.static_top2_baseline)
        use_pml = not self.ablation.disable_pml

        # Precompute PML write columns ONCE per forward pass (both train & eval) to skip 95% no-op tokens!
        precomputed_write_cols: Optional[List[int]] = None
        if use_pml and wb_states is not None and (not read_only_pml) and seq_len > 1:
            scan_end = seq_len - dual_stream_sym_blocks * blk_sz_eff if dual_stream_sym_blocks > 0 else seq_len
            c_t = token_ids[:, 1:scan_end]
            p_t = token_ids[:, : scan_end - 1]
            p2_t = torch.cat([torch.zeros_like(p_t[:, :1]), p_t[:, :-1]], dim=1)
            w_any = self.vocab.is_postfix_write_dst(c_t, p_t, p2_t)
            precomputed_write_cols = [off + 1 for off in w_any.any(dim=0).nonzero(as_tuple=False).view(-1).tolist()]

        precomputed_graph_map = self._precompute_moe_manifolds(device)

        for l_idx in range(self.cfg.num_layers):
            if use_engram and l_idx in self.engram_map:
                eng = self.engrams[self.engram_map[l_idx]]
                if multi_frame_iso and frame_sizes is not None:
                    x, a_eng_raw, a_eng_eff = self._run_engram_per_frame(eng, x, token_ids, frame_sizes)
                else:
                    x, a_eng_raw, a_eng_eff = eng(
                        x,
                        token_ids,
                        is_sym_phase=(phase != CognitivePhase.VERBALIZE),
                        dual_stream_sym_len=dual_stream_sym_blocks * blk_sz_eff,
                        dual_stream_block_size=blk_sz_eff,
                    )
                alpha_eng_broadcast = a_eng_eff
                alpha_eng_raw_list.append(a_eng_raw)

            wb_l = (
                wb_states[self.pml_slot_of_layer[l_idx]]
                if (use_pml and wb_states is not None and l_idx in self.pml_slot_of_layer)
                else None
            )
            pkv_in = past_kvs[l_idx] if past_kvs is not None else None
            x, v_raw, u_epi, u_acute, pkv = self.attn_layers[l_idx](
                x,
                token_ids=token_ids,
                positions=positions,
                wb_state=wb_l,
                d_b=d_b,
                attn_mask=attn_mask,
                v_anchor=v_anchor,
                past_kv=pkv_in,
                reassert_flags=reassert_flags,
                write_to_draft=write_to_draft,
                read_only_pml=read_only_pml,
                dual_stream_sym_blocks=dual_stream_sym_blocks,
                dual_stream_block_size=blk_sz_eff,
                precomputed_write_cols=precomputed_write_cols,
                x_val_embed=x_tok_emb,
                frame_sizes=frame_sizes,
                clean_token_ids_for_pml=clean_token_ids_for_pml,
                clean_val_embed_for_pml=clean_tok_emb,
            )
            present_kvs.append(pkv)
            if l_idx == 0:
                v_anchor = v_raw
            if use_pml and l_idx in self.pml_slot_of_layer:
                u_epi_latest = u_epi
                u_acute_latest = torch.maximum(u_acute_latest, u_acute)

            if l_idx == self.cfg.dir_vector_layer_index and d_b is None:
                if multi_frame_iso and frame_sizes is not None:
                    d_b = self._compute_dir_per_frame(x, token_ids, u_epi_latest, u_acute_latest, frame_sizes)
                else:
                    d_b = self.dir_head(
                        x,
                        valid_mask=(token_ids != SymbolicVocab.PAD_ID),
                        u_epistemic_mean=u_epi_latest.mean(dim=-1),
                        u_acute_mean=u_acute_latest.mean(dim=-1),
                        t_k=t_k,
                        r_clean=r_clean_t,
                        force_breaker_div=force_breaker_div,
                    )
                    if self.ablation.disable_wet_cement:
                        d_b = d_b.clone()
                        d_b[:, 1] = 0.0

            ffn = self.ffn_layers[l_idx]
            if isinstance(ffn, TriStateMoELayer):
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    # Gradient checkpointing over large 4.5B MoE FFN (bind layer module & pass all tensors explicitly!)
                    def _make_moe_ckpt(mod: TriStateMoELayer):
                        def _fn(
                            h_in: torch.Tensor,
                            a_prev_in: Optional[torch.Tensor],
                            a_eng_in: Optional[torch.Tensor],
                            u_epi_in: torch.Tensor,
                            u_ac_in: torch.Tensor,
                            d_b_in: Optional[torch.Tensor],
                            m_syn_in: Optional[torch.Tensor],
                            m_adv_in: Optional[torch.Tensor],
                            m_inter_in: Optional[torch.Tensor],
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                            pm = (m_syn_in, m_adv_in, m_inter_in) if m_syn_in is not None else None
                            y_o, aux_o = mod(
                                h_in,
                                phase=phase,
                                t_k=t_k_eff,  # type: ignore[arg-type]
                                r_clean=r_clean_t,
                                alpha_eng_eff=a_eng_in,
                                u_epistemic=u_epi_in,
                                u_acute=u_ac_in,
                                d_b=d_b_in,
                                exp_state=exp_state,
                                a_prev_layer=a_prev_in,
                                prior_dropout_prob=prior_dropout_prob,
                                static_top2_mode=self.ablation.static_top2_baseline,
                                disable_bipolar_graph=self.ablation.disable_bipolar_graph,
                                disable_pain_imprint=self.ablation.disable_pain_imprint,
                                cache_utility=cache_utility,
                                utility_mask=utility_mask,
                                precomputed_manifolds=pm,
                                segment_specs=segment_specs,
                            )
                            return y_o, aux_o["gates"], aux_o["w_route"], aux_o["block_profile"], aux_o["active_count"], aux_o.get("seg_relay")
                        return _fn

                    gm = precomputed_graph_map.get(l_idx)
                    ms, ma, mi = gm if gm is not None else (None, None, None)
                    x, g_c, wr_c, bp_c, ac_c, sr_c = torch.utils.checkpoint.checkpoint(
                        _make_moe_ckpt(ffn),
                        x,
                        a_prev_layer,
                        alpha_eng_broadcast,
                        u_epi_latest,
                        u_acute_latest,
                        d_b,
                        ms,
                        ma,
                        mi,
                        use_reentrant=False,
                    )
                    moe_aux = {
                        "gates": g_c,
                        "w_route": wr_c,
                        "block_profile": bp_c,
                        "seg_relay": sr_c,
                        "active_count": ac_c,
                        "graph_manifolds": gm,
                    }
                else:
                    x, moe_aux = ffn(
                        x,
                        phase=phase,
                        t_k=t_k_eff,  # type: ignore[arg-type]
                        r_clean=r_clean_t,
                        alpha_eng_eff=alpha_eng_broadcast,
                        u_epistemic=u_epi_latest,
                        u_acute=u_acute_latest,
                        d_b=d_b,
                        exp_state=exp_state,
                        a_prev_layer=a_prev_layer,
                        prior_dropout_prob=prior_dropout_prob,
                        static_top2_mode=self.ablation.static_top2_baseline,
                        disable_bipolar_graph=self.ablation.disable_bipolar_graph,
                        disable_pain_imprint=self.ablation.disable_pain_imprint,
                        cache_utility=cache_utility,
                        utility_mask=utility_mask,
                        precomputed_manifolds=precomputed_graph_map.get(l_idx),
                        segment_specs=segment_specs,
                    )
                a_prev_layer = moe_aux["seg_relay"] if multi_frame_iso else moe_aux["block_profile"]
                moe_aux_list.append(moe_aux)
            else:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    x = torch.utils.checkpoint.checkpoint(ffn, x, use_reentrant=False)
                else:
                    x = ffn(x)

        h_final = self.sym_final_norm(x) if self.sym_final_norm is not None else self.final_norm(x)
        logits = self.sym_embed_head.logits(h_final)
        if return_soft_sketch:
            probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
            next_soft_sketch = torch.matmul(probs, self.sym_embed_head.full_weight_matrix(device))
        else:
            next_soft_sketch = None

        return {
            "logits": logits,
            "hidden": h_final,
            "hidden_raw": x,
            "soft_sketch": next_soft_sketch,
            "u_epistemic": u_epi_latest,
            "u_acute": u_acute_latest,
            "d_b": d_b,
            "alpha_eng_list": alpha_eng_raw_list,
            "moe_aux": moe_aux_list,
            "present_kvs": present_kvs,
        }

    @staticmethod
    def _run_engram_per_frame(
        eng: StaticEngram,
        x_sym: torch.Tensor,
        tok_sym: torch.Tensor,
        frame_sizes: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run StaticEngram strictly inside each physical frame (grouped by frame size -> <=2 batched calls!)."""
        bsz, s_tot, dim = x_sym.shape
        if not frame_sizes or sum(frame_sizes) != s_tot:
            return eng(x_sym, tok_sym, is_sym_phase=True)
        unique_lens = sorted(set(frame_sizes))
        if len(unique_lens) == 1:
            flen = unique_lens[0]
            n_f = len(frame_sizes)
            xb, ab, aeb = eng(x_sym.reshape(bsz * n_f, flen, dim), tok_sym.reshape(bsz * n_f, flen), is_sym_phase=True)
            return xb.reshape(bsz, s_tot, dim), ab.reshape(bsz, s_tot), aeb.reshape(bsz, s_tot)

        spans: List[Tuple[int, int]] = []
        off = 0
        for fl in frame_sizes:
            spans.append((off, off + fl))
            off += fl
        x_out = torch.empty_like(x_sym)
        a_out = torch.empty((bsz, s_tot), device=x_sym.device, dtype=x_sym.dtype)
        ae_out = torch.empty((bsz, s_tot), device=x_sym.device, dtype=x_sym.dtype)
        for fl in unique_lens:
            idxs = [i for i, f_i in enumerate(frame_sizes) if f_i == fl]
            n_g = len(idxs)
            xg = torch.stack([x_sym[:, spans[i][0] : spans[i][1], :] for i in idxs], dim=1).reshape(bsz * n_g, fl, dim)
            tg = torch.stack([tok_sym[:, spans[i][0] : spans[i][1]] for i in idxs], dim=1).reshape(bsz * n_g, fl)
            xo, ao, aeo = eng(xg, tg, is_sym_phase=True)
            xo = xo.reshape(bsz, n_g, fl, dim)
            ao = ao.reshape(bsz, n_g, fl)
            aeo = aeo.reshape(bsz, n_g, fl)
            for g_pos, f_idx in enumerate(idxs):
                s_i, e_i = spans[f_idx]
                x_out[:, s_i:e_i, :] = xo[:, g_pos, :, :]
                a_out[:, s_i:e_i] = ao[:, g_pos, :]
                ae_out[:, s_i:e_i] = aeo[:, g_pos, :]
        return x_out, a_out, ae_out

    def _compute_dir_per_frame(
        self,
        x_sym: torch.Tensor,
        tok_sym: torch.Tensor,
        u_epi_sym: torch.Tensor,
        u_ac_sym: torch.Tensor,
        frame_sizes: List[int],
    ) -> torch.Tensor:
        """Compute 4D direction vector d_b strictly per frame/block (zero future-frame pooling, zero .item() sync!).
        Returns expanded per-token tensor [B, S_sym, 4].
        """
        bsz, s_tot, dim = x_sym.shape
        pos_frames = [fl for fl in frame_sizes if fl > 0]
        if not pos_frames:
            return torch.zeros((bsz, s_tot, 4), device=x_sym.device, dtype=x_sym.dtype)
        uniq = set(pos_frames)
        if len(uniq) == 1 and sum(pos_frames) == s_tot:
            fl = pos_frames[0]
            nf = len(pos_frames)
            tf = tok_sym.reshape(bsz * nf, fl)
            val_m = tf != SymbolicVocab.PAD_ID
            is_val = val_m & (tf != SymbolicVocab.MASK_ID)
            rc_f = is_val.float().sum(dim=-1, keepdim=True) / val_m.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            db_f = self.dir_head(
                x_sym.reshape(bsz * nf, fl, dim),
                valid_mask=val_m,
                u_epistemic_mean=u_epi_sym.reshape(bsz * nf, fl).mean(dim=-1),
                u_acute_mean=u_ac_sym.reshape(bsz * nf, fl).mean(dim=-1),
                t_k=(1.0 - rc_f).detach(),
                r_clean=rc_f,
            )
            if self.ablation.disable_wet_cement:
                db_f = db_f.clone()
                db_f[:, 1] = 0.0
            return db_f.reshape(bsz, nf, 1, 4).expand(-1, -1, fl, -1).reshape(bsz, s_tot, 4)

        out_3d = torch.zeros((bsz, s_tot, 4), device=x_sym.device, dtype=x_sym.dtype)
        off = 0
        for fl in pos_frames:
            end = off + fl
            tf = tok_sym[:, off:end]
            val_m = tf != SymbolicVocab.PAD_ID
            is_val = val_m & (tf != SymbolicVocab.MASK_ID)
            rc_f = is_val.float().sum(dim=-1, keepdim=True) / val_m.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            db_f = self.dir_head(
                x_sym[:, off:end, :],
                valid_mask=val_m,
                u_epistemic_mean=u_epi_sym[:, off:end].mean(dim=-1),
                u_acute_mean=u_ac_sym[:, off:end].mean(dim=-1),
                t_k=(1.0 - rc_f).detach(),
                r_clean=rc_f,
            )
            if self.ablation.disable_wet_cement:
                db_f = db_f.clone()
                db_f[:, 1] = 0.0
            out_3d[:, off:end, :] = db_f.unsqueeze(1).expand(-1, fl, -1)
            off = end
        return out_3d

    def forward_nl_causal(
        self,
        nl_token_ids: torch.Tensor,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        full_sum_eval: bool = False,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Causal AR forward pass over V_nl for Step-0 Qwen3 parity, Level 3 Text-CoT, & Verbalize."""
        bsz, seq_len = nl_token_ids.shape
        device = nl_token_ids.device
        positions = torch.arange(start_pos, start_pos + seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.nl_embed(nl_token_ids)

        attn_mask: Optional[torch.Tensor] = None
        if seq_len > 1 and past_kvs is not None:
            past_len = past_kvs[0][0].shape[2]
            mask_left = torch.ones((seq_len, past_len), dtype=torch.bool, device=device)
            mask_right = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
            attn_mask = torch.cat([mask_left, mask_right], dim=1)

        precomputed_graph_map = self._precompute_moe_manifolds(device, full_sum_eval=full_sum_eval)
        v_anchor: Optional[torch.Tensor] = None
        a_prev_layer: Optional[torch.Tensor] = None
        new_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for l_idx in range(self.cfg.num_layers):
            pkv_in = past_kvs[l_idx] if past_kvs is not None else None
            x, v_raw, _, _, pkv_out = self.attn_layers[l_idx](
                x,
                token_ids=nl_token_ids,
                positions=positions,
                wb_state=None,
                attn_mask=attn_mask,
                v_anchor=v_anchor,
                past_kv=pkv_in,
                read_only_pml=True,
            )
            new_kvs.append(pkv_out)
            if l_idx == 0:
                v_anchor = v_raw

            ffn = self.ffn_layers[l_idx]
            if isinstance(ffn, TriStateMoELayer):
                x, moe_aux = ffn(
                    x,
                    phase=CognitivePhase.VERBALIZE,
                    t_k=0.0,
                    a_prev_layer=a_prev_layer,
                    static_top2_mode=self.ablation.static_top2_baseline,
                    precomputed_manifolds=precomputed_graph_map.get(l_idx),
                    full_sum_eval=full_sum_eval,
                    causal_routing=True,
                )
                a_prev_layer = moe_aux["w_route"]
            else:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    x = torch.utils.checkpoint.checkpoint(ffn, x, use_reentrant=False)
                else:
                    x = ffn(x)

        h_final = self.final_norm(x)
        if return_hidden:
            return h_final, new_kvs
        logits = self.nl_head(h_final)
        return logits, new_kvs

    def forward_dual_domain(
        self,
        prompt_nl_ids: torch.Tensor,
        sym_token_ids: torch.Tensor,
        ans_nl_ids: Optional[torch.Tensor] = None,
        gen_mode: str = "ar",
        sym_frame_sizes: Optional[List[int]] = None,
        wb_states: Optional[List[WhiteboardState]] = None,
        exp_state: Optional[Track2ExperienceState] = None,
        reassert_flags: Optional[torch.Tensor] = None,
        t_k: float = 0.0,
        read_only_pml: bool = False,
        return_present_kvs: bool = False,
        sym_clean_ids: Optional[torch.Tensor] = None,
        ro_span_weights: Optional[torch.Tensor] = None,
        use_copy_gate: Optional[bool] = None,
        dllm_block_size: int = 8,
    ) -> Dict[str, object]:
        """End-to-end dual-domain forward over [Prompt(V_nl) | Thinking(V_sym) | Answer(V_nl)]:
        - gen_mode='ar'         : Strictly causal full-sequence pass (<1e-5 parity with 1-token forward_ar_step);
        - gen_mode='block_dllm' : Supports True Dual-Stream BD3-LM [Prompt_nl | Y_clean | Y_noisy | Answer_nl] when
                                  sym_clean_ids is provided (zero [MASK] whiteboard pollution!), unified 8-slot block
                                  granularity (<1e-5 parity with Prefix-KV block_dllm inference!), AND strict per-block
                                  isolation (zero future-frame or Answer_nl pooling leakage!).
        """
        bsz, p_len = prompt_nl_ids.shape
        if p_len < 1:
            raise ValueError("forward_dual_domain requires prompt_nl_ids with p_len >= 1")
        s_sym = sym_token_ids.shape[1]
        ans_len = ans_nl_ids.shape[1] if ans_nl_ids is not None else 0
        device = prompt_nl_ids.device
        is_ar = gen_mode == "ar"
        use_bd3 = (not is_ar) and (sym_clean_ids is not None) and (s_sym > 0)
        sym_span_len = 2 * s_sym if use_bd3 else s_sym
        tot_len = p_len + sym_span_len + ans_len

        raw_frames = (
            list(sym_frame_sizes)
            if (sym_frame_sizes is not None and sum(sym_frame_sizes) == s_sym)
            else ([8] * (s_sym // 8) if (s_sym > 0 and s_sym % 8 == 0) else ([s_sym] if s_sym > 0 else []))
        )
        if (not is_ar) and dllm_block_size > 0:
            frame_sizes: List[int] = []
            for fl in raw_frames:
                if fl > dllm_block_size and fl % dllm_block_size == 0:
                    frame_sizes.extend([dllm_block_size] * (fl // dllm_block_size))
                else:
                    frame_sizes.append(fl)
        else:
            frame_sizes = raw_frames
        blk_sz_eff = self.cfg.atomic_chunk_size

        x_prompt = self.nl_embed(prompt_nl_ids)
        sym_pos = torch.arange(0, s_sym, device=device).unsqueeze(0).expand(bsz, -1)
        slot_emb = self.chunk_slot_embed(sym_pos % blk_sz_eff)
        if self.use_block_idx_embed and self.block_idx_embed is not None:
            slot_emb = slot_emb + self.block_idx_embed((sym_pos // blk_sz_eff).clamp(0, 63))

        x_sym_noisy_tok = self.sym_embed_head(sym_token_ids)
        if use_bd3:
            assert sym_clean_ids is not None
            x_sym_clean_tok = self.sym_embed_head(sym_clean_ids)
            x_sym_tok_all = torch.cat([x_sym_clean_tok, x_sym_noisy_tok], dim=1)
            sym_ids_all = torch.cat([sym_clean_ids, sym_token_ids], dim=1)
            x_sym_all = torch.cat([x_sym_clean_tok + slot_emb, x_sym_noisy_tok + slot_emb], dim=1)
            attn_mask, base_pos = build_dual_domain_bd3_mask(p_len, frame_sizes, ans_len, device=device)
            positions = base_pos.expand(bsz, -1)
            sym_all_frames = frame_sizes + frame_sizes
            sym_slice_start = p_len + s_sym
        else:
            x_sym_tok_all = x_sym_noisy_tok
            sym_ids_all = sym_token_ids
            x_sym_all = x_sym_noisy_tok + slot_emb
            attn_mask = build_dual_domain_mask(p_len, frame_sizes, ans_len, gen_mode=gen_mode, device=device)
            positions = torch.arange(0, tot_len, device=device).unsqueeze(0).expand(bsz, -1)
            sym_all_frames = frame_sizes
            sym_slice_start = p_len

        parts = [x_prompt, x_sym_all]
        if ans_nl_ids is not None and ans_len > 0:
            parts.append(self.nl_embed(ans_nl_ids))
        x = torch.cat(parts, dim=1)  # [B, tot_len, D]

        segment_specs: Optional[List[Tuple[int, bool]]] = None
        if is_ar:
            r_clean_arg: torch.Tensor = torch.ones(bsz, 1, device=device, dtype=x.dtype)
            t_k_arg: float | torch.Tensor = 0.0
        else:
            segment_specs = [(p_len, True)] + [(fl, False) for fl in sym_all_frames] + ([(ans_len, True)] if ans_len > 0 else [])
            rc_parts = [torch.ones((bsz, p_len), device=device, dtype=torch.float32)]
            tk_parts = [torch.zeros((bsz, p_len), device=device, dtype=torch.float32)]
            if use_bd3:
                rc_parts.append(torch.ones((bsz, s_sym), device=device, dtype=torch.float32))
                tk_parts.append(torch.zeros((bsz, s_sym), device=device, dtype=torch.float32))
            off_rc = 0
            for fl in frame_sizes:
                tf = sym_token_ids[:, off_rc : off_rc + fl]
                non_p = (tf != SymbolicVocab.PAD_ID).float()
                is_v = ((tf != SymbolicVocab.MASK_ID) & (tf != SymbolicVocab.PAD_ID)).float()
                rc_f = (is_v.sum(dim=-1, keepdim=True) / non_p.sum(dim=-1, keepdim=True).clamp(min=1.0)).expand(-1, fl)
                rc_parts.append(rc_f)
                tk_parts.append(1.0 - rc_f)
                off_rc += fl
            if ans_len > 0:
                rc_parts.append(torch.ones((bsz, ans_len), device=device, dtype=torch.float32))
                tk_parts.append(torch.zeros((bsz, ans_len), device=device, dtype=torch.float32))
            r_clean_arg = torch.cat(rc_parts, dim=1).to(x.dtype)
            t_k_arg = torch.cat(tk_parts, dim=1)

        v_anchor: Optional[torch.Tensor] = None
        alpha_eng_full = torch.zeros(bsz, tot_len, device=device, dtype=x.dtype)
        alpha_eng_raw_list: List[torch.Tensor] = []
        u_epi_full = torch.zeros(bsz, tot_len, device=device, dtype=x.dtype)
        u_acute_full = torch.zeros(bsz, tot_len, device=device, dtype=x.dtype)
        u_acute_full[:, p_len : p_len + sym_span_len] = (sym_ids_all == SymbolicVocab.CONTRA_ID).to(x.dtype)
        d_b: Optional[torch.Tensor] = None
        a_prev_layer: Optional[torch.Tensor] = None
        moe_aux_list: List[Dict[str, torch.Tensor]] = []
        present_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        use_engram = (not self.ablation.disable_engram) and (not self.ablation.static_top2_baseline)
        use_pml = not self.ablation.disable_pml

        # Precompute PML write columns inside the clean/scanned V_sym stream ONCE per forward pass!
        precomputed_write_cols: Optional[List[int]] = None
        scan_src_ids = sym_clean_ids if use_bd3 else sym_token_ids
        if use_pml and wb_states is not None and (not read_only_pml) and scan_src_ids is not None and s_sym > 1:
            c_t = scan_src_ids[:, 1:]
            p_t = scan_src_ids[:, :-1]
            p2_t = torch.cat([torch.zeros_like(p_t[:, :1]), p_t[:, :-1]], dim=1)
            w_any = self.vocab.is_postfix_write_dst(c_t, p_t, p2_t)
            precomputed_write_cols = [off + 1 for off in w_any.any(dim=0).nonzero(as_tuple=False).view(-1).tolist()]

        precomputed_graph_map = self._precompute_moe_manifolds(device)

        for l_idx in range(self.cfg.num_layers):
            # 1. Static Engram strictly on V_sym slice (causal in AR; strictly frame-local in Block-dLLM!)
            if use_engram and l_idx in self.engram_map and s_sym > 0:
                eng = self.engrams[self.engram_map[l_idx]]
                if is_ar:
                    x_sym_eng, a_eng_raw, a_eng_eff = eng(x[:, p_len : p_len + s_sym, :], sym_token_ids, is_sym_phase=True)
                else:
                    x_sym_eng, a_eng_raw, a_eng_eff = self._run_engram_per_frame(
                        eng, x[:, p_len : p_len + sym_span_len, :], sym_ids_all, sym_all_frames
                    )
                    if use_bd3:
                        a_eng_raw = a_eng_raw[:, s_sym:]
                x = torch.cat([x[:, :p_len, :], x_sym_eng, x[:, p_len + sym_span_len :, :]], dim=1)
                alpha_eng_full = torch.cat(
                    [
                        torch.zeros(bsz, p_len, device=device, dtype=x.dtype),
                        a_eng_eff,
                        torch.zeros(bsz, ans_len, device=device, dtype=x.dtype),
                    ],
                    dim=1,
                )
                alpha_eng_raw_list.append(a_eng_raw)

            # 2. Left-Brain GQA || Right-Brain Track-1 PML (populating S^RO from prompt + scanning V_sym + reading Answer_nl)
            attn_l = self.attn_layers[l_idx]
            norm_x = attn_l.norm(x)
            h_gqa, v_raw, pkv = attn_l.gqa(norm_x, positions=positions, attn_mask=attn_mask, v_anchor=v_anchor)
            if return_present_kvs:
                present_kvs.append(pkv)
            if l_idx == 0:
                v_anchor = v_raw

            wb_l = (
                wb_states[self.pml_slot_of_layer[l_idx]]
                if (use_pml and wb_states is not None and l_idx in self.pml_slot_of_layer)
                else None
            )
            if use_pml and attn_l.enable_pml and attn_l.pml is not None and wb_l is not None:
                # Populate S^RO from Left-Brain contextualized NL prompt (identical in AR, BD3-LM, & 1-step KV prefill!)
                self._populate_nl_ro_layer(
                    attn_l.pml, wb_l, x[:, :p_len, :] + h_gqa[:, :p_len, :], ro_span_weights=ro_span_weights
                )
                if s_sym > 0:
                    norm_sym = norm_x[:, p_len : p_len + sym_span_len, :]
                    if read_only_pml:
                        h_pml_sym, u_epi_sym = attn_l.pml.read_step_or_block(norm_sym, wb_l, token_ids=sym_ids_all)
                        u_ac_sym = (sym_ids_all == SymbolicVocab.CONTRA_ID).to(x.dtype)
                    else:
                        reassert_in = (
                            torch.cat([reassert_flags, reassert_flags], dim=1)
                            if (use_bd3 and reassert_flags is not None)
                            else reassert_flags
                        )
                        h_pml_sym, u_epi_sym, u_ac_sym = attn_l.pml.sequential_scan_forward(
                            norm_sym,
                            token_ids=sym_ids_all,
                            state=wb_l,
                            reassert_flags=reassert_in,
                            write_to_draft=False,
                            dual_stream_sym_blocks=(len(frame_sizes) if use_bd3 else 0),
                            dual_stream_block_size=(1 if is_ar else 0),
                            precomputed_write_cols=precomputed_write_cols,
                            x_val_embed=x_sym_tok_all,
                            frame_sizes=(None if is_ar else frame_sizes),
                        )
                    if ans_len > 0:
                        h_pml_ans, u_epi_ans = attn_l.pml.read_step_or_block(
                            norm_x[:, p_len + sym_span_len :, :], wb_l, use_committed_only=True, token_ids=None
                        )
                    else:
                        h_pml_ans = torch.zeros(bsz, 0, self.cfg.hidden_size, device=device, dtype=x.dtype)
                        u_epi_ans = torch.zeros(bsz, 0, device=device, dtype=x.dtype)

                    h_pml_full = torch.cat(
                        [torch.zeros(bsz, p_len, self.cfg.hidden_size, device=device, dtype=x.dtype), h_pml_sym, h_pml_ans],
                        dim=1,
                    )
                    if d_b is not None and d_b.dim() == 3:
                        gamma_film: float | torch.Tensor = attn_l.film_proj(d_b.to(x.dtype))
                    elif d_b is not None:
                        gamma_film = attn_l.film_proj(d_b.to(x.dtype)).unsqueeze(1)
                    else:
                        gamma_film = 0.0
                    gate = torch.tanh(attn_l.alpha_calm.view(1, 1, -1) * (1.0 + gamma_film)).to(x.dtype)
                    x = x + h_gqa + gate * h_pml_full

                    u_epi_full = torch.cat(
                        [torch.zeros(bsz, p_len, device=device, dtype=x.dtype), u_epi_sym, u_epi_ans], dim=1
                    )
                    u_ac_pad = torch.cat(
                        [
                            torch.zeros(bsz, p_len, device=device, dtype=x.dtype),
                            u_ac_sym,
                            torch.zeros(bsz, ans_len, device=device, dtype=x.dtype),
                        ],
                        dim=1,
                    )
                    u_acute_full = torch.maximum(u_acute_full, u_ac_pad)
                else:
                    x = x + h_gqa
            else:
                x = x + h_gqa

            if (not is_ar) and l_idx == self.cfg.dir_vector_layer_index and d_b is None and s_sym > 0:
                db_sym_3d = self._compute_dir_per_frame(
                    x[:, p_len : p_len + sym_span_len, :],
                    sym_ids_all,
                    u_epi_full[:, p_len : p_len + sym_span_len],
                    u_acute_full[:, p_len : p_len + sym_span_len],
                    sym_all_frames,
                )
                d_b = torch.cat(
                    [
                        torch.zeros((bsz, p_len, 4), device=device, dtype=x.dtype),
                        db_sym_3d,
                        torch.zeros((bsz, ans_len, 4), device=device, dtype=x.dtype),
                    ],
                    dim=1,
                )

            # 3. Capacity-Preserving Adaptive Top-K MoE / DenseSwiGLU (with gradient checkpointing!)
            ffn = self.ffn_layers[l_idx]
            if isinstance(ffn, TriStateMoELayer):
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    def _make_dual_moe_ckpt(mod: TriStateMoELayer, segs: Optional[List[Tuple[int, bool]]]):
                        def _fn(
                            h_in: torch.Tensor,
                            a_prev_in: Optional[torch.Tensor],
                            a_eng_in: torch.Tensor,
                            u_epi_in: torch.Tensor,
                            u_ac_in: torch.Tensor,
                            d_b_in: Optional[torch.Tensor],
                            m_syn_in: Optional[torch.Tensor],
                            m_adv_in: Optional[torch.Tensor],
                            m_inter_in: Optional[torch.Tensor],
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                            pm = (m_syn_in, m_adv_in, m_inter_in) if m_syn_in is not None else None
                            y_o, aux_o = mod(
                                h_in,
                                phase=CognitivePhase.FORMAL,
                                t_k=t_k_arg,  # type: ignore[arg-type]
                                r_clean=r_clean_arg,
                                alpha_eng_eff=a_eng_in,
                                u_epistemic=u_epi_in,
                                u_acute=u_ac_in,
                                d_b=d_b_in,
                                exp_state=exp_state,
                                a_prev_layer=a_prev_in,
                                static_top2_mode=self.ablation.static_top2_baseline,
                                disable_bipolar_graph=self.ablation.disable_bipolar_graph,
                                disable_pain_imprint=self.ablation.disable_pain_imprint,
                                precomputed_manifolds=pm,
                                causal_routing=is_ar,
                                segment_specs=segs,
                            )
                            return (
                                y_o,
                                aux_o["gates"],
                                aux_o["w_route"],
                                aux_o["block_profile"],
                                aux_o["active_count"],
                                aux_o.get("seg_relay"),
                            )
                        return _fn

                    gm = precomputed_graph_map.get(l_idx)
                    ms, ma, mi = gm if gm is not None else (None, None, None)
                    x, g_c, wr_c, bp_c, ac_c, sr_c = torch.utils.checkpoint.checkpoint(
                        _make_dual_moe_ckpt(ffn, segment_specs),
                        x,
                        a_prev_layer,
                        alpha_eng_full,
                        u_epi_full,
                        u_acute_full,
                        d_b,
                        ms,
                        ma,
                        mi,
                        use_reentrant=False,
                    )
                    moe_aux = {
                        "gates": g_c,
                        "w_route": wr_c,
                        "block_profile": bp_c,
                        "seg_relay": sr_c,
                        "active_count": ac_c,
                        "graph_manifolds": gm,
                    }
                else:
                    x, moe_aux = ffn(
                        x,
                        phase=CognitivePhase.FORMAL,
                        t_k=t_k_arg,  # type: ignore[arg-type]
                        r_clean=r_clean_arg,
                        alpha_eng_eff=alpha_eng_full,
                        u_epistemic=u_epi_full,
                        u_acute=u_acute_full,
                        d_b=d_b,
                        exp_state=exp_state,
                        a_prev_layer=a_prev_layer,
                        static_top2_mode=self.ablation.static_top2_baseline,
                        disable_bipolar_graph=self.ablation.disable_bipolar_graph,
                        disable_pain_imprint=self.ablation.disable_pain_imprint,
                        precomputed_manifolds=precomputed_graph_map.get(l_idx),
                        causal_routing=is_ar,
                        segment_specs=segment_specs,
                    )
                a_prev_layer = moe_aux["w_route"] if is_ar else moe_aux["seg_relay"]
                moe_aux_list.append(moe_aux)
            else:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    x = torch.utils.checkpoint.checkpoint(ffn, x, use_reentrant=False)
                else:
                    x = ffn(x)

        # 4. Separate Domain Heads: V_sym logits (shifted for AR, Y_noisy for BD3-LM) + V_nl Verbalize Answer logits!
        sym_norm_fn = self.sym_final_norm if self.sym_final_norm is not None else self.final_norm
        if is_ar:
            h_sym_for_head = sym_norm_fn(x[:, p_len - 1 : p_len + s_sym - 1, :])
        else:
            h_sym_for_head = sym_norm_fn(x[:, sym_slice_start : sym_slice_start + s_sym, :])
        sym_logits = self.sym_embed_head.logits(h_sym_for_head)  # [B, s_sym, V_sym]

        ans_logits: Optional[torch.Tensor] = None
        do_copy = self.cfg.enable_copy_gate if use_copy_gate is None else use_copy_gate
        if ans_len > 0:
            if use_bd3:
                # Last clean V_sym token (<EOT> at p_len + s_sym - 1) predicts ans_nl_ids[:, 0]; ans_nl_ids[:, :-1] predicts rest!
                if ans_len == 1:
                    h_ans_raw = x[:, p_len + s_sym - 1 : p_len + s_sym, :]
                else:
                    h_ans_raw = torch.cat(
                        [
                            x[:, p_len + s_sym - 1 : p_len + s_sym, :],
                            x[:, p_len + 2 * s_sym : p_len + 2 * s_sym + ans_len - 1, :],
                        ],
                        dim=1,
                    )
            else:
                h_ans_raw = x[:, p_len + s_sym - 1 : p_len + s_sym + ans_len - 1, :]
            h_ans_for_head = self.final_norm(h_ans_raw)
            ans_logits = self.nl_head(h_ans_for_head)  # [B, ans_len, V_nl]
            if do_copy:
                h_prompt_norm = self.final_norm(x[:, :p_len, :])
                ans_logits = self.copy_gate(h_ans_for_head, ans_logits, h_prompt_norm, prompt_nl_ids)

        return {
            "sym_logits": sym_logits,
            "ans_logits": ans_logits,
            "hidden_all": x,
            "u_epistemic": u_epi_full,
            "u_acute": u_acute_full,
            "d_b": d_b,
            "alpha_eng_list": alpha_eng_raw_list,
            "moe_aux": moe_aux_list,
            "present_kvs": present_kvs,
            "p_len": p_len,
            "s_sym": s_sym,
            "sym_slice_start": sym_slice_start,
            "ans_len": ans_len,
        }

    def init_engram_causal_caches(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Dict[str, dict[str, torch.Tensor]]:
        """Initialize O(1) causal StaticEngram caches for 1-token KV-cached AR decoding."""
        return {k: eng.init_causal_cache(batch_size, device, dtype) for k, eng in self.engrams.items()}

    @torch.no_grad()
    def forward_ar_step(
        self,
        token_ids: torch.Tensor,
        domain: str,
        start_pos: int = 0,
        sym_offset: int = 0,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        wb_states: Optional[List[WhiteboardState]] = None,
        engram_caches: Optional[Dict[str, dict[str, torch.Tensor]]] = None,
        exp_state: Optional[Track2ExperienceState] = None,
        prev_sym_token: Optional[torch.Tensor] = None,
        prev2_sym_embed: Optional[torch.Tensor] = None,
        prev2_sym_token: Optional[torch.Tensor] = None,
        prev2_pml_norm_x: Optional[Dict[int, torch.Tensor]] = None,
        ro_span_weights: Optional[torch.Tensor] = None,
        prompt_hidden_for_copy: Optional[torch.Tensor] = None,
        prompt_ids_for_copy: Optional[torch.Tensor] = None,
        read_only_pml: bool = False,
    ) -> Dict[str, object]:
        """KV-cached strictly causal step over 'nl_prompt', 'sym', or 'nl_ans' domains.
        Matches full-sequence forward_dual_domain(gen_mode='ar') to < 1e-5 (TF32-off) with zero prefix recomputation!
        """
        if domain not in ("nl_prompt", "sym", "nl_ans"):
            raise ValueError(f"Unsupported domain={domain!r}; expected 'nl_prompt', 'sym', or 'nl_ans'")

        bsz, seq_len = token_ids.shape
        device = token_ids.device
        positions = torch.arange(start_pos, start_pos + seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        x_sym_tok: Optional[torch.Tensor] = None
        if domain == "sym":
            x_sym_tok = self.sym_embed_head(token_ids)
            sym_pos = torch.arange(sym_offset, sym_offset + seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            blk_sz_eff = self.cfg.atomic_chunk_size
            x = x_sym_tok + self.chunk_slot_embed(sym_pos % blk_sz_eff)
            if self.use_block_idx_embed and self.block_idx_embed is not None:
                x = x + self.block_idx_embed((sym_pos // blk_sz_eff).clamp(0, 63))
        else:
            x = self.nl_embed(token_ids)

        # Build causal mask if seq_len > 1 and past_kvs is present (SDPA disables is_causal when past_kv is not None)
        attn_mask: Optional[torch.Tensor] = None
        if seq_len > 1 and past_kvs is not None:
            past_len = past_kvs[0][0].shape[2]
            mask_left = torch.ones((seq_len, past_len), dtype=torch.bool, device=device)
            mask_right = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
            attn_mask = torch.cat([mask_left, mask_right], dim=1)

        r_clean_t = torch.ones(bsz, 1, device=device, dtype=x.dtype)
        v_anchor: Optional[torch.Tensor] = None
        alpha_eng_cur = torch.zeros(bsz, seq_len, device=device, dtype=x.dtype)
        u_epi_cur = torch.zeros(bsz, seq_len, device=device, dtype=x.dtype)
        u_acute_cur = (
            (token_ids == SymbolicVocab.CONTRA_ID).to(x.dtype)
            if domain == "sym"
            else torch.zeros(bsz, seq_len, device=device, dtype=x.dtype)
        )
        a_prev_layer: Optional[torch.Tensor] = None
        present_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pml_norm_x_cur: Dict[int, torch.Tensor] = {}

        use_engram = (not self.ablation.disable_engram) and (not self.ablation.static_top2_baseline)
        use_pml = not self.ablation.disable_pml
        precomputed_graph_map = self._precompute_moe_manifolds(device)

        # Fast-path for 1-token AR decode: skip 8x PML write kernels on the ~95% of tokens that are not postfix writes!
        is_single_tok_write = True
        if domain == "sym" and seq_len == 1 and (not read_only_pml):
            if prev_sym_token is None:
                is_single_tok_write = False
            else:
                p2_1 = prev2_sym_token if prev2_sym_token is not None else torch.zeros_like(prev_sym_token)
                is_w_1 = self.vocab.is_postfix_write_dst(token_ids[:, 0], prev_sym_token, p2_1)
                is_single_tok_write = bool(is_w_1.any().item())

        for l_idx in range(self.cfg.num_layers):
            if domain == "sym" and use_engram and l_idx in self.engram_map:
                eng_slot = self.engram_map[l_idx]
                eng = self.engrams[eng_slot]
                if seq_len == 1 and engram_caches is not None:
                    x, _, alpha_eng_cur = eng.forward_causal_step(x, token_ids, engram_caches[eng_slot])
                else:
                    x, _, alpha_eng_cur = eng(x, token_ids, is_sym_phase=True)

            attn_l = self.attn_layers[l_idx]
            norm_x = attn_l.norm(x)
            pkv_in = past_kvs[l_idx] if past_kvs is not None else None
            h_gqa, v_raw, pkv_out = attn_l.gqa(
                norm_x, positions=positions, attn_mask=attn_mask, v_anchor=v_anchor, past_kv=pkv_in
            )
            present_kvs.append(pkv_out)
            if l_idx == 0:
                v_anchor = v_raw

            wb_l = (
                wb_states[self.pml_slot_of_layer[l_idx]]
                if (use_pml and wb_states is not None and l_idx in self.pml_slot_of_layer)
                else None
            )
            if use_pml and attn_l.enable_pml and attn_l.pml is not None and wb_l is not None:
                if domain == "sym" and seq_len == 1:
                    pml_norm_x_cur[l_idx] = norm_x[:, 0, :]
                if domain == "nl_prompt":
                    self._populate_nl_ro_layer(attn_l.pml, wb_l, x + h_gqa, ro_span_weights=ro_span_weights)
                    x = x + h_gqa
                else:
                    if domain == "nl_ans":
                        h_pml, u_epi_cur = attn_l.pml.read_step_or_block(
                            norm_x, wb_l, use_committed_only=True, token_ids=None
                        )
                    elif read_only_pml or (seq_len == 1 and not is_single_tok_write):
                        p1_in = prev_sym_token.unsqueeze(1) if (prev_sym_token is not None and seq_len == 1) else None
                        p2_in = (
                            prev2_sym_token.unsqueeze(1)
                            if (prev2_sym_token is not None and seq_len == 1)
                            else (torch.zeros_like(p1_in) if p1_in is not None else None)
                        )
                        h_pml, u_epi_cur = attn_l.pml.read_step_or_block(
                            norm_x, wb_l, token_ids=token_ids, prev_tokens=p1_in, prev2_tokens=p2_in
                        )
                    else:
                        p2_ctx_l = prev2_pml_norm_x.get(l_idx) if prev2_pml_norm_x is not None else None
                        h_pml, u_epi_cur, u_ac_l = attn_l.pml.sequential_scan_forward(
                            norm_x,
                            token_ids=token_ids,
                            state=wb_l,
                            write_to_draft=False,
                            prev_last_token=prev_sym_token,
                            dual_stream_block_size=1,
                            x_val_embed=x_sym_tok,
                            prev2_val_embed=prev2_sym_embed,
                            prev2_last_token=prev2_sym_token,
                            prev2_ctx_hidden=p2_ctx_l,
                        )
                        u_acute_cur = torch.maximum(u_acute_cur, u_ac_l)
                    gate = torch.tanh(attn_l.alpha_calm.view(1, 1, -1)).to(x.dtype)
                    x = x + h_gqa + gate * h_pml
            else:
                x = x + h_gqa

            ffn = self.ffn_layers[l_idx]
            if isinstance(ffn, TriStateMoELayer):
                x, moe_aux = ffn(
                    x,
                    phase=CognitivePhase.FORMAL,
                    t_k=0.0,
                    r_clean=r_clean_t,
                    alpha_eng_eff=alpha_eng_cur,
                    u_epistemic=u_epi_cur,
                    u_acute=u_acute_cur,
                    d_b=None,
                    exp_state=exp_state,
                    a_prev_layer=a_prev_layer,
                    static_top2_mode=self.ablation.static_top2_baseline,
                    disable_bipolar_graph=self.ablation.disable_bipolar_graph,
                    disable_pain_imprint=self.ablation.disable_pain_imprint,
                    precomputed_manifolds=precomputed_graph_map.get(l_idx),
                    causal_routing=True,
                )
                a_prev_layer = moe_aux["w_route"]
            else:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    x = torch.utils.checkpoint.checkpoint(ffn, x, use_reentrant=False)
                else:
                    x = ffn(x)

        sym_norm_fn = self.sym_final_norm if self.sym_final_norm is not None else self.final_norm
        sym_logits = self.sym_embed_head.logits(sym_norm_fn(x[:, -1:, :])) if domain in ("nl_prompt", "sym") else None
        nl_logits: Optional[torch.Tensor] = None
        if domain in ("sym", "nl_ans"):
            h_nl_last = self.final_norm(x[:, -1:, :])
            nl_logits = self.nl_head(h_nl_last)
            if self.cfg.enable_copy_gate and prompt_hidden_for_copy is not None and prompt_ids_for_copy is not None:
                nl_logits = self.copy_gate(h_nl_last, nl_logits, prompt_hidden_for_copy, prompt_ids_for_copy)
        return {
            "sym_logits": sym_logits,
            "nl_logits": nl_logits,
            "hidden": x,
            "x_sym_tok": x_sym_tok,
            "pml_norm_x": pml_norm_x_cur,
            "present_kvs": present_kvs,
            "u_epistemic": u_epi_cur,
            "u_acute": u_acute_cur,
        }

    def compute_dual_domain_losses(
        self,
        out: Dict[str, object],
        sym_target_ids: torch.Tensor,
        sym_loss_mask: torch.Tensor,
        ans_target_ids: Optional[torch.Tensor] = None,
        ans_loss_weight: float = 1.0,
        collect_stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Unified loss over [Thinking(V_sym) + Verbalize(V_nl) + L_pred(0.02) + L_budget(0.02) + L_graph(0.05) + L_balance(0.02)]."""
        sym_logits: torch.Tensor = out["sym_logits"]  # type: ignore[assignment]
        ans_logits: Optional[torch.Tensor] = out["ans_logits"]  # type: ignore[assignment]
        moe_aux_list: List[Dict[str, torch.Tensor]] = out["moe_aux"]  # type: ignore[assignment]

        per_tok_sym = F.cross_entropy(
            sym_logits.reshape(-1, sym_logits.shape[-1]).float(),
            sym_target_ids.reshape(-1),
            reduction="none",
        ).view_as(sym_target_ids)
        esc_w = torch.where(self.vocab.is_escape_anchor_tensor(sym_target_ids), 2.5, 1.0)
        m_f = sym_loss_mask.float() * esc_w
        row_den = m_f.sum(dim=-1)
        valid_r = (row_den > 0.0).float()
        l_sym = ((per_tok_sym * m_f).sum(dim=-1) / row_den.clamp(min=1e-6) * valid_r).sum() / valid_r.sum().clamp(min=1.0)

        if ans_logits is not None and ans_target_ids is not None and ans_target_ids.numel() > 0:
            l_ans = F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]).float(), ans_target_ids.reshape(-1))
        else:
            l_ans = torch.zeros_like(l_sym)

        l_pred = self._collect_pml_pred_loss(l_sym)

        budget_terms, graph_terms, balance_terms = [], [], []
        ent_list, ov_list, act_tensors = [], [], []
        moe_layers = [l for l in self.ffn_layers if isinstance(l, TriStateMoELayer)]
        u_epi: torch.Tensor = out["u_epistemic"]  # type: ignore[assignment]
        alpha_eng_list: List[torch.Tensor] = out["alpha_eng_list"]  # type: ignore[assignment]
        s_sym = int(out["s_sym"])  # type: ignore[arg-type]
        sym_start = int(out.get("sym_slice_start", out["p_len"]))  # type: ignore[arg-type]
        a_eng_sym = alpha_eng_list[-1].detach().float() if alpha_eng_list else torch.zeros(u_epi.shape[0], s_sym, device=u_epi.device)
        u_epi_sym = u_epi[:, sym_start : sym_start + s_sym].detach().float()
        g_target_sym = 0.55 + 0.85 * torch.sigmoid(4.0 * (u_epi_sym - 0.35 - 0.35 * a_eng_sym))

        for m_layer, m_aux in zip(moe_layers, moe_aux_list):
            act_sum_sym = m_aux["gates"][:, sym_start : sym_start + s_sym, :].float().sum(dim=-1)
            if collect_stats:
                act_tensors.append(m_aux["active_count"][:, sym_start : sym_start + s_sym].mean())
            budget_terms.append((act_sum_sym - g_target_sym).pow(2).mean())
            balance_terms.append(m_layer.load_balance_loss(m_aux))
            if not self.ablation.disable_bipolar_graph:
                g_loss, g_met = m_layer.graph.graph_regularization_loss(
                    manifolds=m_aux.get("graph_manifolds"), collect_metrics=collect_stats  # type: ignore[arg-type]
                )
                graph_terms.append(g_loss)
                if collect_stats:
                    ent_list.append(g_met["row_entropy_syn"])
                    ov_list.append(g_met["syn_adv_overlap"])

        l_budget = torch.stack(budget_terms).mean() if budget_terms else torch.zeros_like(l_sym)
        l_graph = torch.stack(graph_terms).mean() if graph_terms else torch.zeros_like(l_sym)
        l_balance = torch.stack(balance_terms).mean() if balance_terms else torch.zeros_like(l_sym)

        total = l_sym + ans_loss_weight * l_ans + 0.02 * l_pred + 0.02 * l_budget + 0.05 * l_graph + 0.02 * l_balance
        if not collect_stats:
            return total, {}

        with torch.no_grad():
            sym_pred = sym_logits.argmax(dim=-1)
            sym_acc = float(((sym_pred == sym_target_ids) & sym_loss_mask).float().sum().item() / max(1.0, float(sym_loss_mask.float().sum().item())))
            if ans_logits is not None and ans_target_ids is not None and ans_target_ids.numel() > 0:
                ans_acc = float((ans_logits.argmax(dim=-1) == ans_target_ids).float().mean().item())
            else:
                ans_acc = 0.0
            if alpha_eng_list:
                cat_eng = torch.cat([a.detach().float().reshape(-1) for a in alpha_eng_list])
                std_eng = float(cat_eng.std(unbiased=False).item()) if cat_eng.numel() > 1 else 0.0
            else:
                std_eng = 0.0

        return total, {
            "loss_total": float(total.detach().item()),
            "loss_sym": float(l_sym.detach().item()),
            "loss_ans": float(l_ans.detach().item()),
            "loss_budget": float(l_budget.detach().item()),
            "loss_graph": float(l_graph.detach().item()),
            "sym_tok_acc": round(100.0 * sym_acc, 2),
            "ans_tok_acc": round(100.0 * ans_acc, 2),
            "active_experts_mean": float(torch.stack(act_tensors).mean().item()) if act_tensors else 0.0,
            "row_entropy_syn": sum(ent_list) / max(1, len(ent_list)),
            "syn_adv_overlap": sum(ov_list) / max(1, len(ov_list)),
            "std_alpha_eng": std_eng,
        }

    def compute_six_losses(
        self,
        out: Dict[str, object],
        target_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        collect_stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits: torch.Tensor = out["logits"]  # type: ignore[assignment]
        moe_aux_list: List[Dict[str, torch.Tensor]] = out["moe_aux"]  # type: ignore[assignment]

        per_tok_ce = F.cross_entropy(
            logits.view(-1, logits.shape[-1]).float(),
            target_ids.view(-1),
            reduction="none",
        ).view_as(target_ids)
        esc_weight = torch.where(self.vocab.is_escape_anchor_tensor(target_ids), 2.5, 1.0)
        m_f = loss_mask.float() * esc_weight
        row_denom = m_f.sum(dim=-1)
        valid_rows = (row_denom > 0.0).float()
        row_loss = (per_tok_ce * m_f).sum(dim=-1) / row_denom.clamp(min=1e-6)
        l_diff = (row_loss * valid_rows).sum() / valid_rows.sum().clamp(min=1.0)

        if self.ablation.static_top2_baseline:
            if not collect_stats:
                return l_diff, {}
            return l_diff, {
                "loss_total": float(l_diff.detach().item()),
                "loss_diff": float(l_diff.detach().item()),
                "loss_align": 0.0,
                "loss_budget": 0.0,
                "loss_graph": 0.0,
                "probe_alignment_cosine": 0.0,
                "active_experts_mean": float(
                    torch.stack([m["active_count"].mean() for m in moe_aux_list]).mean().item()
                )
                if moe_aux_list
                else 2.0,
            }

        budget_terms = []
        graph_terms = []
        balance_terms = []
        ent_list = []
        ov_list = []
        act_tensors = []
        moe_layers = [l for l in self.ffn_layers if isinstance(l, TriStateMoELayer)]
        u_epi: torch.Tensor = out["u_epistemic"]  # type: ignore[assignment]
        alpha_eng_list: List[torch.Tensor] = out["alpha_eng_list"]  # type: ignore[assignment]
        a_eng_ref = alpha_eng_list[-1].detach().float() if alpha_eng_list else torch.zeros_like(u_epi, dtype=torch.float32)
        g_target = 0.55 + 0.85 * torch.sigmoid(4.0 * (u_epi.detach().float() - 0.35 - 0.35 * a_eng_ref))

        for m_layer, m_aux in zip(moe_layers, moe_aux_list):
            act_sum = m_aux["gates"].float().sum(dim=-1)
            if collect_stats:
                act_tensors.append(m_aux["active_count"].mean())
            budget_terms.append((act_sum - g_target).pow(2).mean())
            balance_terms.append(m_layer.load_balance_loss(m_aux))
            if not self.ablation.disable_bipolar_graph:
                g_loss, g_met = m_layer.graph.graph_regularization_loss(
                    manifolds=m_aux.get("graph_manifolds"),  # type: ignore[arg-type]
                    collect_metrics=collect_stats,
                )
                graph_terms.append(g_loss)
                if collect_stats:
                    ent_list.append(g_met["row_entropy_syn"])
                    ov_list.append(g_met["syn_adv_overlap"])

        l_pred = self._collect_pml_pred_loss(l_diff)
        l_budget = torch.stack(budget_terms).mean() if budget_terms else torch.zeros_like(l_diff)
        l_graph = torch.stack(graph_terms).mean() if graph_terms else torch.zeros_like(l_diff)
        l_balance = torch.stack(balance_terms).mean() if balance_terms else torch.zeros_like(l_diff)

        total = l_diff + 0.02 * l_pred + 0.05 * l_budget + 0.05 * l_graph + 0.02 * l_balance
        if not collect_stats:
            return total, {}

        if alpha_eng_list:
            cat_eng = torch.cat([a.detach().float().reshape(-1) for a in alpha_eng_list])
            std_eng = float(cat_eng.std(unbiased=False).item()) if cat_eng.numel() > 1 else 0.0
        else:
            std_eng = 0.0

        stats = {
            "loss_total": float(total.detach().item()),
            "loss_diff": float(l_diff.detach().item()),
            "loss_align": 0.0,
            "loss_budget": float(l_budget.detach().item()),
            "loss_graph": float(l_graph.detach().item()),
            "loss_balance": float(l_balance.detach().item()),
            "probe_alignment_cosine": 0.0,
            "row_entropy_syn": sum(ent_list) / max(1, len(ent_list)),
            "syn_adv_overlap": sum(ov_list) / max(1, len(ov_list)),
            "std_alpha_eng": std_eng,
            "active_experts_mean": float(torch.stack(act_tensors).mean().item()) if act_tensors else 0.0,
        }
        return total, stats

    def run_delayed_scorer_microstep(
        self, scorer_optimizer: torch.optim.Optimizer, tau_u: float = 0.25
    ) -> Tuple[float, float]:
        """Deprecated in AVL-PMLv1.1 (amputated after Stage 0 Gate 1 falsification of L_util)."""
        return 0.0, 0.0
