"""Stage 3 (M4) Unified Dual-Domain Trainer for AVL-PMLv1.2-4.5B (Gate 2 2x2 Matrix + Cloud/Local 5080).

Implements Master Manual §3.1–§3.3, §5.1, & §5.5:
  - Execution Modes:
      * --mode q_avl_5080      : Tier-Ref-5080 (4-bit GQA/Dense/Routed + BF16 LoRA & full BF16 Shared/PML/Engram/FSQ, < 9 GB VRAM)
      * --mode full_bf16_cloud : Tier-L40S / H100 (100% full-parameter 4.13B BF16 + gradient checkpointing)
  - Gate 2 2x2 Factorial Matrix Switches:
      * --gen-mode {ar, block_dllm} : Causal AR (Stage 3 Phase-B default) vs Block-Diffusion
      * --pml-mode {on, off}        : Hard-Indexed Track-1 PML Whiteboard ON vs OFF
  - True Dual-Domain Supervision:
      [Prompt(V_nl) | Thinking(V_sym) | Answer(V_nl)] via Qwen3 tokenizer + Factored FSQ head!
  - Real Periodic Validation on --val-data (strictly unseen Gold v2 Val 10,000):
      * Reports overall & per-domain & step-bucketed (<=8 / 9-16 / 17-24 / 25+) val loss & token accuracy;
      * Computes K_max=0 plug-pull loss delta + 6 mandatory telemetry redlines;
      * Saves best_val safetensors checkpoint & structured JSON report.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.model import AVLPMLModel, AblationConfig, build_nl_ro_span_weights
from avl_pmlv1.engine.q_avl_pml import (
    build_five_group_optimizer,
    build_warmup_cosine_scheduler,
    quantize_to_q_avl_pml,
)
from avl_pmlv1.vocab.nl_three_think import format_nl_three_think_example
from avl_pmlv1.vocab.sym_vocab import SymbolicVocab


DEFAULT_LOCAL_QWEN3_DIR = (
    r"E:\avl-list\hf_cache\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)


def resolve_tokenizer(tokenizer_dir: Optional[str]) -> Any:
    cand = tokenizer_dir or (DEFAULT_LOCAL_QWEN3_DIR if Path(DEFAULT_LOCAL_QWEN3_DIR).exists() else "Qwen/Qwen3-4B-Base")
    is_local = Path(cand).exists()
    tok = AutoTokenizer.from_pretrained(cand, local_files_only=is_local, trust_remote_code=False)
    return tok


def step_bucket(num_steps: int) -> str:
    if num_steps <= 8:
        return "<=8"
    if num_steps <= 16:
        return "9-16"
    if num_steps <= 24:
        return "17-24"
    return "25+"


def clean_reference_cot(rec: Dict[str, Any]) -> str:
    """Format raw mega_seeds reference_solution into clean step-by-step natural-language CoT text."""
    src = str(rec.get("source", "")).split(":")[0]
    raw = str(rec.get("reference_solution", "") or "").strip()
    if not raw:
        return ""
    if src == "openai/gsm8k":
        cleaned = re.sub(r"<<[^<>]*>>", "", raw)
        cleaned = re.sub(r"\n?####\s*[^\n]*$", "", cleaned).strip()
        return cleaned
    if src == "K-and-K/knights-and-knaves":
        try:
            d = json.loads(raw)
            steps = d.get("steps", [])
            if isinstance(steps, list) and steps:
                return "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
        except Exception:
            pass
    if src == "tasksource/proofwriter":
        try:
            d = json.loads(raw)
            proofs = str(d.get("allProofs", "") or "").strip()
            query = str(d.get("query", "") or "").strip()
            if proofs:
                return f"Query: {query}\nProof chain: {proofs}" if query else f"Proof chain: {proofs}"
        except Exception:
            pass
    return raw


def format_nl_cot_example(
    rec: Dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    max_prompt_len: int = 256,
    max_cot_len: int = 384,
) -> Dict[str, Any]:
    """Encode a matched-budget pure V_nl NL-CoT SFT example: [Prompt_nl | CoT_nl | Answer: <gt><eos>]."""
    domain = str(rec.get("domain", "math"))
    question = str(rec.get("question", "")).strip()
    prompt_text = f"Problem ({domain}):\n{question}\nSolve step by step and end with 'Answer: <final answer>'.\n"
    p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not p_ids:
        p_ids = [tokenizer.eos_token_id or 151643]
    p_ids = p_ids[:max_prompt_len]

    gt = str(rec.get("ground_truth_answer") or "").strip()
    eos_str = tokenizer.eos_token or ""
    ans_text = f"\nAnswer: {gt}{eos_str}"
    ans_ids = tokenizer.encode(ans_text, add_special_tokens=False)
    if not ans_ids:
        ans_ids = [tokenizer.eos_token_id or 151643]

    cot_text = clean_reference_cot(rec)
    cot_budget = max(0, max_cot_len - len(ans_ids))
    cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)[:cot_budget] if (cot_text and cot_budget > 0) else []
    target_ids = cot_ids + ans_ids

    full_ids = torch.tensor([p_ids + target_ids], dtype=torch.long, device=device)
    labels = torch.full_like(full_ids, -100)
    labels[0, len(p_ids) :] = full_ids[0, len(p_ids) :]
    ans_start = len(p_ids) + len(cot_ids)

    return {
        "full_ids": full_ids,
        "labels": labels,
        "p_len": len(p_ids),
        "cot_len": len(cot_ids),
        "ans_start": ans_start,
        "domain": domain,
        "num_steps": int(rec.get("num_steps", len(rec.get("frame_sizes", [])) or 8)),
    }


def format_dual_domain_example(
    rec: Dict[str, Any],
    tokenizer: Any,
    gen_mode: str,
    mask_p: float,
    rng: random.Random,
    device: torch.device,
    num_ro_slots: int = 64,
    max_prompt_len: int = 256,
    max_ans_len: int = 48,
    dllm_block_size: int = 8,
    val_deterministic_beats: bool = False,
) -> Dict[str, Any]:
    """Encode a Gold v2 JSONL record into exact [Prompt(V_nl) | Thinking(V_sym) | Answer(V_nl)] tensors."""
    domain = str(rec.get("domain", "math"))
    question = str(rec.get("question", "")).strip()
    prompt_text = f"Problem ({domain}):\n{question}\nThought:\n"
    p_ids_list = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not p_ids_list:
        p_ids_list = [tokenizer.eos_token_id or 151643]
    if len(p_ids_list) > max_prompt_len:
        p_ids_list = p_ids_list[:max_prompt_len]
    prompt_nl_ids = torch.tensor([p_ids_list], dtype=torch.long, device=device)
    ro_span_weights = build_nl_ro_span_weights(prompt_nl_ids, tokenizer=tokenizer, num_ro_slots=num_ro_slots)

    gold_sym = list(rec["token_ids_aligned_fsq"])
    frame_sizes = list(rec["frame_sizes"])
    reassert_mask = list(rec["reassert_mask"])
    tgt_sym = torch.tensor([gold_sym], dtype=torch.long, device=device)
    rm_t = torch.tensor([reassert_mask], dtype=torch.long, device=device)

    if gen_mode == "ar":
        inp_sym = tgt_sym
        sym_loss_mask = tgt_sym != SymbolicVocab.PAD_ID
        t_k = 0.0
    else:
        inp_sym = tgt_sym.clone()
        sym_loss_mask = torch.zeros_like(tgt_sym, dtype=torch.bool)
        # Subdivide frames into exact inference dllm_block_size (8-slot) blocks & sample full Beat-0..2 noise levels!
        sub_blocks: List[int] = []
        for flen in frame_sizes:
            if dllm_block_size > 0 and flen > dllm_block_size and flen % dllm_block_size == 0:
                sub_blocks.extend([dllm_block_size] * (flen // dllm_block_size))
            else:
                sub_blocks.append(flen)
        beat_cycle = (1.0, 0.67, 0.33)
        off = 0
        for b_i, blen in enumerate(sub_blocks):
            if val_deterministic_beats:
                p_b = beat_cycle[b_i % len(beat_cycle)]
            elif mask_p < 0.0:
                # BD3-LM trajectory-aligned per-block schedule: 25% full-block Beat-0 (1.0) + 75% U(0.20, 0.95)
                p_b = 1.0 if rng.random() < 0.25 else rng.uniform(0.20, 0.95)
            else:
                p_b = mask_p
            end_pos = min(off + blen, len(gold_sym))
            for pos in range(off, end_pos):
                if gold_sym[pos] != SymbolicVocab.PAD_ID and (p_b >= 0.999 or rng.random() < p_b):
                    inp_sym[0, pos] = SymbolicVocab.MASK_ID
                    sym_loss_mask[0, pos] = True
            off += blen
        if not sym_loss_mask.any():
            valid_pos = [i for i, tid in enumerate(gold_sym) if tid != SymbolicVocab.PAD_ID]
            pick = valid_pos[-1] if valid_pos else len(gold_sym) - 1
            inp_sym[0, pick] = SymbolicVocab.MASK_ID
            sym_loss_mask[0, pick] = True
        non_pad_cnt = max(1, sum(1 for tid in gold_sym if tid != SymbolicVocab.PAD_ID))
        t_k = float(sym_loss_mask.sum().item()) / float(non_pad_cnt)

    gt = str(rec.get("ground_truth_answer") or rec.get("reference_solution") or "").strip()
    ans_nl_ids: Optional[torch.Tensor] = None
    if gt:
        eos_str = tokenizer.eos_token or ""
        if domain == "code" and ("def " in gt or "return " in gt or "\n" in gt):
            ans_text = f"```python\n{gt}\n```{eos_str}"
            cap_ans = max(max_ans_len, 256)
        else:
            ans_text = f"Answer: {gt}{eos_str}"
            cap_ans = max_ans_len
        a_ids_list = tokenizer.encode(ans_text, add_special_tokens=False)
        if a_ids_list:
            ans_nl_ids = torch.tensor([a_ids_list[:cap_ans]], dtype=torch.long, device=device)

    return {
        "prompt_nl_ids": prompt_nl_ids,
        "ro_span_weights": ro_span_weights,
        "inp_sym": inp_sym,
        "tgt_sym": tgt_sym,
        "sym_loss_mask": sym_loss_mask,
        "ans_nl_ids": ans_nl_ids,
        "frame_sizes": frame_sizes,
        "reassert_flags": rm_t,
        "t_k": t_k,
        "domain": domain,
        "num_steps": int(rec.get("num_steps", len(frame_sizes))),
    }


def load_jsonl_traces(
    path: str,
    max_sym_len: int,
    max_samples: int = 0,
    allowed_domains: Optional[set[str]] = None,
    require_answer: bool = True,
    hydrate_all_refs: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seed_sol_cache: Optional[Dict[str, str]] = None
    p_obj = Path(path)

    def _hydrate_seed_ref(rec_id: str) -> str:
        nonlocal seed_sol_cache
        if seed_sol_cache is None:
            seed_sol_cache = {}
            cand_seeds = p_obj.parent / "mega_seeds.jsonl"
            if cand_seeds.exists():
                with open(cand_seeds, "r", encoding="utf-8") as sf:
                    for sline in sf:
                        if not sline.strip():
                            continue
                        sr = json.loads(sline)
                        if sr.get("reference_solution"):
                            dom_s = str(sr.get("domain", "")).lower()
                            if hydrate_all_refs or dom_s == "code":
                                seed_sol_cache[str(sr["id"])] = str(sr["reference_solution"]).strip()
        return seed_sol_cache.get(rec_id, "")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            dom = str(r.get("domain", "math")).lower()
            if allowed_domains is not None and dom not in allowed_domains:
                continue
            s_len = len(r.get("token_ids_aligned_fsq", []))
            if not (8 <= s_len <= max_sym_len):
                continue
            gt_s = str(r.get("ground_truth_answer") or r.get("reference_solution") or "").strip()
            if not gt_s and dom == "code":
                gt_s = _hydrate_seed_ref(str(r.get("id", "")))
                if gt_s:
                    r["ground_truth_answer"] = gt_s
            if hydrate_all_refs and not r.get("reference_solution"):
                ref_s = _hydrate_seed_ref(str(r.get("id", "")))
                if ref_s:
                    r["reference_solution"] = ref_s
            if require_answer and not gt_s:
                continue
            rows.append(r)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


class SourceBalancedSampler:
    """Deterministic source-balanced sampler (uniform or N^0.5 temperature) guaranteeing identical problem streams across arms."""

    def __init__(self, rows: List[Dict[str, Any]], mode: str = "none") -> None:
        self.rows = rows
        self.mode = mode
        self.by_src: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            src_fam = str(r.get("source", r.get("domain", "other"))).split(":")[0]
            self.by_src.setdefault(src_fam, []).append(r)
        self.sources = sorted(self.by_src.keys())
        if mode == "sqrt":
            raw_w = [math.sqrt(float(len(self.by_src[s]))) for s in self.sources]
        else:
            raw_w = [float(len(self.by_src[s])) for s in self.sources]
        tot_w = sum(raw_w) or 1.0
        self.probs = [w / tot_w for w in raw_w]

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            s: {"count": float(len(self.by_src[s])), "sample_pct": round(100.0 * p, 2)}
            for s, p in zip(self.sources, self.probs)
        }

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        if self.mode == "none":
            return self.rows[rng.randrange(len(self.rows))]
        u = rng.random()
        acc = 0.0
        chosen = self.sources[-1]
        for s, p in zip(self.sources, self.probs):
            acc += p
            if u <= acc:
                chosen = s
                break
        bucket = self.by_src[chosen]
        return bucket[rng.randrange(len(bucket))]


def select_stratified_val(val_rows: List[Dict[str, Any]], n_target: int, seed: int = 2026) -> List[Dict[str, Any]]:
    """Deterministically stratify validation samples across domains (math/code/logic) for apples-to-apples comparison."""
    if n_target <= 0 or len(val_rows) <= n_target:
        return list(val_rows)
    rng = random.Random(seed)
    by_dom: Dict[str, List[Dict[str, Any]]] = {}
    for r in val_rows:
        by_dom.setdefault(str(r.get("domain", "other")), []).append(r)
    for lst in by_dom.values():
        rng.shuffle(lst)
    doms = sorted(by_dom.keys())
    selected: List[Dict[str, Any]] = []
    idx = 0
    while len(selected) < n_target:
        added = False
        for d in doms:
            if idx < len(by_dom[d]) and len(selected) < n_target:
                selected.append(by_dom[d][idx])
                added = True
        if not added:
            break
        idx += 1
    return selected


@torch.no_grad()
def evaluate_validation(
    model: AVLPMLModel,
    val_subset: List[Dict[str, Any]],
    tokenizer: Any,
    gen_mode: str,
    device: torch.device,
    max_prompt_len: int = 256,
    max_ans_len: int = 48,
    ans_loss_weight: float = 1.0,
    val_mask_p: float = 0.35,
    kmax0_probe_count: int = 8,
) -> Dict[str, Any]:
    """Run real dual-domain validation on unseen Gold v2 Val with domain/step-bucket breakdown & K_max=0 plug-pull."""
    was_training = model.training
    model.eval()
    val_rng = random.Random(777)

    tot_losses, sym_losses, ans_losses = [], [], []
    sym_accs, ans_accs = [], []
    acts, ents, ovs, eng_stds = [], [], [], []
    by_domain: Dict[str, Dict[str, List[float]]] = {}
    by_bucket: Dict[str, Dict[str, List[float]]] = {}

    probe_batches: List[Dict[str, Any]] = []
    normal_probe_losses: List[float] = []

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    for i, rec in enumerate(val_subset):
        ex = format_dual_domain_example(
            rec,
            tokenizer=tokenizer,
            gen_mode=gen_mode,
            mask_p=val_mask_p,
            rng=val_rng,
            device=device,
            num_ro_slots=model.cfg.num_ro_slots,
            max_prompt_len=max_prompt_len,
            max_ans_len=max_ans_len,
            val_deterministic_beats=(gen_mode == "block_dllm"),
        )
        wb = model.init_whiteboard_states(1, device, dtype=amp_dtype)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model.forward_dual_domain(
                prompt_nl_ids=ex["prompt_nl_ids"],
                sym_token_ids=ex["inp_sym"],
                ans_nl_ids=ex["ans_nl_ids"],
                gen_mode=gen_mode,
                sym_frame_sizes=ex["frame_sizes"],
                wb_states=wb,
                reassert_flags=ex["reassert_flags"],
                t_k=ex["t_k"],
                sym_clean_ids=(ex["tgt_sym"] if gen_mode == "block_dllm" else None),
                ro_span_weights=ex["ro_span_weights"],
            )
            _, st = model.compute_dual_domain_losses(
                out,
                sym_target_ids=ex["tgt_sym"],
                sym_loss_mask=ex["sym_loss_mask"],
                ans_target_ids=ex["ans_nl_ids"],
                ans_loss_weight=ans_loss_weight,
                collect_stats=True,
            )

        tot_losses.append(st["loss_total"])
        sym_losses.append(st["loss_sym"])
        sym_accs.append(st["sym_tok_acc"])
        acts.append(st["active_experts_mean"])
        ents.append(st["row_entropy_syn"])
        ovs.append(st["syn_adv_overlap"])
        eng_stds.append(st["std_alpha_eng"])
        if ex["ans_nl_ids"] is not None:
            ans_losses.append(st["loss_ans"])
            ans_accs.append(st["ans_tok_acc"])

        dom = ex["domain"]
        d_rec = by_domain.setdefault(dom, {"loss_sym": [], "sym_acc": [], "ans_acc": []})
        d_rec["loss_sym"].append(st["loss_sym"])
        d_rec["sym_acc"].append(st["sym_tok_acc"])
        if ex["ans_nl_ids"] is not None:
            d_rec["ans_acc"].append(st["ans_tok_acc"])

        bkt = step_bucket(ex["num_steps"])
        b_rec = by_bucket.setdefault(bkt, {"loss_sym": [], "sym_acc": [], "ans_acc": []})
        b_rec["loss_sym"].append(st["loss_sym"])
        b_rec["sym_acc"].append(st["sym_tok_acc"])
        if ex["ans_nl_ids"] is not None:
            b_rec["ans_acc"].append(st["ans_tok_acc"])

        if len(probe_batches) < kmax0_probe_count:
            probe_batches.append(ex)
            normal_probe_losses.append(st["loss_sym"])

    # Mandatory K_max=0 Plug-Pull Diagnostic (only when MoE layers exist; skipped in 100% Dense FFN mode!)
    kmax0_probe_losses: List[float] = []
    if len(model.cfg.moe_layer_indices) > 0 and probe_batches:
        prev_k_max = model.cfg.k_max
        model.cfg.k_max = 0
        try:
            for ex in probe_batches:
                wb = model.init_whiteboard_states(1, device, dtype=amp_dtype)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    out0 = model.forward_dual_domain(
                        prompt_nl_ids=ex["prompt_nl_ids"],
                        sym_token_ids=ex["inp_sym"],
                        ans_nl_ids=ex["ans_nl_ids"],
                        gen_mode=gen_mode,
                        sym_frame_sizes=ex["frame_sizes"],
                        wb_states=wb,
                        reassert_flags=ex["reassert_flags"],
                        t_k=ex["t_k"],
                        sym_clean_ids=(ex["tgt_sym"] if gen_mode == "block_dllm" else None),
                        ro_span_weights=ex["ro_span_weights"],
                    )
                    _, st0 = model.compute_dual_domain_losses(
                        out0,
                        sym_target_ids=ex["tgt_sym"],
                        sym_loss_mask=ex["sym_loss_mask"],
                        ans_target_ids=ex["ans_nl_ids"],
                        ans_loss_weight=ans_loss_weight,
                        collect_stats=True,
                    )
                    kmax0_probe_losses.append(st0["loss_sym"])
        finally:
            model.cfg.k_max = prev_k_max

    if was_training:
        model.train()

    def _mean(xs: List[float]) -> float:
        return round(sum(xs) / max(1, len(xs)), 4)

    delta_kmax0 = (
        round((sum(kmax0_probe_losses) - sum(normal_probe_losses)) / max(1, len(kmax0_probe_losses)), 4)
        if kmax0_probe_losses
        else 0.0
    )

    return {
        "n_val": len(val_subset),
        "val_loss_total": _mean(tot_losses),
        "val_loss_sym": _mean(sym_losses),
        "val_loss_ans": _mean(ans_losses) if ans_losses else 0.0,
        "val_sym_tok_acc": round(sum(sym_accs) / max(1, len(sym_accs)), 2),
        "val_ans_tok_acc": round(sum(ans_accs) / max(1, len(ans_accs)), 2) if ans_accs else 0.0,
        "active_experts_mean": _mean(acts),
        "row_entropy_syn": _mean(ents),
        "syn_adv_overlap": _mean(ovs),
        "std_alpha_eng": _mean(eng_stds),
        "delta_loss_kmax0": delta_kmax0,
        "by_domain": {
            d: {
                "n": len(v["loss_sym"]),
                "loss_sym": _mean(v["loss_sym"]),
                "sym_tok_acc": round(sum(v["sym_acc"]) / max(1, len(v["sym_acc"])), 2),
                "ans_tok_acc": round(sum(v["ans_acc"]) / max(1, len(v["ans_acc"])), 2) if v["ans_acc"] else 0.0,
            }
            for d, v in sorted(by_domain.items())
        },
        "by_step_bucket": {
            b: {
                "n": len(v["loss_sym"]),
                "loss_sym": _mean(v["loss_sym"]),
                "sym_tok_acc": round(sum(v["sym_acc"]) / max(1, len(v["sym_acc"])), 2),
                "ans_tok_acc": round(sum(v["ans_acc"]) / max(1, len(v["ans_acc"])), 2) if v["ans_acc"] else 0.0,
            }
            for b, v in sorted(by_bucket.items())
        },
    }


@torch.no_grad()
def evaluate_validation_nl_cot(
    model: AVLPMLModel,
    val_subset: List[Dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    max_prompt_len: int = 256,
    max_cot_len: int = 384,
    three_think: bool = False,
) -> Dict[str, Any]:
    """Evaluate matched-budget pure V_nl causal NL-CoT or Three-Think SFT baseline on unseen Val."""
    was_training = model.training
    model.eval()
    tot_losses, ans_losses, cot_accs, ans_accs = [], [], [], []
    by_domain: Dict[str, Dict[str, List[float]]] = {}
    by_bucket: Dict[str, Dict[str, List[float]]] = {}

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    for rec in val_subset:
        if three_think:
            ex = format_nl_three_think_example(
                rec, tokenizer, device, max_prompt_len=max_prompt_len, max_target_len=max_cot_len, is_val=True
            )
        else:
            ex = format_nl_cot_example(rec, tokenizer, device, max_prompt_len=max_prompt_len, max_cot_len=max_cot_len)
        full_ids = ex["full_ids"]
        labels = ex["labels"]
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, _ = model.forward_nl_causal(full_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_all = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

            ans_st = max(0, ex["ans_start"] - 1)
            ans_lg = shift_logits[:, ans_st:, :]
            ans_lb = shift_labels[:, ans_st:]
            ans_mask = ans_lb != -100
            if ans_mask.any():
                loss_a = F.cross_entropy(ans_lg.reshape(-1, ans_lg.size(-1)), ans_lb.reshape(-1), ignore_index=-100)
                ans_acc = float(((ans_lg.argmax(dim=-1) == ans_lb) & ans_mask).sum().item()) / float(ans_mask.sum().item()) * 100.0
            else:
                loss_a = torch.zeros((), device=device)
                ans_acc = 0.0

            valid_m = shift_labels != -100
            tok_acc = float(((shift_logits.argmax(dim=-1) == shift_labels) & valid_m).sum().item()) / max(1.0, float(valid_m.sum().item())) * 100.0

        l_tot = float(loss_all.item())
        l_ans = float(loss_a.item())
        tot_losses.append(l_tot)
        ans_losses.append(l_ans)
        cot_accs.append(tok_acc)
        ans_accs.append(ans_acc)

        d_rec = by_domain.setdefault(ex["domain"], {"loss_sym": [], "sym_acc": [], "ans_acc": []})
        d_rec["loss_sym"].append(l_tot)
        d_rec["sym_acc"].append(tok_acc)
        d_rec["ans_acc"].append(ans_acc)

        b_rec = by_bucket.setdefault(step_bucket(ex["num_steps"]), {"loss_sym": [], "sym_acc": [], "ans_acc": []})
        b_rec["loss_sym"].append(l_tot)
        b_rec["sym_acc"].append(tok_acc)
        b_rec["ans_acc"].append(ans_acc)

    if was_training:
        model.train()

    def _mean(xs: List[float]) -> float:
        return round(sum(xs) / max(1, len(xs)), 4)

    return {
        "n_val": len(val_subset),
        "val_loss_total": _mean(tot_losses),
        "val_loss_sym": _mean(tot_losses),
        "val_loss_ans": _mean(ans_losses),
        "val_sym_tok_acc": round(sum(cot_accs) / max(1, len(cot_accs)), 2),
        "val_ans_tok_acc": round(sum(ans_accs) / max(1, len(ans_accs)), 2),
        "active_experts_mean": 0.0,
        "row_entropy_syn": 0.0,
        "syn_adv_overlap": 0.0,
        "std_alpha_eng": 0.0,
        "delta_loss_kmax0": 0.0,
        "by_domain": {
            d: {
                "n": len(v["loss_sym"]),
                "loss_sym": _mean(v["loss_sym"]),
                "sym_tok_acc": round(sum(v["sym_acc"]) / max(1, len(v["sym_acc"])), 2),
                "ans_tok_acc": round(sum(v["ans_acc"]) / max(1, len(v["ans_acc"])), 2),
            }
            for d, v in sorted(by_domain.items())
        },
        "by_step_bucket": {
            b: {
                "n": len(v["loss_sym"]),
                "loss_sym": _mean(v["loss_sym"]),
                "sym_tok_acc": round(sum(v["sym_acc"]) / max(1, len(v["sym_acc"])), 2),
                "ans_tok_acc": round(sum(v["ans_acc"]) / max(1, len(v["ans_acc"])), 2),
            }
            for b, v in sorted(by_bucket.items())
        },
    }


def save_checkpoint(model: AVLPMLModel, ckpt_out: Path, is_lora: bool) -> None:
    if is_lora:
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        sd_save = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in trainable_names}
    else:
        sd_save = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k != "nl_head.weight"}
    save_file(sd_save, str(ckpt_out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["q_avl_5080", "full_bf16_cloud"], default="q_avl_5080")
    parser.add_argument("--gen-mode", choices=["ar", "block_dllm", "nl_cot", "nl_three_think"], default="ar")
    parser.add_argument("--pml-mode", choices=["on", "off"], default="on")
    parser.add_argument("--arm-name", type=str, default="")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "model_4b_dense_pml.yaml"))
    parser.add_argument("--init-ckpt", type=str, default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "avl_pmlv12_4b_dense_step0_bf16.safetensors"))
    parser.add_argument("--tokenizer-dir", type=str, default="")
    parser.add_argument("--data", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_train_fsq.jsonl"))
    parser.add_argument("--val-data", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_val_fsq.jsonl"))
    parser.add_argument("--save-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "stage3_phase_b"))
    parser.add_argument("--report-out", type=str, default="")
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-sym-len", type=int, default=384)
    parser.add_argument("--max-ans-len", type=int, default=48)
    parser.add_argument("--ans-loss-weight", type=float, default=1.0)
    parser.add_argument("--domains", type=str, default="math,logic", help="Comma-separated domains (default 'math,logic'; pass 'math,logic,code' to include hydrated Python code).")
    parser.add_argument("--balance-sources", choices=["none", "sqrt"], default="none", help="Source sampling balance: none (uniform over rows) or sqrt (N^0.5 temperature balance across datasets).")
    parser.add_argument("--allow-empty-answer", action="store_true", help="Allow records without ground_truth_answer/reference_solution.")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume-ckpt", type=str, default="", help="Optional LoRA/full trainable checkpoint to resume from.")
    parser.add_argument("--start-step", type=int, default=0, help="Completed step count when resuming.")
    parser.add_argument("--init-best-val", type=float, default=float("inf"), help="Existing best_val_loss when resuming.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    is_nl_cot = args.gen_mode in ("nl_cot", "nl_three_think")
    is_three_think = args.gen_mode == "nl_three_think"
    eff_pml_mode = "off" if is_nl_cot else args.pml_mode
    default_nl_arm = "nl_three_think_sft" if is_three_think else "nl_cot_sft"
    arm_name = args.arm_name or (default_nl_arm if is_nl_cot else f"{args.gen_mode}_pml_{eff_pml_mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AVLPMLConfig.from_yaml(args.config)
    ablation = AblationConfig(disable_pml=(eff_pml_mode == "off"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_out) if args.report_out else (save_dir / f"{arm_name}_train_report.json")

    print(
        f"=== [Stage 3 Dual-Domain Trainer | arm={arm_name} | mode={args.mode} | gen={args.gen_mode} | pml={eff_pml_mode} | "
        f"balance={args.balance_sources} | steps={args.total_steps} x accum={args.grad_accum}] ===",
        flush=True,
    )

    tokenizer = resolve_tokenizer(args.tokenizer_dir or None)
    print(f"[Tokenizer Ready] vocab_size={len(tokenizer)}, eos_id={tokenizer.eos_token_id}", flush=True)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16 if device.type == "cuda" else torch.float32)
    model = AVLPMLModel(cfg, ablation=ablation)
    torch.set_default_dtype(prev_dtype)

    if Path(args.init_ckpt).exists():
        sd = load_file(args.init_ckpt, device="cpu")
        model.load_state_dict(sd, strict=False)
        model.nl_head.weight = model.nl_embed.weight
        del sd
        gc.collect()
        print(f"[Loaded Init Ckpt] {args.init_ckpt}", flush=True)
    else:
        print(f"[Warning] init_ckpt not found ({args.init_ckpt}); running from random init!", flush=True)

    # Freeze unused Stage-0-only modules, disabled PML, and (in nl_cot mode) all symbolic-only modules!
    nl_cot_freeze_tokens = (
        "sym_embed_head.",
        "chunk_slot_embed.",
        "block_idx_embed.",
        "sym_final_norm.",
        "engrams.",
        "dir_head.",
        ".pml.",
        ".alpha_calm",
        ".film_proj.",
        ".film_tau.",
    )
    for name, p in model.named_parameters():
        if any(k in name for k in ("hyper_router.", "sc_proj.", "in_span_tag.", "alpha_span_tag")):
            p.requires_grad_(False)
        if eff_pml_mode == "off" and (".pml." in name or ".alpha_calm" in name or ".film_proj." in name):
            p.requires_grad_(False)
        if is_nl_cot and any(k in name for k in nl_cot_freeze_tokens):
            p.requires_grad_(False)

    is_lora = args.mode == "q_avl_5080"
    if is_lora:
        q_stats = quantize_to_q_avl_pml(model, lora_rank_gqa=16, lora_rank_dense=16, lora_rank_routed=0, group_size=64)
        print(f"[Q-AVL-PML Hybrid Ready] {q_stats}", flush=True)

    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        sd_res = load_file(args.resume_ckpt, device="cpu")
        model.load_state_dict(sd_res, strict=False)
        model.nl_head.weight = model.nl_embed.weight
        del sd_res
        gc.collect()
        print(f"[Resumed Ckpt] {args.resume_ckpt} (start_step={args.start_step}, init_best_val={args.init_best_val})", flush=True)

    model.gradient_checkpointing = True
    model.to(device)
    opt, opt_summary = build_five_group_optimizer(model, stage="B", is_lora_hybrid=is_lora)
    sched = build_warmup_cosine_scheduler(opt, total_steps=args.total_steps, warmup_ratio=0.10)
    if args.start_step > 0:
        for _ in range(args.start_step):
            sched.step()
    print("[5-Group Optimizer]", json.dumps(opt_summary, indent=2), flush=True)

    allowed_doms = {d.strip().lower() for d in args.domains.split(",") if d.strip()} if args.domains.strip().lower() != "all" else None
    req_ans = not args.allow_empty_answer
    train_dataset = load_jsonl_traces(
        args.data,
        max_sym_len=args.max_sym_len,
        max_samples=args.max_train_samples,
        allowed_domains=allowed_doms,
        require_answer=req_ans,
        hydrate_all_refs=is_nl_cot,
    )
    if not train_dataset:
        raise RuntimeError(f"No valid traces loaded from {args.data} with max_sym_len={args.max_sym_len}, domains={allowed_doms}")
    sampler = SourceBalancedSampler(train_dataset, mode=args.balance_sources)
    print(
        f"[Train Dataset Loaded] {len(train_dataset)} Gold v2 traces (S_sym <= {args.max_sym_len}, domains={allowed_doms or 'all'}, "
        f"balance={args.balance_sources})",
        flush=True,
    )
    print("[Source Sampling Mix]", json.dumps(sampler.summary(), indent=2), flush=True)

    val_subset: List[Dict[str, Any]] = []
    if args.val_data and Path(args.val_data).exists() and args.val_samples > 0:
        raw_val = load_jsonl_traces(
            args.val_data,
            max_sym_len=args.max_sym_len,
            max_samples=max(args.val_samples * 5, 2000),
            allowed_domains=allowed_doms,
            require_answer=req_ans,
            hydrate_all_refs=is_nl_cot,
        )
        val_subset = select_stratified_val(raw_val, n_target=args.val_samples, seed=args.seed)
        print(f"[Val Dataset Loaded] {len(val_subset)} stratified unseen Gold v2 Val traces from {args.val_data}", flush=True)
    else:
        print("[Warning] --val-data not found or --val-samples=0; periodic validation skipped.", flush=True)

    # Separate sample_rng from mask_rng so AR, Block-dLLM, & NL-CoT SFT draw the EXACT SAME problems in identical order!
    sample_rng = random.Random(args.seed)
    mask_rng = random.Random(args.seed + 10007)
    if args.start_step > 0:
        for _ in range(args.start_step * args.grad_accum):
            _ = sampler.sample(sample_rng)
            _ = mask_rng.random()
    torch.manual_seed(args.seed + args.start_step)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    history_train: List[Dict[str, Any]] = []
    history_val: List[Dict[str, Any]] = []
    best_val_loss = float(args.init_best_val)
    best_ckpt_path: Optional[str] = (
        str(save_dir / f"stage3_{arm_name}_best_val.safetensors")
        if (args.start_step > 0 and (save_dir / f"stage3_{arm_name}_best_val.safetensors").exists())
        else None
    )

    t0 = time.perf_counter()
    for step in range(args.start_step + 1, args.total_steps + 1):
        opt.zero_grad(set_to_none=True)
        # In block_dllm mode, mask_p=-1.0 samples exact per-8-slot-block Beat-0..2 trajectories (25% full-mask 1.0 + 75% U[0.20, 0.95])
        mask_p = 0.0 if args.gen_mode == "ar" else -1.0
        last_stats: Dict[str, float] = {}

        for micro in range(args.grad_accum):
            rec = sampler.sample(sample_rng)
            if is_nl_cot:
                if is_three_think:
                    ex = format_nl_three_think_example(
                        rec,
                        tokenizer=tokenizer,
                        device=device,
                        rng=mask_rng,
                        max_prompt_len=args.max_prompt_len,
                        max_target_len=args.max_sym_len,
                        counterfactual_post_prob=0.20,
                        is_val=False,
                    )
                else:
                    ex = format_nl_cot_example(
                        rec,
                        tokenizer=tokenizer,
                        device=device,
                        max_prompt_len=args.max_prompt_len,
                        max_cot_len=args.max_sym_len,
                    )
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits, _ = model.forward_nl_causal(ex["full_ids"])
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = ex["labels"][:, 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
                    (loss / float(args.grad_accum)).backward()
                if micro == args.grad_accum - 1:
                    with torch.no_grad():
                        valid_m = shift_labels != -100
                        tok_acc = float(((shift_logits.argmax(dim=-1) == shift_labels) & valid_m).sum().item()) / max(1.0, float(valid_m.sum().item())) * 100.0
                        ans_st = max(0, ex["ans_start"] - 1)
                        ans_lg = shift_logits[:, ans_st:, :]
                        ans_lb = shift_labels[:, ans_st:]
                        ans_m = ans_lb != -100
                        loss_a = float(F.cross_entropy(ans_lg.reshape(-1, ans_lg.size(-1)), ans_lb.reshape(-1), ignore_index=-100).item()) if ans_m.any() else 0.0
                        ans_acc = float(((ans_lg.argmax(dim=-1) == ans_lb) & ans_m).sum().item()) / max(1.0, float(ans_m.sum().item())) * 100.0 if ans_m.any() else 0.0
                        l_val = float(loss.item())
                        last_stats = {
                            "loss_total": l_val,
                            "loss_sym": l_val,
                            "loss_ans": loss_a,
                            "sym_tok_acc": round(tok_acc, 2),
                            "ans_tok_acc": round(ans_acc, 2),
                            "active_experts_mean": 0.0,
                            "row_entropy_syn": 0.0,
                            "syn_adv_overlap": 0.0,
                        }
            else:
                ex = format_dual_domain_example(
                    rec,
                    tokenizer=tokenizer,
                    gen_mode=args.gen_mode,
                    mask_p=mask_p,
                    rng=mask_rng,
                    device=device,
                    num_ro_slots=cfg.num_ro_slots,
                    max_prompt_len=args.max_prompt_len,
                    max_ans_len=args.max_ans_len,
                )
                wb = model.init_whiteboard_states(1, device, dtype=amp_dtype)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    out = model.forward_dual_domain(
                        prompt_nl_ids=ex["prompt_nl_ids"],
                        sym_token_ids=ex["inp_sym"],
                        ans_nl_ids=ex["ans_nl_ids"],
                        gen_mode=args.gen_mode,
                        sym_frame_sizes=ex["frame_sizes"],
                        wb_states=wb,
                        reassert_flags=ex["reassert_flags"],
                        t_k=ex["t_k"],
                        sym_clean_ids=(ex["tgt_sym"] if args.gen_mode == "block_dllm" else None),
                        ro_span_weights=ex["ro_span_weights"],
                    )
                    loss, stats = model.compute_dual_domain_losses(
                        out,
                        sym_target_ids=ex["tgt_sym"],
                        sym_loss_mask=ex["sym_loss_mask"],
                        ans_target_ids=ex["ans_nl_ids"],
                        ans_loss_weight=args.ans_loss_weight,
                        collect_stats=(micro == args.grad_accum - 1),
                    )
                    (loss / float(args.grad_accum)).backward()
                if micro == args.grad_accum - 1:
                    last_stats = stats

        if not math.isfinite(last_stats.get("loss_total", float("nan"))):
            raise RuntimeError(f"Redline halt: non-finite loss at step {step} in arm {arm_name}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if device.type == "cuda" and step % 25 == 0:
            torch.cuda.empty_cache()

        if step == 1 or step % args.log_every == 0 or step == args.total_steps:
            vram = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
            vram_resv = round(torch.cuda.memory_reserved(device) / (1024**3), 2) if device.type == "cuda" else 0.0
            row_log = {
                "step": step,
                "loss_total": round(last_stats["loss_total"], 4),
                "loss_sym": round(last_stats["loss_sym"], 4),
                "loss_ans": round(last_stats["loss_ans"], 4),
                "sym_tok_acc": last_stats["sym_tok_acc"],
                "ans_tok_acc": last_stats["ans_tok_acc"],
                "active_experts_mean": round(last_stats["active_experts_mean"], 3),
                "row_entropy_syn": round(last_stats["row_entropy_syn"], 3),
                "syn_adv_overlap": round(last_stats["syn_adv_overlap"], 4),
                "peak_vram_gb": vram,
                "reserved_vram_gb": vram_resv,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            history_train.append(row_log)
            print(
                f"  [{arm_name} | Step {step:05d}/{args.total_steps}] tot={row_log['loss_total']:.4f} "
                f"sym={row_log['loss_sym']:.4f}({row_log['sym_tok_acc']:.1f}%) "
                f"ans={row_log['loss_ans']:.4f}({row_log['ans_tok_acc']:.1f}%) | "
                f"act={row_log['active_experts_mean']:.2f} ent={row_log['row_entropy_syn']:.3f} "
                f"ov={row_log['syn_adv_overlap']:.4f} | vram={vram}GB(resv={vram_resv}GB) | {row_log['elapsed_sec']}s",
                flush=True,
            )

        if val_subset and (step % args.eval_every == 0 or step == args.total_steps):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            t_val = time.perf_counter()
            if is_nl_cot:
                val_res = evaluate_validation_nl_cot(
                    model=model,
                    val_subset=val_subset,
                    tokenizer=tokenizer,
                    device=device,
                    max_prompt_len=args.max_prompt_len,
                    max_cot_len=args.max_sym_len,
                    three_think=is_three_think,
                )
            else:
                val_res = evaluate_validation(
                    model=model,
                    val_subset=val_subset,
                    tokenizer=tokenizer,
                    gen_mode=args.gen_mode,
                    device=device,
                    max_prompt_len=args.max_prompt_len,
                    max_ans_len=args.max_ans_len,
                    ans_loss_weight=args.ans_loss_weight,
                )
            val_res["step"] = step
            val_res["val_sec"] = round(time.perf_counter() - t_val, 2)
            history_val.append(val_res)
            print(
                f"  [VAL {arm_name} @ Step {step}] tot={val_res['val_loss_total']:.4f} "
                f"sym={val_res['val_loss_sym']:.4f}({val_res['val_sym_tok_acc']:.1f}%) "
                f"ans={val_res['val_loss_ans']:.4f}({val_res['val_ans_tok_acc']:.1f}%) | "
                f"dL_kmax0={val_res['delta_loss_kmax0']:+.4f} | act={val_res['active_experts_mean']:.2f} | "
                f"{val_res['val_sec']}s",
                flush=True,
            )
            if val_res["val_loss_total"] < best_val_loss:
                best_val_loss = val_res["val_loss_total"]
                best_out = save_dir / f"stage3_{arm_name}_best_val.safetensors"
                save_checkpoint(model, best_out, is_lora=is_lora)
                best_ckpt_path = str(best_out)
                print(f"  [New Best Val Ckpt] {best_out} (val_loss_total={best_val_loss:.4f})", flush=True)

        if step % args.save_every == 0 or step == args.total_steps:
            ckpt_out = save_dir / f"stage3_{arm_name}_{args.mode}_step{step}.safetensors"
            save_checkpoint(model, ckpt_out, is_lora=is_lora)
            print(f"  [Saved Checkpoint] {ckpt_out}", flush=True)

    peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
    final_report = {
        "arm_name": arm_name,
        "mode": args.mode,
        "gen_mode": args.gen_mode,
        "pml_mode": eff_pml_mode,
        "balance_sources": args.balance_sources,
        "source_mix": sampler.summary(),
        "total_steps": args.total_steps,
        "grad_accum": args.grad_accum,
        "peak_vram_gb": peak_vram_gb,
        "best_val_loss": None if not math.isfinite(best_val_loss) else round(best_val_loss, 4),
        "best_ckpt": best_ckpt_path,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "train_history": history_train,
        "val_history": history_val,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved Train+Val Report] {report_path}", flush=True)


if __name__ == "__main__":
    main()
