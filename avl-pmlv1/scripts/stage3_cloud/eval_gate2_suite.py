"""Gate 2 Unified 6-Arm Evaluation Suite for AVL-PMLv1.2 (Master Manual §5.5).

Evaluates the exact 2x2 factorial matrix + 2 external reference anchors on identical problems,
measuring true end-to-end wall-clock latency from NL Prompt -> Final Answer:

  Arms 1-4 (2x2 Factorial Matrix on 4.127B):
    1) ar_pml_off         : Causal AR + Hard-Indexed Track-1 Whiteboard OFF
    2) ar_pml_on          : Causal AR + Hard-Indexed Track-1 Whiteboard ON
    3) block_dllm_pml_off : Prefix-KV Block-dLLM + Hard-Indexed Track-1 Whiteboard OFF
    4) block_dllm_pml_on  : Prefix-KV Block-dLLM + Hard-Indexed Track-1 Whiteboard ON

  Arms 5-6 (Mandatory External Anchors):
    5) nl_cot_ref         : Same-budget vanilla Qwen3-4B V_nl causal NL-CoT reference
    6) dsl_ext_exec       : Model-generated V_sym DSL + Deterministic Python Symbolic Executor
                            (isolates symbolic decomposition accuracy vs neural execution accuracy!)

Reports per arm:
  - Overall Exact Match (on gradable Math/Logic problems)
  - Step-bucketed Exact Match (<=6 / 7-14 / 15-24 / 25-32 / >32 steps)
  - Average thinking token length & compression ratio vs nl_cot_ref
  - True E2E latency breakdown (prefill / think / repair / verbalize / total, p50, p95) & throughput (prob/s)
  - Post-Thinking repair audit (fixed_count, broken_count, net_fix, accept_rate_pct, repair_overhead_ms)
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import gc
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.four_phase_runner import FourPhaseRunner
from avl_pmlv1.engine.model import AVLPMLModel, AblationConfig
from avl_pmlv1.engine.q_avl_pml import merge_lora_delta_into_bf16_model, quantize_to_q_avl_pml
from avl_pmlv1.vocab.sym_vocab import SymbolicVocab, extract_prompt_number_spans


DEFAULT_LOCAL_QWEN3_DIR = (
    r"E:\avl-list\hf_cache\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANS_LINE_RE = re.compile(r"(?:Answer|Final Answer|####)\s*:?\s*([^\n]+)", re.IGNORECASE)


def resolve_tokenizer(tokenizer_dir: Optional[str]) -> Any:
    cand = tokenizer_dir or (DEFAULT_LOCAL_QWEN3_DIR if Path(DEFAULT_LOCAL_QWEN3_DIR).exists() else "Qwen/Qwen3-4B-Base")
    return AutoTokenizer.from_pretrained(cand, local_files_only=Path(cand).exists(), trust_remote_code=False)


def _sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def gate2_step_bucket(num_steps: int) -> str:
    if num_steps <= 6:
        return "<=6"
    if num_steps <= 14:
        return "7-14"
    if num_steps <= 24:
        return "15-24"
    if num_steps <= 32:
        return "25-32"
    return ">32"


def parse_numeric(text: str) -> Optional[float]:
    s = text.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    try:
        if "/" in s and s.count("/") == 1:
            num_s, den_s = s.split("/")
            den = float(den_s)
            if abs(den) > 1e-12:
                return float(num_s) / den
        return float(s)
    except Exception:
        return None


def extract_answer_candidate(text: str, domain: str) -> str:
    t = text.strip()
    if not t:
        return ""
    m_box = _BOXED_RE.findall(t)
    if m_box:
        t = m_box[-1].strip()
    else:
        m_ans = _ANS_LINE_RE.findall(t)
        if m_ans:
            t = m_ans[-1].strip()

    dom = domain.lower()
    if dom == "logic":
        low = t.lower()
        # Handle Knights & Knaves boolean vector answers like "[true, false, true, ...]"
        list_matches = re.findall(r"\[(?:\s*(?:true|false|knight|knave)\s*,?\s*)+\]", low)
        if list_matches:
            items = re.findall(r"\b(true|false|knight|knave)\b", list_matches[-1])
            norm = ["true" if it in ("true", "knight") else "false" for it in items]
            return "[" + ", ".join(norm) + "]"
        # Single-label logic (ProofWriter etc.): take the LAST keyword so free-form CoT conclusions aren't shadowed by premises!
        kw_matches = re.findall(r"\b(unknown|true|false|yes|no|knight|knave)\b", low)
        if kw_matches:
            return kw_matches[-1]
        return low.split()[0].strip(".,;:!?") if low.split() else ""

    # Math / general numeric extraction
    nums = _NUM_RE.findall(t)
    if nums:
        return nums[-1].replace(",", "")
    return t.split("\n")[0].strip()


def grade_prediction(pred_text: str, gt_answer: str, domain: str) -> bool:
    gt = gt_answer.strip()
    if not gt:
        return False
    pred_cand = extract_answer_candidate(pred_text, domain)
    gt_cand = extract_answer_candidate(gt, domain) or gt.strip().lower()
    if not pred_cand:
        return False
    if pred_cand.lower() == gt_cand.lower():
        return True
    p_num = parse_numeric(pred_cand)
    g_num = parse_numeric(gt_cand)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) <= 1e-4 * max(1.0, abs(g_num))
    return False


def execute_dsl_trace(sym_token_ids: List[int], question: str, vocab: SymbolicVocab) -> Tuple[str, bool, float]:
    """Deterministic External Symbolic/Arithmetic Executor over V_sym traces (Gate 2 Arm 6).
    Decodes ⟨IN_k⟩, ⟨REG_j⟩, lossless V_lit primitives, and ⟨BIND⟩/⟨COMMIT⟩/⟨VERIFY⟩ writes
    to cleanly separate symbolic problem decomposition from neural whiteboard execution!
    """
    t0 = time.perf_counter()
    # Build inverse token map lazily
    id_to_tok: Dict[int, str] = getattr(vocab, "_cached_id_to_tok", None)  # type: ignore[assignment]
    if id_to_tok is None:
        id_to_tok = {tid: tok for tok, tid in vocab.token_to_id.items()}
        setattr(vocab, "_cached_id_to_tok", id_to_tok)

    # Populate read-only input slots ⟨IN_k⟩ from question numbers in order of appearance (1:1 with compiler & S^RO!)
    in_vals: Dict[int, Any] = {}
    for k, (_, _, m) in enumerate(extract_prompt_number_spans(question)):
        v = parse_numeric(m)
        if v is not None:
            in_vals[k] = v

    regs: Dict[int, Any] = {}
    last_written_val: Any = None

    # Split sequence into frames on frame-starting operators or ⟨EOT⟩
    frame_starts = {
        SymbolicVocab.GOAL_ID,
        SymbolicVocab.HYP_ID,
        SymbolicVocab.DEDUCE_ID,
        SymbolicVocab.SPLIT_ID,
        SymbolicVocab.VERIFY_ID,
        SymbolicVocab.COMMIT_ID,
    }
    frames: List[List[int]] = []
    cur: List[int] = []
    for tid in sym_token_ids:
        if tid == SymbolicVocab.EOT_ID:
            if cur:
                frames.append(cur)
                cur = []
            break
        if tid == SymbolicVocab.PAD_ID or tid == SymbolicVocab.MASK_ID:
            continue
        if tid in frame_starts and cur:
            frames.append(cur)
            cur = [tid]
        else:
            cur.append(tid)
    if cur:
        frames.append(cur)

    bin_ops = {
        "+", "bin_+", "aug_+",
        "-", "bin_-", "aug_-",
        "*", "bin_*", "aug_*",
        "/", "bin_/", "aug_/",
        "%", "bin_%", "aug_%",
        "^", "bin_^",
        "min", "max",
        "==", "cmp_==", "!=", "cmp_!=", "<", "cmp_<", "<=", "cmp_<=", ">", "cmp_>", ">=", "cmp_>=",
    }

    def _resolve_val(tid: int) -> Tuple[bool, Any]:
        if vocab.ro_start <= tid < vocab.ro_end:
            idx = tid - vocab.ro_start
            return (idx in in_vals, in_vals.get(idx))
        if vocab.rw_start <= tid < vocab.rw_end:
            idx = tid - vocab.rw_start
            return (idx in regs, regs.get(idx))
        tok_s = id_to_tok.get(tid, "")
        if not tok_s:
            return False, None
        low = tok_s.lower()
        if low in ("true", "false", "knight", "knave"):
            return True, low
        num = parse_numeric(tok_s)
        if num is not None:
            return True, num
        return False, None

    def _apply_binop(op: str, a: Any, b: Any) -> Optional[Any]:
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            if op in ("==", "cmp_=="):
                return "true" if str(a).lower() == str(b).lower() else "false"
            if op in ("!=", "cmp_!="):
                return "true" if str(a).lower() != str(b).lower() else "false"
            return None
        try:
            if op in ("+", "bin_+", "aug_+"):
                return a + b
            if op in ("-", "bin_-", "aug_-"):
                return a - b
            if op in ("*", "bin_*", "aug_*"):
                return a * b
            if op in ("/", "bin_/", "aug_/"):
                return (a / b) if abs(b) > 1e-12 else None
            if op in ("%", "bin_%", "aug_%"):
                return (a % b) if abs(b) > 1e-12 else None
            if op in ("^", "bin_^"):
                if abs(b) <= 12 and abs(a) <= 1e4:
                    return a ** b
                return None
            if op == "min":
                return min(a, b)
            if op == "max":
                return max(a, b)
            if op in ("==", "cmp_=="):
                return "true" if abs(a - b) <= 1e-6 else "false"
            if op in ("!=", "cmp_!="):
                return "true" if abs(a - b) > 1e-6 else "false"
            if op in ("<", "cmp_<"):
                return "true" if a < b else "false"
            if op in ("<=", "cmp_<="):
                return "true" if a <= b else "false"
            if op in (">", "cmp_>"):
                return "true" if a > b else "false"
            if op in (">=", "cmp_>="):
                return "true" if a >= b else "false"
        except Exception:
            return None
        return None

    for fr in frames:
        # Locate destination write register at the LAST ⟨BIND⟩ ⟨REG_dst⟩ (never index-0 ⟨COMMIT⟩/⟨VERIFY⟩ opcode!)
        dst_reg: Optional[int] = None
        bind_pos: int = len(fr)
        for idx_t in range(len(fr) - 2, 0, -1):
            if fr[idx_t] == SymbolicVocab.BIND_ID:
                nxt = fr[idx_t + 1]
                if vocab.rw_start <= nxt < vocab.rw_end:
                    dst_reg = nxt - vocab.rw_start
                    bind_pos = idx_t
                    break

        # 1) Direct bound value right before ⟨BIND⟩ (t-2 invariant in AVL-PMLv1.2: e.g. '56', 'True', '3')
        direct_val: Optional[Any] = None
        if bind_pos >= 2:
            ok_d, val_d = _resolve_val(fr[bind_pos - 1])
            if ok_d:
                direct_val = val_d

        # 2) Extract input operands & operator strictly BEFORE '=' padding and excluding the bound result slot (bind_pos - 1)
        expr_end = max(1, bind_pos - 1) if bind_pos < len(fr) else len(fr)
        op_found: Optional[str] = None
        operands: List[Any] = []
        for tid in fr[1:expr_end]:
            tok_s = id_to_tok.get(tid, "")
            if tok_s == "=":
                break
            if tok_s in bin_ops and op_found is None:
                op_found = tok_s
            else:
                ok_v, val_v = _resolve_val(tid)
                if ok_v:
                    operands.append(val_v)

        computed_val: Optional[Any] = None
        if op_found is not None and len(operands) >= 2:
            computed_val = _apply_binop(op_found, operands[0], operands[1])

        # Prefer exact deterministic binary re-execution when 2 operands resolve cleanly (recovers even ESC values like 7000!),
        # otherwise use direct lossless t-2 bound literal/register (e.g. '56', 'True', '3')!
        if computed_val is not None:
            final_step_val = computed_val
        elif direct_val is not None:
            final_step_val = direct_val
        elif operands:
            final_step_val = operands[-1]
        else:
            final_step_val = None

        if final_step_val is not None:
            if dst_reg is not None:
                regs[dst_reg] = final_step_val
            last_written_val = final_step_val

    exec_ms = (time.perf_counter() - t0) * 1000.0
    if last_written_val is None:
        return "", False, round(exec_ms, 3)

    if isinstance(last_written_val, float):
        if math.isfinite(last_written_val) and abs(last_written_val - round(last_written_val)) < 1e-6:
            return str(int(round(last_written_val))), True, round(exec_ms, 3)
        return f"{last_written_val:.6g}", True, round(exec_ms, 3)
    return str(last_written_val), True, round(exec_ms, 3)


@torch.no_grad()
def run_nl_cot_reference(
    model: AVLPMLModel,
    prompt_nl_ids: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int = 192,
) -> Tuple[str, int, float, float]:
    """Arm 5: Vanilla Qwen3-4B V_nl causal NL-CoT reference using KV-cached forward_nl_causal."""
    device = prompt_nl_ids.device
    eos_id = getattr(tokenizer, "eos_token_id", None)
    p_len = int(prompt_nl_ids.shape[1])

    t0 = _sync_time(device)
    logits, kvs = model.forward_nl_causal(prompt_nl_ids, past_kvs=None, start_pos=0)
    t1 = _sync_time(device)
    prefill_ms = (t1 - t0) * 1000.0

    gen_ids: List[int] = []
    cur_pos = p_len
    enc_fn = getattr(tokenizer, "encode", None)
    newline_toks = set(enc_fn("\n", add_special_tokens=False)) if callable(enc_fn) else set()
    for step_i in range(max_new_tokens):
        nxt = int(logits[0, -1, :].argmax(dim=-1).item())
        if eos_id is not None and nxt == eos_id:
            break
        gen_ids.append(nxt)
        # Early-stop base Qwen3-4B cleanly once it finishes emitting "Answer: <ans>\n" or starts a new Problem,
        # so base-model non-EOS rambling never artificially inflates NL-CoT latency or V_sym compression ratios!
        if nxt in newline_toks or (step_i + 1) % 16 == 0:
            cur_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if any(s in cur_txt for s in ("\nProblem (", "\nQuestion:", "\n\nProblem")):
                break
            m_done = re.search(r"(?:Answer|Final Answer|####)\s*:?\s*[^\n]+\n", cur_txt, re.IGNORECASE)
            if m_done is not None:
                break
        if len(gen_ids) >= max_new_tokens:
            break
        tok_t = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, kvs = model.forward_nl_causal(tok_t, past_kvs=kvs, start_pos=cur_pos)
        cur_pos += 1

    t2 = _sync_time(device)
    total_ms = (t2 - t0) * 1000.0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    for stop_seq in ("\nProblem (", "\nQuestion:", "\n\nProblem"):
        if stop_seq in text:
            text = text.split(stop_seq, 1)[0].strip()
    return text, len(gen_ids), round(prefill_ms, 2), round(total_ms, 2)


def load_arm_model(
    cfg: AVLPMLConfig,
    init_ckpt: str,
    arm_ckpt: str,
    mode: str,
    disable_pml: bool,
    device: torch.device,
    lora_scaling: float = 2.0,
) -> AVLPMLModel:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16 if device.type == "cuda" else torch.float32)
    model = AVLPMLModel(cfg, ablation=AblationConfig(disable_pml=disable_pml))
    torch.set_default_dtype(prev_dtype)

    if init_ckpt and Path(init_ckpt).exists():
        sd = load_file(init_ckpt, device="cpu")
        model.load_state_dict(sd, strict=False)
        model.nl_head.weight = model.nl_embed.weight
        del sd
        gc.collect()

    if mode == "q_avl_5080":
        quantize_to_q_avl_pml(model, lora_rank_gqa=16, lora_rank_dense=16, lora_rank_routed=0, group_size=64)

    if arm_ckpt and Path(arm_ckpt).exists():
        sd_arm = load_file(arm_ckpt, device="cpu")
        if mode == "bf16_merged":
            merge_lora_delta_into_bf16_model(model, sd_arm, lora_scaling=lora_scaling)
        else:
            model.load_state_dict(sd_arm, strict=False)
            model.nl_head.weight = model.nl_embed.weight
        del sd_arm
        gc.collect()

    model.eval()
    model.to(device)
    return model


def load_eval_problems(
    eval_file: str,
    max_samples: int,
    seed: int = 2026,
    allowed_domains: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            dom = str(r.get("domain", "math")).lower()
            if allowed_domains is not None and dom not in allowed_domains:
                continue
            # Support both Gold v2 Val/Test (ground_truth_answer) and Official Benchmarks (ground_truth / answer)
            raw_gt = r.get("ground_truth_answer")
            if raw_gt is None or str(raw_gt).strip() == "":
                raw_gt = r.get("ground_truth")
            if raw_gt is None or str(raw_gt).strip() == "":
                raw_gt = r.get("answer") or r.get("solution_answer") or ""
            gt = str(raw_gt).strip()
            q = str(r.get("question") or r.get("problem") or r.get("prompt") or "").strip()
            if not q:
                continue
            rows.append(
                {
                    "id": str(r.get("id", f"item_{len(rows)}")),
                    "domain": str(r.get("domain", "math")),
                    "source": str(r.get("source", r.get("domain", "unknown"))),
                    "question": q,
                    "ground_truth_answer": gt,
                    "num_steps": int(r.get("num_steps", len(r.get("frame_sizes", [])) or 8)),
                    "gold_sym_tokens": r.get("token_ids_aligned_fsq"),
                }
            )
    # Prioritize gradable problems (with non-empty GT) when subsampling for Gate 2 EM!
    gradable = [r for r in rows if r["ground_truth_answer"]]
    pool = gradable if len(gradable) >= max_samples else rows
    if max_samples > 0 and len(pool) > max_samples:
        rng = random.Random(seed)
        # Stratify across step buckets when available
        by_bkt: Dict[str, List[Dict[str, Any]]] = {}
        for r in pool:
            by_bkt.setdefault(gate2_step_bucket(r["num_steps"]), []).append(r)
        for lst in by_bkt.values():
            rng.shuffle(lst)
        selected: List[Dict[str, Any]] = []
        bkts = sorted(by_bkt.keys())
        idx = 0
        while len(selected) < max_samples:
            added = False
            for b in bkts:
                if idx < len(by_bkt[b]) and len(selected) < max_samples:
                    selected.append(by_bkt[b][idx])
                    added = True
            if not added:
                break
            idx += 1
        return selected
    return pool


def summarize_arm(records: List[Dict[str, Any]], ref_nl_tokens: Optional[float] = None) -> Dict[str, Any]:
    n = len(records)
    gradable = [r for r in records if r["gradable"]]
    n_g = len(gradable)
    em_pct = round(100.0 * sum(1 for r in gradable if r["correct"]) / max(1, n_g), 2)

    by_bkt: Dict[str, List[bool]] = {}
    by_dom: Dict[str, List[bool]] = {}
    by_src: Dict[str, List[bool]] = {}
    for r in gradable:
        by_bkt.setdefault(r["step_bucket"], []).append(bool(r["correct"]))
        by_dom.setdefault(str(r.get("domain", "other")), []).append(bool(r["correct"]))
        src_fam = str(r.get("source", "other")).split(":")[0]
        by_src.setdefault(src_fam, []).append(bool(r["correct"]))
    bucket_acc = {
        b: {"n": len(v), "exact_match_pct": round(100.0 * sum(v) / max(1, len(v)), 2)} for b, v in sorted(by_bkt.items())
    }
    domain_acc = {
        d: {"n": len(v), "exact_match_pct": round(100.0 * sum(v) / max(1, len(v)), 2)} for d, v in sorted(by_dom.items())
    }
    source_acc = {
        s: {"n": len(v), "exact_match_pct": round(100.0 * sum(v) / max(1, len(v)), 2)} for s, v in sorted(by_src.items())
    }

    e2e_list = sorted(r["total_e2e_ms"] for r in records)
    p50 = e2e_list[len(e2e_list) // 2] if e2e_list else 0.0
    p95 = e2e_list[min(len(e2e_list) - 1, int(0.95 * len(e2e_list)))] if e2e_list else 0.0
    avg_e2e = sum(e2e_list) / max(1, n)
    throughput = 1000.0 / max(1e-6, avg_e2e) if n > 0 else 0.0

    avg_think_toks = sum(r["think_tokens"] for r in records) / max(1, n)
    comp_ratio = round(ref_nl_tokens / max(1.0, avg_think_toks), 2) if (ref_nl_tokens and avg_think_toks > 0) else None

    fixed_cnt = sum(1 for r in gradable if (not r["draft_correct"]) and r["correct"])
    broken_cnt = sum(1 for r in gradable if r["draft_correct"] and (not r["correct"]))
    rep_acc_cnt = sum(1 for r in records if r.get("repair_accepted", False))
    avg_rep_ms = sum(r.get("repair_ms", 0.0) for r in records) / max(1, n)

    return {
        "n_total": n,
        "n_gradable": n_g,
        "exact_match_pct": em_pct,
        "by_domain": domain_acc,
        "by_source": source_acc,
        "by_step_bucket": bucket_acc,
        "avg_think_tokens": round(avg_think_toks, 1),
        "compression_vs_nl_cot": comp_ratio,
        "avg_prefill_ms": round(sum(r.get("prefill_ms", 0.0) for r in records) / max(1, n), 2),
        "avg_think_ms": round(sum(r.get("think_ms", 0.0) for r in records) / max(1, n), 2),
        "avg_repair_ms": round(avg_rep_ms, 2),
        "avg_verbalize_ms": round(sum(r.get("verbalize_ms", 0.0) for r in records) / max(1, n), 2),
        "avg_total_e2e_ms": round(avg_e2e, 2),
        "p50_e2e_ms": round(p50, 2),
        "p95_e2e_ms": round(p95, 2),
        "throughput_prob_per_sec": round(throughput, 3),
        "repair_audit": {
            "fixed_count": fixed_cnt,
            "broken_count": broken_cnt,
            "net_fix": fixed_cnt - broken_cnt,
            "repair_accept_rate_pct": round(100.0 * rep_acc_cnt / max(1, n), 2),
            "repair_overhead_ms": round(avg_rep_ms, 2),
            "repair_overhead_pct": round(100.0 * avg_rep_ms / max(1e-6, avg_e2e), 2),
        },
    }


def render_markdown_table(report: Dict[str, Any]) -> str:
    lines = [
        f"# Gate 2 Evaluation Report (`{report['eval_file']}`, n={report['n_problems']})",
        "",
        "| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for arm_name, st in report["arms"].items():
        comp = f"{st['compression_vs_nl_cot']}×" if st["compression_vs_nl_cot"] is not None else "1.0× (ref)"
        ra = st["repair_audit"]
        rep_str = f"{ra['fixed_count']} / {ra['broken_count']} / {ra['net_fix']:+d}"
        lines.append(
            f"| `{arm_name}` | {st['exact_match_pct']:.2f}% | {st['avg_think_tokens']:.1f} | {comp} | "
            f"{st['avg_total_e2e_ms']:.1f} | {st['throughput_prob_per_sec']:.2f} | {rep_str} | {ra['repair_overhead_ms']:.1f} |"
        )
    if "gold_dsl_executor_upper_bound" in report and report["gold_dsl_executor_upper_bound"] is not None:
        gb = report["gold_dsl_executor_upper_bound"]
        lines.append("")
        lines.append(
            f"> **Gold-DSL Deterministic Executor Upper-Bound Check:** "
            f"`{gb['exact_match_pct']:.2f}%` EM ({gb['correct']}/{gb['n_gradable']}, "
            f"avg exec `{gb['avg_exec_ms']:.3f} ms/problem`)."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["bf16_merged", "q_avl_5080", "full_bf16_cloud"],
        default="bf16_merged",
        help="bf16_merged folds Q-AVL-PML LoRA deltas into pure native BF16 (~7.8GB VRAM on 5080, zero 4-bit overhead!)",
    )
    parser.add_argument("--lora-scaling", type=float, default=2.0)
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "model_4b_dense_pml.yaml"))
    parser.add_argument("--init-ckpt", type=str, default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "avl_pmlv12_4b_dense_step0_bf16.safetensors"))
    parser.add_argument("--tokenizer-dir", type=str, default="")
    parser.add_argument("--eval-file", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_val_fsq.jsonl"))
    parser.add_argument("--domains", type=str, default="math,logic", help="Comma-separated domains to evaluate (default 'math,logic'; pass 'all' for all domains).")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--arms", type=str, default="ar_pml_off,ar_pml_on,block_dllm_pml_off,block_dllm_pml_on,nl_cot_ref,dsl_ext_exec")
    parser.add_argument("--ckpt-ar-pml-off", type=str, default="")
    parser.add_argument("--ckpt-ar-pml-on", type=str, default="")
    parser.add_argument("--ckpt-dllm-pml-off", type=str, default="")
    parser.add_argument("--ckpt-dllm-pml-on", type=str, default="")
    parser.add_argument("--ckpt-nl-cot-sft", type=str, default="")
    parser.add_argument("--ckpt-nl-three-think-sft", type=str, default="")
    parser.add_argument("--executor-source-arm", type=str, default="ar_pml_on")
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-sym-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--denoise-beats", type=int, default=3)
    parser.add_argument("--max-ans-tokens", type=int, default=32)
    parser.add_argument("--max-nl-cot-tokens", type=int, default=256)
    parser.add_argument("--enable-repair", action="store_true")
    parser.add_argument("--out-json", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "gate2_eval_report.json"))
    parser.add_argument("--out-md", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "gate2_eval_report.md"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AVLPMLConfig.from_yaml(args.config)
    tokenizer = resolve_tokenizer(args.tokenizer_dir or None)
    allowed_doms = (
        {d.strip().lower() for d in args.domains.split(",") if d.strip()}
        if args.domains.strip().lower() != "all"
        else None
    )
    problems = load_eval_problems(args.eval_file, max_samples=args.max_samples, seed=args.seed, allowed_domains=allowed_doms)
    arms_requested = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"=== [Gate 2 Unified Suite | mode={args.mode} | n={len(problems)} | arms={arms_requested}] ===", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ckpt_map = {
        "ar_pml_off": args.ckpt_ar_pml_off,
        "ar_pml_on": args.ckpt_ar_pml_on,
        "block_dllm_pml_off": args.ckpt_dllm_pml_off,
        "block_dllm_pml_on": args.ckpt_dllm_pml_on,
    }
    arm_specs = {
        "ar_pml_off": ("ar", True),
        "ar_pml_on": ("ar", False),
        "block_dllm_pml_off": ("block_dllm", True),
        "block_dllm_pml_on": ("block_dllm", False),
    }

    raw_arm_records: Dict[str, List[Dict[str, Any]]] = {}
    saved_sym_traces: Dict[str, List[Tuple[List[int], float]]] = {}
    vocab_ref = SymbolicVocab(cfg)

    # Load base model ONCE onto GPU; swap only arm deltas in <0.5s (both q_avl_5080 and bf16_merged!)
    shared_model: Optional[AVLPMLModel] = None
    base_trainable_cpu: Optional[Dict[str, torch.Tensor]] = None
    base_bf16_cpu: Optional[Dict[str, torch.Tensor]] = None
    touched_bf16_keys: set[str] = set()

    if args.mode == "q_avl_5080":
        t_load = time.perf_counter()
        shared_model = load_arm_model(cfg, args.init_ckpt, "", args.mode, disable_pml=False, device=device)
        trainable_keys = {n for n, p in shared_model.named_parameters() if p.requires_grad}
        base_trainable_cpu = {k: v.detach().cpu().clone() for k, v in shared_model.state_dict().items() if k in trainable_keys}
        print(f"[Base Q-AVL-PML Loaded Once] {len(base_trainable_cpu)} delta tensors cached in {time.perf_counter() - t_load:.1f}s", flush=True)
    elif args.mode == "bf16_merged":
        t_load = time.perf_counter()
        shared_model = load_arm_model(cfg, args.init_ckpt, "", args.mode, disable_pml=False, device=device)
        if args.init_ckpt and Path(args.init_ckpt).exists():
            base_bf16_cpu = load_file(args.init_ckpt, device="cpu")
        else:
            base_bf16_cpu = {k: v.detach().cpu().clone() for k, v in shared_model.state_dict().items() if k != "nl_head.weight"}
        print(f"[Base Pure-BF16 Loaded Once] {len(base_bf16_cpu)} base tensors cached in {time.perf_counter() - t_load:.1f}s", flush=True)

    def _prepare_model_for_arm(arm_ckpt: str, disable_pml: bool) -> AVLPMLModel:
        if shared_model is not None and base_trainable_cpu is not None:
            shared_model.load_state_dict(base_trainable_cpu, strict=False)
            if arm_ckpt and Path(arm_ckpt).exists():
                sd_arm = load_file(arm_ckpt, device="cpu")
                shared_model.load_state_dict(sd_arm, strict=False)
                del sd_arm
            shared_model.nl_head.weight = shared_model.nl_embed.weight
            shared_model.ablation.disable_pml = disable_pml
            shared_model.eval()
            return shared_model
        if shared_model is not None and base_bf16_cpu is not None:
            if touched_bf16_keys:
                pmap = dict(shared_model.named_parameters())
                for k_restore in touched_bf16_keys:
                    if k_restore in pmap and k_restore in base_bf16_cpu:
                        pmap[k_restore].data.copy_(base_bf16_cpu[k_restore].to(device=pmap[k_restore].device, dtype=pmap[k_restore].dtype))
                touched_bf16_keys.clear()
            if arm_ckpt and Path(arm_ckpt).exists():
                sd_arm = load_file(arm_ckpt, device="cpu")
                for k_a in sd_arm:
                    if k_a.endswith(".lora_a"):
                        touched_bf16_keys.add(f"{k_a[:-len('.lora_a')]}.weight")
                    elif not k_a.endswith(".lora_b") and k_a != "nl_head.weight":
                        touched_bf16_keys.add(k_a)
                m_info = merge_lora_delta_into_bf16_model(shared_model, sd_arm, lora_scaling=args.lora_scaling)
                del sd_arm
                print(f"  [Merged BF16 Delta] {m_info}", flush=True)
            shared_model.nl_head.weight = shared_model.nl_embed.weight
            shared_model.ablation.disable_pml = disable_pml
            shared_model.eval()
            return shared_model
        return load_arm_model(cfg, args.init_ckpt, arm_ckpt, args.mode, disable_pml, device, lora_scaling=args.lora_scaling)

    # Pure BF16 modes run WITHOUT autocast dispatcher overhead (verified dtype-safe by test_p0_1!)
    amp_enabled = (device.type == "cuda") and (args.mode == "q_avl_5080")

    # 1. Run requested AVL-PML 2x2 factorial arms
    for arm_name in ("ar_pml_off", "ar_pml_on", "block_dllm_pml_off", "block_dllm_pml_on"):
        if arm_name not in arms_requested:
            continue
        gen_mode, disable_pml = arm_specs[arm_name]
        t_arm = time.perf_counter()
        print(f"\n--- Running Arm [{arm_name}] (gen={gen_mode}, disable_pml={disable_pml}) ---", flush=True)
        model = _prepare_model_for_arm(ckpt_map[arm_name], disable_pml)
        runner = FourPhaseRunner(model)
        recs: List[Dict[str, Any]] = []
        traces_for_exec: List[Tuple[List[int], float]] = []

        for idx, prob in enumerate(problems):
            prompt_text = f"Problem ({prob['domain']}):\n{prob['question']}\nThought:\n"
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[: args.max_prompt_len]
            if not p_ids:
                p_ids = [tokenizer.eos_token_id or 151643]
            p_t = torch.tensor([p_ids], dtype=torch.long, device=device)
            do_repair = bool(args.enable_repair and gen_mode == "block_dllm")

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                res = runner.run_end_to_end_nl(
                    prompt_nl_ids=p_t,
                    tokenizer=tokenizer,
                    gen_mode=gen_mode,
                    max_sym_tokens=args.max_sym_tokens,
                    block_size=args.block_size,
                    denoise_beats=args.denoise_beats,
                    max_ans_tokens=args.max_ans_tokens,
                    enable_repair=do_repair,
                    audit_draft_for_netfix=do_repair,
                )

            gradable = bool(prob["ground_truth_answer"]) and prob["domain"].lower() in ("math", "logic")
            ok_final = grade_prediction(res.final_answer_text, prob["ground_truth_answer"], prob["domain"]) if gradable else False
            ok_draft = grade_prediction(res.draft_answer_text, prob["ground_truth_answer"], prob["domain"]) if gradable else False
            recs.append(
                {
                    "id": prob["id"],
                    "domain": prob["domain"],
                    "source": prob["source"],
                    "step_bucket": gate2_step_bucket(prob["num_steps"]),
                    "gradable": gradable,
                    "correct": ok_final,
                    "draft_correct": ok_draft,
                    "think_tokens": len(res.sym_tokens_final),
                    "prefill_ms": res.prefill_ms,
                    "think_ms": res.think_ms,
                    "repair_ms": res.repair_ms,
                    "verbalize_ms": res.verbalize_ms,
                    "total_e2e_ms": res.total_e2e_ms,
                    "repair_accepted": res.repair_accepted,
                }
            )
            traces_for_exec.append((res.sym_tokens_final, res.prefill_ms + res.think_ms + res.repair_ms))

        raw_arm_records[arm_name] = recs
        saved_sym_traces[arm_name] = traces_for_exec
        avg_ms_arm = sum(r["total_e2e_ms"] for r in recs) / max(1, len(recs))
        em_arm = 100.0 * sum(1 for r in recs if r["gradable"] and r["correct"]) / max(1, sum(1 for r in recs if r["gradable"]))
        print(f"  [Done {arm_name}] EM={em_arm:.2f}% | avg_e2e={avg_ms_arm:.1f}ms | wall={time.perf_counter() - t_arm:.1f}s", flush=True)
        del runner
        if shared_model is None:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # 2. Run NL-CoT Anchors: nl_cot_ref (Zero-Shot), nl_cot_sft (1-Phase SFT), & nl_three_think_sft (3-Phase Pre->Formal->Post SFT!)
    ref_nl_tokens: Optional[float] = None
    nl_arm_specs = (
        ("nl_cot_ref", "", False),
        ("nl_cot_sft", args.ckpt_nl_cot_sft, False),
        ("nl_three_think_sft", args.ckpt_nl_three_think_sft, True),
    )
    for nl_arm_name, nl_ckpt_path, use_three_prompt in nl_arm_specs:
        if nl_arm_name not in arms_requested:
            continue
        t_nl = time.perf_counter()
        print(f"\n--- Running Anchor [{nl_arm_name}] (Causal V_nl | ckpt={nl_ckpt_path or 'Step-0 Base'}) ---", flush=True)
        model_nl = _prepare_model_for_arm(nl_ckpt_path, disable_pml=True)
        recs_nl: List[Dict[str, Any]] = []
        for prob in problems:
            if use_three_prompt:
                prompt_text = (
                    f"Problem ({prob['domain']}):\n{prob['question']}\n"
                    f"Use three-phase thinking ([Pre] plan, [Formal] derivation, [Post] verification) "
                    f"and end with 'Answer: <final answer>'.\n"
                )
            else:
                prompt_text = f"Problem ({prob['domain']}):\n{prob['question']}\nSolve step by step and end with 'Answer: <final answer>'.\n"
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[: args.max_prompt_len]
            if not p_ids:
                p_ids = [tokenizer.eos_token_id or 151643]
            p_t = torch.tensor([p_ids], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                ans_text, n_toks, pre_ms, tot_ms = run_nl_cot_reference(
                    model_nl, p_t, tokenizer, max_new_tokens=args.max_nl_cot_tokens
                )
            gradable = bool(prob["ground_truth_answer"]) and prob["domain"].lower() in ("math", "logic")
            ok = grade_prediction(ans_text, prob["ground_truth_answer"], prob["domain"]) if gradable else False
            recs_nl.append(
                {
                    "id": prob["id"],
                    "domain": prob["domain"],
                    "source": prob["source"],
                    "step_bucket": gate2_step_bucket(prob["num_steps"]),
                    "gradable": gradable,
                    "correct": ok,
                    "draft_correct": ok,
                    "think_tokens": n_toks,
                    "prefill_ms": pre_ms,
                    "think_ms": round(max(0.0, tot_ms - pre_ms), 2),
                    "repair_ms": 0.0,
                    "verbalize_ms": 0.0,
                    "total_e2e_ms": tot_ms,
                    "repair_accepted": False,
                }
            )
        raw_arm_records[nl_arm_name] = recs_nl
        avg_toks_nl = sum(r["think_tokens"] for r in recs_nl) / max(1, len(recs_nl))
        # Prefer matched-budget nl_cot_sft as compression denominator if run, else zero-shot nl_cot_ref
        if nl_arm_name == "nl_cot_sft" or ref_nl_tokens is None:
            ref_nl_tokens = avg_toks_nl
        em_nl = 100.0 * sum(1 for r in recs_nl if r["gradable"] and r["correct"]) / max(1, sum(1 for r in recs_nl if r["gradable"]))
        print(f"  [Done {nl_arm_name}] EM={em_nl:.2f}% | avg_toks={avg_toks_nl:.1f} | wall={time.perf_counter() - t_nl:.1f}s", flush=True)
        if shared_model is None:
            del model_nl
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # 3. Run Anchor 2: dsl_ext_exec (Model-Generated V_sym DSL + Deterministic External Executor for EVERY evaluated symbolic arm!)
    if "dsl_ext_exec" in arms_requested and saved_sym_traces:
        exec_sources = list(saved_sym_traces.keys())
        for src_arm in exec_sources:
            print(f"\n--- Running Anchor [dsl_ext_exec({src_arm})] ---", flush=True)
            recs_ex: List[Dict[str, Any]] = []
            for prob, (sym_ids, gen_ms) in zip(problems, saved_sym_traces[src_arm]):
                pred_str, exec_ok, exec_ms = execute_dsl_trace(sym_ids, prob["question"], vocab_ref)
                gradable = bool(prob["ground_truth_answer"]) and prob["domain"].lower() in ("math", "logic")
                ok = grade_prediction(pred_str, prob["ground_truth_answer"], prob["domain"]) if (gradable and exec_ok) else False
                recs_ex.append(
                    {
                        "id": prob["id"],
                        "domain": prob["domain"],
                        "source": prob["source"],
                        "step_bucket": gate2_step_bucket(prob["num_steps"]),
                        "gradable": gradable,
                        "correct": ok,
                        "draft_correct": ok,
                        "think_tokens": len(sym_ids),
                        "prefill_ms": 0.0,
                        "think_ms": round(gen_ms, 2),
                        "repair_ms": 0.0,
                        "verbalize_ms": exec_ms,
                        "total_e2e_ms": round(gen_ms + exec_ms, 2),
                        "repair_accepted": False,
                    }
                )
            em_ex = 100.0 * sum(1 for r in recs_ex if r["gradable"] and r["correct"]) / max(1, sum(1 for r in recs_ex if r["gradable"]))
            print(f"  [Done dsl_ext_exec({src_arm})] EM={em_ex:.2f}%", flush=True)
            raw_arm_records[f"dsl_ext_exec({src_arm})"] = recs_ex

    # Optional Gold-DSL Deterministic Executor Upper-Bound Sanity Audit (when gold traces exist in eval_file)
    gold_exec_summary: Optional[Dict[str, Any]] = None
    gold_gradable = [p for p in problems if p.get("gold_sym_tokens") and p["ground_truth_answer"]]
    if gold_gradable:
        g_hits = 0
        g_ms = []
        for p in gold_gradable:
            pred_s, ok_ex, ems = execute_dsl_trace(p["gold_sym_tokens"], p["question"], vocab_ref)
            g_ms.append(ems)
            if ok_ex and grade_prediction(pred_s, p["ground_truth_answer"], p["domain"]):
                g_hits += 1
        gold_exec_summary = {
            "n_gradable": len(gold_gradable),
            "correct": g_hits,
            "exact_match_pct": round(100.0 * g_hits / max(1, len(gold_gradable)), 2),
            "avg_exec_ms": round(sum(g_ms) / max(1, len(g_ms)), 3),
        }

    peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
    arm_summaries = {k: summarize_arm(v, ref_nl_tokens=ref_nl_tokens) for k, v in raw_arm_records.items()}
    report = {
        "eval_file": args.eval_file,
        "mode": args.mode,
        "n_problems": len(problems),
        "peak_vram_gb": peak_vram_gb,
        "arms": arm_summaries,
        "gold_dsl_executor_upper_bound": gold_exec_summary,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_text = render_markdown_table(report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")

    print("\n" + md_text, flush=True)
    print(f"[Peak VRAM] {peak_vram_gb} GB | [Saved JSON] {out_json} | [Saved MD] {out_md}", flush=True)


if __name__ == "__main__":
    main()
