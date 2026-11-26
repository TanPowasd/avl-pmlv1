"""Pure V_nl Three-Phase Thinking (Pre -> Formal -> Post -> Answer) Formatter for Causal AR Control.

Tests the isolated value of the AVL-PMLv1 'Three-Think, One-Express' (三思一表) cognitive structure
directly on pretrained Qwen3-4B-Base native V_nl vocabulary (ZERO cold-start V_sym vocabulary tax!):
  1) [Pre]    : Extract prompt givens/query & high-level reasoning skeleton (zero answer leakage!)
  2) [Formal] : Step-by-step natural-language / PoT derivation (identical source CoT as nl_cot_sft)
  3) [Post]   : Consistency verification (80%) OR counterfactual error-recovery conditioning (20%,
                with corrupted [Formal] tokens masked to label=-100 so the model never learns to
                make errors, only to catch & fix them in [Post]!)
  4) Answer   : Final concise answer strictly in 'Answer: <gt><eos>' format.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from avl_pmlv1.vocab.sym_vocab import extract_prompt_number_spans


def clean_reference_cot_text(rec: Dict[str, Any]) -> str:
    """Extract clean natural-language/PoT reasoning text from mega_seeds reference_solution."""
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


def build_pre_skeleton(rec: Dict[str, Any]) -> str:
    """Build leakage-free [Pre] skeleton purely from prompt givens/query and coarse step plan."""
    domain = str(rec.get("domain", "math")).lower()
    question = str(rec.get("question", "")).strip()
    src = str(rec.get("source", "")).split(":")[0]
    n_steps = int(rec.get("num_steps", len(rec.get("frame_sizes", [])) or 4))

    if domain == "logic":
        if src == "K-and-K/knights-and-knaves":
            m_names = re.search(r"inhabitants?:\s*([^.\n]+)", question, re.IGNORECASE)
            who = m_names.group(1).strip() if m_names else "inhabitants"
            return (
                f"[Pre]\nTask: knight/knave deduction for {who}. "
                f"Plan: test truth-teller/liar assignments and eliminate contradictions."
            )
        # ProofWriter / general logic: extract target query sentence if present
        q_tail = question.splitlines()[-1].strip() if question else ""
        if len(q_tail) > 120:
            q_tail = q_tail[-120:]
        return (
            f"[Pre]\nTask: deductive verification. Target: {q_tail}\n"
            f"Plan: chain known facts and rules (~{max(1, n_steps)} steps) to determine truth value."
        )

    # Math domain: extract prompt numeric givens via exact same span extractor used by S^RO!
    spans = extract_prompt_number_spans(question)
    nums = [txt for _, _, txt in spans[:10]]
    givens_str = ", ".join(nums) if nums else "symbolic quantities"
    return (
        f"[Pre]\nTask: multi-step math ({src.split('/')[-1] or 'math'}). "
        f"Givens: {givens_str}. Plan: solve in ~{max(1, n_steps)} steps then verify."
    )


def _make_plausible_wrong_answer(gt: str, domain: str, rng: random.Random) -> Optional[str]:
    """Produce a plausible perturbed draft answer for 20% [Post] self-correction conditioning."""
    g = gt.strip()
    if not g:
        return None
    low = g.lower()
    if domain == "logic":
        if low == "true":
            return "False"
        if low == "false":
            return "True"
        if low.startswith("[") and low.endswith("]") and ("true" in low or "false" in low):
            items = re.findall(r"\b(true|false)\b", low)
            if items:
                flip_i = rng.randrange(len(items))
                items[flip_i] = "false" if items[flip_i] == "true" else "true"
                return "[" + ", ".join(items) + "]"
        return None
    # Numeric perturbation (off-by-one, sign flip, or small delta)
    try:
        val = float(g.replace(",", ""))
        if abs(val - round(val)) < 1e-6:
            iv = int(round(val))
            delta = rng.choice([-2, -1, 1, 2, 10, -10])
            if iv + delta == iv:
                delta = 1
            return str(iv + delta)
        return f"{val + rng.choice([-1.0, 1.0, 0.5]):.4g}"
    except Exception:
        return None


def _corrupt_formal_draft(formal_text: str, gt: str, wrong_ans: str) -> Tuple[str, bool]:
    """Subtly inject a wrong final draft value in [Formal] (only used when [Formal] loss is masked to -100!)."""
    if not formal_text:
        return f"Draft estimate gives {wrong_ans}.", True
    # Replace last occurrence of gt if present, else append a drifted draft conclusion line
    idx = formal_text.rfind(gt)
    if idx != -1 and len(gt) >= 1:
        corrupted = formal_text[:idx] + wrong_ans + formal_text[idx + len(gt) :]
        return corrupted, True
    return f"{formal_text}\nTentative result: {wrong_ans}.", True


def format_nl_three_think_example(
    rec: Dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    rng: Optional[random.Random] = None,
    max_prompt_len: int = 256,
    max_target_len: int = 384,
    counterfactual_post_prob: float = 0.20,
    is_val: bool = False,
) -> Dict[str, Any]:
    """Encode a pure V_nl Three-Phase Thinking AR example:
      Prompt -> [Pre] -> [Formal] -> [Post] -> Answer: <gt><eos>

    When counterfactual error-recovery triggers (training only, p=0.20):
      - [Formal] contains a subtly perturbed draft value;
      - Labels on [Pre] & corrupted [Formal] are set to -100 so gradients NEVER reinforce errors;
      - Loss is computed strictly on [Post] (catching & correcting the error) + Answer: <gt><eos>!
    """
    domain = str(rec.get("domain", "math")).lower()
    question = str(rec.get("question", "")).strip()
    prompt_text = (
        f"Problem ({domain}):\n{question}\n"
        f"Use three-phase thinking ([Pre] plan, [Formal] derivation, [Post] verification) "
        f"and end with 'Answer: <final answer>'.\n"
    )
    p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not p_ids:
        p_ids = [tokenizer.eos_token_id or 151643]
    p_ids = p_ids[:max_prompt_len]

    gt = str(rec.get("ground_truth_answer") or "").strip()
    eos_str = tokenizer.eos_token or ""
    pre_text = build_pre_skeleton(rec) + "\n"
    formal_clean = clean_reference_cot_text(rec)

    is_cf = False
    wrong_ans: Optional[str] = None
    if (not is_val) and rng is not None and counterfactual_post_prob > 0.0 and rng.random() < counterfactual_post_prob:
        wrong_ans = _make_plausible_wrong_answer(gt, domain, rng)
        if wrong_ans is not None:
            is_cf = True

    if is_cf and wrong_ans is not None:
        formal_body, _ = _corrupt_formal_draft(formal_clean, gt, wrong_ans)
        post_text = (
            f"\n[Post]\nCheck: draft value {wrong_ans} is inconsistent with the problem constraints; "
            f"re-verifying derivation yields {gt}."
        )
    else:
        formal_body = formal_clean
        post_text = f"\n[Post]\nCheck: constraints and arithmetic verified; consistent result is {gt}."

    ans_text = f"\nAnswer: {gt}{eos_str}"

    pre_ids = tokenizer.encode(pre_text, add_special_tokens=False)
    post_ids = tokenizer.encode(post_text, add_special_tokens=False)
    ans_ids = tokenizer.encode(ans_text, add_special_tokens=False)
    if not ans_ids:
        ans_ids = [tokenizer.eos_token_id or 151643]

    # Always reserve full space for [Pre], [Post], and Answer: <gt><eos>; truncate only [Formal] if needed
    formal_prefix_ids = tokenizer.encode("[Formal]\n", add_special_tokens=False)
    fixed_len = len(pre_ids) + len(formal_prefix_ids) + len(post_ids) + len(ans_ids)
    formal_budget = max(0, max_target_len - fixed_len)
    formal_body_ids = tokenizer.encode(formal_body, add_special_tokens=False)[:formal_budget] if formal_body else []
    formal_ids = formal_prefix_ids + formal_body_ids

    target_ids = pre_ids + formal_ids + post_ids + ans_ids
    full_ids = torch.tensor([p_ids + target_ids], dtype=torch.long, device=device)
    labels = torch.full_like(full_ids, -100)

    post_start = len(p_ids) + len(pre_ids) + len(formal_ids)
    ans_start = post_start + len(post_ids)
    if is_cf:
        # Strictly supervise ONLY [Post] correction + Answer (never train model to emit corrupted [Formal]!)
        labels[0, post_start:] = full_ids[0, post_start:]
    else:
        labels[0, len(p_ids) :] = full_ids[0, len(p_ids) :]

    return {
        "full_ids": full_ids,
        "labels": labels,
        "p_len": len(p_ids),
        "post_start": post_start,
        "ans_start": ans_start,
        "is_counterfactual": is_cf,
        "domain": domain,
        "num_steps": int(rec.get("num_steps", len(rec.get("frame_sizes", [])) or 8)),
    }
