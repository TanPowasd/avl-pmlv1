"""Unit tests for pure V_nl Three-Phase Thinking (Pre -> Formal -> Post -> Answer) formatter."""

from __future__ import annotations

import random
import torch

from avl_pmlv1.vocab.nl_three_think import (
    build_pre_skeleton,
    format_nl_three_think_example,
)


class _WordTok:
    eos_token = "<|eos|>"
    eos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [abs(hash(tok)) % 5000 + 1 for tok in text.split()]


def test_pre_skeleton_zero_answer_leakage() -> None:
    rec = {
        "domain": "math",
        "source": "openai/gsm8k:train",
        "question": "Natalia sold 48 clips in April and half as many in May. How many total?",
        "ground_truth_answer": "72",
        "num_steps": 2,
    }
    pre = build_pre_skeleton(rec)
    assert "[Pre]" in pre
    assert "48" in pre
    assert "72" not in pre, "Pre skeleton must never leak ground-truth answer!"


def test_clean_and_counterfactual_post_masking() -> None:
    tok = _WordTok()
    rec = {
        "domain": "math",
        "source": "openai/gsm8k:train",
        "question": "A box has 12 apples and you buy 5 more. How many apples?",
        "reference_solution": "12 + 5 = <<12+5=17>>17 apples.\n#### 17",
        "ground_truth_answer": "17",
        "num_steps": 1,
    }
    # 1. Clean path supervises [Pre] + [Formal] + [Post] + Answer (prompt masked to -100)
    ex_clean = format_nl_three_think_example(
        rec, tok, torch.device("cpu"), rng=random.Random(1), counterfactual_post_prob=0.0, is_val=False
    )
    assert ex_clean["is_counterfactual"] is False
    p_len = ex_clean["p_len"]
    assert (ex_clean["labels"][0, :p_len] == -100).all()
    assert (ex_clean["labels"][0, p_len:] != -100).all()

    # 2. Counterfactual error-recovery path masks corrupted [Formal] to -100 and supervises ONLY [Post] + Answer!
    ex_cf = format_nl_three_think_example(
        rec, tok, torch.device("cpu"), rng=random.Random(1), counterfactual_post_prob=1.0, is_val=False
    )
    assert ex_cf["is_counterfactual"] is True
    post_start = ex_cf["post_start"]
    assert post_start > ex_cf["p_len"]
    assert (ex_cf["labels"][0, :post_start] == -100).all(), "Corrupted [Formal] must be masked to -100!"
    assert (ex_cf["labels"][0, post_start:] != -100).all(), "[Post] correction + Answer must be supervised!"
