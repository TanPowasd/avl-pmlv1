"""Tri-partite symbolic vocabulary partition & slot-aware domain logit masks."""

from __future__ import annotations

from enum import Enum
import re
from typing import Dict, List, Optional, Tuple
import torch

from ..config import AVLPMLConfig


_PROMPT_NUM_CAND_RE = re.compile(
    r"(?<![A-Za-z0-9_^])(?:(?<![A-Za-z0-9_)\]\-])-(?=\d))?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"
)
_ORDINAL_SUFFIX_RE = re.compile(r"^(?:st|nd|rd|th)\b", re.IGNORECASE)
_DIM_SUFFIX_RE = re.compile(r"^[dD]\b")
_HEX_BIN_SUFFIX_RE = re.compile(r"^[xXbBoO][0-9A-Fa-f]")


def extract_prompt_number_spans(text: str) -> List[Tuple[int, int, str]]:
    """Canonical prompt-number span extractor shared 1:1 across DSL compilers, S^RO span pooling, & Gate-2 executor.
    Returns list of (start_char, end_char, matched_str) in order of appearance:
      - Captures integers, comma-grouped thousands ('3,456'), decimals ('0.58'), simple fractions ('3/4'),
        standalone negatives ('-12' without turning '10-15' into '-15'), and unit-suffixed quantities ('35oz', '750m^2');
      - Rejects ordinals ('1st', '2nd', '17th'), variable/model identifiers ('v1', 'D3', 'MP3'), exponents ('^2'),
        and dimensional tags ('2D', '3D').
    """
    out: List[Tuple[int, int, str]] = []
    n_len = len(text)
    for m in _PROMPT_NUM_CAND_RE.finditer(text):
        s_i, e_i = m.start(), m.end()
        raw = m.group(0)
        suffix = text[e_i : min(n_len, e_i + 6)]
        if suffix and (suffix[0].isdigit() or suffix[0] == "_"):
            continue
        if _ORDINAL_SUFFIX_RE.match(suffix):
            continue
        if raw in ("1", "2", "3", "4") and _DIM_SUFFIX_RE.match(suffix):
            continue
        if raw in ("0", "-0") and _HEX_BIN_SUFFIX_RE.match(suffix):
            continue
        out.append((s_i, e_i, raw))
    return out


class CognitivePhase(str, Enum):
    PRE = "Pre"
    FORMAL = "Formal"
    POST = "Post"
    VERBALIZE = "Verbalize"


class SymbolicVocab:
    """Static partition contract for V_sym (4,096 in Stage 0; 19,328 in 4.5B/8.9B)."""

    MASK_ID = 0
    PAD_ID = 1
    GOAL_ID = 2
    HYP_ID = 3
    DEDUCE_ID = 4
    SPLIT_ID = 5
    BIND_ID = 6
    CONTRA_ID = 7
    VERIFY_ID = 8
    PRUNE_ID = 9
    COMMIT_ID = 10
    LONG_FRAME_ID = 11
    EOT_ID = 12
    LOAD_ID = 13
    REJECTED_DRAFT_ID = 14
    SHORT_FRAME_ID = 15
    LIT_FAST_ID = 254
    LIT_BIDIR_ID = 255

    CORE_OPS: Dict[str, int] = {
        "[MASK]": MASK_ID,
        "<PAD>": PAD_ID,
        "⟨GOAL⟩": GOAL_ID,
        "⟨HYP⟩": HYP_ID,
        "⟨DEDUCE⟩": DEDUCE_ID,
        "⟨SPLIT⟩": SPLIT_ID,
        "⟨BIND⟩": BIND_ID,
        "⟨CONTRA⟩": CONTRA_ID,
        "⟨VERIFY⟩": VERIFY_ID,
        "⟨PRUNE⟩": PRUNE_ID,
        "⟨COMMIT⟩": COMMIT_ID,
        "⟨LONG_FRAME⟩": LONG_FRAME_ID,
        "⟨EOT⟩": EOT_ID,
        "⟨LOAD⟩": LOAD_ID,
        "⟨REJECTED_DRAFT⟩": REJECTED_DRAFT_ID,
        "⟨SHORT_FRAME⟩": SHORT_FRAME_ID,
        "⟨LIT_FAST⟩": LIT_FAST_ID,
        "⟨LIT_BIDIR⟩": LIT_BIDIR_ID,
    }

    def __init__(self, cfg: AVLPMLConfig) -> None:
        self.cfg = cfg
        self.op_start = 0
        self.op_end = cfg.op_vocab_size
        self.fsq_coarse_start = self.op_end
        self.fsq_coarse_end = self.fsq_coarse_start + cfg.single_fsq_size
        self.fsq_fine_start = self.fsq_coarse_end
        self.fsq_fine_end = self.fsq_fine_start + cfg.single_fsq_size
        self.prim_start = self.fsq_fine_end
        self.prim_end = self.prim_start + cfg.lit_prim_size
        self.escape_start = self.prim_end
        self.escape_end = self.escape_start + cfg.lit_escape_size
        self.ro_start = cfg.ro_token_start
        self.ro_end = cfg.rw_token_start
        self.rw_start = cfg.rw_token_start
        self.rw_end = cfg.sym_vocab_size

        self.token_to_id: Dict[str, int] = dict(self.CORE_OPS)
        for k in range(cfg.num_ro_slots):
            self.token_to_id[f"⟨IN_{k}⟩"] = self.ro_start + k
        for j in range(cfg.num_rw_slots):
            self.token_to_id[f"⟨REG_{j}⟩"] = self.rw_start + j

        # Populate common math/code/logic lossless primitives deterministically (all 512 slots, 0..40 unchanged!)
        prims = [
            # Slots 0..40 (FROZEN for 100% backward compatibility with all Stage 0 & unit tests!)
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "+", "-", "*", "/", "%", "=", "==", "!=", "<", "<=", ">", ">=",
            "^", "(", ")", "[", "]", "{", "}", ",", ".", ":", "->",
            "and", "or", "not", "forall", "exists", "mod", "min", "max",
        ]
        # Slots 41..299: Exact lossless integers 10..256 and -1..-12 (zero hash collision!)
        prims.extend(str(n) for n in range(10, 257))
        prims.extend(str(n) for n in range(-1, -13, -1))
        # High-frequency larger integers, fractions, constants, and math variables
        prims.extend([
            "300", "360", "400", "500", "600", "720", "800", "900", "1000", "1024", "2000", "10000",
            "1/2", "1/3", "2/3", "1/4", "3/4", "1/5", "2/5", "3/5", "4/5", "1/6", "5/6", "1/8", "3/8",
            "-1/2", "0.5", "0.25", "0.75", "0.1", "0.2", "1.5", "2.5",
            "pi", "inf", "x", "y", "z", "n", "k", "i", "j", "a", "b", "c", "m", "r", "t", "s", "p", "q",
        ])
        # Exact lossless Logic, Code & Symbolic Reasoning tokens
        prims.extend([
            "True", "False", "None", "knight", "knave", "not_knight", "not_knave", "backtrack", "consistent",
            "return", "return_none", "branch_if", "loop_for", "loop_while", "index",
            "list", "dict", "tuple", "set", "neg", "str", "int", "float", "bool",
            "len", "range", "sum", "abs", "sorted", "append", "pop", "split", "join", "enumerate", "zip",
            "solve", "simplify", "symbols", "sqrt", "gcd", "lcm", "comb", "perm", "sin", "cos", "tan",
            "expand", "factor", "diff", "integrate", "Rational", "Matrix", "Eq", "Mod", "print",
            "bin_+", "bin_-", "bin_*", "bin_/", "bin_%", "bin_^",
            "cmp_==", "cmp_!=", "cmp_<", "cmp_<=", "cmp_>", "cmp_>=",
            "aug_+", "aug_-", "aug_*", "aug_/", "aug_%",
            "pw_d1", "pw_d2", "pw_d3", "pw_d4", "pw_d5",
            "kk_2p", "kk_3p", "kk_4p", "kk_5p", "kk_6p", "kk_7p", "kk_8p",
        ])
        prims.extend(f"rule{r_i}" for r_i in range(1, 25))
        # Fill remaining unused tail slots (459..511) with common decimal/unit constants (preserving 0..458 bit-for-bit!)
        prims.extend([
            "0.01", "0.02", "0.04", "0.05", "0.08", "0.12", "0.15", "0.20", "0.3", "0.4", "0.6", "0.7", "0.8", "0.9",
            "1.2", "1.25", "1.75", "2.25", "3.5", "4.5", "3.14",
            "260", "270", "280", "320", "350", "365", "450", "480", "540", "640", "700", "750", "840", "960",
            "1200", "1500", "1600", "1800", "2400", "2500", "3000", "3600", "4000", "5000", "6000", "7500", "8000",
            "5280", "1760", "144", "100000", "1000000",
        ])
        # Deduplicate preserving order and cap strictly at cfg.lit_prim_size (512)
        seen_p = set()
        slot_i = 0
        for p in prims:
            if p in seen_p:
                continue
            if slot_i >= cfg.lit_prim_size:
                break
            seen_p.add(p)
            self.token_to_id[p] = self.prim_start + slot_i
            slot_i += 1

        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

    def in_slot_to_id(self, slot: int) -> int:
        if not (0 <= slot < self.cfg.num_ro_slots):
            raise IndexError(f"RO slot {slot} out of range [0, {self.cfg.num_ro_slots})")
        return self.ro_start + slot

    def reg_slot_to_id(self, slot: int) -> int:
        if not (0 <= slot < self.cfg.num_rw_slots):
            raise IndexError(f"RW slot {slot} out of range [0, {self.cfg.num_rw_slots})")
        return self.rw_start + slot

    def is_rw_register_tensor(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.rw_start) & (token_ids < self.rw_end)

    def rw_slot_index_tensor(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids - self.rw_start).clamp(0, self.cfg.num_rw_slots - 1)

    def is_escape_anchor_tensor(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.escape_start) & (token_ids < self.escape_end)

    def is_value_bearing_tensor(self, token_ids: torch.Tensor) -> torch.Tensor:
        """True iff token_ids lies in [FSQ / V_lit] excluding structural '=' and ':' delimiters."""
        eq_id = self.token_to_id.get("=", -1)
        sep_id = self.token_to_id.get(":", -1)
        return (
            (token_ids >= self.cfg.op_vocab_size)
            & (token_ids < self.ro_start)
            & (token_ids != eq_id)
            & (token_ids != sep_id)
        )

    def is_postfix_write_dst(
        self,
        token_ids: torch.Tensor,
        prev_tokens: torch.Tensor,
        prev2_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """True iff token_ids is an RW register ⟨REG_j⟩ in a postfix write slot:
        - Always True for [..., ⟨BIND⟩, ⟨REG_j⟩];
        - For [..., ⟨VERIFY⟩/⟨COMMIT⟩, ⟨REG_j⟩], True ONLY when preceded at t-2 by a value-bearing token
          (postfix re-assertion), NEVER when ⟨VERIFY⟩/⟨COMMIT⟩ is a frame-header opcode reading an operand!
        """
        is_reg = self.is_rw_register_tensor(token_ids)
        is_bind = prev_tokens == self.BIND_ID
        is_reassert_op = (prev_tokens == self.VERIFY_ID) | (prev_tokens == self.COMMIT_ID)
        if prev2_tokens is None:
            if token_ids.dim() == 2 and token_ids.shape[1] > 1:
                z = torch.zeros_like(prev_tokens[:, :1])
                prev2_tokens = torch.cat([z, prev_tokens[:, :-1]], dim=1)
            else:
                prev2_tokens = torch.zeros_like(prev_tokens)
        is_postfix_reassert = is_reassert_op & self.is_value_bearing_tensor(prev2_tokens)
        return is_reg & (is_bind | is_postfix_reassert)

    def apply_domain_mask(
        self,
        sym_logits: torch.Tensor,
        allow_mask_token: bool = False,
    ) -> torch.Tensor:
        """Mask out [MASK] and <PAD> from output symbol sampling unless explicitly requested."""
        masked = sym_logits.clone()
        if not allow_mask_token:
            masked[..., self.MASK_ID] = float("-inf")
        masked[..., self.PAD_ID] = float("-inf")
        return masked
