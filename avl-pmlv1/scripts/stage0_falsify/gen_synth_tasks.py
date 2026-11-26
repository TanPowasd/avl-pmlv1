"""Stage 0 Procedural Benchmark Suite: 100% Packed 16-Slot Frames & Disjoint Splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import List, Tuple
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.vocab.sym_vocab import SymbolicVocab

# Fixed nonlinear 4x8 S-boxes for Level 1 Skill 1 (memorizable by Static Engram)
SBOX_TABLE: Tuple[Tuple[int, ...], ...] = (
    (5, 2, 7, 0, 3, 6, 1, 4),
    (3, 6, 1, 4, 7, 2, 5, 0),
    (6, 1, 4, 7, 0, 5, 2, 3),
    (2, 7, 0, 5, 4, 1, 6, 3),
)

# Fixed 4x6x6 shortest-path distance tables (mod 8) for Level 1 Skill 3
GRAPH_DIST_TABLE: Tuple[Tuple[Tuple[int, ...], ...], ...] = (
    (
        (0, 2, 3, 5, 4, 6),
        (2, 0, 1, 3, 4, 5),
        (3, 1, 0, 2, 3, 4),
        (5, 3, 2, 0, 1, 2),
        (4, 4, 3, 1, 0, 1),
        (6, 5, 4, 2, 1, 0),
    ),
    (
        (0, 3, 1, 4, 2, 5),
        (3, 0, 2, 1, 5, 3),
        (1, 2, 0, 3, 1, 4),
        (4, 1, 3, 0, 2, 2),
        (2, 5, 1, 2, 0, 3),
        (5, 3, 4, 2, 3, 0),
    ),
    (
        (0, 1, 4, 2, 5, 3),
        (1, 0, 3, 2, 4, 2),
        (4, 3, 0, 1, 2, 5),
        (2, 2, 1, 0, 3, 4),
        (5, 4, 2, 3, 0, 1),
        (3, 2, 5, 4, 1, 0),
    ),
    (
        (0, 4, 2, 3, 1, 5),
        (4, 0, 3, 1, 5, 2),
        (2, 3, 0, 4, 2, 1),
        (3, 1, 4, 0, 3, 2),
        (1, 5, 2, 3, 0, 4),
        (5, 2, 1, 2, 4, 0),
    ),
)


def _is_test_instance(key_str: str, test_ratio: float = 0.15) -> bool:
    digest = hashlib.md5(key_str.encode("utf-8")).digest()
    val = int.from_bytes(digest[:4], "little") / float(2**32)
    return val < test_ratio


@dataclass
class TaskBatch:
    prompt_ids: torch.Tensor       # [B, P_len]
    target_sym_ids: torch.Tensor   # [B, S_sym] (zero <PAD> inside active blocks!)
    reassert_mask: torch.Tensor    # [B, S_sym]
    answer_pos: torch.Tensor       # [B] index inside target_sym_ids where final answer token sits
    answer_tok_ids: torch.Tensor   # [B] ground-truth token ID at answer_pos
    answer_digits: torch.Tensor    # [B] integer 0..7


class Stage0SynthGenerator:
    """Disjoint Train/Test procedural benchmark generator with 100% packed 16-slot frames."""

    def __init__(self, cfg: AVLPMLConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.vocab = SymbolicVocab(cfg)
        self.rng = random.Random(seed)
        self.sep_id = self.vocab.token_to_id[":"]
        self.eq_id = self.vocab.token_to_id["="]

    def digit_id(self, d: int) -> int:
        return self.vocab.token_to_id[str(int(d) % 10)]

    def id_to_digit(self, tok_id: int) -> int:
        for d in range(10):
            if self.vocab.token_to_id[str(d)] == tok_id:
                return d
        return -1

    def _pack16_frame(
        self,
        op_id: int,
        ref1: int,
        ref2: int,
        body_tokens: List[int],
        val_tok: int,
        write_op: int,
        out_reg: int,
        tail_flag: int | None = None,
    ) -> List[int]:
        """Construct an exact 16-slot zero-PAD postfix symbolic frame:
        Slots 0..2: [op_id, ref1, ref2] (Header)
        Slots 3..12 (or 3..11 if tail_flag): body_tokens + deterministic '=' fill + val_tok
        Trailing 2 (or 3) slots: [write_op, out_reg (, tail_flag)]
        """
        tail = [write_op, out_reg] if tail_flag is None else [write_op, out_reg, tail_flag]
        body_budget = 16 - 3 - 1 - len(tail)
        trimmed = body_tokens[:body_budget]
        fill = [self.eq_id] * (body_budget - len(trimmed))
        frame = [op_id, ref1, ref2] + trimmed + fill + [val_tok] + tail
        assert len(frame) == 16
        return frame

    # -------------------------------------------------------------------------
    # LEVEL 1: 4-Skill Compositional Pipeline (6,144 instances, disjoint split)
    # -------------------------------------------------------------------------
    def sample_level1_batch(self, batch_size: int, split: str = "train") -> TaskBatch:
        prompts, syms, rmasks, ans_pos, ans_ids, ans_digits = [], [], [], [], [], []
        want_test = split == "test"

        while len(prompts) < batch_size:
            c = self.rng.randint(0, 3)
            a0 = self.rng.randint(0, 7)
            a1 = self.rng.randint(0, 7)
            a2 = self.rng.randint(0, 3)
            a3 = self.rng.randint(0, 5)
            key = f"L1:{c}:{a0}:{a1}:{a2}:{a3}"
            if _is_test_instance(key) != want_test:
                continue

            v1 = SBOX_TABLE[c][a0]
            v2 = (3 * v1 + a1 + 1) % 8
            v3 = GRAPH_DIST_TABLE[a2][v2 % 6][a3] % 8
            ans = ((v3 ^ v1) + (1 if v3 >= 4 else 0)) % 8

            p_seq = [
                SymbolicVocab.GOAL_ID,
                self.vocab.in_slot_to_id(0), self.digit_id(c),
                self.vocab.in_slot_to_id(1), self.digit_id(a0),
                self.vocab.in_slot_to_id(2), self.digit_id(a1),
                self.vocab.in_slot_to_id(3), self.digit_id(a2),
                self.vocab.in_slot_to_id(4), self.digit_id(a3),
                self.sep_id, self.sep_id, self.sep_id, self.sep_id, self.eq_id,
            ]
            assert len(p_seq) == 16

            # 2 packed 16-slot blocks = 32 tokens (Block 0: SBox -> ModArith; Block 1: GraphDist -> ParityCheck)
            # Wait: 4 full 16-slot blocks = 64 tokens tests deeper cross-block expert handoff! Keep 4 blocks = 64 tokens:
            b0 = self._pack16_frame(
                SymbolicVocab.GOAL_ID,
                self.vocab.in_slot_to_id(0),
                self.vocab.in_slot_to_id(1),
                [
                    self.vocab.fsq_coarse_start + 10 + c,
                    self.vocab.fsq_fine_start + a0,
                    self.digit_id(c),
                    self.digit_id(a0),
                ],
                self.digit_id(v1),
                SymbolicVocab.BIND_ID,
                self.vocab.reg_slot_to_id(0),
            )
            b1 = self._pack16_frame(
                SymbolicVocab.DEDUCE_ID,
                self.vocab.reg_slot_to_id(0),
                self.vocab.in_slot_to_id(2),
                [
                    self.vocab.fsq_coarse_start + 50 + v1,
                    self.vocab.fsq_fine_start + a1,
                    self.vocab.token_to_id["+"],
                    self.vocab.token_to_id["%"],
                ],
                self.digit_id(v2),
                SymbolicVocab.BIND_ID,
                self.vocab.reg_slot_to_id(1),
            )
            b2 = self._pack16_frame(
                SymbolicVocab.DEDUCE_ID,
                self.vocab.reg_slot_to_id(1),
                self.vocab.in_slot_to_id(3),
                [
                    self.vocab.in_slot_to_id(4),
                    self.vocab.fsq_coarse_start + 90 + a2,
                    self.vocab.fsq_fine_start + a3,
                    self.vocab.token_to_id["min"],
                ],
                self.digit_id(v3),
                SymbolicVocab.BIND_ID,
                self.vocab.reg_slot_to_id(2),
            )
            # In b3 (with tail_flag=EOT_ID), tail has length 3 (slots 13,14,15), so val_tok sits at slot 12! (48 + 12 = 60)
            b3 = self._pack16_frame(
                SymbolicVocab.COMMIT_ID,
                self.vocab.reg_slot_to_id(0),
                self.vocab.reg_slot_to_id(2),
                [
                    self.vocab.fsq_coarse_start + 140 + v3,
                    self.vocab.fsq_fine_start + v1,
                    self.vocab.token_to_id["=="],
                ],
                self.digit_id(ans),
                SymbolicVocab.BIND_ID,
                self.vocab.reg_slot_to_id(3),
                tail_flag=SymbolicVocab.EOT_ID,
            )
            sym_seq = b0 + b1 + b2 + b3
            prompts.append(p_seq)
            syms.append(sym_seq)
            rmasks.append([0] * 64)
            ans_pos.append(48 + 12)
            ans_ids.append(self.digit_id(ans))
            ans_digits.append(ans)

        return TaskBatch(
            prompt_ids=torch.tensor(prompts, dtype=torch.long),
            target_sym_ids=torch.tensor(syms, dtype=torch.long),
            reassert_mask=torch.tensor(rmasks, dtype=torch.long),
            answer_pos=torch.tensor(ans_pos, dtype=torch.long),
            answer_tok_ids=torch.tensor(ans_ids, dtype=torch.long),
            answer_digits=torch.tensor(ans_digits, dtype=torch.long),
        )

    # -------------------------------------------------------------------------
    # LEVEL 2(a/b): Zero-Shot Rule Transfer via S^RO + Register Overwrite Chain
    # -------------------------------------------------------------------------
    def sample_level2_rule_and_chain_batch(
        self, batch_size: int, split: str = "train", num_chain_steps: int = 4
    ) -> TaskBatch:
        prompts, syms, rmasks, ans_pos, ans_ids, ans_digits = [], [], [], [], [], []
        want_test = split == "test"

        while len(prompts) < batch_size:
            rule = [self.rng.randint(0, 7) for _ in range(4)]
            key = f"L2Rule:{rule[0]}:{rule[1]}:{rule[2]}:{rule[3]}"
            if _is_test_instance(key, test_ratio=0.20) != want_test:
                continue

            queries = [self.rng.randint(0, 3) for _ in range(num_chain_steps)]
            p_raw = [
                self.vocab.in_slot_to_id(0), self.digit_id(rule[0]),
                self.vocab.in_slot_to_id(1), self.digit_id(rule[1]),
                self.vocab.in_slot_to_id(2), self.digit_id(rule[2]),
                self.vocab.in_slot_to_id(3), self.digit_id(rule[3]),
            ]
            for idx, q in enumerate(queries):
                p_raw.extend([self.vocab.in_slot_to_id(4 + idx), self.digit_id(q)])
            while len(p_raw) % 16 != 0:
                p_raw.append(self.sep_id)

            acc = 0
            sym_blocks: List[int] = []
            for s_idx, q in enumerate(queries):
                mapped = rule[q]
                acc = (acc + mapped) % 8
                in_reg = self.vocab.reg_slot_to_id((s_idx - 1) % 2) if s_idx > 0 else self.vocab.in_slot_to_id(4)
                out_reg = self.vocab.reg_slot_to_id(s_idx % 2)
                is_last = s_idx == num_chain_steps - 1
                op_tok = SymbolicVocab.HYP_ID if s_idx == 0 else (SymbolicVocab.COMMIT_ID if is_last else SymbolicVocab.DEDUCE_ID)
                blk = self._pack16_frame(
                    op_tok,
                    in_reg,
                    self.vocab.in_slot_to_id(q),
                    [self.vocab.in_slot_to_id(4 + s_idx), self.digit_id(q), self.digit_id(mapped)],
                    self.digit_id(acc),
                    SymbolicVocab.BIND_ID,
                    out_reg,
                    tail_flag=SymbolicVocab.EOT_ID if is_last else None,
                )
                sym_blocks.extend(blk)

            prompts.append(p_raw)
            syms.append(sym_blocks)
            rmasks.append([0] * len(sym_blocks))
            ans_pos.append((num_chain_steps - 1) * 16 + 12)
            ans_ids.append(self.digit_id(acc))
            ans_digits.append(acc)

        return TaskBatch(
            prompt_ids=torch.tensor(prompts, dtype=torch.long),
            target_sym_ids=torch.tensor(syms, dtype=torch.long),
            reassert_mask=torch.tensor(rmasks, dtype=torch.long),
            answer_pos=torch.tensor(ans_pos, dtype=torch.long),
            answer_tok_ids=torch.tensor(ans_ids, dtype=torch.long),
            answer_digits=torch.tensor(ans_digits, dtype=torch.long),
        )

    # -------------------------------------------------------------------------
    # LEVEL 2(c): Coupled Constraint Backtracking (Trap vs True Root)
    # -------------------------------------------------------------------------
    def sample_level2_backtrack_batch(
        self, batch_size: int, split: str = "train", inject_trap_in_train: bool = False
    ) -> Tuple[TaskBatch, torch.Tensor]:
        prompts, syms, rmasks, ans_pos, ans_ids, ans_digits, traps = [], [], [], [], [], [], []

        for _ in range(batch_size):
            z1 = self.rng.randint(0, 3)
            z2 = self.rng.randint(0, 3)
            c1 = (z1 + z2) % 4
            c2 = (2 * z1 + z2) % 4
            trap_z1 = (z1 + self.rng.choice([1, 2, 3])) % 4
            cue = trap_z1 if self.rng.random() < 0.65 else z1

            p_seq = [
                SymbolicVocab.HYP_ID,
                self.vocab.in_slot_to_id(0), self.digit_id(c1),
                self.vocab.in_slot_to_id(1), self.digit_id(c2),
                self.vocab.in_slot_to_id(2), self.digit_id(cue),
            ] + [self.sep_id] * 9
            assert len(p_seq) == 16

            cand_z1 = trap_z1 if inject_trap_in_train else z1
            cand_z2 = (c1 - cand_z1) % 4
            check_lhs = (2 * cand_z1 + cand_z2) % 4
            true_ans = (2 * z1 + z2) % 8

            # In b0 (no tail_flag), val_tok (cand_z1) sits at slot 13!
            b0 = self._pack16_frame(
                SymbolicVocab.HYP_ID,
                self.vocab.in_slot_to_id(0),
                self.vocab.in_slot_to_id(2),
                [self.digit_id(c1), self.digit_id(c2), self.digit_id(cand_z2)],
                self.digit_id(cand_z1),
                SymbolicVocab.BIND_ID,
                self.vocab.reg_slot_to_id(0),
            )
            b1 = self._pack16_frame(
                SymbolicVocab.VERIFY_ID,
                self.vocab.reg_slot_to_id(0),
                self.vocab.in_slot_to_id(1),
                [self.digit_id(check_lhs), self.digit_id(c2)],
                self.digit_id(true_ans),
                SymbolicVocab.VERIFY_ID,
                self.vocab.reg_slot_to_id(0),
                tail_flag=SymbolicVocab.CONTRA_ID if inject_trap_in_train else SymbolicVocab.EOT_ID,
            )
            sym_seq = b0 + b1
            rm = [0] * 16 + [0] * 13 + [0, 1, 0]

            prompts.append(p_seq)
            syms.append(sym_seq)
            rmasks.append(rm)
            ans_pos.append(16 + 12)
            ans_ids.append(self.digit_id(true_ans))
            ans_digits.append(z1)
            traps.append(trap_z1)

        return (
            TaskBatch(
                prompt_ids=torch.tensor(prompts, dtype=torch.long),
                target_sym_ids=torch.tensor(syms, dtype=torch.long),
                reassert_mask=torch.tensor(rmasks, dtype=torch.long),
                answer_pos=torch.tensor(ans_pos, dtype=torch.long),
                answer_tok_ids=torch.tensor(ans_ids, dtype=torch.long),
                answer_digits=torch.tensor(ans_digits, dtype=torch.long),
            ),
            torch.tensor(traps, dtype=torch.long),
        )

    # -------------------------------------------------------------------------
    # LEVEL 3: Paired AST -> Verbose NL-CoT (160 toks AR) vs Compact Sym (32 toks Block-dLLM)
    # -------------------------------------------------------------------------
    def sample_level3_paired_ast_batch(
        self, batch_size: int, split: str = "train"
    ) -> Tuple[TaskBatch, torch.Tensor, torch.Tensor, int]:
        prompts, syms, rmasks, ans_pos, ans_ids, ans_digits = [], [], [], [], [], []
        nl_seqs, nl_ans_toks = [], []
        want_test = split == "test"

        while len(prompts) < batch_size:
            x0 = self.rng.randint(0, 7)
            k1 = self.rng.randint(1, 5)
            k2 = self.rng.randint(1, 5)
            k3 = self.rng.randint(1, 5)
            k4 = self.rng.randint(1, 5)
            key = f"L3AST:{x0}:{k1}:{k2}:{k3}:{k4}"
            if _is_test_instance(key, test_ratio=0.15) != want_test:
                continue

            s1 = (x0 + k1) % 8
            s2 = (s1 * 2 + k2) % 8
            s3 = (s2 ^ k3) % 8
            s4 = (s3 + k4) % 8

            # Symmetric AST dual compilation:
            # Both Symbolic (32 toks, 2x16 Block-dLLM) and Verbose NL-CoT (160 toks, KV-AR)
            # receive typed input entities in Prompt and dereference each step's parameter before computing s_i!
            def _lit(role: int, val: int) -> int:
                return self.vocab.escape_start + (role % 16) * 8 + (val % 8)

            p_seq = [
                SymbolicVocab.GOAL_ID,
                self.vocab.in_slot_to_id(0), _lit(0, x0),
                self.vocab.in_slot_to_id(1), _lit(1, k1),
                self.vocab.in_slot_to_id(2), _lit(2, k2),
                self.vocab.in_slot_to_id(3), _lit(3, k3),
                self.vocab.in_slot_to_id(4), _lit(4, k4),
                self.sep_id, self.sep_id, self.sep_id, self.sep_id, self.eq_id,
            ]

            # Exact 2 packed 16-slot blocks = 32 tokens (vs 144 CoT tokens = 4.5x compression!)
            # Beat 0 unmasks op/refs & k1/k3 (slot 3); Beat 1 unmasks s1/s3 (slot 4) & k2/k4 (slot 11); Beat 2 unmasks s2/s4 (slot 13/12)!
            blk0 = [
                SymbolicVocab.GOAL_ID, self.vocab.in_slot_to_id(0), self.vocab.in_slot_to_id(1),
                _lit(1, k1), _lit(8, s1), SymbolicVocab.BIND_ID, self.vocab.reg_slot_to_id(0),
                SymbolicVocab.DEDUCE_ID, self.vocab.reg_slot_to_id(0), self.vocab.in_slot_to_id(2), self.eq_id,
                _lit(2, k2), self.eq_id, _lit(8, s2), SymbolicVocab.BIND_ID, self.vocab.reg_slot_to_id(1),
            ]
            blk1 = [
                SymbolicVocab.DEDUCE_ID, self.vocab.reg_slot_to_id(1), self.vocab.in_slot_to_id(3),
                _lit(3, k3), _lit(8, s3), SymbolicVocab.BIND_ID, self.vocab.reg_slot_to_id(0),
                SymbolicVocab.COMMIT_ID, self.vocab.reg_slot_to_id(0), self.vocab.in_slot_to_id(4), self.eq_id,
                _lit(4, k4), _lit(8, s4), SymbolicVocab.BIND_ID, self.vocab.reg_slot_to_id(1), SymbolicVocab.EOT_ID,
            ]
            sym_seq = blk0 + blk1  # Answer s4 sits at index 16 + 12 = 28

            nl_prompt = [20 + x0, 30 + k1, 40 + k2, 50 + k3, 60 + k4] + [2] * 11  # 16 tokens
            nl_cot: List[int] = []
            for idx_st, (kv, sv) in enumerate([(k1, s1), (k2, s2), (k3, s3), (k4, s4)]):
                boilerplate = [100 + ((idx_st * 37 + w * 7) % 300) for w in range(34)] + [30 + idx_st * 10 + kv]
                nl_cot.extend(boilerplate + [10 + sv])
            full_nl = nl_prompt + nl_cot  # 160 tokens total

            prompts.append(p_seq)
            syms.append(sym_seq)
            rmasks.append([0] * 32)
            ans_pos.append(16 + 12)
            ans_ids.append(_lit(8, s4))
            ans_digits.append(s4)
            nl_seqs.append(full_nl)
            nl_ans_toks.append(10 + s4)

        return (
            TaskBatch(
                prompt_ids=torch.tensor(prompts, dtype=torch.long),
                target_sym_ids=torch.tensor(syms, dtype=torch.long),
                reassert_mask=torch.tensor(rmasks, dtype=torch.long),
                answer_pos=torch.tensor(ans_pos, dtype=torch.long),
                answer_tok_ids=torch.tensor(ans_ids, dtype=torch.long),
                answer_digits=torch.tensor(ans_digits, dtype=torch.long),
            ),
            torch.tensor(nl_seqs, dtype=torch.long),
            torch.tensor(nl_ans_toks, dtype=torch.long),
            16,
        )
