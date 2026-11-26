"""Escape-free line-oriented DSL compiler + Deterministic Linear-Scan Register Allocator."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Optional, Tuple

from ..config import AVLPMLConfig
from .sym_vocab import SymbolicVocab


class CompilerValidationError(ValueError):
    """Raised when any of the 5 static hard quality checks fail (incl. Check 5: 1-Causal-Hop-Per-Beat)."""


@dataclass
class MicroStepIR:
    op: str
    in_refs: List[str] = field(default_factory=list)
    var_deps: List[str] = field(default_factory=list)
    fsq_pairs: List[Tuple[int, int]] = field(default_factory=list)
    lit_tokens: List[str] = field(default_factory=list)
    out_var: Optional[str] = None
    write_op: str = "⟨BIND⟩"  # "⟨BIND⟩" for fresh/recycled SSA, "⟨VERIFY⟩"/"⟨COMMIT⟩" for re-assertion
    flag: Optional[str] = None
    val_token: Optional[str] = None  # Explicit bound value token placed strictly at t-2 before ⟨BIND⟩ ⟨REG_j⟩ (v1.2)
    expr_text: str = ""              # Raw human-readable micro-step phrase (used for FSQ training & oscilloscope!)
    operands: List[str] = field(default_factory=list)  # Strict left-to-right ordered operands (⟨IN_k⟩, $v_m, or literal const)

    def __post_init__(self) -> None:
        self.sync_from_operands()

    def sync_from_operands(self) -> None:
        """Keep in_refs and var_deps strictly synchronized whenever ordered operands are present."""
        if self.operands:
            norm_ops: List[str] = []
            for item in self.operands:
                s = item.strip()
                if not s:
                    continue
                if s.startswith("IN_") and not s.startswith("⟨"):
                    s = f"⟨{s}⟩"
                norm_ops.append(s)
            self.operands = norm_ops
            self.in_refs = [x for x in dict.fromkeys(self.operands) if x.startswith("⟨IN_")]
            self.var_deps = [x for x in dict.fromkeys(self.operands) if x.startswith("$")]


@dataclass
class CompiledTrace:
    token_ids: List[int]
    reassert_mask: List[int]
    reg_map: Dict[str, int]
    frame_sizes: List[int] = field(default_factory=list)  # Exact 8-slot or 16-slot physical frame lengths


BINARY_LOGIC_OPS = {"+", "-", "*", "/", "%", "^", "==", "!=", "<", "<=", ">", ">=", "and", "or", "mod", "min", "max"}


class LinearScanDSLCompiler:
    """Deterministic Backward-Liveness Linear-Scan Register Allocator with 5 static guards (AVL-PMLv1.2)."""

    def __init__(self, cfg: AVLPMLConfig, vocab: SymbolicVocab) -> None:
        self.cfg = cfg
        self.vocab = vocab

    @staticmethod
    def validate_single_causal_hop(step: MicroStepIR, step_idx: int) -> None:
        """Hard Check 5 (Stage 0 Gate 1 Iron Law): Strictly 1 causal hop per micro-step / beat!
        - Forbids cramming >=2 chained binary operators into a single micro-step (must SSA-split!);
        - Forbids self-referential fresh ⟨BIND⟩ cycles inside one step;
        - Requires a non-empty value-bearing token right before ⟨BIND⟩ ⟨REG_j⟩ for v1.2 hard-indexed S^RW write!
        """
        bin_op_count = sum(1 for tok in step.lit_tokens if tok in BINARY_LOGIC_OPS)
        if bin_op_count > 1:
            raise CompilerValidationError(
                f"Step {step_idx} [Check 5: 1-Causal-Hop-Per-Beat]: contains {bin_op_count} chained binary ops "
                f"({step.lit_tokens}); must split into atomic 1-hop SSA micro-steps!"
            )
        if step.out_var is not None and step.write_op == "⟨BIND⟩" and step.out_var in step.var_deps:
            # In-place accumulation like $x = $x + 1 is valid ONLY when $x was ALREADY defined in an earlier step
            # (checked by Check 2); however multi-hop indirect pointer nesting ($a[$b]) is forbidden!
            pass
        if any("[" in tok and "$" in tok for tok in step.lit_tokens):
            raise CompilerValidationError(
                f"Step {step_idx} [Check 5: 1-Causal-Hop-Per-Beat]: nested indirect pointer indexing in single step!"
            )
        if step.out_var is not None:
            has_val = (step.val_token is not None) or (len(step.fsq_pairs) > 0) or (len(step.lit_tokens) > 0)
            if not has_val:
                raise CompilerValidationError(
                    f"Step {step_idx} [Check 5: v1.2 Hard-Indexed Write Alignment]: missing bound value token before ⟨BIND⟩ ⟨REG_j⟩!"
                )

    @staticmethod
    def parse_line_dsl(dsl_text: str) -> List[MicroStepIR]:
        """Parse escape-free line DSL: OP | in=IN_0,$v1 | fsq=12:4,88:2 | lit=+,1 | out=$v2 | flag=NONE"""
        steps: List[MicroStepIR] = []
        for raw_line in dsl_text.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            op = parts[0]
            in_refs: List[str] = []
            var_deps: List[str] = []
            operands: List[str] = []
            fsq_pairs: List[Tuple[int, int]] = []
            lit_tokens: List[str] = []
            out_var: Optional[str] = None
            write_op = "⟨BIND⟩"
            flag: Optional[str] = None
            val_token: Optional[str] = None
            expr_text: str = ""

            for field_str in parts[1:]:
                if "=" not in field_str:
                    continue
                k, v = [x.strip() for x in field_str.split("=", 1)]
                if not v or v == "NONE":
                    continue
                if k == "in":
                    for item in v.split(","):
                        item = item.strip()
                        if not item:
                            continue
                        if item.startswith("$"):
                            if item not in var_deps:
                                var_deps.append(item)
                            operands.append(item)
                        elif item.startswith("⟨IN_") or item.startswith("IN_"):
                            if not item.startswith("⟨"):
                                item = f"⟨{item}⟩"
                            if item not in in_refs:
                                in_refs.append(item)
                            operands.append(item)
                        else:
                            operands.append(item)
                elif k == "fsq":
                    for pair in v.split(","):
                        c_str, f_str = pair.strip().split(":")
                        fsq_pairs.append((int(c_str), int(f_str)))
                elif k == "lit":
                    lit_tokens = [tok.strip() for tok in v.split(",") if tok.strip()]
                elif k == "out":
                    out_var = v
                elif k == "wop":
                    write_op = v if v.startswith("⟨") else f"⟨{v}⟩"
                elif k == "val":
                    val_token = v
                elif k == "expr":
                    expr_text = v
                elif k == "flag":
                    flag = v if v.startswith("⟨") else f"⟨{v}⟩"
            steps.append(
                MicroStepIR(
                    op=op if op.startswith("⟨") else f"⟨{op}⟩",
                    in_refs=in_refs,
                    var_deps=var_deps,
                    fsq_pairs=fsq_pairs,
                    lit_tokens=lit_tokens,
                    out_var=out_var,
                    write_op=write_op,
                    flag=flag,
                    val_token=val_token,
                    expr_text=expr_text,
                    operands=operands,
                )
            )
        return steps

    def _encode_operand_token(self, op_str: str, var_to_reg: Dict[str, int]) -> int:
        if op_str.startswith("⟨"):
            return self.vocab.token_to_id[op_str]
        if op_str.startswith("$"):
            return self.vocab.reg_slot_to_id(var_to_reg[op_str])
        tid = self.vocab.token_to_id.get(op_str)
        if tid is not None:
            return tid
        h_esc = sum(ord(ch) * (i + 31) for i, ch in enumerate(op_str)) % self.cfg.lit_escape_size
        return self.vocab.escape_start + h_esc

    def compile(self, steps: List[MicroStepIR], final_live_vars: Optional[List[str]] = None) -> CompiledTrace:
        final_live = set(final_live_vars or [])

        # Backward liveness analysis
        last_use: Dict[str, int] = {v: len(steps) + 1 for v in final_live}
        defined_at: Dict[str, int] = {}
        for idx, step in enumerate(steps):
            step.sync_from_operands()
            # Hard Check 5: 1-Causal-Hop-Per-Beat & v1.2 Hard-Indexed Write Alignment
            self.validate_single_causal_hop(step, idx)
            # Hard Check 1: Dependency & proposition granularity cap
            dep_count = max(len(step.in_refs) + len(step.var_deps), len(step.operands))
            if dep_count > 4:
                raise CompilerValidationError(
                    f"Step {idx}: exceeds max 4 input dependencies ({dep_count})"
                )
            for dep in step.var_deps:
                # Hard Check 2: Zero uninitialized variable read
                if dep not in defined_at:
                    raise CompilerValidationError(f"Step {idx}: reads uninitialized variable '{dep}'")
            if step.out_var is not None and step.out_var not in defined_at:
                defined_at[step.out_var] = idx

        for idx in range(len(steps) - 1, -1, -1):
            for dep in steps[idx].var_deps:
                if dep not in last_use:
                    last_use[dep] = idx

        # Min-heap deterministic register pool
        free_pool: List[int] = list(range(self.cfg.num_rw_slots))
        heapq.heapify(free_pool)
        var_to_reg: Dict[str, int] = {}

        token_ids: List[int] = []
        reassert_mask: List[int] = []
        chunk_written_regs: Dict[int, set[int]] = {}

        for idx, step in enumerate(steps):
            step_tokens: List[int] = [self.vocab.token_to_id[step.op]]
            step_reassert: List[int] = [0]

            if step.operands:
                for op_item in step.operands:
                    step_tokens.append(self._encode_operand_token(op_item, var_to_reg))
                    step_reassert.append(0)
            else:
                for in_ref in step.in_refs:
                    step_tokens.append(self.vocab.token_to_id[in_ref])
                    step_reassert.append(0)
                for dep in step.var_deps:
                    reg_slot = var_to_reg[dep]
                    step_tokens.append(self.vocab.reg_slot_to_id(reg_slot))
                    step_reassert.append(0)

            for c_id, f_id in step.fsq_pairs:
                step_tokens.append(self.vocab.fsq_coarse_start + c_id)
                step_reassert.append(0)
                step_tokens.append(self.vocab.fsq_fine_start + f_id)
                step_reassert.append(0)

            for lit in step.lit_tokens:
                tok_id = self.vocab.token_to_id.get(lit, self.vocab.prim_start)
                step_tokens.append(tok_id)
                step_reassert.append(0)

            # Reclaim dead input registers BEFORE allocating output if safe, or right after read
            reclaimed_now: List[int] = []
            for dep in step.var_deps:
                if last_use.get(dep, -1) == idx and dep != step.out_var:
                    reclaimed_now.append(var_to_reg[dep])

            if step.out_var is not None:
                is_reassert = step.out_var in var_to_reg or step.write_op in ("⟨VERIFY⟩", "⟨COMMIT⟩")
                if step.out_var in var_to_reg:
                    target_slot = var_to_reg[step.out_var]
                else:
                    # Hard Check 3: Avoid writing same physical slot twice inside the same 16-slot atomic chunk
                    prospective_pos = len(token_ids) + len(step_tokens) + 1
                    chunk_id = prospective_pos // self.cfg.atomic_chunk_size
                    forbidden = chunk_written_regs.get(chunk_id, set())

                    deferred: List[int] = []
                    target_slot = -1
                    while free_pool:
                        cand = heapq.heappop(free_pool)
                        if cand not in forbidden:
                            target_slot = cand
                            break
                        deferred.append(cand)
                    for d_slot in deferred:
                        heapq.heappush(free_pool, d_slot)

                    # Hard Check 4: Register file capacity bound
                    if target_slot < 0:
                        raise CompilerValidationError(
                            f"Step {idx}: register pool exhausted or intra-chunk write collision on chunk {chunk_id}"
                        )
                    var_to_reg[step.out_var] = target_slot

                wop_id = self.vocab.token_to_id[step.write_op]
                reg_id = self.vocab.reg_slot_to_id(target_slot)
                write_global_pos = len(token_ids) + len(step_tokens) + 1
                w_chunk = write_global_pos // self.cfg.atomic_chunk_size
                if target_slot in chunk_written_regs.get(w_chunk, set()):
                    raise CompilerValidationError(
                        f"Step {idx}: duplicate write to ⟨REG_{target_slot}⟩ inside 16-slot atomic chunk {w_chunk}"
                    )
                chunk_written_regs.setdefault(w_chunk, set()).add(target_slot)

                step_tokens.extend([wop_id, reg_id])
                step_reassert.extend([0, 1 if is_reassert else 0])

                # If output variable is never read later, release after this step
                if last_use.get(step.out_var, -1) < idx and step.out_var not in final_live:
                    reclaimed_now.append(target_slot)

            if step.flag is not None:
                step_tokens.append(self.vocab.token_to_id[step.flag])
                step_reassert.append(0)

            if not (2 <= len(step_tokens) <= 28):
                raise CompilerValidationError(
                    f"Step {idx}: compiled symbol length {len(step_tokens)} outside [2, 28]"
                )

            for r_slot in set(reclaimed_now):
                heapq.heappush(free_pool, r_slot)

            token_ids.extend(step_tokens)
            reassert_mask.extend(step_reassert)

        return CompiledTrace(token_ids=token_ids, reassert_mask=reassert_mask, reg_map=var_to_reg)

    def compile_aligned_frames(
        self, steps: List[MicroStepIR], final_live_vars: Optional[List[str]] = None, prefer_8slot: bool = True
    ) -> CompiledTrace:
        """AVL-PMLv1.2 Physical Frame Compiler:
        Enforces all 5 static hard checks and packs EVERY 1-hop MicroStepIR into an exact 8-slot or 16-slot frame
        with the bound value token locked strictly at t-2 right before [⟨BIND⟩, ⟨REG_j⟩ (, flag)]!
        """
        # Run full validation + register assignment check first
        base = self.compile(steps, final_live_vars=final_live_vars)
        eq_id = self.vocab.token_to_id["="]
        final_live = set(final_live_vars or [])

        last_use: Dict[str, int] = {v: len(steps) + 1 for v in final_live}
        for idx in range(len(steps) - 1, -1, -1):
            for dep in steps[idx].var_deps:
                if dep not in last_use:
                    last_use[dep] = idx

        free_pool: List[int] = list(range(self.cfg.num_rw_slots))
        heapq.heapify(free_pool)
        var_to_reg: Dict[str, int] = {}

        all_tokens: List[int] = []
        all_reassert: List[int] = []
        frame_sizes: List[int] = []

        for idx, step in enumerate(steps):
            header: List[int] = [self.vocab.token_to_id[step.op]]
            if step.operands:
                for op_item in step.operands:
                    header.append(self._encode_operand_token(op_item, var_to_reg))
            else:
                for in_ref in step.in_refs:
                    header.append(self.vocab.token_to_id[in_ref])
                for dep in step.var_deps:
                    header.append(self.vocab.reg_slot_to_id(var_to_reg[dep]))

            body: List[int] = []
            for c_id, f_id in step.fsq_pairs:
                body.append(self.vocab.fsq_coarse_start + (c_id % self.cfg.single_fsq_size))
                body.append(self.vocab.fsq_fine_start + (f_id % self.cfg.single_fsq_size))
            for lit in step.lit_tokens:
                body.append(self.vocab.token_to_id.get(lit, self.vocab.prim_start))

            val_tid: Optional[int] = None
            if step.val_token is not None:
                val_tid = self.vocab.token_to_id.get(step.val_token)
                if val_tid is None:
                    # Deterministic escape literal hash for multi-digit/symbolic values
                    h_esc = sum(ord(ch) * (i + 31) for i, ch in enumerate(step.val_token)) % self.cfg.lit_escape_size
                    val_tid = self.vocab.escape_start + h_esc
            elif body:
                val_tid = body.pop()

            reclaimed: List[int] = [
                var_to_reg[dep] for dep in step.var_deps if last_use.get(dep, -1) == idx and dep != step.out_var
            ]

            tail: List[int] = []
            tail_reassert: List[int] = []
            if step.out_var is not None:
                is_reassert = step.out_var in var_to_reg or step.write_op in ("⟨VERIFY⟩", "⟨COMMIT⟩")
                if step.out_var in var_to_reg:
                    slot = var_to_reg[step.out_var]
                else:
                    if not free_pool:
                        raise CompilerValidationError(f"Step {idx}: register pool exhausted")
                    slot = heapq.heappop(free_pool)
                    var_to_reg[step.out_var] = slot
                wop_id = self.vocab.token_to_id[step.write_op]
                reg_id = self.vocab.reg_slot_to_id(slot)
                tail = [wop_id, reg_id]
                tail_reassert = [0, 1 if is_reassert else 0]
                if last_use.get(step.out_var, -1) < idx and step.out_var not in final_live:
                    reclaimed.append(slot)

            flag_id = self.vocab.token_to_id[step.flag] if step.flag is not None else eq_id
            # Exact slot layout: [header + body + fill(=) + VAL + BIND + REG + flag]
            # Guarantees VAL sits at exact t-2 right before ⟨BIND⟩ ⟨REG_j⟩!
            val_part = [val_tid] if val_tid is not None else []
            raw_len = len(header) + len(body) + len(val_part) + len(tail) + 1
            frame_len = 8 if (prefer_8slot and raw_len <= 8) else 16
            if raw_len > 16:
                raise CompilerValidationError(f"Step {idx}: aligned frame length {raw_len} exceeds 16 slots!")

            fill_len = frame_len - raw_len
            frame = header + body + [eq_id] * fill_len + val_part + tail + [flag_id]
            f_reassert = [0] * (len(header) + len(body) + fill_len + len(val_part)) + tail_reassert + [0]
            assert len(frame) == frame_len

            for r_s in set(reclaimed):
                heapq.heappush(free_pool, r_s)

            all_tokens.extend(frame)
            all_reassert.extend(f_reassert)
            frame_sizes.append(frame_len)

        return CompiledTrace(
            token_ids=all_tokens,
            reassert_mask=all_reassert,
            reg_map=var_to_reg,
            frame_sizes=frame_sizes,
        )
