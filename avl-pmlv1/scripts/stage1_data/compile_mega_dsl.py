"""Stage 1 Mega Gold v2 0-RMB Parallel Deterministic Compiler (Zero Error Blocks, Zero Truncation, Concrete Numeric Eval).

Hardened after honest real-sample spot-check to eliminate ALL 4 real data-quality flaws:
  1. `NuminaMath-TIR` / `OpenMath1`: Extracts ONLY the LAST clean (error-free!) code block, rejects any sample with
     unresolved Traceback/Error or missing ground-truth answer, and DROPS any trace exceeding 32 steps (ZERO truncation!).
  2. `MathInstruct-PoT` & Arithmetic `TIR`: Pure in-memory safe AST numeric evaluator (zero `exec()`) fills concrete
     intermediate numeric values (`val=120` instead of `bin_+`) AND computes/verifies the concrete final answer `gt`
     for `MathInstruct-PoT` (drops any unresolvable sample without a real `gt`!).
  3. `KodCode-V1` / `MBPP`: `strict_no_truncate=True` (drops >30-step partial functions instead of truncating mid-loop!).
  4. `Knights-and-Knaves` & `ProofWriter`: Zero truncation (`K&K <= 32` steps, `ProofWriter <= 14` steps) + lossless
     459-primitive `V_lit` vocabulary (integers -12..256, True/False, knight/knave, rule1..24 zero hash collision!).
"""

from __future__ import annotations

import argparse
import ast
import collections
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.vocab.dsl_compiler import CompilerValidationError, LinearScanDSLCompiler, MicroStepIR
from avl_pmlv1.vocab.sym_vocab import SymbolicVocab, extract_prompt_number_spans

from scripts.stage1_data.compile_dsl_pipeline import (
    _ast_binop_sym,
    _ast_cmp_sym,
    _fast_parse_float,
    _fmt_num,
    approx_nl_token_count,
    compile_code_deterministic,
    compile_math_or_logic_deterministic,
)

ERROR_MARKERS = (
    "Traceback",
    "SyntaxError",
    "NameError",
    "ZeroDivisionError",
    "TypeError",
    "ValueError",
    "AttributeError",
    "IndexError",
    "RuntimeError",
    "TimeoutError",
)


def _extract_clean_final_python_block(sol: str, is_raw_pot: bool = False) -> Tuple[Optional[str], str]:
    """Select ONLY the final clean (error-free) executable Python block; reject error-tainted blocks!"""
    # Match paired ```python ... ``` optionally followed by ```output ... ```
    paired = re.findall(
        r"```python\s*\n(.*?)```(?:\s*```output\s*\n(.*?)```)?", sol, flags=re.DOTALL
    )
    if not paired:
        paired = re.findall(
            r"<llm-code>\s*\n?(.*?)</llm-code>(?:\s*<llm-code-output>\s*\n?(.*?)</llm-code-output>)?",
            sol,
            flags=re.DOTALL,
        )
    if paired:
        # Walk backwards: pick the LAST code block whose output has ZERO Python errors!
        for code_blk, out_blk in reversed(paired):
            if any(err in (out_blk or "") for err in ERROR_MARKERS):
                continue
            if code_blk.strip():
                return code_blk.strip(), "ok"
        return None, "tir_all_blocks_had_python_error"

    if is_raw_pot:
        if any(err in sol for err in ERROR_MARKERS):
            return None, "pot_has_error"
        # Strip optional markdown fences around PoT script
        blk = re.sub(r"^```[a-zA-Z]*\s*\n|```$", "", sol.strip(), flags=re.MULTILINE).strip()
        return (blk, "ok") if blk else (None, "pot_empty")

    return None, "tir_no_code_block"


def _eval_binop_safe(op_s: str, lv: float, rv: float) -> Optional[float]:
    try:
        if op_s == "+":
            res = lv + rv
        elif op_s == "-":
            res = lv - rv
        elif op_s == "*":
            res = lv * rv
        elif op_s == "/":
            if abs(rv) < 1e-12:
                return None
            res = lv / rv
        elif op_s == "%":
            if abs(rv) < 1e-12:
                return None
            res = lv % rv
        elif op_s == "^":
            if abs(rv) > 16 or abs(lv) > 1e5:
                return None
            res = lv ** rv
        else:
            return None
        if not math.isfinite(res) or abs(res) > 1e12:
            return None
        return float(res)
    except Exception:
        return None


def _eval_math_call_safe(fn_tail: str, arg_vals: List[Optional[float]]) -> Optional[float]:
    if not arg_vals or any(v is None for v in arg_vals):
        return None
    vs = [float(v) for v in arg_vals if v is not None]
    try:
        if fn_tail == "sqrt" and len(vs) == 1 and vs[0] >= 0:
            res = math.sqrt(vs[0])
        elif fn_tail == "abs" and len(vs) == 1:
            res = abs(vs[0])
        elif fn_tail == "round" and len(vs) == 1:
            res = float(round(vs[0]))
        elif fn_tail == "floor" and len(vs) == 1:
            res = float(math.floor(vs[0]))
        elif fn_tail == "ceil" and len(vs) == 1:
            res = float(math.ceil(vs[0]))
        elif fn_tail == "min" and len(vs) >= 2:
            res = min(vs)
        elif fn_tail == "max" and len(vs) >= 2:
            res = max(vs)
        elif fn_tail == "pow" and len(vs) == 2 and abs(vs[1]) <= 16 and abs(vs[0]) <= 1e5:
            res = vs[0] ** vs[1]
        elif fn_tail == "comb" and len(vs) == 2 and 0 <= int(vs[1]) <= int(vs[0]) <= 60:
            res = float(math.comb(int(round(vs[0])), int(round(vs[1]))))
        elif fn_tail == "gcd" and len(vs) == 2:
            res = float(math.gcd(int(round(vs[0])), int(round(vs[1]))))
        elif fn_tail == "factorial" and len(vs) == 1 and 0 <= int(vs[0]) <= 18:
            res = float(math.factorial(int(round(vs[0]))))
        else:
            return None
        return res if (math.isfinite(res) and abs(res) <= 1e12) else None
    except Exception:
        return None


def compile_tir_python_gold(row: Dict[str, Any], max_steps: int = 32) -> Tuple[List[MicroStepIR], bool, str, str]:
    """Gold v2 TIR/PoT compiler: clean final block only + in-memory safe numeric evaluator + ZERO truncation + mandatory GT!"""
    sol = row["reference_solution"]
    src = str(row.get("source", ""))
    is_pot = "MathInstruct" in src
    gt = str(row.get("ground_truth_answer", "")).strip()

    # For NuminaMath-TIR / OpenMath1, ground truth MUST already exist!
    if (not is_pot) and (not gt):
        return [], False, "tir_missing_gt", ""

    blk, blk_rs = _extract_clean_final_python_block(sol, is_raw_pot=is_pot)
    if blk is None:
        return [], False, blk_rs, gt

    try:
        mod = ast.parse(blk)
    except SyntaxError:
        return [], False, "tir_syntax_error", gt
    if not mod.body:
        return [], False, "tir_empty_ast", gt

    steps: List[MicroStepIR] = []
    name_to_ref: Dict[str, str] = {}
    ref_to_num: Dict[str, float] = {}
    v_cnt = 0
    truncated = False
    last_printed_num: Optional[float] = None
    last_computed_num: Optional[float] = None

    def _nv() -> str:
        nonlocal v_cnt
        v_cnt += 1
        return f"$v{v_cnt}"

    # Extract problem numbers as ⟨IN_0..k⟩ in exact prompt span order (1:1 aligned with S^RO & executor!)
    q_nums = [m_s for _, _, m_s in extract_prompt_number_spans(row["question"])]
    num_to_in: Dict[str, str] = {}
    for k_idx, ns in enumerate(q_nums[:48]):
        fv = _fast_parse_float(ns)
        if fv is None or not math.isfinite(fv) or abs(fv) > 1e12:
            continue
        key = _fmt_num(fv)
        slot_s = f"⟨IN_{k_idx}⟩"
        ref_to_num[slot_s] = fv
        if key != "OVERFLOW" and key not in num_to_in:
            num_to_in[key] = slot_s

    def _emit(
        op: str,
        refs: List[str],
        lit_toks: List[str],
        val_tok: str,
        expr_txt: str,
        num_val: Optional[float] = None,
        flag: Optional[str] = None,
    ) -> str:
        nonlocal last_computed_num
        ov = _nv()
        if num_val is not None:
            ref_to_num[ov] = num_val
            last_computed_num = num_val
        ord_ops = [r for r in refs if r][:4]
        steps.append(
            MicroStepIR(
                op=op,
                operands=ord_ops,
                lit_tokens=lit_toks[:2],
                val_token=(val_tok or "0")[:16],
                out_var=ov,
                flag=flag,
                expr_text=" ".join(expr_txt.split())[:120],
            )
        )
        return ov

    def _lower_expr(expr: ast.AST) -> Tuple[str, Optional[float]]:
        nonlocal truncated, last_printed_num
        if len(steps) >= max_steps - 1:
            truncated = True
            return (f"$v{v_cnt}" if v_cnt > 0 else "⟨IN_0⟩"), None
        if isinstance(expr, ast.Name):
            if expr.id in name_to_ref:
                r = name_to_ref[expr.id]
                return r, ref_to_num.get(r)
            r = _emit("⟨DEDUCE⟩", [], [], expr.id[:12], f"ref {expr.id}")
            return r, None
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                s = str(expr.value)
                return _emit("⟨DEDUCE⟩", [], [], s, f"const {s}"), None
            if isinstance(expr.value, (int, float)) and math.isfinite(float(expr.value)) and abs(float(expr.value)) <= 1e12:
                fv = float(expr.value)
                sv = _fmt_num(fv)
                if sv in num_to_in:
                    return num_to_in[sv], fv
                return _emit("⟨DEDUCE⟩", [], [], sv, f"const {sv}", num_val=fv), fv
            s = str(expr.value)[:14] if expr.value is not None else "None"
            return _emit("⟨DEDUCE⟩", [], [], s or "0", f"const {s}"), None
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            r, rv = _lower_expr(expr.operand)
            nv = (-rv) if rv is not None else None
            vt = _fmt_num(nv) if nv is not None else "neg"
            et = f"-{r} -> {vt}" if nv is not None else f"-{r}"
            return _emit("⟨DEDUCE⟩", [r], ["-"], vt, et, num_val=nv), nv
        if isinstance(expr, ast.BinOp):
            lr, lv = _lower_expr(expr.left)
            rr, rv = _lower_expr(expr.right)
            ops = _ast_binop_sym(expr.op)
            nv = _eval_binop_safe(ops, lv, rv) if (lv is not None and rv is not None) else None
            vt = _fmt_num(nv) if nv is not None else f"bin_{ops}"
            et = f"{lr} {ops} {rr} -> {vt}" if nv is not None else f"{lr} {ops} {rr}"
            return _emit("⟨DEDUCE⟩", [lr, rr], [ops], vt, et, num_val=nv), nv
        if isinstance(expr, ast.Compare) and expr.ops and expr.comparators:
            lr, lv = _lower_expr(expr.left)
            rr, rv = _lower_expr(expr.comparators[0])
            cs = _ast_cmp_sym(expr.ops[0])
            vt = f"cmp_{cs}"
            if lv is not None and rv is not None:
                if cs == "==": vt = str(abs(lv - rv) < 1e-6)
                elif cs == "!=": vt = str(abs(lv - rv) >= 1e-6)
                elif cs == "<": vt = str(lv < rv)
                elif cs == "<=": vt = str(lv <= rv)
                elif cs == ">": vt = str(lv > rv)
                elif cs == ">=": vt = str(lv >= rv)
            return _emit("⟨DEDUCE⟩", [lr, rr], [cs], vt, f"{lr} {cs} {rr}"), None
        if isinstance(expr, ast.Call):
            fn_name = ast.unparse(expr.func)[:24] if hasattr(ast, "unparse") else "call"
            fn_tail = fn_name.split(".")[-1]
            arg_res = [_lower_expr(a) for a in expr.args[:3]]
            arg_refs = [x[0] for x in arg_res]
            arg_vals = [x[1] for x in arg_res]
            if fn_tail == "print":
                for av in reversed(arg_vals):
                    if av is not None:
                        last_printed_num = av
                        break
                # Skip noisy bare print() step if we already computed the numeric value!
                if arg_refs and last_printed_num is not None:
                    return arg_refs[-1], last_printed_num
            nv = _eval_math_call_safe(fn_tail, arg_vals)
            vt = _fmt_num(nv) if nv is not None else fn_tail[:12]
            et = f"call {fn_name}({', '.join(arg_refs)})" + (f" -> {vt}" if nv is not None else "")
            return _emit("⟨DEDUCE⟩", arg_refs, [], vt, et, num_val=nv), nv
        if isinstance(expr, ast.Subscript):
            vr, _ = _lower_expr(expr.value)
            sr, _ = _lower_expr(expr.slice)
            return _emit("⟨DEDUCE⟩", [vr, sr], [], "index", f"{vr}[{sr}]"), None
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            raw = ast.unparse(expr)[:40] if hasattr(ast, "unparse") else "coll"
            return _emit("⟨DEDUCE⟩", [], [], type(expr).__name__.lower(), f"init {raw}"), None
        raw = ast.unparse(expr)[:48] if hasattr(ast, "unparse") else type(expr).__name__
        return _emit("⟨DEDUCE⟩", [], [], type(expr).__name__.lower()[:12], raw), None

    def _walk(stmts: List[ast.stmt]) -> None:
        nonlocal truncated
        for st in stmts:
            if len(steps) >= max_steps - 1:
                truncated = True
                break
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                continue
            elif isinstance(st, ast.Assign):
                if len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    r, _ = _lower_expr(st.value)
                    name_to_ref[st.targets[0].id] = r
                elif len(st.targets) == 1 and isinstance(st.targets[0], ast.Tuple) and isinstance(st.value, ast.Tuple):
                    for t_el, v_el in zip(st.targets[0].elts, st.value.elts):
                        if len(steps) >= max_steps - 1:
                            truncated = True
                            break
                        if isinstance(t_el, ast.Name):
                            r, _ = _lower_expr(v_el)
                            name_to_ref[t_el.id] = r
                else:
                    _lower_expr(st.value)
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                cur = name_to_ref.get(st.target.id, "⟨IN_0⟩")
                cur_v = ref_to_num.get(cur)
                rhs, rhs_v = _lower_expr(st.value)
                ops = _ast_binop_sym(st.op)
                nv = _eval_binop_safe(ops, cur_v, rhs_v) if (cur_v is not None and rhs_v is not None) else None
                vt = _fmt_num(nv) if nv is not None else f"aug_{ops}"
                name_to_ref[st.target.id] = _emit(
                    "⟨DEDUCE⟩", [cur, rhs], [ops], vt, f"{st.target.id} {ops}= {rhs}", num_val=nv
                )
            elif isinstance(st, ast.FunctionDef):
                for idx_a, a in enumerate(st.args.args[:8]):
                    name_to_ref[a.arg] = f"⟨IN_{idx_a}⟩"
                _walk(st.body)
            elif isinstance(st, ast.For):
                ir, _ = _lower_expr(st.iter)
                lv = _emit("⟨SPLIT⟩", [ir], [], "loop_for", f"for iter {ir}")
                if isinstance(st.target, ast.Name):
                    name_to_ref[st.target.id] = lv
                _walk(st.body)
            elif isinstance(st, ast.While):
                cr, _ = _lower_expr(st.test)
                _emit("⟨SPLIT⟩", [cr], [], "loop_while", f"while {cr}")
                _walk(st.body)
            elif isinstance(st, ast.If):
                cr, _ = _lower_expr(st.test)
                _emit("⟨SPLIT⟩", [cr], [], "branch_if", f"if {cr}")
                _walk(st.body)
                if st.orelse:
                    _walk(st.orelse)
            elif isinstance(st, ast.Return) and st.value is not None:
                rr, rv = _lower_expr(st.value)
                vt = _fmt_num(rv) if rv is not None else (gt[:14] if gt else "return")
                _emit("⟨VERIFY⟩", [rr], [], vt, f"return {rr}", num_val=rv)
            elif isinstance(st, ast.Expr):
                if isinstance(st.value, ast.Constant):
                    continue
                _lower_expr(st.value)

    _walk(mod.body)
    if truncated:
        return [], False, "tir_truncated_dropped", gt
    if len(steps) < 2:
        return [], False, "tir_too_few_steps", gt

    # For MathInstruct-PoT (which had empty gt), REQUIRE our safe numeric evaluator to have resolved a concrete final answer!
    if is_pot and not gt:
        resolved = last_printed_num if last_printed_num is not None else last_computed_num
        if resolved is None:
            return [], False, "pot_unresolved_gt_dropped", ""
        gt = _fmt_num(resolved)
        if gt == "OVERFLOW":
            return [], False, "pot_overflow_gt_dropped", ""

    steps[0].op = "⟨GOAL⟩"
    last_ref = steps[-1].out_var or "$v1"
    _emit("⟨COMMIT⟩", [last_ref], [], gt[:14], f"commit_ans = {last_ref} -> {gt[:14]}", flag="⟨EOT⟩")
    if len(steps) > max_steps:
        return [], False, "tir_truncated_dropped", gt
    return steps, True, "ok", gt


def compile_knights_knaves_gold(row: Dict[str, Any], max_steps: int = 32) -> Tuple[List[MicroStepIR], bool, str]:
    """Gold v2 K&K compiler: ZERO truncation! Drops puzzles needing > 32 steps cleanly."""
    payload = json.loads(row["reference_solution"])
    names: List[str] = payload["names"]
    raw_steps: List[str] = payload["steps"]
    solution: List[bool] = payload["solution"]
    if not names or not raw_steps:
        return [], False, "kk_empty"
    if len(raw_steps) + 2 > max_steps:
        return [], False, "kk_truncated_dropped"

    name_to_in = {nm: f"⟨IN_{i}⟩" for i, nm in enumerate(names[:16])}
    name_to_var: Dict[str, str] = {}
    steps: List[MicroStepIR] = []
    v_cnt = 0

    def _nv() -> str:
        nonlocal v_cnt
        v_cnt += 1
        return f"$v{v_cnt}"

    g_var = _nv()
    steps.append(
        MicroStepIR(
            op="⟨GOAL⟩",
            in_refs=[f"⟨IN_{i}⟩" for i in range(min(4, len(names)))],
            lit_tokens=[],
            val_token=f"kk_{len(names)}p",
            out_var=g_var,
            expr_text=f"logic_goal kk_{len(names)}p",
        )
    )

    for raw in raw_steps:
        s = raw.strip()
        mentioned = [nm for nm in names if nm in s]
        refs: List[str] = []
        for nm in mentioned[:4]:
            if nm in name_to_var:
                refs.append(name_to_var[nm])
            elif nm in name_to_in:
                refs.append(name_to_in[nm])
        in_r = [r for r in dict.fromkeys(refs) if r.startswith("⟨")][:4]
        v_d = [r for r in dict.fromkeys(refs) if r.startswith("$")][: max(0, 4 - len(in_r))]

        ov = _nv()
        sl = s.lower()
        if ("cannot be a" in sl or "would contradict" in sl) and ("no contradiction" not in sl):
            role_val = "not_knight" if "cannot be a knight" in sl else "not_knave"
            steps.append(
                MicroStepIR(
                    op="⟨VERIFY⟩",
                    in_refs=in_r,
                    var_deps=v_d,
                    lit_tokens=["!="],
                    val_token=role_val,
                    out_var=ov,
                    flag="⟨CONTRA⟩",
                    expr_text=f"contra {role_val} on {len(mentioned)} entities",
                )
            )
        elif "exhausted" in sl or "go back" in sl or "reconsider" in sl:
            steps.append(
                MicroStepIR(
                    op="⟨SPLIT⟩",
                    in_refs=in_r,
                    var_deps=v_d,
                    lit_tokens=[],
                    val_token="backtrack",
                    out_var=ov,
                    expr_text=f"backtrack_split {len(mentioned)} entities",
                )
            )
        elif sl.startswith("assume "):
            role_val = "knight" if "is a knight" in sl else "knave"
            steps.append(
                MicroStepIR(
                    op="⟨HYP⟩",
                    in_refs=in_r,
                    var_deps=v_d,
                    lit_tokens=["=="],
                    val_token=role_val,
                    out_var=ov,
                    expr_text=f"hyp_assume {role_val}",
                )
            )
            if mentioned:
                name_to_var[mentioned[0]] = ov
        else:
            steps.append(
                MicroStepIR(
                    op="⟨DEDUCE⟩",
                    in_refs=in_r,
                    var_deps=v_d,
                    lit_tokens=[],
                    val_token="consistent",
                    out_var=ov,
                    expr_text="deduce consistent",
                )
            )
            if mentioned:
                name_to_var[mentioned[0]] = ov

    bit_str = "".join("1" if b else "0" for b in solution)
    final_deps = list(dict.fromkeys(name_to_var.values()))[-4:]
    cv = _nv()
    steps.append(
        MicroStepIR(
            op="⟨COMMIT⟩",
            in_refs=[],
            var_deps=final_deps,
            lit_tokens=[],
            val_token=f"sol_{bit_str}",
            out_var=cv,
            flag="⟨EOT⟩",
            expr_text=f"commit_kk {len(solution)}p -> {bit_str}",
        )
    )
    return steps, True, "ok"


def compile_proofwriter_gold(row: Dict[str, Any], max_steps: int = 14) -> Tuple[List[MicroStepIR], bool, str]:
    """Gold v2 ProofWriter compiler: crisp multi-hop deduction (drops bloated >14-step theories!)."""
    payload = json.loads(row["reference_solution"])
    proofs_str: str = payload["allProofs"]
    ans: str = payload["answer"]
    qdep: int = int(payload.get("qdep", 1))

    level_chunks = re.findall(r"@(\d+):\s*(.*?)(?=@\d+:|$)", proofs_str, flags=re.DOTALL)
    if not level_chunks:
        return [], False, "pw_no_levels"

    triple_to_in: Dict[str, str] = {}
    fact_to_var: Dict[str, str] = {}
    steps: List[MicroStepIR] = []
    v_cnt = 0

    def _nv() -> str:
        nonlocal v_cnt
        v_cnt += 1
        return f"$v{v_cnt}"

    for lvl_s, body in level_chunks:
        lvl = int(lvl_s)
        entries = re.findall(r"([^.\[\]]+\.)\[([^\]]+)\]", body)
        if lvl == 0:
            for _, pexpr in entries:
                for tid in re.findall(r"triple\d+", pexpr):
                    if tid not in triple_to_in and len(triple_to_in) < 48:
                        triple_to_in[tid] = f"⟨IN_{len(triple_to_in)}⟩"
            g_v = _nv()
            steps.append(
                MicroStepIR(
                    op="⟨GOAL⟩",
                    in_refs=list(triple_to_in.values())[:4],
                    lit_tokens=[],
                    val_token=f"pw_d{qdep}",
                    out_var=g_v,
                    expr_text=f"proof_goal depth={qdep}",
                )
            )
        else:
            for stmt, pexpr in entries:
                if len(steps) >= max_steps - 1:
                    return [], False, "pw_bloated_dropped"
                rules = re.findall(r"rule\d+", pexpr)
                tr_ids = re.findall(r"triple\d+", pexpr)
                in_r = [triple_to_in[t] for t in dict.fromkeys(tr_ids) if t in triple_to_in][:3]
                prev_v = list(fact_to_var.values())[-2:] if fact_to_var else ([steps[-1].out_var] if steps else [])
                v_d = [v for v in prev_v if v][: max(0, 4 - len(in_r))]
                r_tok = rules[-1] if rules else f"rule_d{lvl}"
                ov = _nv()
                words = re.findall(r"[A-Za-z]+", stmt.lower())
                pred = "_".join(words[-2:]) if len(words) >= 2 else "fact"
                is_neg = "not" in words
                steps.append(
                    MicroStepIR(
                        op="⟨DEDUCE⟩",
                        in_refs=in_r,
                        var_deps=v_d,
                        lit_tokens=["!=" if is_neg else "=="],
                        val_token=r_tok[:12],
                        out_var=ov,
                        expr_text=f"apply_rule d{lvl} {'not_' if is_neg else ''}{pred[:16]}",
                    )
                )
                fact_to_var[stmt.strip()] = ov

    if len(steps) < 2:
        return [], False, "pw_too_few_steps"

    cv = _nv()
    steps.append(
        MicroStepIR(
            op="⟨COMMIT⟩",
            in_refs=[],
            var_deps=[steps[-1].out_var] if steps[-1].out_var else [],
            lit_tokens=[],
            val_token=ans,
            out_var=cv,
            flag="⟨EOT⟩",
            expr_text=f"commit_proof -> {ans}",
        )
    )
    return steps, True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_seeds.jsonl"))
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_gold_v2_compiled.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-sandbox", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_4_5b.yaml")
    vocab = SymbolicVocab(cfg)
    compiler = LinearScanDSLCompiler(cfg, vocab)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_path.with_name(out_path.stem + "_report.json")

    seeds: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))
    print(f"=== [Gold v2 Zero-Truncation Deterministic Compiler] Loaded {len(seeds):,} seeds | workers={args.workers} ===", flush=True)

    def _compile_one(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]:
        fmt = row.get("format", "")
        gt_out = row.get("ground_truth_answer", "")
        if fmt == "gsm8k_eq":
            steps, ok, reason = compile_math_or_logic_deterministic(row)
        elif fmt == "tir_python":
            steps, ok, reason, gt_out = compile_tir_python_gold(row, max_steps=32)
        elif fmt == "python_fn":
            steps, ok, reason = compile_code_deterministic(
                row, verify_sandbox=(not args.no_sandbox), strict_no_truncate=True, max_steps=30
            )
            if ok and not str(gt_out).strip():
                gt_out = str(row.get("reference_solution", "")).strip()
        elif fmt == "knights_knaves":
            steps, ok, reason = compile_knights_knaves_gold(row, max_steps=32)
        elif fmt == "proofwriter":
            steps, ok, reason = compile_proofwriter_gold(row, max_steps=14)
        else:
            return row, None, f"unknown_fmt:{fmt}"

        if not ok or not steps:
            return row, None, reason

        try:
            final_live = [steps[-1].out_var] if steps[-1].out_var else []
            raw_tr = compiler.compile(steps, final_live_vars=final_live)
            aln_tr = compiler.compile_aligned_frames(steps, final_live_vars=final_live, prefer_8slot=True)
        except CompilerValidationError as e:
            return row, None, f"static_check:{str(e).split(':')[0]}"

        nl_len = approx_nl_token_count(row["reference_solution"])
        rec = {
            "id": row["id"],
            "domain": row["domain"],
            "source": row["source"],
            "format": fmt,
            "split_hint": row.get("split_hint", ""),
            "question": row["question"],
            "ground_truth_answer": gt_out,
            "num_steps": len(steps),
            "frame_sizes": aln_tr.frame_sizes,
            "token_ids_aligned": aln_tr.token_ids,
            "reassert_mask": aln_tr.reassert_mask,
            "step_expressions": [s.expr_text for s in steps if s.expr_text],
            "steps_ir": [
                {
                    "op": s.op,
                    "operands": s.operands,
                    "in_refs": s.in_refs,
                    "var_deps": s.var_deps,
                    "lit_tokens": s.lit_tokens,
                    "val_token": s.val_token,
                    "out_var": s.out_var,
                    "flag": s.flag,
                    "expr_text": s.expr_text,
                }
                for s in steps
            ],
            "nl_tokens_est": nl_len,
            "sym_tokens_compact": len(raw_tr.token_ids),
            "sym_tokens_aligned": len(aln_tr.token_ids),
            "max_regs": len(aln_tr.reg_map),
        }
        return row, rec, "ok"

    t0 = time.perf_counter()
    by_dom: Dict[str, Dict[str, int]] = {"math": {"tot": 0, "ok": 0}, "code": {"tot": 0, "ok": 0}, "logic": {"tot": 0, "ok": 0}}
    by_fmt: Dict[str, Dict[str, int]] = {}
    fails = collections.Counter()
    ok_cnt = fail_parse = fail_static = 0
    nl_sum = raw_sum = aln_sum = steps_sum = max_regs = 0

    with open(out_path, "w", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for idx, (row, rec, reason) in enumerate(pool.map(_compile_one, seeds), 1):
            dom = row["domain"]
            fmt = row.get("format", "unknown")
            by_dom.setdefault(dom, {"tot": 0, "ok": 0})
            by_dom[dom]["tot"] += 1
            by_fmt.setdefault(fmt, {"tot": 0, "ok": 0})
            by_fmt[fmt]["tot"] += 1
            if rec is None:
                if reason.startswith("static_check:"):
                    fail_static += 1
                else:
                    fail_parse += 1
                fails[reason] += 1
            else:
                ok_cnt += 1
                by_dom[dom]["ok"] += 1
                by_fmt[fmt]["ok"] += 1
                nl_sum += rec["nl_tokens_est"]
                raw_sum += rec["sym_tokens_compact"]
                aln_sum += rec["sym_tokens_aligned"]
                steps_sum += rec["num_steps"]
                max_regs = max(max_regs, rec.pop("max_regs"))
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if idx % 30000 == 0:
                print(f"  [Gold v2 Compile] {idx:,}/{len(seeds):,} | ok={ok_cnt:,} ({100.0*ok_cnt/idx:.1f}%) | {time.perf_counter()-t0:.1f}s", flush=True)

    elapsed = round(time.perf_counter() - t0, 2)
    attempted_static = ok_cnt + fail_static
    report = {
        "version": "gold_v2",
        "total_seeds": len(seeds),
        "compiled_ok": ok_cnt,
        "overall_pass_pct": round(100.0 * ok_cnt / max(1, len(seeds)), 2),
        "static_5check_pass_pct": round(100.0 * ok_cnt / max(1, attempted_static), 2),
        "failed_or_strictly_filtered": fail_parse,
        "failed_static_5checks": fail_static,
        "by_domain": by_dom,
        "by_format": by_fmt,
        "top_filters_and_failures": dict(fails.most_common(12)),
        "avg_steps_per_problem": round(steps_sum / max(1, ok_cnt), 2),
        "max_registers_used": max_regs,
        "avg_nl_tokens": round(nl_sum / max(1, ok_cnt), 1),
        "avg_sym_tokens_compact": round(raw_sum / max(1, ok_cnt), 1),
        "avg_sym_tokens_aligned_8_16": round(aln_sum / max(1, ok_cnt), 1),
        "mean_compression_compact_x": round(nl_sum / max(1, raw_sum), 2),
        "mean_compression_aligned_x": round(nl_sum / max(1, aln_sum), 2),
        "elapsed_sec": elapsed,
        "output_path": str(out_path),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== [Gold v2 Compile DONE in {elapsed}s] ===", flush=True)
    print(f"Compiled OK: {ok_cnt:,} / {len(seeds):,} ({report['overall_pass_pct']}%) | 5-Check Pass: {report['static_5check_pass_pct']}% | Max Regs: {max_regs}/64", flush=True)
    print(f"By Domain: {by_dom}", flush=True)
    print(f"By Format: {by_fmt}", flush=True)
    print(f"Strict Filters Dropped: {report['top_filters_and_failures']}", flush=True)


if __name__ == "__main__":
    main()
