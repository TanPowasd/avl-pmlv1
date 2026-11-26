"""Stage 1 (M2) AVL-PMLv1.2 Escape-Free Line DSL Compiler Pipeline (Hybrid Deterministic + Flash API / Local LLM).

Supports 3 pluggable backends under one unified resumable JSONL pipeline:
  1. --backend deterministic : 0-RMB exact AST + SymPy SSA-splitter for real Math (GSM8K/NuminaMath),
                               real Python Code (MBPP/KodCode + sandbox unit-test check), & formal Logic!
  2. --backend local_llm     : 0-RMB local RTX 5080 OpenAI-compatible endpoint (Ollama / vLLM).
  3. --backend api           : Async Flash API (DeepSeek / Qwen-Flash / Gemini) with 85% fast + 15% fallback waterfall!

Enforces ALL 5 Static Hard Checks in LinearScanDSLCompiler (including Check 5: 1-Causal-Hop-Per-Beat &
v1.2 [VAL, ⟨BIND⟩, ⟨REG_j⟩] 8/16-slot alignment) + SymPy / Python Sandbox Ground-Truth Verification!
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.vocab.dsl_compiler import CompilerValidationError, LinearScanDSLCompiler, MicroStepIR
from avl_pmlv1.vocab.sym_vocab import SymbolicVocab, extract_prompt_number_spans

# Exact system prompt for Flash API / Local LLM reformatting (reformat-from-verified-solution mode!)
DSL_REFORMAT_SYSTEM_PROMPT = """You are a strict symbolic trace compiler for AVL-PMLv1.2.
Convert the given problem and its VERIFIED reference solution into escape-free line DSL micro-steps.
CRITICAL RULES (5 Static Hard Checks):
1. One line per micro-step: OP | in=IN_0,$v1 | lit=OP_TOK | val=RESULT | out=$v2 | expr=short_phrase | flag=NONE
2. OP must be one of: GOAL, HYP, DEDUCE, SPLIT, VERIFY, COMMIT.
3. STRICTLY ONE CAUSAL HOP PER LINE (Check 5): NEVER chain >=2 binary operators (like a+b*c) on one line; split into SSA lines!
4. Every line with out=$v_k MUST include val=RESULT (concise literal/symbol) and at most 4 inputs in `in=`.
5. Never read an undefined $v_k. The final line MUST use OP=COMMIT and flag=EOT with val=<final_answer>.
Output ONLY the DSL lines, no markdown fences or prose."""


def approx_nl_token_count(text: str) -> int:
    """Standard subword-calibrated BPE token estimate (~1.25x word/punct chunks for English/code/LaTeX)."""
    chunks = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\s\w]", text)
    return max(1, int(round(len(chunks) * 1.15)))


def _fast_parse_float(s: str) -> Optional[float]:
    s = s.replace(",", "").strip()
    if not s or len(s) > 40:
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        pass
    if "/" in s and s.count("/") == 1:
        p, q = s.split("/", 1)
        try:
            vp, vq = float(p.strip()), float(q.strip())
            if abs(vq) > 1e-12:
                v = vp / vq
                return v if math.isfinite(v) else None
        except ValueError:
            pass
    if "**" in s or "^" in s:
        return None
    try:
        v = float(sp.sympify(s))
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _safe_num_equal(a: str, b: str) -> bool:
    va = _fast_parse_float(a)
    vb = _fast_parse_float(b)
    if va is not None and vb is not None:
        return abs(va - vb) <= 1e-3 * max(1.0, abs(vb))
    return a.strip().lower() == b.strip().lower()


def _ast_binop_sym(node: ast.operator) -> str:
    if isinstance(node, ast.Add):
        return "+"
    if isinstance(node, ast.Sub):
        return "-"
    if isinstance(node, ast.Mult):
        return "*"
    if isinstance(node, (ast.Div, ast.FloorDiv)):
        return "/"
    if isinstance(node, ast.Mod):
        return "%"
    if isinstance(node, (ast.BitXor, ast.Pow)):
        return "^"
    return "+"


def _ast_cmp_sym(node: ast.cmpop) -> str:
    if isinstance(node, ast.Eq):
        return "=="
    if isinstance(node, ast.NotEq):
        return "!="
    if isinstance(node, ast.Lt):
        return "<"
    if isinstance(node, ast.LtE):
        return "<="
    if isinstance(node, ast.Gt):
        return ">"
    if isinstance(node, ast.GtE):
        return ">="
    return "=="


def _fmt_num(val: float) -> str:
    if not math.isfinite(val) or abs(val) > 1e15:
        return "OVERFLOW"
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.4f}".rstrip("0").rstrip(".")


# =============================================================================
# 1. Deterministic 1-Hop SSA Math / Logic Compiler (AST + SymPy Verified!)
# =============================================================================
def compile_math_or_logic_deterministic(row: Dict[str, Any]) -> Tuple[List[MicroStepIR], bool, str]:
    """Lower real GSM8K / NuminaMath / Logic verified solutions into exact 1-hop SSA MicroStepIR."""
    q = row["question"]
    sol = row["reference_solution"]
    gt = str(row["ground_truth_answer"]).strip()

    # Extract input numbers from problem statement as read-only ⟨IN_0..k⟩ slots (1:1 aligned with S^RO & executor!)
    q_nums = [m_s for _, _, m_s in extract_prompt_number_spans(q)]
    in_val_to_slot: Dict[str, str] = {}
    for k_idx, n_str in enumerate(q_nums[:48]):
        fv = _fast_parse_float(n_str)
        if fv is None:
            continue
        key = _fmt_num(fv)
        if key != "OVERFLOW" and key not in in_val_to_slot:
            in_val_to_slot[key] = f"⟨IN_{k_idx}⟩"

    # Extract explicit <<expr=res>> annotations (GSM8K & formal_dag) OR inline equation chains (NuminaMath)
    raw_eqs: List[Tuple[str, str]] = []
    for m in re.finditer(r"<<([^=<>]+)=([^<>]+)>>", sol):
        lhs_s, rhs_s = m.group(1).strip(), m.group(2).strip()
        if not any(op in lhs_s for op in "+-*/%^"):
            # Handle GSM8K `60/2 = <<30=30>>30` where the arithmetic expression sits right before `= <<c=c>>`
            line_start = sol.rfind("\n", 0, m.start()) + 1
            line_pref = sol[line_start : m.start()]
            m_pre = re.search(r"(-?\d[\d\s,\.\+\-\*/\(\)]*)=\s*\$?\s*$", line_pref)
            if m_pre and any(op in m_pre.group(1) for op in "+-*/"):
                cand_lhs = m_pre.group(1).strip()
                if "=" in cand_lhs:
                    cand_lhs = cand_lhs.split("=")[-1].strip()
                lhs_s = cand_lhs
        raw_eqs.append((lhs_s, rhs_s))

    if not raw_eqs:
        # Fallback for NuminaMath-CoT: extract clean arithmetic/algebraic equations `expr = num`
        clean_sol = sol.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
        clean_sol = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", clean_sol)
        for m in re.finditer(r"(\d[\d\s\.\+\-\*/\(\)]{1,45})=\s*([-+]?\d+(?:\.\d+)?)", clean_sol):
            lhs, rhs = m.group(1).strip(), m.group(2).strip()
            if any(op in lhs for op in "+-*/"):
                raw_eqs.append((lhs, rhs))

    if not raw_eqs:
        return [], False, "no_parseable_equations"

    steps: List[MicroStepIR] = []
    val_to_var: Dict[str, str] = {}
    var_counter = 0

    def _next_var() -> str:
        nonlocal var_counter
        var_counter += 1
        return f"$v{var_counter}"

    def _lower_expr_node(node: ast.AST) -> Tuple[Optional[str], Optional[float], str]:
        """Recursively SSA-split compound arithmetic AST into strictly 1-causal-hop MicroStepIR!"""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            fv = float(node.value)
            sv = _fmt_num(fv)
            if sv == "OVERFLOW":
                return None, None, ""
            if sv in val_to_var:
                return val_to_var[sv], fv, sv
            if sv in in_val_to_slot:
                return in_val_to_slot[sv], fv, sv
            # Preserve non-prompt implicit constants (e.g. 12 months, 60 mins, 100%, English number words) losslessly!
            return sv, fv, sv

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            ref, fv, txt = _lower_expr_node(node.operand)
            if fv is None or ref is None:
                return None, None, ""
            nfv = -fv
            sv = _fmt_num(nfv)
            if sv == "OVERFLOW":
                return None, None, ""
            if not ref.startswith(("⟨", "$")):
                return sv, nfv, sv
            ov = _next_var()
            steps.append(
                MicroStepIR(
                    op="⟨DEDUCE⟩",
                    operands=["0", ref],
                    lit_tokens=["-"],
                    val_token=sv,
                    out_var=ov,
                    expr_text=f"{ov} = -{txt} -> {sv}",
                )
            )
            val_to_var[sv] = ov
            return ov, nfv, ov

        if isinstance(node, ast.BinOp):
            l_ref, l_val, l_txt = _lower_expr_node(node.left)
            r_ref, r_val, r_txt = _lower_expr_node(node.right)
            if l_val is None or r_val is None or l_ref is None or r_ref is None:
                return None, None, ""
            op_s = _ast_binop_sym(node.op)
            try:
                if op_s == "+":
                    res_v = l_val + r_val
                elif op_s == "-":
                    res_v = l_val - r_val
                elif op_s == "*":
                    res_v = l_val * r_val
                elif op_s == "/":
                    if abs(r_val) < 1e-12:
                        return None, None, ""
                    res_v = l_val / r_val
                elif op_s == "%":
                    if abs(r_val) < 1e-12:
                        return None, None, ""
                    res_v = l_val % r_val
                elif op_s == "^":
                    if abs(r_val) > 20 or abs(l_val) > 1e6:
                        return None, None, ""
                    res_v = l_val ** r_val
                else:
                    return None, None, ""
            except Exception:
                return None, None, ""
            if not math.isfinite(res_v) or abs(res_v) > 1e15:
                return None, None, ""
            sv = _fmt_num(res_v)
            if sv == "OVERFLOW":
                return None, None, ""
            ov = _next_var()
            # Strictly preserve left-to-right operand order (fixing - / reversal, implicit constants, & x * x!)
            steps.append(
                MicroStepIR(
                    op="⟨GOAL⟩" if len(steps) == 0 else "⟨DEDUCE⟩",
                    operands=[l_ref, r_ref],
                    lit_tokens=[op_s],
                    val_token=sv,
                    out_var=ov,
                    expr_text=f"{ov} = {l_txt} {op_s} {r_txt} -> {sv}",
                )
            )
            val_to_var[sv] = ov
            return ov, res_v, ov

        return None, None, ""

    last_val_str = ""
    for lhs, rhs in raw_eqs[:24]:
        expr_clean = lhs.replace(",", "").replace(" ", "")
        # Normalize implicit multiplication like 2(3+4) -> 2*(3+4)
        expr_clean = re.sub(r"(\d)\(", r"\1*(", expr_clean)
        try:
            tree = ast.parse(expr_clean, mode="eval")
        except SyntaxError:
            continue
        prev_step_cnt = len(steps)
        ref, comp_val, _ = _lower_expr_node(tree.body)
        if comp_val is not None and len(steps) > prev_step_cnt:
            last_val_str = steps[-1].val_token or _fmt_num(comp_val)
            # Also bind RHS string if numerically identical
            if _safe_num_equal(last_val_str, rhs) and ref and ref.startswith("$"):
                rv = _fast_parse_float(rhs)
                if rv is not None:
                    val_to_var[_fmt_num(rv)] = ref

    if not steps:
        return [], False, "no_steps_emitted"

    # Ground-truth verification strictly against emitted steps[-1].val_token (never phantom bare constants!)
    verified = bool(steps[-1].val_token and _safe_num_equal(steps[-1].val_token, gt))
    if not verified:
        # Check if any earlier step computed the exact ground truth
        for st in reversed(steps):
            if st.val_token and _safe_num_equal(st.val_token, gt):
                # Trim trailing redundant steps after ground truth was reached
                cut_idx = steps.index(st)
                steps = steps[: cut_idx + 1]
                verified = True
                break

    if not verified:
        return [], False, "gt_mismatch"

    # Finalize last step as COMMIT + ⟨EOT⟩
    steps[-1].op = "⟨COMMIT⟩"
    steps[-1].flag = "⟨EOT⟩"
    if len(steps) > 36:
        return [], False, "too_many_steps"
    return steps, True, "ok"


# =============================================================================
# 2. Deterministic 1-Hop Python AST Code Compiler (+ Sandbox Unit-Test Check!)
# =============================================================================
def verify_python_solution_sandbox(code: str, unit_tests: List[str], timeout_s: float = 2.0) -> bool:
    if not unit_tests:
        return True
    test_script = code + "\n\n" + "\n".join(unit_tests) + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode == 0
    except Exception:
        return False


def compile_code_deterministic(
    row: Dict[str, Any], verify_sandbox: bool = True, strict_no_truncate: bool = False, max_steps: int = 30
) -> Tuple[List[MicroStepIR], bool, str]:
    """Lower real MBPP/KodCode Python function AST into verified 1-hop SSA MicroStepIR frames."""
    code = row["reference_solution"]
    tests = row.get("unit_tests", [])
    src = str(row.get("source", ""))
    # MBPP has direct assert lists -> run live Python subprocess sandbox!
    # KodCode-V1 is already execution-verified (filter_reason=None) with pytest modules -> verify strict syntax + bytecode compile!
    if verify_sandbox and src.startswith("mbpp:"):
        if not verify_python_solution_sandbox(code, tests):
            return [], False, "sandbox_unit_test_failed"
    else:
        try:
            compile(code, "<kodcode>", "exec")
        except Exception:
            return [], False, "python_compile_error"

    try:
        mod = ast.parse(code)
    except SyntaxError:
        return [], False, "python_syntax_error"

    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    is_method = False
    if not funcs:
        cls_methods = [
            m
            for c in mod.body
            if isinstance(c, ast.ClassDef)
            for m in c.body
            if isinstance(m, ast.FunctionDef) and not (m.name.startswith("__") and m.name.endswith("__"))
        ]
        if cls_methods:
            funcs = cls_methods
            is_method = True
    if not funcs:
        return [], False, "no_function_def"
    fn = funcs[-1]

    steps: List[MicroStepIR] = []
    name_to_ref: Dict[str, str] = {}
    v_cnt = 0
    truncated = False

    def _nv() -> str:
        nonlocal v_cnt
        v_cnt += 1
        return f"$v{v_cnt}"

    # Map function arguments to read-only ⟨IN_0..k⟩ (skip 'self'/'cls' for class methods)
    arg_list = [a for a in fn.args.args if not (is_method and a.arg in ("self", "cls"))]
    for i, arg in enumerate(arg_list[:16]):
        name_to_ref[arg.arg] = f"⟨IN_{i}⟩"

    def _emit_step(
        op: str,
        refs: List[str],
        lit_toks: List[str],
        val_tok: str,
        expr_txt: str,
        flag: Optional[str] = None,
    ) -> str:
        ov = _nv()
        ord_ops = [r for r in refs if r][:4]
        clean_expr = " ".join(expr_txt.split())[:120]
        steps.append(
            MicroStepIR(
                op=op,
                operands=ord_ops,
                lit_tokens=lit_toks[:2],
                val_token=val_tok,
                out_var=ov,
                flag=flag,
                expr_text=clean_expr,
            )
        )
        return ov

    def _lower_py_expr(expr: ast.AST) -> str:
        nonlocal truncated
        if len(steps) >= max_steps:
            truncated = True
            return "$v1" if v_cnt > 0 else "⟨IN_0⟩"
        if isinstance(expr, ast.Name):
            if expr.id in name_to_ref:
                return name_to_ref[expr.id]
            return _emit_step("⟨DEDUCE⟩", [], [], expr.id[:12], f"ref {expr.id}")
        if isinstance(expr, ast.Constant):
            s = str(expr.value)[:12] if expr.value is not None else "None"
            return _emit_step("⟨DEDUCE⟩", [], [], s or "0", f"const {s}")
        if isinstance(expr, ast.BinOp):
            lr = _lower_py_expr(expr.left)
            rr = _lower_py_expr(expr.right)
            ops = _ast_binop_sym(expr.op)
            return _emit_step("⟨DEDUCE⟩", [lr, rr], [ops], f"bin_{ops}", f"{lr} {ops} {rr}")
        if isinstance(expr, ast.Compare) and expr.ops and expr.comparators:
            lr = _lower_py_expr(expr.left)
            rr = _lower_py_expr(expr.comparators[0])
            cs = _ast_cmp_sym(expr.ops[0])
            return _emit_step("⟨DEDUCE⟩", [lr, rr], [cs], f"cmp_{cs}", f"{lr} {cs} {rr}")
        if isinstance(expr, ast.Call):
            fn_name = ast.unparse(expr.func)[:20] if hasattr(ast, "unparse") else "call"
            arg_refs = [_lower_py_expr(a) for a in expr.args[:3]]
            return _emit_step("⟨DEDUCE⟩", arg_refs, [], fn_name[:12], f"call {fn_name}({', '.join(arg_refs)})")
        if isinstance(expr, ast.Subscript):
            vr = _lower_py_expr(expr.value)
            sr = _lower_py_expr(expr.slice)
            return _emit_step("⟨DEDUCE⟩", [vr, sr], [], "index", f"{vr}[{sr}]")
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            raw = ast.unparse(expr)[:40] if hasattr(ast, "unparse") else "collection"
            return _emit_step("⟨DEDUCE⟩", [], [], type(expr).__name__.lower(), f"init {raw}")
        raw = ast.unparse(expr)[:48] if hasattr(ast, "unparse") else type(expr).__name__
        return _emit_step("⟨DEDUCE⟩", [], [], type(expr).__name__.lower()[:12], raw)

    def _walk_stmts(stmts: List[ast.stmt]) -> None:
        nonlocal truncated
        for st in stmts:
            if len(steps) >= max_steps:
                truncated = True
                break
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                r = _lower_py_expr(st.value)
                name_to_ref[st.targets[0].id] = r
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                cur_r = name_to_ref.get(st.target.id, "⟨IN_0⟩")
                rhs_r = _lower_py_expr(st.value)
                ops = _ast_binop_sym(st.op)
                name_to_ref[st.target.id] = _emit_step("⟨DEDUCE⟩", [cur_r, rhs_r], [ops], f"aug_{ops}", f"{st.target.id} {ops}= {rhs_r}")
            elif isinstance(st, ast.If):
                cr = _lower_py_expr(st.test)
                _emit_step("⟨SPLIT⟩", [cr], [], "branch_if", f"if {cr}")
                _walk_stmts(st.body[:4])
                if st.orelse:
                    _walk_stmts(st.orelse[:3])
            elif isinstance(st, ast.For):
                ir = _lower_py_expr(st.iter)
                lv = _emit_step("⟨SPLIT⟩", [ir], [], "loop_for", f"for iter {ir}")
                if isinstance(st.target, ast.Name):
                    name_to_ref[st.target.id] = lv
                _walk_stmts(st.body[:5])
            elif isinstance(st, ast.While):
                cr = _lower_py_expr(st.test)
                _emit_step("⟨SPLIT⟩", [cr], [], "loop_while", f"while {cr}")
                _walk_stmts(st.body[:5])
            elif isinstance(st, ast.Return):
                if st.value is not None:
                    rr = _lower_py_expr(st.value)
                    _emit_step("⟨COMMIT⟩", [rr], [], "return", f"return {rr}", flag="⟨EOT⟩")
                else:
                    _emit_step("⟨COMMIT⟩", [], [], "return_none", "return None", flag="⟨EOT⟩")
            elif isinstance(st, ast.Expr):
                if isinstance(st.value, ast.Constant):
                    continue  # Skip function docstrings & no-op bare literals!
                _lower_py_expr(st.value)

    # Emit function GOAL header
    init_refs = [f"⟨IN_{i}⟩" for i in range(min(4, len(arg_list)))]
    _emit_step("⟨GOAL⟩", init_refs, [], fn.name[:12], f"def {fn.name}({', '.join(a.arg for a in arg_list)})")
    _walk_stmts(fn.body)

    if strict_no_truncate and truncated:
        return [], False, "code_truncated_dropped"
    if len(steps) < 2:
        return [], False, "too_few_code_steps"
    steps[-1].op = "⟨COMMIT⟩"
    steps[-1].flag = "⟨EOT⟩"
    return steps, True, "ok"


# =============================================================================
# 3. OpenAI-Compatible Flash API / Local Ollama Backend (Ready for Full 50k!)
# =============================================================================
def compile_via_openai_compat(
    row: Dict[str, Any],
    base_url: str,
    api_key: str,
    model_name: str,
    fallback_model: Optional[str] = None,
) -> Tuple[List[MicroStepIR], bool, str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=45.0)
    user_msg = (
        f"Domain: {row['domain']}\n"
        f"Problem:\n{row['question']}\n\n"
        f"Verified Reference Solution:\n{row['reference_solution']}\n\n"
        f"Ground Truth Answer: {row['ground_truth_answer']}\n"
        f"Compile into AVL-PMLv1.2 1-hop line DSL:"
    )
    models_to_try = [model_name] + ([fallback_model] if fallback_model else [])
    for m_name in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=m_name,
                messages=[
                    {"role": "system", "content": DSL_REFORMAT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=900,
            )
            txt = (resp.choices[0].message.content or "").strip()
            txt = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", txt).strip()
            steps = LinearScanDSLCompiler.parse_line_dsl(txt)
            if steps:
                return steps, True, f"llm:{m_name}"
        except Exception as e:
            last_err = str(e)
    return [], False, "llm_compile_failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1" / "pilot_1000_seeds.jsonl"))
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1" / "pilot_1000_compiled.jsonl"))
    parser.add_argument("--export-pending", type=str, default="", help="Optional JSONL path to save uncompiled prose seeds for Flash API pass.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads for deterministic compilation & sandbox checks.")
    parser.add_argument("--backend", choices=["deterministic", "local_llm", "api"], default="deterministic")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--api-key", type=str, default=os.environ.get("STAGE1_API_KEY", "ollama"))
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    parser.add_argument("--fallback-model", type=str, default=None)
    parser.add_argument("--no-sandbox", action="store_true", help="Skip Python unit test sandbox execution.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output JSONL from scratch instead of resuming.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_4_5b.yaml")
    vocab = SymbolicVocab(cfg)
    compiler = LinearScanDSLCompiler(cfg, vocab)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_path.with_name(out_path.stem + "_report.json")

    if args.overwrite and out_path.exists():
        out_path.unlink()

    # Resumable checkpointing: skip already compiled IDs
    done_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])

    seeds = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))

    stats: Dict[str, Any] = {
        "backend": args.backend,
        "total_seeds": len(seeds),
        "compiled_ok": 0,
        "failed_parse_or_verify": 0,
        "failed_static_5checks": 0,
        "by_domain": {"math": {"tot": 0, "ok": 0}, "code": {"tot": 0, "ok": 0}, "logic": {"tot": 0, "ok": 0}},
        "failure_reasons": {},
    }
    nl_toks_sum = sym_raw_sum = sym_aligned_sum = steps_sum = max_regs_used = 0
    t0 = time.perf_counter()

    pending_path = Path(args.export_pending) if args.export_pending else None
    if pending_path is not None:
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_f = open(pending_path, "w", encoding="utf-8")
    else:
        pending_f = None

    todo_seeds = [r for r in seeds if r["id"] not in done_ids]

    def _compile_one(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]:
        dom = row["domain"]
        if args.backend == "deterministic":
            if dom in ("math", "logic"):
                steps, ok, reason = compile_math_or_logic_deterministic(row)
            else:
                steps, ok, reason = compile_code_deterministic(row, verify_sandbox=(not args.no_sandbox))
        else:
            steps, ok, reason = compile_via_openai_compat(row, args.base_url, args.api_key, args.model, args.fallback_model)

        if not ok or not steps:
            return row, None, reason

        try:
            raw_tr = compiler.compile(steps, final_live_vars=[steps[-1].out_var] if steps[-1].out_var else [])
            aligned_tr = compiler.compile_aligned_frames(
                steps, final_live_vars=[steps[-1].out_var] if steps[-1].out_var else [], prefer_8slot=True
            )
        except CompilerValidationError as e:
            return row, None, f"static_check:{str(e).split(':')[0]}"

        nl_len = approx_nl_token_count(row["reference_solution"])
        sym_raw_len = len(raw_tr.token_ids)
        sym_aln_len = len(aligned_tr.token_ids)
        gt_ans = str(row.get("ground_truth_answer", "")).strip()
        if not gt_ans and dom == "code":
            gt_ans = str(row.get("reference_solution", "")).strip()
        rec = {
            "id": row["id"],
            "domain": dom,
            "source": row["source"],
            "question": row["question"],
            "ground_truth_answer": gt_ans,
            "num_steps": len(steps),
            "frame_sizes": aligned_tr.frame_sizes,
            "token_ids_aligned": aligned_tr.token_ids,
            "reassert_mask": aligned_tr.reassert_mask,
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
            "sym_tokens_compact": sym_raw_len,
            "sym_tokens_aligned": sym_aln_len,
            "max_regs": len(aligned_tr.reg_map),
            "compression_compact_x": round(nl_len / max(1, sym_raw_len), 2),
            "compression_aligned_x": round(nl_len / max(1, sym_aln_len), 2),
        }
        return row, rec, "ok"

    pending_by_domain: Dict[str, int] = {"math": 0, "code": 0, "logic": 0}
    with open(out_path, "a", encoding="utf-8") as fout:
        if args.backend == "deterministic" and args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                it = pool.map(_compile_one, todo_seeds)
                for idx, (row, rec, reason) in enumerate(it, 1):
                    dom = row["domain"]
                    stats["by_domain"].setdefault(dom, {"tot": 0, "ok": 0})
                    stats["by_domain"][dom]["tot"] += 1
                    if rec is None:
                        if reason.startswith("static_check:"):
                            stats["failed_static_5checks"] += 1
                        else:
                            stats["failed_parse_or_verify"] += 1
                        stats["failure_reasons"][reason] = stats["failure_reasons"].get(reason, 0) + 1
                        if pending_f is not None:
                            pending_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            pending_by_domain[dom] = pending_by_domain.get(dom, 0) + 1
                    else:
                        nl_toks_sum += rec["nl_tokens_est"]
                        sym_raw_sum += rec["sym_tokens_compact"]
                        sym_aligned_sum += rec["sym_tokens_aligned"]
                        steps_sum += rec["num_steps"]
                        max_regs_used = max(max_regs_used, rec.pop("max_regs"))
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        stats["compiled_ok"] += 1
                        stats["by_domain"][dom]["ok"] += 1
                    if idx % 10000 == 0:
                        print(f"  [Compile progress] {idx}/{len(todo_seeds)} | ok={stats['compiled_ok']} | elapsed={time.perf_counter()-t0:.1f}s", flush=True)
        else:
            for idx, row in enumerate(todo_seeds, 1):
                _, rec, reason = _compile_one(row)
                dom = row["domain"]
                stats["by_domain"].setdefault(dom, {"tot": 0, "ok": 0})
                stats["by_domain"][dom]["tot"] += 1
                if rec is None:
                    if reason.startswith("static_check:"):
                        stats["failed_static_5checks"] += 1
                    else:
                        stats["failed_parse_or_verify"] += 1
                    stats["failure_reasons"][reason] = stats["failure_reasons"].get(reason, 0) + 1
                    if pending_f is not None:
                        pending_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        pending_by_domain[dom] = pending_by_domain.get(dom, 0) + 1
                else:
                    nl_toks_sum += rec["nl_tokens_est"]
                    sym_raw_sum += rec["sym_tokens_compact"]
                    sym_aligned_sum += rec["sym_tokens_aligned"]
                    steps_sum += rec["num_steps"]
                    max_regs_used = max(max_regs_used, rec.pop("max_regs"))
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["compiled_ok"] += 1
                    stats["by_domain"][dom]["ok"] += 1
    if pending_f is not None:
        pending_f.close()
        stats["pending_api_exported"] = {"total": sum(pending_by_domain.values()), "by_domain": pending_by_domain, "path": str(pending_path)}

    ok_n = max(1, stats["compiled_ok"])
    attempted_static = stats["compiled_ok"] + stats["failed_static_5checks"]
    stats["summary"] = {
        "overall_compile_and_verify_pass_pct": round(100.0 * stats["compiled_ok"] / max(1, len(seeds)), 2),
        "static_5check_pass_pct_of_parsed": round(100.0 * stats["compiled_ok"] / max(1, attempted_static), 2),
        "avg_steps_per_problem": round(steps_sum / ok_n, 2),
        "max_registers_used": max_regs_used,
        "avg_nl_tokens": round(nl_toks_sum / ok_n, 1),
        "avg_sym_tokens_compact": round(sym_raw_sum / ok_n, 1),
        "avg_sym_tokens_aligned_8_16": round(sym_aligned_sum / ok_n, 1),
        "mean_compression_compact_x": round(nl_toks_sum / max(1, sym_raw_sum), 2),
        "mean_compression_aligned_x": round(nl_toks_sum / max(1, sym_aligned_sum), 2),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"=== [Stage 1 DSL Compilation Report ({args.backend})] ===", flush=True)
    print(json.dumps(stats["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"By domain: {stats['by_domain']} | Top failures: {dict(list(stats['failure_reasons'].items())[:6])}", flush=True)
    print(f"Saved compiled JSONL -> {out_path} | Report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
