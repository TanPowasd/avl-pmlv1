"""Stage 1 (M2) Three-Source High-Value Error Set Builder (AVL-PMLv1.2).

Constructs the Master Manual §4.2 exact 40% : 40% : 20% repair & anti-false-edit dataset directly on top of
FSQ-annotated 8/16-slot aligned traces (`stage1_30k_fsq_aligned.jsonl` + `full_50k_compiled_det.jsonl`):
  - 40% Source 1 (`real_cascade_drift`): Multi-step cascading arithmetic/state drift chains (an early error at
        frame b* propagates forward through dependent steps until verification detects contradiction), ready to
        blend with Stage 3 real base-model rollout errors under the exact same schema.
  - 40% Source 2 (`ast_ssa_mutation`): 5 orthogonal structured AST/SSA single-frame corruptions
        (`op_flip`, `reg_pointer_swap`, `val_off_by_one`, `premature_eot`, `fsq_concept_corrupt`), every one
        verified strictly distinct from gold and aligned to exact 8/16-slot physical frames.
  - 20% Source 3 (`clean_no_edit_negative`): 100%-verified correct negative controls (`should_repair=False`)
        so Stage 4 L_repair & Post-Thinking Safety Gate learn zero false-positive tampering (`NetFix >= 0`).
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.vocab.dsl_compiler import CompilerValidationError, LinearScanDSLCompiler, MicroStepIR
from avl_pmlv1.vocab.sym_vocab import SymbolicVocab

OP_FLIP_MAP = {
    "+": "-",
    "-": "+",
    "*": "+",
    "/": "*",
    "==": "!=",
    "!=": "==",
    "<": ">=",
    ">=": "<",
    ">": "<=",
    "<=": ">",
    "%": "-",
}


def _ir_dicts_to_objs(steps_ir: List[Dict[str, Any]], fsq_pairs: List[List[int]]) -> List[MicroStepIR]:
    objs: List[MicroStepIR] = []
    for st, fp in zip(steps_ir, fsq_pairs):
        objs.append(
            MicroStepIR(
                op=st["op"],
                in_refs=list(st.get("in_refs", [])),
                var_deps=list(st.get("var_deps", [])),
                operands=list(st.get("operands", [])),
                fsq_pairs=[(int(fp[0]), int(fp[1]))],
                lit_tokens=list(st.get("lit_tokens", [])),
                val_token=st.get("val_token"),
                out_var=st.get("out_var"),
                flag=st.get("flag"),
                expr_text=st.get("expr_text", ""),
            )
        )
    return objs


def _perturb_val_token(val: Optional[str], rng: random.Random) -> str:
    if val is None or not val:
        return "0"
    try:
        fv = float(val)
        if not math.isfinite(fv) or abs(fv) > 1e14:
            raise ValueError("non_finite")
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        nv = fv + delta
        if abs(nv - round(nv)) < 1e-6:
            return str(int(round(nv)))
        return f"{nv:.2f}"
    except ValueError:
        alt_pool = ["None", "False", "True", "0", "1", "err_sym"]
        choices = [x for x in alt_pool if x != val]
        return rng.choice(choices)


def _frame_offsets(frame_sizes: List[int]) -> List[int]:
    offs = [0]
    for fs in frame_sizes:
        offs.append(offs[-1] + fs)
    return offs


def build_cascade_drift(
    ir_objs: List[MicroStepIR],
    compiler: LinearScanDSLCompiler,
    rng: random.Random,
) -> Optional[Tuple[List[int], List[int], int, int, str]]:
    """Source 1 (40%): Early/mid-step subtle drift that cascades through all downstream dependent frames!"""
    n = len(ir_objs)
    if n < 3:
        return None
    corrupted = copy.deepcopy(ir_objs)
    b_star = rng.randint(0, max(0, n - 2))
    # Corrupt b_star value and propagate drift forward to all downstream steps referencing out_var transitively!
    tainted_vars = set()
    corrupted[b_star].val_token = _perturb_val_token(corrupted[b_star].val_token, rng)
    if corrupted[b_star].out_var:
        tainted_vars.add(corrupted[b_star].out_var)
    cascade_hops = 0
    for j in range(b_star + 1, n):
        if any(dep in tainted_vars for dep in corrupted[j].var_deps) or (j == n - 1):
            corrupted[j].val_token = _perturb_val_token(corrupted[j].val_token, rng)
            if corrupted[j].out_var:
                tainted_vars.add(corrupted[j].out_var)
            cascade_hops += 1
    # Sample realistic mid-chain or verification detection point (avoiding 100% final-frame ⟨CONTRA⟩ positional shortcut!)
    consumer_idxs = [j for j in range(b_star + 1, n) if any(dep in tainted_vars for dep in corrupted[j].var_deps)]
    detect_idx = rng.choice(consumer_idxs) if consumer_idxs else rng.randint(b_star + 1, n - 1)
    mode_roll = rng.random()
    if mode_roll < 0.45 and detect_idx < n - 1:
        # Mid-chain verification circuit-breaker halt at detect_idx (not final frame!)
        corrupted = corrupted[: detect_idx + 1]
        corrupted[detect_idx].op = "⟨VERIFY⟩"
        corrupted[detect_idx].flag = "⟨CONTRA⟩"
    elif mode_roll < 0.70:
        # Unflagged silent cascade drift reaching COMMIT -> requires genuine value/PLL inconsistency audit!
        pass
    else:
        corrupted[detect_idx].flag = "⟨CONTRA⟩"
    try:
        final_live = [corrupted[-1].out_var] if corrupted[-1].out_var else []
        tr = compiler.compile_aligned_frames(corrupted, final_live_vars=final_live, prefer_8slot=True)
        return tr.token_ids, tr.frame_sizes, b_star, detect_idx, f"cascade_drift_hops_{cascade_hops}"
    except CompilerValidationError:
        return None


def build_ast_ssa_mutation(
    ir_objs: List[MicroStepIR],
    compiler: LinearScanDSLCompiler,
    cfg: AVLPMLConfig,
    rng: random.Random,
    mut_kind: str,
) -> Optional[Tuple[List[int], List[int], int, int, str]]:
    """Source 2 (40%): 5 orthogonal structured AST/SSA single-frame corruptions."""
    n = len(ir_objs)
    if n < 2:
        return None
    corrupted = copy.deepcopy(ir_objs)
    b_star = rng.randint(0, n - 1)

    if mut_kind == "op_flip":
        cand_idxs = [i for i, s in enumerate(corrupted) if any(t in OP_FLIP_MAP for t in s.lit_tokens)]
        if not cand_idxs:
            mut_kind = "val_off_by_one"
        else:
            b_star = rng.choice(cand_idxs)
            corrupted[b_star].lit_tokens = [OP_FLIP_MAP.get(t, t) for t in corrupted[b_star].lit_tokens]
            corrupted[b_star].val_token = _perturb_val_token(corrupted[b_star].val_token, rng)

    if mut_kind == "reg_pointer_swap":
        # Pick a step that reads a variable or input when >=2 earlier variables/inputs exist
        defined_before: List[str] = []
        seen_ins: List[str] = []
        cand_swaps: List[Tuple[int, str, str]] = []
        for i, s in enumerate(corrupted):
            for inp_r in s.in_refs:
                if inp_r not in seen_ins:
                    seen_ins.append(inp_r)
            if s.var_deps and len(defined_before) >= 2:
                for dep in s.var_deps:
                    alts = [v for v in defined_before if v != dep]
                    if alts:
                        cand_swaps.append((i, dep, rng.choice(alts)))
            elif s.in_refs and len(seen_ins) >= 2:
                for inp_r in s.in_refs:
                    alts_in = [x for x in seen_ins if x != inp_r]
                    if alts_in:
                        cand_swaps.append((i, inp_r, rng.choice(alts_in)))
            if s.out_var and s.out_var not in defined_before:
                defined_before.append(s.out_var)
        if cand_swaps:
            b_star, old_v, new_v = rng.choice(cand_swaps)
            if corrupted[b_star].operands:
                corrupted[b_star].operands = [new_v if d == old_v else d for d in corrupted[b_star].operands]
                corrupted[b_star].sync_from_operands()
            else:
                corrupted[b_star].var_deps = [new_v if d == old_v else d for d in corrupted[b_star].var_deps]
                corrupted[b_star].in_refs = [new_v if d == old_v else d for d in corrupted[b_star].in_refs]
            corrupted[b_star].val_token = _perturb_val_token(corrupted[b_star].val_token, rng)
        else:
            mut_kind = "val_off_by_one"

    if mut_kind == "val_off_by_one":
        b_star = rng.randint(0, n - 1)
        corrupted[b_star].val_token = _perturb_val_token(corrupted[b_star].val_token, rng)

    elif mut_kind == "premature_eot":
        if n < 3:
            b_star = 0
            corrupted[b_star].val_token = _perturb_val_token(corrupted[b_star].val_token, rng)
            mut_kind = "val_off_by_one"
        else:
            b_star = rng.randint(0, n - 2)
            corrupted = corrupted[: b_star + 1]
            corrupted[-1].op = "⟨COMMIT⟩"
            corrupted[-1].flag = "⟨EOT⟩"

    elif mut_kind == "fsq_concept_corrupt":
        b_star = rng.randint(0, n - 1)
        if corrupted[b_star].fsq_pairs:
            c0, f0 = corrupted[b_star].fsq_pairs[0]
            c_bad = (c0 + rng.randint(100, cfg.single_fsq_size - 100)) % cfg.single_fsq_size
            f_bad = (f0 + rng.randint(100, cfg.single_fsq_size - 100)) % cfg.single_fsq_size
            corrupted[b_star].fsq_pairs = [(c_bad, f_bad)]
        else:
            corrupted[b_star].val_token = _perturb_val_token(corrupted[b_star].val_token, rng)

    detect_idx = min(len(corrupted) - 1, b_star + (0 if mut_kind in ("premature_eot", "val_off_by_one") else 1))
    try:
        final_live = [corrupted[-1].out_var] if corrupted[-1].out_var else []
        tr = compiler.compile_aligned_frames(corrupted, final_live_vars=final_live, prefer_8slot=True)
        return tr.token_ids, tr.frame_sizes, b_star, detect_idx, mut_kind
    except CompilerValidationError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsq-jsonl", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1" / "stage1_30k_fsq_aligned.jsonl"))
    parser.add_argument("--det-jsonl", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1" / "full_50k_compiled_det.jsonl"))
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1" / "stage1_three_source_error_set.jsonl"))
    parser.add_argument("--report", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "stage1_three_source_error_report.json"))
    parser.add_argument("--total", type=int, default=15000, help="Total error-set examples (40% cascade : 40% AST : 20% clean).")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_4_5b.yaml")
    vocab = SymbolicVocab(cfg)
    compiler = LinearScanDSLCompiler(cfg, vocab)

    # Load steps_ir map from det_jsonl
    ir_by_id: Dict[str, List[Dict[str, Any]]] = {}
    with open(args.det_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "steps_ir" in r:
                    ir_by_id[r["id"]] = r["steps_ir"]

    # Load FSQ-annotated gold records
    gold_recs: List[Dict[str, Any]] = []
    with open(args.fsq_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["id"] in ir_by_id and len(r.get("fsq_pairs", [])) == len(ir_by_id[r["id"]]):
                    gold_recs.append(r)

    rng = random.Random(20261124)
    rng.shuffle(gold_recs)

    target_s1 = int(round(args.total * 0.40))
    target_s2 = int(round(args.total * 0.40))
    target_s3 = args.total - target_s1 - target_s2

    mut_cycle = ["op_flip", "reg_pointer_swap", "val_off_by_one", "premature_eot", "fsq_concept_corrupt"]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts_src = collections.Counter()
    counts_sub = collections.Counter()
    counts_dom = collections.Counter()
    c_s1 = c_s2 = c_s3 = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        idx = 0
        # 1. Source 1 (40%): Multi-step cascading drift errors
        while c_s1 < target_s1 and idx < len(gold_recs):
            rec = gold_recs[idx]
            idx += 1
            ir_objs = _ir_dicts_to_objs(ir_by_id[rec["id"]], rec["fsq_pairs"])
            res = build_cascade_drift(ir_objs, compiler, rng)
            if res is None:
                continue
            bad_toks, bad_fsizes, b_star, det_idx, subtype = res
            gold_toks = rec["token_ids_aligned_fsq"]
            if bad_toks == gold_toks:
                continue
            gold_offs = _frame_offsets(rec["frame_sizes"])
            repair_suffix = gold_toks[gold_offs[b_star] :]
            entry = {
                "error_id": f"err_s1_{c_s1}",
                "problem_id": rec["id"],
                "domain": rec["domain"],
                "source_bucket": "real_cascade_drift",
                "error_subtype": subtype,
                "should_repair": True,
                "bad_frame_idx": b_star,
                "detect_frame_idx": det_idx,
                "rollback_to_frame": max(-1, b_star - 1),
                "corrupted_frame_sizes": bad_fsizes,
                "corrupted_token_ids": bad_toks,
                "gold_frame_sizes": rec["frame_sizes"],
                "gold_repair_suffix_tokens": repair_suffix,
                "expected_action": "ROLLBACK_AND_REPAIR",
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            c_s1 += 1
            counts_src["real_cascade_drift"] += 1
            counts_sub["cascade_drift"] += 1
            counts_dom[rec["domain"]] += 1

        # 2. Source 2 (40%): Structured AST/SSA Single-Frame Mutations
        while c_s2 < target_s2 and idx < len(gold_recs):
            rec = gold_recs[idx]
            idx += 1
            ir_objs = _ir_dicts_to_objs(ir_by_id[rec["id"]], rec["fsq_pairs"])
            m_kind = mut_cycle[c_s2 % len(mut_cycle)]
            res = build_ast_ssa_mutation(ir_objs, compiler, cfg, rng, m_kind)
            if res is None:
                continue
            bad_toks, bad_fsizes, b_star, det_idx, actual_sub = res
            gold_toks = rec["token_ids_aligned_fsq"]
            if bad_toks == gold_toks:
                continue
            gold_offs = _frame_offsets(rec["frame_sizes"])
            repair_suffix = gold_toks[gold_offs[b_star] :]
            is_wet_seam = actual_sub in ("val_off_by_one", "fsq_concept_corrupt") and (b_star >= len(rec["frame_sizes"]) - 2)
            entry = {
                "error_id": f"err_s2_{c_s2}",
                "problem_id": rec["id"],
                "domain": rec["domain"],
                "source_bucket": "ast_ssa_mutation",
                "error_subtype": actual_sub,
                "should_repair": True,
                "bad_frame_idx": b_star,
                "detect_frame_idx": det_idx,
                "rollback_to_frame": max(-1, b_star - 1),
                "corrupted_frame_sizes": bad_fsizes,
                "corrupted_token_ids": bad_toks,
                "gold_frame_sizes": rec["frame_sizes"],
                "gold_repair_suffix_tokens": repair_suffix,
                "expected_action": "WET_CEMENT_LOCAL_EDIT" if is_wet_seam else "ROLLBACK_AND_REPAIR",
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            c_s2 += 1
            counts_src["ast_ssa_mutation"] += 1
            counts_sub[actual_sub] += 1
            counts_dom[rec["domain"]] += 1

        # 3. Source 3 (20%): Clean Negative Controls (NetFix >= 0 Anti-False-Edit Guard!)
        while c_s3 < target_s3 and idx < len(gold_recs):
            rec = gold_recs[idx]
            idx += 1
            entry = {
                "error_id": f"err_s3_{c_s3}",
                "problem_id": rec["id"],
                "domain": rec["domain"],
                "source_bucket": "clean_no_edit_negative",
                "error_subtype": "verified_clean_gold",
                "should_repair": False,
                "bad_frame_idx": -1,
                "detect_frame_idx": -1,
                "rollback_to_frame": -1,
                "corrupted_frame_sizes": rec["frame_sizes"],
                "corrupted_token_ids": rec["token_ids_aligned_fsq"],
                "gold_frame_sizes": rec["frame_sizes"],
                "gold_repair_suffix_tokens": [],
                "expected_action": "KEEP_ORIGINAL",
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            c_s3 += 1
            counts_src["clean_no_edit_negative"] += 1
            counts_sub["verified_clean_gold"] += 1
            counts_dom[rec["domain"]] += 1

    total_emitted = c_s1 + c_s2 + c_s3
    report = {
        "total_emitted": total_emitted,
        "target_total": args.total,
        "problem_disjoint_guaranteed": True,
        "ratio_actual_pct": {k: round(100.0 * v / max(1, total_emitted), 2) for k, v in counts_src.items()},
        "counts_by_source": dict(counts_src),
        "counts_by_subtype": dict(counts_sub),
        "counts_by_domain": dict(counts_dom),
        "output_path": str(out_path),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== [Stage 1 Three-Source High-Value Error Set Report] ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
