"""Deep Random Spot-Check & Quantitative Quality / Flaw Audit across ALL 7 Stage 1 Mega Sources."""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.vocab.sym_vocab import SymbolicVocab


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_4_5b.yaml")
    vocab = SymbolicVocab(cfg)
    id2tok = {v: k for k, v in vocab.token_to_id.items()}

    def decode_tok(tid: int) -> str:
        if tid in id2tok:
            return id2tok[tid]
        if vocab.fsq_coarse_start <= tid < vocab.fsq_fine_start:
            return f"C{tid - vocab.fsq_coarse_start}"
        if vocab.fsq_fine_start <= tid < vocab.prim_start:
            return f"F{tid - vocab.fsq_fine_start}"
        if vocab.escape_start <= tid < vocab.ro_start:
            return f"ESC_{tid - vocab.escape_start}"
        return f"T{tid}"

    # 1. Seed-level authenticity & noise check
    seed_q = collections.Counter()
    with open(root / "data" / "stage1_mega" / "mega_seeds.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r["source"].split(":")[0]
            seed_q[f"{src}::tot"] += 1
            if not str(r.get("ground_truth_answer", "")).strip():
                seed_q[f"{src}::no_gt"] += 1
            sol = str(r.get("reference_solution", ""))
            if any(err in sol for err in ("Traceback", "SyntaxError", "NameError", "ZeroDivisionError", "TypeError")):
                seed_q[f"{src}::has_python_error_in_sol"] += 1
            if src == "AI-MO/NuminaMath-TIR" and "```output" not in sol:
                seed_q[f"{src}::no_output_block"] += 1

    print("=== 1. SEED-LEVEL AUTHENTICITY & NOISE AUDIT ===")
    for k in sorted(seed_q):
        print(f"  {k:52s}: {seed_q[k]:6d}")

    # 2. Compiled-level utility & flaw check + Reservoir sampling 2 random rows per source
    comp_q = collections.Counter()
    val_kind = collections.Counter()
    samples_by_src = collections.defaultdict(list)
    rng = random.Random(20261124)

    with open(root / "data" / "stage1_mega" / "mega_gold_v2_compiled.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r["source"].split(":")[0]
            comp_q[f"{src}::tot"] += 1
            if r["domain"] in ("math", "logic") and not str(r.get("ground_truth_answer", "")).strip():
                comp_q[f"{src}::missing_gt"] += 1
            if r["num_steps"] > 32:
                comp_q[f"{src}::over_32_steps"] += 1
            irs = r["steps_ir"]
            if any(s.get("flag") == "⟨CONTRA⟩" for s in irs):
                comp_q[f"{src}::has_contra_step"] += 1
            for s in irs:
                vt = str(s.get("val_token") or "")
                if vt.startswith(("bin_", "cmp_", "aug_", "branch_", "loop_")) or vt in ("index", "return", "list", "dict", "tuple", "set"):
                    val_kind[f"{src}::abstract_op_val"] += 1
                else:
                    val_kind[f"{src}::concrete_or_sym_val"] += 1
            if src == "tasksource/proofwriter" and r["num_steps"] >= 15:
                comp_q[f"{src}::many_irrelevant_facts_15+"] += 1

            bucket = samples_by_src[src]
            if len(bucket) < 2:
                bucket.append(r)
            else:
                j = rng.randint(0, comp_q[f"{src}::tot"] - 1)
                if j < 2:
                    bucket[j] = r

    print("\n=== 2. GOLD v2 COMPILED-LEVEL QUALITY & ZERO-TRUNCATION AUDIT ===")
    for k in sorted(comp_q):
        print(f"  {k:52s}: {comp_q[k]:6d}")

    print("\n=== 3. GOLD v2 VAL_TOKEN CONCRETENESS AUDIT ===")
    for k in sorted(val_kind):
        print(f"  {k:52s}: {val_kind[k]:8d}")

    sample_ids = {r["id"] for lst in samples_by_src.values() for r in lst}
    fsq_by_id = {}
    esc_count = prim_count = 0
    with open(root / "data" / "stage1_mega" / "mega_gold_v2_fsq_aligned.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = r["token_ids_aligned_fsq"]
            for t in toks:
                if vocab.prim_start <= t < vocab.prim_end:
                    prim_count += 1
                elif vocab.escape_start <= t < vocab.escape_end:
                    esc_count += 1
            if r["id"] in sample_ids:
                fsq_by_id[r["id"]] = r
    print(f"\n=== 3b. V_LIT LOSSLESS PRIMITIVE vs ESCAPE HASH RATIO ===")
    print(f"  Lossless Primitives: {prim_count:,} | Escape Hash: {esc_count:,} | Lossless Share: {100.0*prim_count/max(1,prim_count+esc_count):.2f}%")

    print("\n=== 4. RANDOM REAL SPOT-CHECK SAMPLES (EVERY SOURCE) ===")
    for src, items in sorted(samples_by_src.items()):
        r = items[0]
        print(f"\n--- [{src}] id={r['id']} | steps={r['num_steps']} | gt={str(r['ground_truth_answer'])[:50]} ---")
        print("  Q:", r["question"][:280].replace("\n", " "))
        for idx, st in enumerate(r["steps_ir"][:5]):
            print(
                f"    step {idx:02d}: {st['op']:9s} in={st['in_refs']} deps={st['var_deps']} "
                f"lit={st['lit_tokens']} val={st['val_token']} out={st['out_var']} flag={st['flag']} | {st['expr_text'][:55]}"
            )
        if len(r["steps_ir"]) > 5:
            st = r["steps_ir"][-1]
            print(
                f"    ...LAST ({len(r['steps_ir'])-1:02d}): {st['op']:9s} in={st['in_refs']} deps={st['var_deps']} "
                f"lit={st['lit_tokens']} val={st['val_token']} out={st['out_var']} flag={st['flag']} | {st['expr_text'][:55]}"
            )
        fr = fsq_by_id.get(r["id"])
        if fr:
            toks = fr["token_ids_aligned_fsq"]
            f0 = fr["frame_sizes"][0]
            fl = fr["frame_sizes"][-1]
            print("  Decoded Frame 0:", [decode_tok(t) for t in toks[:f0]])
            print("  Decoded Last Fr:", [decode_tok(t) for t in toks[-fl:]])


if __name__ == "__main__":
    main()
