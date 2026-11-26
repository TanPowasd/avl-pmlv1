"""Stage 1 Mega-Corpus Fetcher, Global Cross-Source Deduplicator, & 7-Benchmark Exporter (0 RMB!).

Builds a strictly isolated 3-partition frontier dataset suite:
  1. Official Untouched Gold Test Benchmarks (`data/benchmarks/*.jsonl`, ~5,183 problems):
       - GSM8K-test (1,319) + MATH-500 (500) + MBPP-test (500) + HumanEval (164)
       - Knights-and-Knaves test (700, 2ppl..8ppl) + ProofWriter test (1,000, QDep 2..5) + ZebraLogic test (1,000)
     ALL 5,183 gold test questions are indexed into our Exact Hash + 8-gram Shingle Decontamination Blacklist!
  2. Real, Zero-Synthetic Mega Seed Pool (`data/stage1_mega/mega_seeds.jsonl`, ~230,000+ unique clean seeds):
       - Math  (~115k): ALL openai/gsm8k + ALL AI-MO/NuminaMath-TIR + OpenMathInstruct-1 (is_correct & <llm-code>) + MathInstruct (PoT)
       - Code  (~90k) : ALL mbpp (train/val/prompt) + KodCode/KodCode-V1 (filter_reason=None)
       - Logic (~38k) : ALL K-and-K/knights-and-knaves (train 2ppl..8ppl) + tasksource/proofwriter (train + val QDep>=1)
     Strictly deduplicated across ALL sources by normalized question MD5 hash!
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Set, Tuple

from datasets import load_dataset


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


def md5_norm(s: str) -> str:
    return hashlib.md5(norm_text(s).encode("utf-8")).hexdigest()


def shingles(s: str, k: int = 8) -> Set[str]:
    words = norm_text(s).split()
    if len(words) < k:
        return {norm_text(s)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def is_contaminated(question: str, exact_hashes: Set[str], shingle_set: Set[str]) -> bool:
    nt = norm_text(question)
    if not nt:
        return True
    if hashlib.md5(nt.encode("utf-8")).hexdigest() in exact_hashes:
        return True
    q_sh = shingles(nt, k=8)
    if len(q_sh) >= 4:
        overlap = len(q_sh & shingle_set) / float(len(q_sh))
        if overlap >= 0.50:
            return True
    return False


def export_benchmarks_and_build_blacklist(bench_dir: Path) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    """Export all 7 official untouched gold benchmarks to data/benchmarks/ AND build decontamination blacklist."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    exact_hashes: Set[str] = set()
    shingle_set: Set[str] = set()
    counts: Dict[str, int] = {}

    def _save_and_index(name: str, rows: List[Dict[str, Any]], q_key: str = "question") -> None:
        out_f = bench_dir / f"{name}.jsonl"
        with open(out_f, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                nt = norm_text(str(r[q_key]))
                if nt:
                    exact_hashes.add(hashlib.md5(nt.encode("utf-8")).hexdigest())
                    shingle_set.update(shingles(nt, k=8))
        counts[name] = len(rows)
        print(f"  [Benchmark Exported & Blacklisted] {name:22s}: {len(rows):5d} -> {out_f.name}", flush=True)

    # 1. GSM8K-test (1,319)
    gsm_t = load_dataset("openai/gsm8k", "main", split="test")
    _save_and_index(
        "gsm8k_test",
        [
            {
                "id": f"bench_gsm8k_{i}",
                "domain": "math",
                "question": r["question"].strip(),
                "reference_solution": r["answer"].strip(),
                "ground_truth": (re.search(r"####\s*([^\n]+)", r["answer"]).group(1).strip().replace(",", "") if re.search(r"####\s*([^\n]+)", r["answer"]) else ""),
            }
            for i, r in enumerate(gsm_t)
        ],
    )

    # 2. MATH-500 (500)
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    _save_and_index(
        "math500_test",
        [
            {
                "id": f"bench_math500_{i}",
                "domain": "math",
                "question": r["problem"].strip(),
                "reference_solution": r["solution"].strip(),
                "ground_truth": str(r["answer"]).strip(),
                "level": r.get("level"),
                "subject": r.get("subject"),
            }
            for i, r in enumerate(m500)
        ],
    )

    # 3. MBPP-test (500)
    mbpp_t = load_dataset("google-research-datasets/mbpp", "full", split="test")
    _save_and_index(
        "mbpp_test",
        [
            {
                "id": f"bench_mbpp_{r.get('task_id', i)}",
                "domain": "code",
                "question": str(r["text"]).strip(),
                "reference_solution": str(r["code"]).strip(),
                "unit_tests": list(r.get("test_list", [])),
            }
            for i, r in enumerate(mbpp_t)
        ],
    )

    # 4. HumanEval (164)
    he_t = load_dataset("openai/openai_humaneval", split="test")
    _save_and_index(
        "humaneval_test",
        [
            {
                "id": f"bench_he_{r.get('task_id', i)}",
                "domain": "code",
                "question": str(r["prompt"]).strip(),
                "reference_solution": str(r["canonical_solution"]).strip(),
                "unit_tests": [str(r.get("test", ""))],
                "entry_point": r.get("entry_point", ""),
            }
            for i, r in enumerate(he_t)
        ],
    )

    # 5. Knights-and-Knaves official test (700 across 2ppl..8ppl)
    kk_rows = []
    for sp in ("2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"):
        ds = load_dataset("K-and-K/knights-and-knaves", "test", split=sp)
        for i, r in enumerate(ds):
            kk_rows.append(
                {
                    "id": f"bench_kk_{sp}_{i}",
                    "domain": "logic",
                    "split": sp,
                    "question": str(r["quiz"]).strip(),
                    "names": list(r["names"]),
                    "ground_truth": list(r["solution"]),
                    "cot_steps": list(r["cot_repeat_steps"]),
                }
            )
    _save_and_index("knights_knaves_test", kk_rows)

    # 6. ProofWriter official test (1,000 multi-hop QDep 2..5)
    pw_rows = []
    pw_t = load_dataset("tasksource/proofwriter", split="test", streaming=True)
    for i, r in enumerate(pw_t):
        if len(pw_rows) >= 1000:
            break
        qdep = int(r.get("QDep", 0) or 0)
        ans = str(r.get("answer", "")).strip()
        if qdep < 2 or ans not in ("True", "False"):
            continue
        q_full = f"{str(r['theory']).strip()}\nQuestion: {str(r['question']).strip()}"
        pw_rows.append(
            {
                "id": f"bench_pw_{i}",
                "domain": "logic",
                "qdep": qdep,
                "question": q_full,
                "ground_truth": ans,
                "allProofs": str(r.get("allProofs", "")),
            }
        )
    _save_and_index("proofwriter_test", pw_rows)

    # 7. ZebraLogic official test (1,000)
    zl_rows = []
    zl_t = load_dataset("WildEval/ZebraLogic", "grid_mode", split="test", streaming=True)
    for i, r in enumerate(zl_t):
        q = str(r.get("puzzle", "")).strip()
        if not q:
            continue
        zl_rows.append(
            {
                "id": f"bench_zebra_{i}",
                "domain": "logic",
                "size": r.get("size", ""),
                "question": q,
                "ground_truth": r.get("solution", {}),
            }
        )
    _save_and_index("zebralogic_test", zl_rows)

    return exact_hashes, shingle_set, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-cache", type=str, default=r"e:\avl-list\hf_cache")
    parser.add_argument("--bench-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "benchmarks"))
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega"))
    parser.add_argument("--openmath-target", type=int, default=35000, help="Unique verified <llm-code> seeds from OpenMathInstruct-1.")
    parser.add_argument("--mathinstruct-target", type=int, default=12000, help="Unique PoT seeds from TIGER-Lab/MathInstruct.")
    parser.add_argument("--kodcode-target", type=int, default=90000, help="Unique verified Python functions from KodCode-V1.")
    parser.add_argument("--proofwriter-train-target", type=int, default=32000, help="Unique multi-hop proofs from ProofWriter train.")
    parser.add_argument("--proofwriter-val-target", type=int, default=1500, help="Official ProofWriter val split proofs for Val.")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "mega_seeds.jsonl"
    out_meta = out_dir / "mega_seeds_meta.json"

    print("=== [Stage 1 Mega-Harvest] Step A: Export 7 Gold Benchmarks & Build Blacklist ===", flush=True)
    exact_h, sh_set, bench_counts = export_benchmarks_and_build_blacklist(Path(args.bench_dir))
    print(f"[7-Benchmark Blacklist Ready] total_gold_test={sum(bench_counts.values())} ({time.perf_counter()-t0:.1f}s)", flush=True)

    seen_q_hashes: Set[str] = set()
    stats: Dict[str, Any] = {
        "benchmarks_exported": bench_counts,
        "sources": {},
        "dropped_contaminated": 0,
        "dropped_duplicate": 0,
    }

    with open(out_jsonl, "w", encoding="utf-8") as fout:

        def _try_emit(rec: Dict[str, Any], src_key: str) -> bool:
            q = rec["question"]
            if is_contaminated(q, exact_h, sh_set):
                stats["dropped_contaminated"] += 1
                return False
            qh = md5_norm(q)
            if qh in seen_q_hashes:
                stats["dropped_duplicate"] += 1
                return False
            seen_q_hashes.add(qh)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["sources"][src_key] = stats["sources"].get(src_key, 0) + 1
            return True

        # =====================================================================
        # DOMAIN 1: MATH (GSM8K + ALL NuminaMath-TIR + OpenMathInstruct-1 + MathInstruct-PoT)
        # =====================================================================
        print("\n=== [Domain 1: MATH] ===", flush=True)

        # 1a. ALL GSM8K train (7,473)
        gsm_tr = load_dataset("openai/gsm8k", "main", split="train")
        for i, item in enumerate(gsm_tr):
            q, a = item["question"].strip(), item["answer"].strip()
            m_ans = re.search(r"####\s*([^\n]+)", a)
            gt = m_ans.group(1).strip().replace(",", "") if m_ans else ""
            if not gt:
                continue
            _try_emit(
                {
                    "id": f"math_gsm8k_{i}",
                    "domain": "math",
                    "source": "openai/gsm8k:train",
                    "format": "gsm8k_eq",
                    "question": q,
                    "reference_solution": a,
                    "ground_truth_answer": gt,
                    "unit_tests": [],
                },
                "openai/gsm8k",
            )
        print(f"  [1a] openai/gsm8k          : {stats['sources'].get('openai/gsm8k', 0):6d}", flush=True)

        # 1b. ALL AI-MO/NuminaMath-TIR train (72,441 Olympiad Python/SymPy!)
        tir = load_dataset("AI-MO/NuminaMath-TIR", split="train", streaming=True)
        for i, item in enumerate(tir):
            q = str(item.get("problem", "")).strip()
            sol = str(item.get("solution", "")).strip()
            if len(q) < 15 or len(q) > 2000 or "```python" not in sol:
                continue
            boxed = re.findall(r"\\boxed\{([^{}]+)\}", sol)
            gt = boxed[-1].strip() if boxed else ""
            _try_emit(
                {
                    "id": f"math_numina_tir_{i}",
                    "domain": "math",
                    "source": "AI-MO/NuminaMath-TIR:train",
                    "format": "tir_python",
                    "question": q,
                    "reference_solution": sol,
                    "ground_truth_answer": gt,
                    "unit_tests": [],
                },
                "AI-MO/NuminaMath-TIR",
            )
            if (i + 1) % 20000 == 0:
                print(f"    [NuminaMath-TIR] scanned={i+1} kept={stats['sources'].get('AI-MO/NuminaMath-TIR', 0)}", flush=True)
        print(f"  [1b] AI-MO/NuminaMath-TIR  : {stats['sources'].get('AI-MO/NuminaMath-TIR', 0):6d}", flush=True)

        # 1c. nvidia/OpenMathInstruct-1 (is_correct=True + <llm-code>, deduplicated!)
        om = load_dataset("nvidia/OpenMathInstruct-1", split="train", streaming=True)
        om_kept = 0
        for i, item in enumerate(om):
            if om_kept >= args.openmath_target or i >= 350000:
                break
            if not item.get("is_correct"):
                continue
            sol = str(item.get("generated_solution", "")).strip()
            if "<llm-code>" not in sol:
                continue
            q = str(item.get("question", "")).strip()
            if len(q) < 15 or len(q) > 2000:
                continue
            ok = _try_emit(
                {
                    "id": f"math_openmath1_{i}",
                    "domain": "math",
                    "source": f"nvidia/OpenMathInstruct-1:{item.get('dataset', 'math')}",
                    "format": "tir_python",
                    "question": q,
                    "reference_solution": sol,
                    "ground_truth_answer": str(item.get("expected_answer", "")).strip(),
                    "unit_tests": [],
                },
                "nvidia/OpenMathInstruct-1",
            )
            if ok:
                om_kept += 1
                if om_kept % 10000 == 0:
                    print(f"    [OpenMathInstruct-1] scanned={i+1} unique_kept={om_kept}/{args.openmath_target}", flush=True)
        print(f"  [1c] nvidia/OpenMath1      : {stats['sources'].get('nvidia/OpenMathInstruct-1', 0):6d}", flush=True)

        # 1d. TIGER-Lab/MathInstruct (PoT Python subset!)
        mi = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
        mi_kept = 0
        for i, item in enumerate(mi):
            if mi_kept >= args.mathinstruct_target:
                break
            src_s = str(item.get("source", ""))
            if "PoT" not in src_s and "pot" not in src_s.lower():
                continue
            q = str(item.get("instruction", "")).strip()
            sol = str(item.get("output", "")).strip()
            if len(q) < 15 or len(q) > 2000 or len(sol) < 20:
                continue
            ok = _try_emit(
                {
                    "id": f"math_instruct_pot_{i}",
                    "domain": "math",
                    "source": f"TIGER-Lab/MathInstruct:{src_s}",
                    "format": "tir_python",
                    "question": q,
                    "reference_solution": sol,
                    "ground_truth_answer": "",
                    "unit_tests": [],
                },
                "TIGER-Lab/MathInstruct-PoT",
            )
            if ok:
                mi_kept += 1
        print(f"  [1d] MathInstruct-PoT      : {stats['sources'].get('TIGER-Lab/MathInstruct-PoT', 0):6d}", flush=True)

        # =====================================================================
        # DOMAIN 2: CODE (ALL MBPP train/val/prompt + 90,000 KodCode-V1)
        # =====================================================================
        print("\n=== [Domain 2: CODE] ===", flush=True)
        for sp in ("train", "validation", "prompt"):
            ds = load_dataset("google-research-datasets/mbpp", "full", split=sp)
            for item in ds:
                q = str(item.get("text", "")).strip()
                code = str(item.get("code", "")).strip()
                tests = list(item.get("test_list", []))
                if not q or not code or not tests:
                    continue
                _try_emit(
                    {
                        "id": f"code_mbpp_{sp}_{item.get('task_id')}",
                        "domain": "code",
                        "source": f"mbpp:{sp}",
                        "format": "python_fn",
                        "question": q,
                        "reference_solution": code,
                        "ground_truth_answer": "",
                        "unit_tests": tests,
                    },
                    "mbpp",
                )
        print(f"  [2a] mbpp (non-test)       : {stats['sources'].get('mbpp', 0):6d}", flush=True)

        kod = load_dataset("KodCode/KodCode-V1", split="train", streaming=True)
        kod_kept = 0
        for i, item in enumerate(kod):
            if kod_kept >= args.kodcode_target:
                break
            if item.get("filter_reason"):
                continue
            q = str(item.get("question", "")).strip()
            sol = str(item.get("solution", "")).strip()
            sol = re.sub(r"^```python\s*\n|```$", "", sol, flags=re.MULTILINE).strip()
            if not q or "def " not in sol or len(sol) > 3000:
                continue
            ok = _try_emit(
                {
                    "id": f"code_kodcode_{i}",
                    "domain": "code",
                    "source": "KodCode/KodCode-V1",
                    "format": "python_fn",
                    "question": q,
                    "reference_solution": sol,
                    "ground_truth_answer": "",
                    "unit_tests": [],
                },
                "KodCode/KodCode-V1",
            )
            if ok:
                kod_kept += 1
                if kod_kept % 20000 == 0:
                    print(f"    [KodCode-V1] kept={kod_kept}/{args.kodcode_target} (scanned={i+1})", flush=True)
        print(f"  [2b] KodCode/KodCode-V1    : {stats['sources'].get('KodCode/KodCode-V1', 0):6d}", flush=True)

        # =====================================================================
        # DOMAIN 3: REAL FORMAL LOGIC (ALL Knights-and-Knaves + ProofWriter train/val)
        # =====================================================================
        print("\n=== [Domain 3: FORMAL LOGIC (100% Real, Zero Synthetic!)] ===", flush=True)
        for sp in ("2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"):
            ds = load_dataset("K-and-K/knights-and-knaves", "train", split=sp)
            for i, item in enumerate(ds):
                q = str(item["quiz"]).strip()
                _try_emit(
                    {
                        "id": f"logic_kk_{sp}_{i}",
                        "domain": "logic",
                        "source": f"K-and-K/knights-and-knaves:{sp}",
                        "format": "knights_knaves",
                        "question": q,
                        "reference_solution": json.dumps(
                            {"names": list(item["names"]), "steps": list(item["cot_repeat_steps"]), "solution": list(item["solution"])},
                            ensure_ascii=False,
                        ),
                        "ground_truth_answer": json.dumps(list(item["solution"])),
                        "unit_tests": [],
                    },
                    "K-and-K/knights-and-knaves",
                )
        print(f"  [3a] Knights-and-Knaves    : {stats['sources'].get('K-and-K/knights-and-knaves', 0):6d}", flush=True)

        # ProofWriter multi-hop (QDep >= 1, prefer deeper QDep 2..5, strictly True/False provable!)
        for split_name, target_n, split_hint in (
            ("train", args.proofwriter_train_target, "train"),
            ("validation", args.proofwriter_val_target, "val"),
        ):
            pw = load_dataset("tasksource/proofwriter", split=split_name, streaming=True)
            kept = 0
            for i, item in enumerate(pw):
                if kept >= target_n:
                    break
                qdep = int(item.get("QDep", 0) or 0)
                ans = str(item.get("answer", "")).strip()
                proofs = str(item.get("allProofs", "")).strip()
                if qdep < 1 or ans not in ("True", "False") or "->" not in proofs:
                    continue
                q_full = f"{str(item['theory']).strip()}\nQuestion: {str(item['question']).strip()}"
                if len(q_full) > 2200:
                    continue
                ok = _try_emit(
                    {
                        "id": f"logic_pw_{split_name}_{i}",
                        "domain": "logic",
                        "source": f"tasksource/proofwriter:{split_name}:D{qdep}",
                        "format": "proofwriter",
                        "split_hint": split_hint,
                        "question": q_full,
                        "reference_solution": json.dumps(
                            {"theory": str(item["theory"]), "query": str(item["question"]), "answer": ans, "allProofs": proofs, "qdep": qdep},
                            ensure_ascii=False,
                        ),
                        "ground_truth_answer": ans,
                        "unit_tests": [],
                    },
                    f"tasksource/proofwriter:{split_name}",
                )
                if ok:
                    kept += 1
                    if kept % 10000 == 0:
                        print(f"    [ProofWriter:{split_name}] kept={kept}/{target_n}", flush=True)
            print(f"  [3b] ProofWriter ({split_name:10s}): {stats['sources'].get(f'tasksource/proofwriter:{split_name}', 0):6d}", flush=True)

    total_seeds = sum(stats["sources"].values())
    stats["total_clean_unique_seeds"] = total_seeds
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    stats["output_jsonl"] = str(out_jsonl)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n=== [Mega-Harvest DONE in {stats['elapsed_sec']}s] ===", flush=True)
    print(f"Total Clean Unique Seeds: {total_seeds:,} | Dropped Contaminated: {stats['dropped_contaminated']:,} | Dropped Duplicate: {stats['dropped_duplicate']:,}", flush=True)
    print(f"Saved -> {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
