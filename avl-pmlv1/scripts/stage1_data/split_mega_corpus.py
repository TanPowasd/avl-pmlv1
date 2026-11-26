"""Split FSQ-Annotated Mega-Corpus into Strictly Isolated Train / Val / Internal-Test Partitions & Audit Zero Benchmark Leakage."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Set


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


def md5_norm(s: str) -> str:
    return hashlib.md5(norm_text(s).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_fsq_aligned.jsonl"))
    parser.add_argument("--bench-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "benchmarks"))
    parser.add_argument("--train-out", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_train_fsq.jsonl"))
    parser.add_argument("--val-out", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_val_fsq.jsonl"))
    parser.add_argument("--test-out", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1_mega" / "mega_test_fsq.jsonl"))
    parser.add_argument("--report-out", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "stage1_mega_split_report.json"))
    parser.add_argument("--val-math", type=int, default=4000)
    parser.add_argument("--val-code", type=int, default=4000)
    parser.add_argument("--val-logic", type=int, default=2000)
    parser.add_argument("--test-math", type=int, default=4000)
    parser.add_argument("--test-code", type=int, default=4000)
    parser.add_argument("--test-logic", type=int, default=2000)
    args = parser.parse_args()

    # Load all 7 official public benchmark hashes for final zero-leakage double-audit!
    bench_hashes: Set[str] = set()
    bench_counts: Dict[str, int] = {}
    for bf in sorted(Path(args.bench_dir).glob("*.jsonl")):
        c = 0
        with open(bf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    bench_hashes.add(md5_norm(str(r["question"])))
                    c += 1
        bench_counts[bf.stem] = c

    # Read all FSQ-annotated records
    by_dom: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    leaked_caught = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if md5_norm(r["question"]) in bench_hashes:
                leaked_caught += 1
                continue
            by_dom[r["domain"]].append(r)

    rng = random.Random(20261124)
    val_quotas = {"math": args.val_math, "code": args.val_code, "logic": args.val_logic}
    test_quotas = {"math": args.test_math, "code": args.test_code, "logic": args.test_logic}

    train_recs: List[Dict[str, Any]] = []
    val_recs: List[Dict[str, Any]] = []
    test_recs: List[Dict[str, Any]] = []

    for dom in ("math", "code", "logic"):
        recs = by_dom.get(dom, [])
        # Put official split_hint == 'validation' (ProofWriter official validation split!) strictly into Val!
        val_hinted = [r for r in recs if ":validation:" in r.get("source", "")]
        others = [r for r in recs if ":validation:" not in r.get("source", "")]
        rng.shuffle(val_hinted)
        rng.shuffle(others)

        qv = min(val_quotas[dom], len(recs) // 5)
        qt = min(test_quotas[dom], len(recs) // 5)

        chosen_val = val_hinted[:qv]
        cursor = 0
        if len(chosen_val) < qv:
            need = qv - len(chosen_val)
            chosen_val.extend(others[cursor : cursor + need])
            cursor += need

        chosen_test = others[cursor : cursor + qt]
        cursor += qt

        rem_train = others[cursor:] + val_hinted[qv:]
        val_recs.extend(chosen_val)
        test_recs.extend(chosen_test)
        train_recs.extend(rem_train)

    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    rng.shuffle(test_recs)

    # Strict 4-way pairwise disjointness verification (ID AND normalized question MD5)!
    tr_ids, va_ids, te_ids = ({r["id"] for r in lst} for lst in (train_recs, val_recs, test_recs))
    tr_q, va_q, te_q = ({md5_norm(r["question"]) for r in lst} for lst in (train_recs, val_recs, test_recs))
    assert len(tr_ids & va_ids) == 0 and len(tr_ids & te_ids) == 0 and len(va_ids & te_ids) == 0, "ID overlap!"
    assert len(tr_q & va_q) == 0 and len(tr_q & te_q) == 0 and len(va_q & te_q) == 0, "Question overlap!"
    assert len(tr_q & bench_hashes) == 0 and len(va_q & bench_hashes) == 0 and len(te_q & bench_hashes) == 0, "Benchmark leak!"

    for path_s, lst in ((args.train_out, train_recs), (args.val_out, val_recs), (args.test_out, test_recs)):
        with open(path_s, "w", encoding="utf-8") as f:
            for r in lst:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "train_total": len(train_recs),
        "train_by_domain": dict(collections.Counter(r["domain"] for r in train_recs)),
        "train_by_source": dict(collections.Counter(r["source"].split(":")[0] for r in train_recs)),
        "val_total": len(val_recs),
        "val_by_domain": dict(collections.Counter(r["domain"] for r in val_recs)),
        "val_by_source": dict(collections.Counter(r["source"].split(":")[0] for r in val_recs)),
        "internal_blind_test_total": len(test_recs),
        "internal_blind_test_by_domain": dict(collections.Counter(r["domain"] for r in test_recs)),
        "internal_blind_test_by_source": dict(collections.Counter(r["source"].split(":")[0] for r in test_recs)),
        "official_public_benchmarks_total": sum(bench_counts.values()),
        "official_public_benchmarks": bench_counts,
        "combined_total_test_problems": len(test_recs) + sum(bench_counts.values()),
        "zero_4way_overlap_verified": True,
        "zero_benchmark_leakage_verified": True,
        "late_leaks_dropped": leaked_caught,
        "train_path": args.train_out,
        "val_path": args.val_out,
        "test_path": args.test_out,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("=== [Mega Corpus 4-Way Strict Split Complete] ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
