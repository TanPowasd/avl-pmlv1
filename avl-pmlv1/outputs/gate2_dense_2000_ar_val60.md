# Gate 2 Evaluation Report (`e:\avl-list\avl-pmlv1\data\stage1_mega\mega_val_fsq.jsonl`, n=60)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ar_pml_off` | 35.00% | 206.3 | 1.0× (ref) | 9498.4 | 0.10 | 0 / 0 / +0 | 0.0 |
| `ar_pml_on` | 31.67% | 216.5 | 1.0× (ref) | 12422.4 | 0.08 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_off)` | 21.67% | 206.3 | 1.0× (ref) | 9081.0 | 0.11 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_on)` | 13.33% | 216.5 | 1.0× (ref) | 12012.0 | 0.08 | 0 / 0 / +0 | 0.0 |

> **Gold-DSL Deterministic Executor Upper-Bound Check:** `85.00%` EM (51/60, avg exec `0.050 ms/problem`).
