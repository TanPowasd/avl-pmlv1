# Gate 2 Evaluation Report (`e:\avl-list\avl-pmlv1\data\stage1_mega\mega_val_gsm8k_pw_kk_40.jsonl`, n=40)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ar_pml_off` | 27.50% | 180.4 | 1.0× (ref) | 8136.2 | 0.12 | 0 / 0 / +0 | 0.0 |
| `ar_pml_on` | 35.00% | 149.6 | 1.0× (ref) | 8559.7 | 0.12 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_off)` | 20.00% | 180.4 | 1.0× (ref) | 7863.8 | 0.13 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_on)` | 7.50% | 149.6 | 1.0× (ref) | 8182.2 | 0.12 | 0 / 0 / +0 | 0.0 |

> **Gold-DSL Deterministic Executor Upper-Bound Check:** `72.50%` EM (29/40, avg exec `0.027 ms/problem`).
