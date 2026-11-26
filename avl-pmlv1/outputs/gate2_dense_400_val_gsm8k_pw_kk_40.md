# Gate 2 Evaluation Report (`e:\avl-list\avl-pmlv1\data\stage1_mega\mega_val_gsm8k_pw_kk_40.jsonl`, n=40)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ar_pml_off` | 35.00% | 145.6 | 1.45× | 6947.6 | 0.14 | 0 / 0 / +0 | 0.0 |
| `ar_pml_on` | 32.50% | 169.2 | 1.25× | 9820.9 | 0.10 | 0 / 0 / +0 | 0.0 |
| `block_dllm_pml_off` | 40.00% | 232.6 | 0.91× | 6290.1 | 0.16 | 0 / 0 / +0 | 346.1 |
| `block_dllm_pml_on` | 40.00% | 114.6 | 1.84× | 10086.4 | 0.10 | 0 / 0 / +0 | 5275.6 |
| `nl_cot_ref` | 30.00% | 210.8 | 1.0× | 9203.4 | 0.11 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_off)` | 12.50% | 145.6 | 1.45× | 6572.4 | 0.15 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_on)` | 12.50% | 169.2 | 1.25× | 9626.3 | 0.10 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(block_dllm_pml_off)` | 5.00% | 232.6 | 0.91× | 5994.6 | 0.17 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(block_dllm_pml_on)` | 12.50% | 114.6 | 1.84× | 9755.3 | 0.10 | 0 / 0 / +0 | 0.0 |

> **Gold-DSL Deterministic Executor Upper-Bound Check:** `72.50%` EM (29/40, avg exec `0.026 ms/problem`).
