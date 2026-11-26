# Gate 2 Evaluation Report (`data\stage1_mega\mega_val_fsq.jsonl`, n=6)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ar_pml_on` | 0.00% | 128.0 | 1.29× | 18658.6 | 0.05 | 0 / 0 / +0 | 0.0 |
| `block_dllm_pml_on` | 0.00% | 128.0 | 1.29× | 15620.0 | 0.06 | 0 / 0 / +0 | 4972.5 |
| `nl_cot_ref` | 16.67% | 165.7 | 1.0× | 18905.9 | 0.05 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_on)` | 0.00% | 128.0 | 1.29× | 15982.4 | 0.06 | 0 / 0 / +0 | 0.0 |

> **Gold-DSL Deterministic Executor Upper-Bound Check:** `33.33%` EM (2/6, avg exec `0.077 ms/problem`).
