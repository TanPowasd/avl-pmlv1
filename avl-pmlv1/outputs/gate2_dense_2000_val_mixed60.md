# Gate 2 Evaluation Report (`e:\avl-list\avl-pmlv1\data\stage1_mega\mega_val_fsq.jsonl`, n=60)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `block_dllm_pml_off` | 50.00% | 205.9 | 1.04× | 5267.6 | 0.19 | 0 / 0 / +0 | 0.0 |
| `block_dllm_pml_on` | 60.00% | 135.6 | 1.57× | 4420.8 | 0.23 | 0 / 0 / +0 | 0.0 |
| `nl_cot_ref` | 10.00% | 229.5 | 0.93× | 9797.5 | 0.10 | 0 / 0 / +0 | 0.0 |
| `nl_cot_sft` | 41.67% | 213.2 | 1.0× | 10447.5 | 0.10 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(block_dllm_pml_off)` | 20.00% | 205.9 | 1.04× | 4922.3 | 0.20 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(block_dllm_pml_on)` | 15.00% | 135.6 | 1.57× | 4176.7 | 0.24 | 0 / 0 / +0 | 0.0 |

> **Gold-DSL Deterministic Executor Upper-Bound Check:** `85.00%` EM (51/60, avg exec `0.043 ms/problem`).
