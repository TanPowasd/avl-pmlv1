# Gate 2 Evaluation Report (`e:\avl-list\avl-pmlv1\data\stage1_mega\mega_val_fsq.jsonl`, n=60)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ar_pml_off` | 43.33% | 191.6 | 1.2× | 8989.5 | 0.11 | 0 / 0 / +0 | 0.0 |
| `ar_pml_on` | 43.33% | 201.6 | 1.14× | 11735.5 | 0.09 | 0 / 0 / +0 | 0.0 |
| `block_dllm_pml_off` | 61.67% | 234.1 | 0.98× | 6479.5 | 0.15 | 1 / 0 / +1 | 368.3 |
| `block_dllm_pml_on` | 58.33% | 175.7 | 1.31× | 13137.7 | 0.08 | 0 / 0 / +0 | 6985.3 |
| `nl_cot_ref` | 10.00% | 229.5 | 1.0× | 10044.6 | 0.10 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_off)` | 6.67% | 191.6 | 1.2× | 8690.8 | 0.12 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(ar_pml_on)` | 6.67% | 201.6 | 1.14× | 11583.4 | 0.09 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(block_dllm_pml_off)` | 18.33% | 234.1 | 0.98× | 6262.6 | 0.16 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(block_dllm_pml_on)` | 18.33% | 175.7 | 1.31× | 12902.8 | 0.08 | 0 / 0 / +0 | 0.0 |

> **Gold-DSL Deterministic Executor Upper-Bound Check:** `85.00%` EM (51/60, avg exec `0.042 ms/problem`).
