# Gate 2 Evaluation Report (`e:\avl-list\avl-pmlv1\data\stage1_mega\mega_val_gsm8k_pw_kk_40.jsonl`, n=40)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `block_dllm_pml_off` | 40.00% | 108.2 | 1.62× | 2938.4 | 0.34 | 0 / 0 / +0 | 0.0 |
| `block_dllm_pml_on` | 37.50% | 82.6 | 2.12× | 2879.2 | 0.35 | 0 / 0 / +0 | 0.0 |
| `nl_cot_ref` | 30.00% | 210.8 | 0.83× | 9216.0 | 0.11 | 0 / 0 / +0 | 0.0 |
| `nl_cot_sft` | 47.50% | 175.3 | 1.0× | 7628.0 | 0.13 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(block_dllm_pml_off)` | 22.50% | 108.2 | 1.62× | 2647.0 | 0.38 | 0 / 0 / +0 | 0.0 |
| `dsl_ext_exec(block_dllm_pml_on)` | 12.50% | 82.6 | 2.12× | 2550.8 | 0.39 | 0 / 0 / +0 | 0.0 |

> **Gold-DSL Deterministic Executor Upper-Bound Check:** `72.50%` EM (29/40, avg exec `0.026 ms/problem`).
