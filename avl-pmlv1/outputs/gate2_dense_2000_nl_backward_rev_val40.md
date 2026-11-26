# Gate 2 Evaluation Report (`E:\avl-list\avl-pmlv1\data\stage1_mega\mega_val_gsm8k_pw_kk_40.jsonl`, n=40)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rev_forward_only_b3_t256` | 7.50% | 212.2 | 1.0× (ref) | 4389.1 | 0.23 | 0 / 0 / +0 | 0.0 |
| `rev_wet_cement_b3_t256` | 10.00% | 209.8 | 1.0× (ref) | 4805.5 | 0.21 | 2 / 1 / +1 | 610.7 |
| `rev_post_inplace_b3_t256` | 10.00% | 210.6 | 1.0× (ref) | 6731.9 | 0.15 | 1 / 0 / +1 | 2400.5 |
| `rev_wet_plus_post_b3_t256` | 12.50% | 207.8 | 1.0× (ref) | 7348.3 | 0.14 | 2 / 1 / +1 | 3110.9 |
| `rev_forward_only_b3_t384` | 27.50% | 252.1 | 1.0× (ref) | 5209.6 | 0.19 | 0 / 0 / +0 | 0.0 |
| `rev_wet_cement_b3_t384` | 32.50% | 249.5 | 1.0× (ref) | 5760.1 | 0.17 | 4 / 2 / +2 | 760.6 |
| `rev_post_inplace_b3_t384` | 32.50% | 253.6 | 1.0× (ref) | 7723.7 | 0.13 | 2 / 0 / +2 | 2610.5 |
| `rev_wet_plus_post_b3_t384` | 35.00% | 249.8 | 1.0× (ref) | 8796.2 | 0.11 | 2 / 1 / +1 | 3889.8 |
