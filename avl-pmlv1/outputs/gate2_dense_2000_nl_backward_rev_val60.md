# Gate 2 Evaluation Report (`E:\avl-list\avl-pmlv1\data\stage1_mega\mega_val_fsq.jsonl`, n=60)

| Arm | Exact Match (%) | Avg Think Tok | Compression vs NL-CoT | Avg E2E (ms) | Throughput (prob/s) | Fixed / Broken / NetFix | Repair Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rev_forward_only_b3_t256` | 15.00% | 227.3 | 1.0× (ref) | 4473.3 | 0.22 | 0 / 0 / +0 | 0.0 |
| `rev_wet_cement_b3_t256` | 11.67% | 230.4 | 1.0× (ref) | 5362.5 | 0.19 | 2 / 4 / -2 | 1015.6 |
| `rev_post_inplace_b3_t256` | 15.00% | 227.9 | 1.0× (ref) | 7124.3 | 0.14 | 0 / 0 / +0 | 2638.5 |
| `rev_wet_plus_post_b3_t256` | 13.33% | 229.5 | 1.0× (ref) | 7814.4 | 0.13 | 2 / 1 / +1 | 3458.6 |
| `rev_forward_only_b3_t384` | 43.33% | 291.8 | 1.0× (ref) | 5797.7 | 0.17 | 0 / 0 / +0 | 0.0 |
| `rev_wet_cement_b3_t384` | 48.33% | 304.5 | 1.0× (ref) | 7224.0 | 0.14 | 7 / 4 / +3 | 1462.3 |
| `rev_post_inplace_b3_t384` | 41.67% | 292.6 | 1.0× (ref) | 9550.0 | 0.10 | 2 / 3 / -1 | 3766.6 |
| `rev_wet_plus_post_b3_t384` | 43.33% | 304.0 | 1.0× (ref) | 10876.4 | 0.09 | 2 / 5 / -3 | 5144.4 |
