# Pruning validation (leave-one-model-out)

Each row holds out one model, builds the difficulty prior from the other
two only (no leakage), and estimates the held-out model's full-benchmark
accuracy from the coreset. MAE/p90 are over the 3 held-out models x 40 seeds.
Baselines: `random` = uniform sampling (forbidden); `hardest` = top-k by
difficulty (forbidden). Lower is better.

## live_code_bench_v5  (n=315)

full-set accuracy: `gpt-oss-120b`=0.765, `kimi-k2.5`=0.629, `minimax-m2.5`=0.619

| keep % | n | coreset MAE | coreset p90 | random MAE | random p90 | hardest MAE | flip core | flip rand |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10% | 32 | **0.0531** | 0.109 | 0.0585 | 0.119 | 0.4313 | 0.12 | 0.15 |
| 15% | 47 | **0.0380** | 0.075 | 0.0468 | 0.097 | 0.4298 | 0.04 | 0.09 |
| 20% | 63 | **0.0376** | 0.074 | 0.0461 | 0.090 | 0.3958 | 0.03 | 0.08 |
| 30% | 94 | **0.0240** | 0.045 | 0.0283 | 0.063 | 0.3482 | 0.00 | 0.02 |

## aa_lcr  (n=100)

full-set accuracy: `gpt-oss-120b`=0.480, `kimi-k2.5`=0.660, `minimax-m2.5`=0.640

| keep % | n | coreset MAE | coreset p90 | random MAE | random p90 | hardest MAE | flip core | flip rand |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10% | 10 | **0.1127** | 0.240 | 0.1122 | 0.240 | 0.2933 | 0.18 | 0.23 |
| 15% | 15 | **0.0982** | 0.175 | 0.0927 | 0.193 | 0.3489 | 0.27 | 0.21 |
| 20% | 20 | **0.0729** | 0.140 | 0.0741 | 0.160 | 0.3600 | 0.16 | 0.12 |
| 30% | 30 | **0.0599** | 0.127 | 0.0631 | 0.141 | 0.3267 | 0.22 | 0.14 |
