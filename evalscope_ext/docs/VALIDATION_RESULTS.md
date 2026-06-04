# Pruning validation (leave-one-model-out)

Hold one model out, build the difficulty prior from the other two only
(no leakage), and predict the held-out model's full score from the coreset.
The coreset is deterministic, so its MAE/flip are over the 3 held-out models;
random is averaged over 40 seeds. `hardest` = top-k by difficulty (forbidden).
Lower is better; "flip" = fraction of go/no-go calls that disagree with the full run.

## live_code_bench_v5  (n=315)

full-set accuracy: `gpt-oss-120b`=0.765, `kimi-k2.5`=0.629, `minimax-m2.5`=0.619

| keep % | n | coreset MAE | random MAE | random p90 | hardest MAE | coreset flip | random flip |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 10% | 32 | **0.0879** | 0.0585 | 0.119 | 0.4313 | **0.00** | 0.15 |
| 15% | 47 | **0.0531** | 0.0468 | 0.097 | 0.4298 | **0.00** | 0.09 |
| 20% | 63 | **0.0233** | 0.0461 | 0.090 | 0.3958 | **0.00** | 0.08 |
| 30% | 94 | **0.0214** | 0.0283 | 0.063 | 0.3482 | **0.00** | 0.02 |

## aa_lcr  (n=100)

full-set accuracy: `gpt-oss-120b`=0.480, `kimi-k2.5`=0.660, `minimax-m2.5`=0.640

| keep % | n | coreset MAE | random MAE | random p90 | hardest MAE | coreset flip | random flip |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 10% | 10 | **0.1067** | 0.1122 | 0.240 | 0.2933 | **0.33** | 0.23 |
| 15% | 15 | **0.0733** | 0.0927 | 0.193 | 0.3489 | **0.33** | 0.21 |
| 20% | 20 | **0.0733** | 0.0741 | 0.160 | 0.3600 | **0.33** | 0.12 |
| 30% | 30 | **0.0622** | 0.0631 | 0.141 | 0.3267 | **0.33** | 0.14 |
