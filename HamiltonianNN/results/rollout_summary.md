# Rollout wMAPE summary

Test trajectory: run 2  |  rollout length: 280 steps  |  physical horizon: 560.0 ms

Burn-in skipped for the *after-burn-in* columns: first 5 saved states (= 10.0 ms).

## Displacement  q

| model | median (post-burn-in) | mean (post-burn-in) | step 10 | step 50 | step 100 | final (step 300) |
|---|---:|---:|---:|---:|---:|---:|
| FNO | 110.22% | 112.36% | 117.93% | 104.52% | 105.55% | 125.22% |
| SKINO | 101.21% | 100.54% | 88.43% | 101.92% | 101.67% | 101.00% |

## Momentum  p

| model | median (post-burn-in) | mean (post-burn-in) | step 10 | step 50 | step 100 | final (step 300) |
|---|---:|---:|---:|---:|---:|---:|
| FNO | 108.13% | 109.99% | 123.86% | 102.88% | 104.43% | 129.11% |
| SKINO | 101.19% | 100.43% | 79.99% | 101.19% | 101.16% | 101.11% |

## Rollout wall-clock

| model | seconds for 300 autoregressive steps (CPU) |
|---|---:|
| FNO | 23.28 |
| SKINO | 5.65 |
