# Multi-seed results (mean +/- std over seeds)

Relative RMS at t=200; usable horizon in steps.

## advection  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `skino_plain` | 25,067 | 0.001006 | 0.000365 | 500 | 0 | good |
| `skino_seq2seq` | 21,135 | 0.001375 | 0.000518 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.002568 | 0.00046 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.002873 | 0.000509 | 500 | 0 | good |
| `strict_plain` | 20,779 | 0.007908 | 0.00484 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.01004 | 0.000491 | 500 | 0 | good |
| `tfno_plain` | 28,289 | 0.0118 | 0.00373 | 500 | 0 | good |
| `ufno_plain` | 18,193 | 0.01367 | 0.000602 | 500 | 0 | good |
| `tfno_noise` | 28,289 | 0.03614 | 0.00797 | 500 | 0 | good |
| `fno_plain` | 25,985 | 0.04324 | 0.00372 | 500 | 0 | good |
| `nosymp_noise` | 22,755 | 0.04726 | 0.00517 | 500 | 0 | good |
| `ufno_noise` | 18,193 | 0.04804 | 0.00242 | 500 | 0 | good |
| `skino_noise` | 25,067 | 0.04839 | 0.0116 | 500 | 0 | good |
| `strict_noise` | 20,779 | 0.05274 | 0.00904 | 500 | 0 | good |
| `fno_noise` | 25,985 | 0.2148 | 0.0221 | 488 | 16 | good |

Best: **`skino_plain`** (0.001006 +/- 0.000365). Significantly worse (gap exceeds combined std): `ufno_seq2seq`, `tfno_seq2seq`, `strict_plain`, `fno_seq2seq`, `tfno_plain`, `ufno_plain`, `tfno_noise`, `fno_plain`, `nosymp_noise`, `ufno_noise`, `skino_noise`, `strict_noise`, `fno_noise`.

## heat  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `tfno_plain` | 28,289 | 0.001165 | 0.000289 | 500 | 0 | good |
| `skino_plain` | 25,067 | 0.002736 | 0.00118 | 500 | 0 | good |
| `skino_seq2seq` | 21,135 | 0.005407 | 0.00199 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.006431 | 0.00053 | 500 | 0 | good |
| `ufno_plain` | 18,193 | 0.00701 | 0.00259 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.008546 | 0.00166 | 500 | 0 | good |
| `strict_plain` | 20,779 | 0.009527 | 0.00232 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.01643 | 0.00164 | 500 | 0 | good |
| `fno_plain` | 25,985 | 0.02338 | 0.00866 | 500 | 0 | good |
| `tfno_noise` | 28,289 | 0.0315 | 0.0102 | 500 | 0 | good |
| `ufno_noise` | 18,193 | 0.04083 | 0.00793 | 500 | 0 | good |
| `nosymp_noise` | 22,755 | 0.04473 | 0.00137 | 500 | 0 | good |
| `strict_noise` | 20,779 | 0.04744 | 0.0109 | 500 | 0 | good |
| `skino_noise` | 25,067 | 0.04753 | 0.00744 | 500 | 0 | good |
| `fno_noise` | 25,985 | 0.1984 | 0.0197 | 500 | 0 | good |

Best: **`tfno_plain`** (0.001165 +/- 0.000289). Significantly worse (gap exceeds combined std): `skino_plain`, `skino_seq2seq`, `tfno_seq2seq`, `ufno_plain`, `ufno_seq2seq`, `strict_plain`, `fno_seq2seq`, `fno_plain`, `tfno_noise`, `ufno_noise`, `nosymp_noise`, `strict_noise`, `skino_noise`, `fno_noise`.

## wave1d  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `tfno_seq2seq` | 21,514 | 0.002247 | 0.000211 | 500 | 0 | good |
| `skino_seq2seq` | 25,614 | 0.002672 | 0.000117 | 500 | 0 | good |
| `ufno_seq2seq` | 22,082 | 0.002674 | 0.000326 | 500 | 0 | good |
| `fno_seq2seq` | 27,666 | 0.01681 | 0.000808 | 500 | 0 | good |
| `tfno_noise` | 26,338 | 0.07643 | 0.00811 | 500 | 0 | good |
| `ufno_plain` | 18,218 | 0.1556 | 0.00187 | 462 | 38 | good |
| `ufno_noise` | 18,218 | 0.162 | 0.0436 | 500 | 0 | good |
| `skino_plain` | 21,110 | 0.1959 | 0.0614 | 392 | 80 | good |
| `strict_plain` | 18,870 | 0.3708 | 0.219 | 310 | 139 | degraded/good |
| `skino_noise` | 21,110 | 0.4406 | 0.0634 | 145 | 84 | degraded/good |
| `tfno_plain` | 26,338 | 0.4478 | 0.574 | 327 | 135 | degraded/good |
| `strict_noise` | 18,870 | 0.4697 | 0.00861 | 112 | 2 | degraded |
| `nosymp_noise` | 20,814 | 0.5011 | 0.0189 | 62 | 2 | degraded |
| `fno_noise` | 26,018 | 0.5658 | 0.00587 | 82 | 21 | degraded |
| `fno_plain` | 26,018 | 0.7906 | 0.267 | 120 | 7 | degraded |

Best: **`tfno_seq2seq`** (0.002247 +/- 0.000211). Significantly worse (gap exceeds combined std): `skino_seq2seq`, `fno_seq2seq`, `tfno_noise`, `ufno_plain`, `ufno_noise`, `skino_plain`, `strict_plain`, `skino_noise`, `strict_noise`, `nosymp_noise`, `fno_noise`, `fno_plain`.

## burgers  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `tfno_plain` | 28,289 | 0.003001 | 0.000487 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.01005 | 0.00017 | 500 | 0 | good |
| `ufno_plain` | 18,193 | 0.01123 | 0.00308 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.01358 | 0.00339 | 500 | 0 | good |
| `tfno_noise` | 28,289 | 0.03619 | 0.00936 | 500 | 0 | good |
| `ufno_noise` | 18,193 | 0.04329 | 0.00408 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.0689 | 0.0026 | 500 | 0 | good |
| `fno_plain` | 25,985 | 0.1364 | 0.028 | 500 | 0 | good |
| `fno_noise` | 25,985 | 0.3038 | 0.0374 | 452 | 34 | good |
| `skino_seq2seq` | 21,135 | 0.4226 | 0.000477 | 238 | 2 | good |
| `nosymp_noise` | 22,755 | 0.4306 | 0.00774 | 225 | 7 | good |
| `skino_noise` | 25,067 | 0.4391 | 0.0106 | 207 | 6 | good |
| `skino_plain` | 25,067 | 0.44 | 0.00879 | 208 | 2 | good |
| `strict_noise` | 20,779 | 0.4434 | 0.00855 | 195 | 0 | degraded |
| `strict_plain` | 20,779 | 0.4561 | 0.0113 | 180 | 4 | degraded |

Best: **`tfno_plain`** (0.003001 +/- 0.000487). Significantly worse (gap exceeds combined std): `tfno_seq2seq`, `ufno_plain`, `ufno_seq2seq`, `tfno_noise`, `ufno_noise`, `fno_seq2seq`, `fno_plain`, `fno_noise`, `skino_seq2seq`, `nosymp_noise`, `skino_noise`, `skino_plain`, `strict_noise`, `strict_plain`.

## kdv  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `skino_seq2seq` | 21,135 | 0.001658 | 0.000596 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.00279 | 8.73e-05 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.004822 | 0.0027 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.006595 | 0.000815 | 500 | 0 | good |
| `skino_noise` | 25,067 | 0.04621 | 0.018 | 447 | 75 | good |
| `nosymp_noise` | 22,755 | 0.05474 | 0.00798 | 478 | 21 | good |
| `ufno_noise` | 18,193 | 0.08356 | 0.0256 | 367 | 108 | good |
| `tfno_noise` | 28,289 | 0.08798 | 0.0136 | 337 | 56 | good |
| `fno_plain` | 25,985 | 0.186 | 0.00809 | 140 | 39 | degraded |
| `fno_noise` | 25,985 | 0.3637 | 0.267 | 172 | 93 | decorrelated/degraded/good |
| `ufno_plain` | 18,193 | 1.778 | 2.45 | 363 | 193 | blow-up/good |
| `strict_noise` | 20,779 | 333.6 | 471 | 113 | 65 | blow-up/degraded |
| `tfno_plain` | 28,289 | 847.7 | 215 | 82 | 52 | blow-up |
| `skino_plain` | 25,067 | 1000 | 0 | 33 | 2 | blow-up |
| `strict_plain` | 20,779 | 1000 | 0 | 22 | 6 | blow-up |

Best: **`skino_seq2seq`** (0.001658 +/- 0.000596). Significantly worse (gap exceeds combined std): `tfno_seq2seq`, `fno_seq2seq`, `skino_noise`, `nosymp_noise`, `ufno_noise`, `tfno_noise`, `fno_plain`, `fno_noise`, `tfno_plain`, `skino_plain`, `strict_plain`.

## wave2d  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `skino_seq2seq` | 21,418 | 0.08717 | 0.0409 | 250 | 0 | good |
| `strict_noise` | 21,062 | 0.1084 | 0.00223 | 250 | 0 | good |
| `skino_noise` | 25,414 | 0.1162 | 0.00116 | 250 | 0 | good |
| `fno_seq2seq` | 25,774 | 0.1536 | 0.0088 | 250 | 0 | good |
| `fno_noise` | 29,402 | 3.272 | 4.08 | 172 | 73 | blow-up/good |

Best: **`skino_seq2seq`** (0.08717 +/- 0.0409). Significantly worse (gap exceeds combined std): `fno_seq2seq`.

## wave3d  (3 seeds, reported at t=150)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `skino_seq2seq` | 24,122 | 0.1386 | 0.0373 | 150 | 0 | good |
| `skino_noise` | 23,894 | 0.5334 | 0.586 | 107 | 61 | decorrelated/good |
| `fno_seq2seq` | 26,977 | 0.6394 | 0.104 | 2 | 2 | degraded |
| `strict_noise` | 20,502 | 0.8195 | 0.0635 | 15 | 0 | degraded |
| `fno_noise` | 25,177 | 453.6 | 394 | 22 | 8 | blow-up |

Best: **`skino_seq2seq`** (0.1386 +/- 0.0373). Significantly worse (gap exceeds combined std): `fno_seq2seq`, `strict_noise`, `fno_noise`.
