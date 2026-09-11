# Multi-seed results (mean +/- std over seeds)

Relative RMS at t=200; usable horizon in steps.

## advection  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `skino_plain` | 25,067 | 0.001092 | 0.000569 | 500 | 0 | good |
| `sno_seq2seq` | 31,013 | 0.001101 | 9.01e-05 | 500 | 0 | good |
| `skino_seq2seq` | 21,135 | 0.001321 | 0.000441 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.00257 | 0.000457 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.002903 | 0.000529 | 500 | 0 | good |
| `strict_plain` | 20,779 | 0.007916 | 0.00441 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.01005 | 0.000529 | 500 | 0 | good |
| `tfno_plain` | 28,289 | 0.01208 | 0.00285 | 500 | 0 | good |
| `skino_pinn` | 25,067 | 0.01215 | 0.00175 | 500 | 0 | good |
| `ufno_plain` | 18,193 | 0.01372 | 0.000702 | 500 | 0 | good |
| `sno_plain` | 18,513 | 0.01709 | 0.00726 | 500 | 0 | good |
| `fno_pinn` | 25,985 | 0.03261 | 0.000124 | 500 | 0 | good |
| `fno_plain` | 25,985 | 0.04215 | 0.00353 | 500 | 0 | good |
| `tfno_noise` | 28,289 | 0.04416 | 0.00621 | 500 | 0 | good |
| `nosymp_noise` | 22,755 | 0.04797 | 0.0111 | 500 | 0 | good |
| `skino_noise` | 25,067 | 0.04918 | 0.00354 | 500 | 0 | good |
| `strict_noise` | 20,779 | 0.06225 | 0.00659 | 500 | 0 | good |
| `ufno_noise` | 18,193 | 0.06749 | 0.00971 | 500 | 0 | good |
| `sno_noise` | 18,513 | 0.1088 | 0.0135 | 500 | 0 | good |
| `fno_noise` | 25,985 | 0.233 | 0.011 | 443 | 57 | good |
| `unet_noise` | 18,785 | 0.6005 | 0.00228 | 128 | 2 | degraded |
| `deeponet_noise` | 20,929 | 0.85 | 0.0823 | 33 | 17 | degraded |
| `transformer_noise` | 19,233 | 1.082 | 0.00257 | 10 | 0 | decorrelated |
| `generic_noise` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |

Best: **`skino_plain`** (0.001092 +/- 0.000569). Significantly worse (gap exceeds combined std): `ufno_seq2seq`, `tfno_seq2seq`, `strict_plain`, `fno_seq2seq`, `tfno_plain`, `skino_pinn`, `ufno_plain`, `sno_plain`, `fno_pinn`, `fno_plain`, `tfno_noise`, `nosymp_noise`, `skino_noise`, `strict_noise`, `ufno_noise`, `sno_noise`, `fno_noise`, `unet_noise`, `deeponet_noise`, `transformer_noise`, `generic_noise`, `generic_plain`.

## heat  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `tfno_plain` | 28,289 | 0.00109 | 0.000307 | 500 | 0 | good |
| `skino_plain` | 25,067 | 0.00283 | 0.000911 | 500 | 0 | good |
| `sno_plain` | 18,513 | 0.003091 | 0.00187 | 500 | 0 | good |
| `skino_seq2seq` | 21,135 | 0.005026 | 0.00144 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.006429 | 0.000534 | 500 | 0 | good |
| `sno_seq2seq` | 31,013 | 0.006906 | 0.00136 | 500 | 0 | good |
| `strict_plain` | 20,779 | 0.008164 | 0.00254 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.008554 | 0.00171 | 500 | 0 | good |
| `ufno_plain` | 18,193 | 0.008579 | 0.00254 | 500 | 0 | good |
| `skino_pinn` | 25,067 | 0.01307 | 0.00525 | 500 | 0 | good |
| `fno_pinn` | 25,985 | 0.01409 | 0.00691 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.01643 | 0.00166 | 500 | 0 | good |
| `fno_plain` | 25,985 | 0.02499 | 0.0121 | 500 | 0 | good |
| `tfno_noise` | 28,289 | 0.03713 | 0.0108 | 500 | 0 | good |
| `ufno_noise` | 18,193 | 0.03966 | 8.06e-05 | 500 | 0 | good |
| `strict_noise` | 20,779 | 0.04322 | 0.0141 | 500 | 0 | good |
| `skino_noise` | 25,067 | 0.04353 | 0.0104 | 500 | 0 | good |
| `nosymp_noise` | 22,755 | 0.05152 | 0.0158 | 500 | 0 | good |
| `unet_noise` | 18,785 | 0.07705 | 0.00384 | 500 | 0 | good |
| `sno_noise` | 18,513 | 0.2085 | 0.0363 | 497 | 5 | good |
| `fno_noise` | 25,985 | 0.224 | 0.0137 | 500 | 0 | good |
| `transformer_noise` | 19,233 | 0.4814 | 0.0144 | 232 | 5 | good |
| `deeponet_noise` | 20,929 | 0.6895 | 0.00479 | 170 | 4 | degraded |
| `generic_noise` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |

Best: **`tfno_plain`** (0.00109 +/- 0.000307). Significantly worse (gap exceeds combined std): `skino_plain`, `skino_seq2seq`, `tfno_seq2seq`, `sno_seq2seq`, `strict_plain`, `ufno_seq2seq`, `ufno_plain`, `skino_pinn`, `fno_pinn`, `fno_seq2seq`, `fno_plain`, `tfno_noise`, `ufno_noise`, `strict_noise`, `skino_noise`, `nosymp_noise`, `unet_noise`, `sno_noise`, `fno_noise`, `transformer_noise`, `deeponet_noise`, `generic_noise`, `generic_plain`.

## wave1d  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `sno_seq2seq` | 25,282 | 0.001298 | 0.000164 | 500 | 0 | good |
| `tfno_seq2seq` | 21,514 | 0.002244 | 0.000223 | 500 | 0 | good |
| `skino_seq2seq` | 25,614 | 0.002591 | 0.000231 | 500 | 0 | good |
| `ufno_seq2seq` | 22,082 | 0.002681 | 0.000326 | 500 | 0 | good |
| `sno_plain` | 18,562 | 0.01073 | 0.00367 | 500 | 0 | good |
| `fno_seq2seq` | 27,666 | 0.01682 | 0.00083 | 500 | 0 | good |
| `tfno_noise` | 26,338 | 0.1027 | 0.0221 | 500 | 0 | good |
| `ufno_plain` | 18,218 | 0.1621 | 0.00974 | 452 | 52 | good |
| `skino_plain` | 21,110 | 0.1956 | 0.0579 | 390 | 80 | good |
| `ufno_noise` | 18,218 | 0.2043 | 0.0257 | 442 | 41 | good |
| `skino_pinn` | 21,110 | 0.2182 | 0.0095 | 365 | 41 | good |
| `sno_noise` | 18,562 | 0.2248 | 0.0304 | 425 | 58 | good |
| `strict_plain` | 18,870 | 0.3797 | 0.182 | 295 | 146 | degraded/good |
| `strict_noise` | 18,870 | 0.4759 | 0.00695 | 112 | 2 | degraded |
| `skino_noise` | 21,110 | 0.4781 | 0.0708 | 130 | 63 | degraded |
| `nosymp_noise` | 20,814 | 0.494 | 0.0229 | 82 | 21 | degraded |
| `unet_noise` | 18,914 | 0.5252 | 0.007 | 98 | 16 | degraded |
| `fno_pinn` | 26,018 | 0.5294 | 0.112 | 118 | 45 | degraded/good |
| `fno_noise` | 26,018 | 0.5918 | 0.00979 | 63 | 2 | degraded |
| `fno_plain` | 26,018 | 0.7167 | 0.145 | 117 | 6 | degraded |
| `deeponet_noise` | 25,090 | 0.8725 | 0.184 | 47 | 22 | decorrelated/degraded |
| `transformer_noise` | 18,274 | 1.108 | 0.125 | 18 | 2 | decorrelated |
| `tfno_plain` | 26,338 | 7.318 | 10.3 | 323 | 154 | blow-up/good |
| `generic_noise` | 31,747 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 31,747 | 1000 | 0 | 0 | 0 | blow-up |

Best: **`sno_seq2seq`** (0.001298 +/- 0.000164). Significantly worse (gap exceeds combined std): `tfno_seq2seq`, `skino_seq2seq`, `ufno_seq2seq`, `sno_plain`, `fno_seq2seq`, `tfno_noise`, `ufno_plain`, `skino_plain`, `ufno_noise`, `skino_pinn`, `sno_noise`, `strict_plain`, `strict_noise`, `skino_noise`, `nosymp_noise`, `unet_noise`, `fno_pinn`, `fno_noise`, `fno_plain`, `deeponet_noise`, `transformer_noise`, `generic_noise`, `generic_plain`.

## burgers  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `tfno_plain` | 28,289 | 0.003963 | 0.000979 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.01005 | 0.000171 | 500 | 0 | good |
| `ufno_plain` | 18,193 | 0.01109 | 0.00288 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.01357 | 0.00337 | 500 | 0 | good |
| `sno_seq2seq` | 31,013 | 0.01618 | 0.00168 | 500 | 0 | good |
| `sno_plain` | 18,513 | 0.01888 | 0.00343 | 500 | 0 | good |
| `tfno_noise` | 28,289 | 0.03631 | 0.00767 | 500 | 0 | good |
| `ufno_noise` | 18,193 | 0.03991 | 0.00977 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.06889 | 0.00261 | 500 | 0 | good |
| `fno_pinn` | 25,985 | 0.1088 | 0.0114 | 500 | 0 | good |
| `sno_noise` | 18,513 | 0.1136 | 0.0259 | 500 | 0 | good |
| `fno_plain` | 25,985 | 0.1358 | 0.0268 | 500 | 0 | good |
| `unet_noise` | 18,785 | 0.173 | 0.0218 | 500 | 0 | good |
| `fno_noise` | 25,985 | 0.2888 | 0.0744 | 410 | 100 | good |
| `skino_seq2seq` | 21,135 | 0.4226 | 0.000477 | 238 | 2 | good |
| `nosymp_noise` | 22,755 | 0.4326 | 0.00218 | 225 | 0 | good |
| `skino_plain` | 25,067 | 0.4393 | 0.00753 | 208 | 2 | good |
| `skino_noise` | 25,067 | 0.4404 | 0.00285 | 210 | 4 | good |
| `strict_noise` | 20,779 | 0.445 | 0.000828 | 195 | 4 | degraded/good |
| `skino_pinn` | 25,067 | 0.4462 | 0.0137 | 208 | 5 | good |
| `strict_plain` | 20,779 | 0.4556 | 0.0122 | 180 | 4 | degraded |
| `transformer_noise` | 19,233 | 0.8393 | 0.0126 | 80 | 0 | degraded |
| `deeponet_noise` | 20,929 | 1.458 | 0.031 | 75 | 0 | degraded |
| `generic_noise` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |

Best: **`tfno_plain`** (0.003963 +/- 0.000979). Significantly worse (gap exceeds combined std): `tfno_seq2seq`, `ufno_plain`, `ufno_seq2seq`, `sno_seq2seq`, `sno_plain`, `tfno_noise`, `ufno_noise`, `fno_seq2seq`, `fno_pinn`, `sno_noise`, `fno_plain`, `unet_noise`, `fno_noise`, `skino_seq2seq`, `nosymp_noise`, `skino_plain`, `skino_noise`, `strict_noise`, `skino_pinn`, `strict_plain`, `transformer_noise`, `deeponet_noise`, `generic_noise`, `generic_plain`.

## kdv  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `sno_seq2seq` | 31,013 | 0.0004666 | 6.32e-05 | 500 | 0 | good |
| `skino_seq2seq` | 21,135 | 0.001649 | 0.000597 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.00279 | 8.73e-05 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.00482 | 0.0027 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.006402 | 0.00092 | 500 | 0 | good |
| `skino_noise` | 25,067 | 0.02967 | 0.00709 | 500 | 0 | good |
| `nosymp_noise` | 22,755 | 0.03968 | 0.012 | 492 | 12 | good |
| `sno_plain` | 18,513 | 0.04743 | 0.00805 | 500 | 0 | good |
| `sno_noise` | 18,513 | 0.05615 | 0.00891 | 480 | 28 | good |
| `tfno_noise` | 28,289 | 0.08138 | 0.0222 | 378 | 102 | good |
| `ufno_noise` | 18,193 | 0.08467 | 0.0447 | 393 | 151 | degraded/good |
| `transformer_noise` | 19,233 | 0.1416 | 0.0109 | 187 | 13 | degraded/good |
| `fno_plain` | 25,985 | 0.1531 | 0.0197 | 195 | 15 | degraded/good |
| `deeponet_noise` | 20,929 | 0.3725 | 0.0837 | 35 | 39 | decorrelated/degraded |
| `unet_noise` | 18,785 | 0.4222 | 0.0359 | 7 | 2 | decorrelated/degraded |
| `ufno_plain` | 18,193 | 1.783 | 2.45 | 363 | 193 | blow-up/good |
| `fno_noise` | 25,985 | 333.5 | 471 | 162 | 78 | blow-up/degraded/good |
| `strict_noise` | 20,779 | 333.5 | 471 | 105 | 59 | blow-up/degraded |
| `tfno_plain` | 28,289 | 335.2 | 470 | 95 | 32 | blow-up |
| `generic_noise` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |
| `skino_plain` | 25,067 | 1000 | 0 | 33 | 2 | blow-up |
| `strict_plain` | 20,779 | 1000 | 0 | 20 | 8 | blow-up |

Best: **`sno_seq2seq`** (0.0004666 +/- 6.32e-05). Significantly worse (gap exceeds combined std): `skino_seq2seq`, `tfno_seq2seq`, `ufno_seq2seq`, `fno_seq2seq`, `skino_noise`, `nosymp_noise`, `sno_plain`, `sno_noise`, `tfno_noise`, `ufno_noise`, `transformer_noise`, `fno_plain`, `deeponet_noise`, `unet_noise`, `generic_noise`, `generic_plain`, `skino_plain`, `strict_plain`.

## wave2d  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `skino_plain` | 25,414 | 0.02887 | 0.00832 | 250 | 0 | good |
| `skino_pinn` | 25,414 | 0.07177 | 0.00699 | 250 | 0 | good |
| `strict_plain` | 21,062 | 0.07288 | 0.00737 | 250 | 0 | good |
| `skino_seq2seq` | 21,418 | 0.08717 | 0.0409 | 250 | 0 | good |
| `skino_noise` | 25,414 | 0.116 | 0.00263 | 250 | 0 | good |
| `strict_noise` | 21,062 | 0.1229 | 0.0156 | 250 | 0 | good |
| `fno_seq2seq` | 25,774 | 0.1536 | 0.0088 | 250 | 0 | good |
| `fno_noise` | 29,402 | 3.436 | 4.29 | 173 | 70 | blow-up/good |
| `fno_pinn` | 29,402 | 6.5 | 8.49 | 125 | 39 | blow-up/degraded |
| `fno_plain` | 29,402 | 666.9 | 471 | 63 | 51 | blow-up/degraded |

Best: **`skino_plain`** (0.02887 +/- 0.00832). Significantly worse (gap exceeds combined std): `skino_pinn`, `strict_plain`, `skino_seq2seq`, `skino_noise`, `strict_noise`, `fno_seq2seq`, `fno_plain`.

## wave3d  (3 seeds, reported at t=150)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `skino_seq2seq` | 24,122 | 0.1386 | 0.0373 | 150 | 0 | good |
| `skino_pinn` | 23,894 | 0.1612 | 0.0432 | 150 | 0 | good |
| `skino_plain` | 23,894 | 0.5104 | 0.572 | 107 | 61 | degraded/good |
| `skino_noise` | 23,894 | 0.5197 | 0.572 | 107 | 61 | degraded/good |
| `fno_seq2seq` | 26,977 | 0.6394 | 0.104 | 2 | 2 | degraded |
| `strict_noise` | 20,502 | 0.8252 | 0.14 | 15 | 0 | degraded |
| `strict_plain` | 20,502 | 0.8331 | 0.137 | 15 | 0 | degraded |
| `fno_pinn` | 25,177 | 6.261 | 1.82 | 5 | 0 | blow-up |
| `fno_noise` | 25,177 | 367.6 | 448 | 22 | 8 | blow-up |
| `fno_plain` | 25,177 | 413.7 | 416 | 22 | 8 | blow-up |

Best: **`skino_seq2seq`** (0.1386 +/- 0.0373). Significantly worse (gap exceeds combined std): `fno_seq2seq`, `strict_noise`, `strict_plain`, `fno_pinn`.

## ns2d  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `skino_seq2seq` | 25,245 | 0.1876 | 0.00194 | 250 | 0 | good |
| `fno_seq2seq` | 22,499 | 0.2658 | 0.00533 | 250 | 0 | good |
| `fno_noise` | 29,381 | 0.3238 | 0.00822 | 250 | 0 | good |
| `fno_plain` | 29,381 | 0.3313 | 0.00998 | 250 | 0 | good |
| `strict_noise` | 24,979 | 0.385 | 0.014 | 240 | 14 | good |
| `strict_plain` | 24,979 | 0.3953 | 0.01 | 232 | 9 | good |
| `skino_noise` | 26,187 | 0.4412 | 0.00603 | 193 | 5 | degraded/good |
| `skino_plain` | 26,187 | 0.451 | 0.00353 | 193 | 2 | degraded |

Best: **`skino_seq2seq`** (0.1876 +/- 0.00194). Significantly worse (gap exceeds combined std): `fno_seq2seq`, `fno_noise`, `fno_plain`, `strict_noise`, `strict_plain`, `skino_noise`, `skino_plain`.
