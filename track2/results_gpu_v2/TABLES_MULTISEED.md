# Multi-seed results (mean +/- std over seeds)

Relative RMS at t=200; usable horizon in steps.

## advection  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `ckino_plain` | 25,067 | 0.001092 | 0.000569 | 500 | 0 | good |
| `sno_seq2seq` | 31,013 | 0.001101 | 9.01e-05 | 500 | 0 | good |
| `ckino_seq2seq` | 21,135 | 0.001321 | 0.000441 | 500 | 0 | good |
| `naive_seq2seq` | 28,969 | 0.001508 | 0.000233 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.00257 | 0.000457 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.002903 | 0.000529 | 500 | 0 | good |
| `sacheb_seq2seq` | 28,969 | 0.003556 | 0.000508 | 500 | 0 | good |
| `strict_plain` | 20,779 | 0.007916 | 0.00441 | 500 | 0 | good |
| `naive_plain` | 24,069 | 0.009045 | 0.00258 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.01005 | 0.000529 | 500 | 0 | good |
| `tfno_plain` | 28,289 | 0.01208 | 0.00285 | 500 | 0 | good |
| `ckino_pinn` | 25,067 | 0.01215 | 0.00175 | 500 | 0 | good |
| `ufno_plain` | 18,193 | 0.01372 | 0.000702 | 500 | 0 | good |
| `sno_plain` | 18,513 | 0.01709 | 0.00726 | 500 | 0 | good |
| `fno_pinn` | 25,985 | 0.03261 | 0.000124 | 500 | 0 | good |
| `sacheb_plain` | 24,069 | 0.03618 | 0.0334 | 430 | 99 | good |
| `fno_plain` | 25,985 | 0.04215 | 0.00353 | 500 | 0 | good |
| `tfno_noise` | 28,289 | 0.04416 | 0.00621 | 500 | 0 | good |
| `nosymp_noise` | 22,755 | 0.04797 | 0.0111 | 500 | 0 | good |
| `ckino_noise` | 25,067 | 0.04918 | 0.00354 | 500 | 0 | good |
| `strict_noise` | 20,779 | 0.06225 | 0.00659 | 500 | 0 | good |
| `ufno_noise` | 18,193 | 0.06749 | 0.00971 | 500 | 0 | good |
| `sacheb_noise` | 24,069 | 0.07473 | 0.00958 | 500 | 0 | good |
| `sno_noise` | 18,513 | 0.1088 | 0.0135 | 500 | 0 | good |
| `fno_noise` | 25,985 | 0.233 | 0.011 | 443 | 57 | good |
| `unet_noise` | 18,785 | 0.6005 | 0.00228 | 128 | 2 | degraded |
| `deeponet_noise` | 20,929 | 0.85 | 0.0823 | 33 | 17 | degraded |
| `transformer_noise` | 19,233 | 1.082 | 0.00257 | 10 | 0 | decorrelated |
| `generic_noise` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |

Best: **`ckino_plain`** (0.001092 +/- 0.000569). Significantly worse (gap exceeds combined std): `ufno_seq2seq`, `tfno_seq2seq`, `sacheb_seq2seq`, `strict_plain`, `naive_plain`, `fno_seq2seq`, `tfno_plain`, `ckino_pinn`, `ufno_plain`, `sno_plain`, `fno_pinn`, `sacheb_plain`, `fno_plain`, `tfno_noise`, `nosymp_noise`, `ckino_noise`, `strict_noise`, `ufno_noise`, `sacheb_noise`, `sno_noise`, `fno_noise`, `unet_noise`, `deeponet_noise`, `transformer_noise`, `generic_noise`, `generic_plain`.

## heat  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `tfno_plain` | 28,289 | 0.00109 | 0.000307 | 500 | 0 | good |
| `ckino_plain` | 25,067 | 0.00283 | 0.000911 | 500 | 0 | good |
| `sno_plain` | 18,513 | 0.003091 | 0.00187 | 500 | 0 | good |
| `naive_seq2seq` | 28,969 | 0.003491 | 0.000305 | 500 | 0 | good |
| `ckino_seq2seq` | 21,135 | 0.005026 | 0.00144 | 500 | 0 | good |
| `sacheb_seq2seq` | 28,969 | 0.005565 | 0.000578 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.006429 | 0.000534 | 500 | 0 | good |
| `sno_seq2seq` | 31,013 | 0.006906 | 0.00136 | 500 | 0 | good |
| `strict_plain` | 20,779 | 0.008164 | 0.00254 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.008554 | 0.00171 | 500 | 0 | good |
| `ufno_plain` | 18,193 | 0.008579 | 0.00254 | 500 | 0 | good |
| `naive_plain` | 24,069 | 0.009845 | 0.0035 | 500 | 0 | good |
| `ckino_pinn` | 25,067 | 0.01307 | 0.00525 | 500 | 0 | good |
| `fno_pinn` | 25,985 | 0.01409 | 0.00691 | 500 | 0 | good |
| `sacheb_plain` | 24,069 | 0.01558 | 0.00663 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.01643 | 0.00166 | 500 | 0 | good |
| `fno_plain` | 25,985 | 0.02499 | 0.0121 | 500 | 0 | good |
| `tfno_noise` | 28,289 | 0.03713 | 0.0108 | 500 | 0 | good |
| `ufno_noise` | 18,193 | 0.03966 | 8.06e-05 | 500 | 0 | good |
| `strict_noise` | 20,779 | 0.04322 | 0.0141 | 500 | 0 | good |
| `ckino_noise` | 25,067 | 0.04353 | 0.0104 | 500 | 0 | good |
| `nosymp_noise` | 22,755 | 0.05152 | 0.0158 | 500 | 0 | good |
| `sacheb_noise` | 24,069 | 0.05597 | 0.00316 | 500 | 0 | good |
| `unet_noise` | 18,785 | 0.07705 | 0.00384 | 500 | 0 | good |
| `sno_noise` | 18,513 | 0.2085 | 0.0363 | 497 | 5 | good |
| `fno_noise` | 25,985 | 0.224 | 0.0137 | 500 | 0 | good |
| `transformer_noise` | 19,233 | 0.4814 | 0.0144 | 232 | 5 | good |
| `deeponet_noise` | 20,929 | 0.6895 | 0.00479 | 170 | 4 | degraded |
| `generic_noise` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |

Best: **`tfno_plain`** (0.00109 +/- 0.000307). Significantly worse (gap exceeds combined std): `ckino_plain`, `naive_seq2seq`, `ckino_seq2seq`, `sacheb_seq2seq`, `tfno_seq2seq`, `sno_seq2seq`, `strict_plain`, `ufno_seq2seq`, `ufno_plain`, `naive_plain`, `ckino_pinn`, `fno_pinn`, `sacheb_plain`, `fno_seq2seq`, `fno_plain`, `tfno_noise`, `ufno_noise`, `strict_noise`, `ckino_noise`, `nosymp_noise`, `sacheb_noise`, `unet_noise`, `sno_noise`, `fno_noise`, `transformer_noise`, `deeponet_noise`, `generic_noise`, `generic_plain`.

## wave1d  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `sno_seq2seq` | 25,282 | 0.001298 | 0.000164 | 500 | 0 | good |
| `naive_seq2seq` | 25,410 | 0.00153 | 0.000135 | 500 | 0 | good |
| `tfno_seq2seq` | 21,514 | 0.002244 | 0.000223 | 500 | 0 | good |
| `ckino_seq2seq` | 25,614 | 0.002591 | 0.000231 | 500 | 0 | good |
| `ufno_seq2seq` | 22,082 | 0.002681 | 0.000326 | 500 | 0 | good |
| `sacheb_seq2seq` | 25,410 | 0.005176 | 0.000144 | 500 | 0 | good |
| `sno_plain` | 18,562 | 0.01073 | 0.00367 | 500 | 0 | good |
| `fno_seq2seq` | 27,666 | 0.01682 | 0.00083 | 500 | 0 | good |
| `pureunif_plain` | 21,008 | 0.08809 | 0.00748 | 500 | 0 | good |
| `tfno_noise` | 26,338 | 0.1027 | 0.0221 | 500 | 0 | good |
| `naive_plain` | 24,938 | 0.1334 | 0.016 | 500 | 0 | good |
| `ufno_plain` | 18,218 | 0.1621 | 0.00974 | 452 | 52 | good |
| `sacheb_plain` | 24,938 | 0.1876 | 0.0239 | 470 | 42 | good |
| `ckino_plain` | 21,110 | 0.1956 | 0.0579 | 390 | 80 | good |
| `ufno_noise` | 18,218 | 0.2043 | 0.0257 | 442 | 41 | good |
| `ckino_pinn` | 21,110 | 0.2182 | 0.0095 | 365 | 41 | good |
| `sno_noise` | 18,562 | 0.2248 | 0.0304 | 425 | 58 | good |
| `sacheb_noise` | 24,938 | 0.3443 | 0.0104 | 280 | 21 | degraded/good |
| `strict_plain` | 18,870 | 0.3797 | 0.182 | 295 | 146 | degraded/good |
| `strict_noise` | 18,870 | 0.4759 | 0.00695 | 112 | 2 | degraded |
| `ckino_noise` | 21,110 | 0.4781 | 0.0708 | 130 | 63 | degraded |
| `nosymp_noise` | 20,814 | 0.494 | 0.0229 | 82 | 21 | degraded |
| `unet_noise` | 18,914 | 0.5252 | 0.007 | 98 | 16 | degraded |
| `fno_pinn` | 26,018 | 0.5294 | 0.112 | 118 | 45 | degraded/good |
| `fno_noise` | 26,018 | 0.5918 | 0.00979 | 63 | 2 | degraded |
| `fno_plain` | 26,018 | 0.7167 | 0.145 | 117 | 6 | degraded |
| `deeponet_noise` | 25,090 | 0.8725 | 0.184 | 47 | 22 | decorrelated/degraded |
| `transformer_noise` | 18,274 | 1.108 | 0.125 | 18 | 2 | decorrelated |
| `purecheb_plain` | 21,008 | 1.518 | 0.0109 | 15 | 0 | decorrelated |
| `tfno_plain` | 26,338 | 7.318 | 10.3 | 323 | 154 | blow-up/good |
| `generic_noise` | 31,747 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 31,747 | 1000 | 0 | 0 | 0 | blow-up |

Best: **`sno_seq2seq`** (0.001298 +/- 0.000164). Significantly worse (gap exceeds combined std): `tfno_seq2seq`, `ckino_seq2seq`, `ufno_seq2seq`, `sacheb_seq2seq`, `sno_plain`, `fno_seq2seq`, `pureunif_plain`, `tfno_noise`, `naive_plain`, `ufno_plain`, `sacheb_plain`, `ckino_plain`, `ufno_noise`, `ckino_pinn`, `sno_noise`, `sacheb_noise`, `strict_plain`, `strict_noise`, `ckino_noise`, `nosymp_noise`, `unet_noise`, `fno_pinn`, `fno_noise`, `fno_plain`, `deeponet_noise`, `transformer_noise`, `purecheb_plain`, `generic_noise`, `generic_plain`.

## wave1d_dir  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `naive_seq2seq` | 21,038 | 0.02739 | 0.00112 | 500 | 0 | good |
| `ckino_seq2seq` | 29,710 | 0.03538 | 0.000262 | 500 | 0 | good |
| `sacheb_seq2seq` | 21,038 | 0.03667 | 0.00468 | 500 | 0 | good |
| `strict_noise` | 20,918 | 0.07382 | 0.0092 | 500 | 0 | good |
| `ufno_seq2seq` | 22,082 | 0.08072 | 0.00252 | 500 | 0 | good |
| `pureunif_plain` | 28,044 | 0.08242 | 0.00374 | 500 | 0 | good |
| `fno_seq2seq` | 27,666 | 0.08287 | 0.00451 | 500 | 0 | good |
| `nosymp_noise` | 22,862 | 0.0839 | 0.0161 | 500 | 0 | good |
| `sacheb_noise` | 24,110 | 0.08757 | 0.0136 | 500 | 0 | good |
| `ckino_noise` | 25,206 | 0.0878 | 0.00935 | 500 | 0 | good |
| `unet_noise` | 18,914 | 0.09336 | 0.0143 | 500 | 0 | good |
| `strict_plain` | 20,918 | 0.09457 | 0.0112 | 398 | 73 | good |
| `purecheb_plain` | 28,044 | 0.102 | 0.0122 | 500 | 0 | good |
| `sno_seq2seq` | 25,282 | 0.135 | 0.0256 | 500 | 0 | good |
| `tfno_seq2seq` | 21,514 | 0.1471 | 0.0204 | 500 | 0 | good |
| `ckino_plain` | 25,206 | 0.1685 | 0.017 | 288 | 12 | good |
| `ufno_noise` | 18,218 | 0.178 | 0.0345 | 467 | 47 | good |
| `transformer_noise` | 19,298 | 0.2171 | 0.0117 | 300 | 4 | good |
| `fno_noise` | 26,018 | 0.2365 | 0.0392 | 307 | 39 | good |
| `deeponet_noise` | 29,186 | 0.237 | 0.0233 | 308 | 142 | good |
| `tfno_noise` | 28,386 | 0.2832 | 0.0803 | 242 | 58 | degraded/good |
| `sno_noise` | 18,562 | 0.331 | 0.0656 | 183 | 70 | degraded/good |
| `sno_plain` | 18,562 | 0.4516 | 0.0333 | 148 | 2 | degraded |
| `ckino_pinn` | 25,206 | 2.234 | 0.597 | 23 | 2 | blow-up/decorrelated/degraded |
| `fno_plain` | 26,018 | 2.47 | 2.77 | 130 | 43 | blow-up/degraded |
| `ufno_plain` | 18,218 | 15.65 | 18.2 | 128 | 57 | blow-up/degraded |
| `tfno_plain` | 28,386 | 68.35 | 90.3 | 85 | 25 | blow-up/degraded |
| `fno_pinn` | 26,018 | 1000 | 0 | 7 | 2 | blow-up |
| `generic_noise` | 17,091 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 17,091 | 1000 | 0 | 0 | 0 | blow-up |
| `naive_plain` | 24,110 | 1000 | 0 | 33 | 15 | blow-up |
| `sacheb_plain` | 24,110 | 1000 | 0 | 33 | 12 | blow-up |

Best: **`naive_seq2seq`** (0.02739 +/- 0.00112). Significantly worse (gap exceeds combined std): `ckino_seq2seq`, `sacheb_seq2seq`, `strict_noise`, `ufno_seq2seq`, `pureunif_plain`, `fno_seq2seq`, `nosymp_noise`, `sacheb_noise`, `ckino_noise`, `unet_noise`, `strict_plain`, `purecheb_plain`, `sno_seq2seq`, `tfno_seq2seq`, `ckino_plain`, `ufno_noise`, `transformer_noise`, `fno_noise`, `deeponet_noise`, `tfno_noise`, `sno_noise`, `sno_plain`, `ckino_pinn`, `fno_pinn`, `generic_noise`, `generic_plain`, `naive_plain`, `sacheb_plain`.

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
| `naive_seq2seq` | 28,969 | 0.08567 | 0.00622 | 500 | 0 | good |
| `sacheb_seq2seq` | 28,969 | 0.09196 | 0.00802 | 500 | 0 | good |
| `fno_pinn` | 25,985 | 0.1088 | 0.0114 | 500 | 0 | good |
| `naive_plain` | 24,069 | 0.1096 | 0.0299 | 500 | 0 | good |
| `sno_noise` | 18,513 | 0.1136 | 0.0259 | 500 | 0 | good |
| `sacheb_plain` | 24,069 | 0.1222 | 0.0293 | 500 | 0 | good |
| `fno_plain` | 25,985 | 0.1358 | 0.0268 | 500 | 0 | good |
| `unet_noise` | 18,785 | 0.173 | 0.0218 | 500 | 0 | good |
| `sacheb_noise` | 24,069 | 0.1771 | 0.0067 | 500 | 0 | good |
| `fno_noise` | 25,985 | 0.2888 | 0.0744 | 410 | 100 | good |
| `ckino_seq2seq` | 21,135 | 0.4226 | 0.000477 | 238 | 2 | good |
| `nosymp_noise` | 22,755 | 0.4326 | 0.00218 | 225 | 0 | good |
| `ckino_plain` | 25,067 | 0.4393 | 0.00753 | 208 | 2 | good |
| `ckino_noise` | 25,067 | 0.4404 | 0.00285 | 210 | 4 | good |
| `strict_noise` | 20,779 | 0.445 | 0.000828 | 195 | 4 | degraded/good |
| `ckino_pinn` | 25,067 | 0.4462 | 0.0137 | 208 | 5 | good |
| `strict_plain` | 20,779 | 0.4556 | 0.0122 | 180 | 4 | degraded |
| `transformer_noise` | 19,233 | 0.8393 | 0.0126 | 80 | 0 | degraded |
| `deeponet_noise` | 20,929 | 1.458 | 0.031 | 75 | 0 | degraded |
| `generic_noise` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |
| `generic_plain` | 28,467 | 1000 | 0 | 0 | 0 | blow-up |

Best: **`tfno_plain`** (0.003963 +/- 0.000979). Significantly worse (gap exceeds combined std): `tfno_seq2seq`, `ufno_plain`, `ufno_seq2seq`, `sno_seq2seq`, `sno_plain`, `tfno_noise`, `ufno_noise`, `fno_seq2seq`, `naive_seq2seq`, `sacheb_seq2seq`, `fno_pinn`, `naive_plain`, `sno_noise`, `sacheb_plain`, `fno_plain`, `unet_noise`, `sacheb_noise`, `fno_noise`, `ckino_seq2seq`, `nosymp_noise`, `ckino_plain`, `ckino_noise`, `strict_noise`, `ckino_pinn`, `strict_plain`, `transformer_noise`, `deeponet_noise`, `generic_noise`, `generic_plain`.

## kdv  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `sno_seq2seq` | 31,013 | 0.0004666 | 6.32e-05 | 500 | 0 | good |
| `naive_seq2seq` | 28,969 | 0.000627 | 0.000101 | 500 | 0 | good |
| `ckino_seq2seq` | 21,135 | 0.001649 | 0.000597 | 500 | 0 | good |
| `sacheb_seq2seq` | 28,969 | 0.002693 | 0.000528 | 500 | 0 | good |
| `tfno_seq2seq` | 30,325 | 0.00279 | 8.73e-05 | 500 | 0 | good |
| `ufno_seq2seq` | 24,693 | 0.00482 | 0.0027 | 500 | 0 | good |
| `fno_seq2seq` | 22,197 | 0.006402 | 0.00092 | 500 | 0 | good |
| `ckino_noise` | 25,067 | 0.02967 | 0.00709 | 500 | 0 | good |
| `naive_plain` | 24,069 | 0.03177 | 0.00199 | 500 | 0 | good |
| `sacheb_plain` | 24,069 | 0.03358 | 0.00139 | 500 | 0 | good |
| `nosymp_noise` | 22,755 | 0.03968 | 0.012 | 492 | 12 | good |
| `sno_plain` | 18,513 | 0.04743 | 0.00805 | 500 | 0 | good |
| `sacheb_noise` | 24,069 | 0.0513 | 0.00646 | 500 | 0 | good |
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
| `ckino_plain` | 25,067 | 1000 | 0 | 33 | 2 | blow-up |
| `strict_plain` | 20,779 | 1000 | 0 | 20 | 8 | blow-up |

Best: **`sno_seq2seq`** (0.0004666 +/- 6.32e-05). Significantly worse (gap exceeds combined std): `ckino_seq2seq`, `sacheb_seq2seq`, `tfno_seq2seq`, `ufno_seq2seq`, `fno_seq2seq`, `ckino_noise`, `naive_plain`, `sacheb_plain`, `nosymp_noise`, `sno_plain`, `sacheb_noise`, `sno_noise`, `tfno_noise`, `ufno_noise`, `transformer_noise`, `fno_plain`, `deeponet_noise`, `unet_noise`, `generic_noise`, `generic_plain`, `ckino_plain`, `strict_plain`.

## wave2d  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `naive_seq2seq` | 21,134 | 0.007771 | 0.00154 | 250 | 0 | good |
| `sno_seq2seq` | 27,022 | 0.008073 | 0.00153 | 250 | 0 | good |
| `naive_plain` | 24,302 | 0.01127 | 0.000591 | 250 | 0 | good |
| `sno_plain` | 18,522 | 0.01444 | 0.00406 | 250 | 0 | good |
| `sacheb_plain` | 24,302 | 0.02337 | 0.00319 | 250 | 0 | good |
| `ckino_plain` | 25,414 | 0.02887 | 0.00832 | 250 | 0 | good |
| `pureunif_plain` | 28,428 | 0.04316 | 0.00254 | 250 | 0 | good |
| `sacheb_seq2seq` | 21,134 | 0.06249 | 0.017 | 250 | 0 | good |
| `ckino_pinn` | 25,414 | 0.07177 | 0.00699 | 250 | 0 | good |
| `strict_plain` | 21,062 | 0.07288 | 0.00737 | 250 | 0 | good |
| `sacheb_noise` | 24,302 | 0.07944 | 0.0221 | 250 | 0 | good |
| `ckino_seq2seq` | 21,418 | 0.08717 | 0.0409 | 250 | 0 | good |
| `ckino_noise` | 25,414 | 0.116 | 0.00263 | 250 | 0 | good |
| `sno_noise` | 18,522 | 0.1174 | 0.0117 | 250 | 0 | good |
| `strict_noise` | 21,062 | 0.1229 | 0.0156 | 250 | 0 | good |
| `fno_seq2seq` | 25,774 | 0.1536 | 0.0088 | 250 | 0 | good |
| `purecheb_plain` | 28,428 | 0.9332 | 0.0204 | 40 | 0 | degraded |
| `fno_noise` | 29,402 | 3.436 | 4.29 | 173 | 70 | blow-up/good |
| `fno_pinn` | 29,402 | 6.5 | 8.49 | 125 | 39 | blow-up/degraded |
| `fno_plain` | 29,402 | 666.9 | 471 | 63 | 51 | blow-up/degraded |

Best: **`naive_seq2seq`** (0.007771 +/- 0.00154). Significantly worse (gap exceeds combined std): `naive_plain`, `sno_plain`, `sacheb_plain`, `ckino_plain`, `pureunif_plain`, `sacheb_seq2seq`, `ckino_pinn`, `strict_plain`, `sacheb_noise`, `ckino_seq2seq`, `ckino_noise`, `sno_noise`, `strict_noise`, `fno_seq2seq`, `purecheb_plain`, `fno_plain`.

## wave3d  (3 seeds, reported at t=150)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `pureunif_plain` | 22,668 | 0.07237 | 0.00888 | 150 | 0 | good |
| `naive_seq2seq` | 24,198 | 0.07392 | 0.0179 | 150 | 0 | good |
| `naive_plain` | 23,746 | 0.0998 | 0.00112 | 150 | 0 | good |
| `sacheb_seq2seq` | 24,198 | 0.1201 | 0.0187 | 150 | 0 | good |
| `ckino_seq2seq` | 24,122 | 0.1386 | 0.0373 | 150 | 0 | good |
| `sacheb_plain` | 23,746 | 0.158 | 0.072 | 150 | 0 | good |
| `ckino_pinn` | 23,894 | 0.1612 | 0.0432 | 150 | 0 | good |
| `sacheb_noise` | 23,746 | 0.1802 | 0.0827 | 150 | 0 | good |
| `sno_seq2seq` | 22,402 | 0.3476 | 0.00119 | 35 | 0 | good |
| `ckino_plain` | 23,894 | 0.5104 | 0.572 | 107 | 61 | degraded/good |
| `ckino_noise` | 23,894 | 0.5197 | 0.572 | 107 | 61 | degraded/good |
| `fno_seq2seq` | 26,977 | 0.6394 | 0.104 | 2 | 2 | degraded |
| `strict_noise` | 20,502 | 0.8252 | 0.14 | 15 | 0 | degraded |
| `strict_plain` | 20,502 | 0.8331 | 0.137 | 15 | 0 | degraded |
| `purecheb_plain` | 22,668 | 1.027 | 0.0909 | 40 | 0 | degraded |
| `sno_noise` | 18,502 | 1.102 | 0.00091 | 35 | 0 | degraded |
| `sno_plain` | 18,502 | 1.142 | 0.000952 | 35 | 0 | decorrelated |
| `fno_pinn` | 25,177 | 6.261 | 1.82 | 5 | 0 | blow-up |
| `fno_noise` | 25,177 | 367.6 | 448 | 22 | 8 | blow-up |
| `fno_plain` | 25,177 | 413.7 | 416 | 22 | 8 | blow-up |

Best: **`pureunif_plain`** (0.07237 +/- 0.00888). Significantly worse (gap exceeds combined std): `naive_plain`, `sacheb_seq2seq`, `ckino_seq2seq`, `sacheb_plain`, `ckino_pinn`, `sacheb_noise`, `sno_seq2seq`, `fno_seq2seq`, `strict_noise`, `strict_plain`, `purecheb_plain`, `sno_noise`, `sno_plain`, `fno_pinn`.

## ns2d  (3 seeds, reported at t=200)

| config | params | RMS mean | RMS std | UH mean | UH std | verdicts |
|---|---:|---:|---:|---:|---:|---|
| `naive_seq2seq` | 25,043 | 0.1745 | 0.00144 | 250 | 0 | good |
| `sacheb_seq2seq` | 25,043 | 0.1861 | 0.00488 | 250 | 0 | good |
| `ckino_seq2seq` | 25,245 | 0.1876 | 0.00194 | 250 | 0 | good |
| `sno_seq2seq` | 22,739 | 0.2526 | 5.38e-05 | 250 | 0 | good |
| `sno_noise` | 18,489 | 0.2539 | 0.00178 | 250 | 0 | good |
| `sno_plain` | 18,489 | 0.2582 | 0.00251 | 250 | 0 | good |
| `fno_seq2seq` | 22,499 | 0.2658 | 0.00533 | 250 | 0 | good |
| `fno_noise` | 29,381 | 0.3238 | 0.00822 | 250 | 0 | good |
| `fno_plain` | 29,381 | 0.3313 | 0.00998 | 250 | 0 | good |
| `sacheb_noise` | 28,869 | 0.3439 | 0.0493 | 240 | 14 | good |
| `sacheb_plain` | 28,869 | 0.3604 | 0.0493 | 235 | 21 | good |
| `naive_plain` | 28,869 | 0.3764 | 0.0461 | 232 | 26 | degraded/good |
| `strict_noise` | 24,979 | 0.385 | 0.014 | 240 | 14 | good |
| `strict_plain` | 24,979 | 0.3953 | 0.01 | 232 | 9 | good |
| `ckino_noise` | 26,187 | 0.4412 | 0.00603 | 193 | 5 | degraded/good |
| `ckino_plain` | 26,187 | 0.451 | 0.00353 | 193 | 2 | degraded |

Best: **`naive_seq2seq`** (0.1745 +/- 0.00144). Significantly worse (gap exceeds combined std): `sacheb_seq2seq`, `ckino_seq2seq`, `sno_seq2seq`, `sno_noise`, `sno_plain`, `fno_seq2seq`, `fno_noise`, `fno_plain`, `sacheb_noise`, `sacheb_plain`, `naive_plain`, `strict_noise`, `strict_plain`, `ckino_noise`, `ckino_plain`.
