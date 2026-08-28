# HamiltonianNN — FNO vs SKINO on the 3‑D elastic‑lattice simulator

This folder contains the full code, data, models, plots and a video for a head‑to‑head comparative study of two neural operators on the project's own 3‑D elastic‑lattice simulator [`elm1.py`](elm1.py):

- **FNO3D** (Fourier Neural Operator) — implemented from scratch in [`fno_model.py`](fno_model.py), 5,902,302 parameters.
- **SKINO3D** (Symplectic Kernel‑Integral Neural Operator) — re‑used from the workspace package at [`skino/nd.py`](../skino/nd.py), 8,818 parameters.

Both are trained on the same data, with the same recipe, the same residual learning, and the same train/val/test split.

> **The main comparative report is [`comparative_study.md`](comparative_study.md).**
> **The comparison video is [`video/comparison.mp4`](video/comparison.mp4).**

## Quick start

```powershell
# Anaconda Python 3.13.5 works; PyTorch 2.12.0+cpu, numba 0.61, ffmpeg via conda.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"   # OpenMP runtime is shared between MKL and matplotlib on Windows

python generate_data.py --n-runs 3 --n-steps 600 --save-every 2 --output-dir output2
# FNO uses one-step training; SKINO uses a K=[1,4] push-forward curriculum.
python train.py --epochs 24 --burn-in 20 --stride 2 --skino-unroll-schedule 1 4
# Roll out from the training burn-in step so the network sees an in-distribution
# initial state (step 0 of the simulator has zero state -- the source kick is
# never seen by the network during training and cannot be reproduced from zero).
python rollout.py --test-run 2 --start-step 20 --video-stride 2 --fps 18
python analyze_rollout.py
```

Or run all four stages with [`run_all.py`](run_all.py).

## Layout

| File / folder | What it does |
|---|---|
| [`elm1.py`](elm1.py) | Numba 3‑D velocity‑Verlet elastic‑lattice simulator (input data generator). |
| [`run_elm1.py`](run_elm1.py) | Original wrapper around `elm1.run_simulation` (kept untouched). |
| [`generate_data.py`](generate_data.py) | Deterministic multi‑seed driver that writes `output2/small_run_{0,1,2}.npz`. |
| [`fno_model.py`](fno_model.py) | Self‑contained `SpectralConv3d` + `FNO3D`. |
| [`data_utils.py`](data_utils.py) | Trajectory loading, normalization stats, in‑memory `PairDataset`, `wmape_per_step`. |
| [`train.py`](train.py) | Trains FNO3D *and* SKINO3D with identical hyperparameters, saves checkpoints to `models/`. |
| [`rollout.py`](rollout.py) | Autoregressive rollout, per‑step wMAPE, 3‑panel mp4 with wMAPE labels. |
| [`analyze_rollout.py`](analyze_rollout.py) | Post‑processing: log‑scale wMAPE plot + robust summary tables. |
| [`run_all.py`](run_all.py) | Orchestrator (generate → train → rollout → analyze). |
| [`output2/`](output2) | Simulator output (3 trajectories). |
| [`models/`](models) | Trained checkpoints, `stats.json`, `train_log.json`. |
| [`results/`](results) | Rollout arrays + per‑step metrics + markdown summary. |
| [`figures/`](figures) | wMAPE curves (linear + log‑scale). |
| [`video/`](video) | The comparison animation. |

## Headline numbers

| | FNO3D | SKINO3D |
|---|---:|---:|
| Trainable parameters | 5,902,302 | 8,818 (≈ **669× fewer**) |
| Best one‐step val MSE | 1.64 × 10⁻⁶ | **3.49 × 10⁻⁷** |
| Training wall‐clock (CPU, 24 epochs) | 391 s (one‐step) | 2 383 s (K‐curriculum [1, 4]) |
| 280‐step rollout wall‐clock (CPU) | 23.28 s | **5.65 s** (≈ 4.1× faster) |
| Per‐step rollout inference | 83 ms | **20 ms** |
| Median post‐burn‐in q‐wMAPE (in‐distribution start) | 110.22 % | **101.21 %** |
| Rollout behaviour | **Frozen** at the initial wavefield (outputs ≈ 0 delta) | **Propagates** the wave; tracks decay envelope for the first ~100 ms |

**SKINO actually behaves as a dynamics model**, while FNO collapses to a near‐static prediction at one‐step training. Both numbers were obtained with the in‐distribution rollout protocol (`--start-step 20`) — see [`comparative_study.md`](comparative_study.md) for why starting from step 0 (zero state) is out‐of‐distribution. SKINO is also smaller (669× fewer parameters) and ~4× faster per inference step. See [`figures/comparison_frames.png`](figures/comparison_frames.png) for the static contact sheet and [`video/comparison.mp4`](video/comparison.mp4) for the animation.

### What changed in this iteration

Two training/inference‐recipe changes brought SKINO from a near‐zero prediction to a recognizable wavefield, *without touching the model architecture*:

1. **Push‐forward curriculum for SKINO.** Train one‐step for 12 epochs (`K=1`), then 4‐step unrolled MSE for 12 epochs (`K=4`). The 4‐step phase penalises error accumulation that one‐step training is blind to.
2. **Start the rollout in‐distribution.** The simulator's first 20 saved states are the source‐firing transient (zero → peak amplitude). Training uses `burn_in=20`, so the network has *never* seen the source kick. Starting the rollout at step 20 (where the truth is already excited and lies in the network's training distribution) gives both models a fair chance.


TBD:
1. Work on the simpler version, by removing parameterial