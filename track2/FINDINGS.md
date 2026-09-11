# Track 2 findings — making CKINO usable at autoregressive rollout

Research program directed by **Dr Gareth O'Brien**. This document is updated as
experiments complete. Numeric result tables are populated from the JSON written
under [`results/`](results); anything not yet measured is marked _pending_.

---

## 1. Premise and goal

The 3-D elastic-lattice seismic study established that **both FNO and CKINO fail
the long-horizon autoregressive rollout** (~100 % wMAPE — no better than
predicting no motion). Ranking two failing models is not meaningful. Track 2
therefore drops the head-to-head framing and asks a single **absolute** question
on deliberately simplified, smooth problems:

> Can CKINO be trained to roll out to a *useful* horizon at all — and which
> ingredients extend that horizon?

Success is defined against an absolute bar (relative RMS staying under a
threshold for N steps, and beating a trivial persistence predictor), **not**
against FNO.

## 2. Execution order (as instructed)

Phase 1 → items **1, 2, 4, 5, 6**; Phase 2 → items **3, 7**.

| # | Ingredient | Where implemented |
|---|---|---|
| 1 | Noise injection into training inputs | [`train.py`](train.py) `_unroll_loss` |
| 2 | Homogeneous / smooth simplification | [`pde_solvers.py`](pde_solvers.py) (constant-c wave, KdV) |
| 3 | RMS-vs-horizon / breakdown-point eval | [`evaluate.py`](evaluate.py) |
| 4 | Energy-trajectory-matching penalty | [`train.py`](train.py) `_rel_energy_error` |
| 5 | Extended push-forward horizon K | [`train.py`](train.py) K-curriculum |
| 6 | Two-step input stencil (u_{t-1}, u_t)→u_{t+1} | [`data.py`](data.py) window rule + `stencil` |
| 7 | Teacher forcing + recursive loss | [`train.py`](train.py) scheduled sampling |

Note on ordering: item 3 (evaluation) is formally Phase 2, but a basic
relative-RMS measurement is used throughout Phase 1 to tell whether ingredients
1/4/5/6 help. The formal breakdown-point protocol and this write-up are the
Phase-2 deliverable.

## 3. Testbeds — smooth PDEs

Two smooth equations (the mentor asked for *different equations*), both on the
periodic domain `[-1, 1)`:

| Problem | Equation | Type | Grid N | macro dt | Invariant |
|---|---|---|---:|---:|---|
| `wave1d` | u_tt = c² u_xx  (c=1) | linear, **conservative** | 32 | 0.02 | H = ½∫(v² + c²u_x²) |
| `kdv` | u_t + 6u u_x − u_xxx = 0 | nonlinear, solitons | 64 | 0.001 | ∫(u_x² − 2u³), mass ∫u |

`wave1d` is simultaneously the **homogeneous-velocity** simplification (item 2):
constant wave speed, no heterogeneity — exactly "isolate the learning dynamics
before introducing heterogeneity."

Reference-solver fidelity (200-step rollout of the ground-truth integrators,
[`pde_solvers.py`](pde_solvers.py) self-test):

| Problem | mean rel. energy drift | mass drift |
|---|---:|---:|
| wave1d | 0.023 % | — |
| kdv | 0.262 % | 2.5 × 10⁻⁷ |

So the "truth" the operators are trained against conserves its invariants to
sub-percent accuracy; any large energy error at rollout is the operator's, not
the reference solver's.

## 4. Sampling / downsampling discipline (mentor's caution)

Handled explicitly and documented in code:

* **Temporal.** The operator only ever learns the map at the fixed macro-step
  `dt`. The reference integrator subdivides `dt` into `n_inner` micro-steps for
  accuracy; the network never sees them. Rollout uses exactly `dt` per step, so
  train and eval share one temporal lattice.
* **Spatial band-limiting.** Initial conditions use ≤3 (wave) / ≤4 (KdV) Fourier
  modes, well below the grid Nyquist (16 / 32), so nothing is aliased at t=0.
* **Anti-aliased resampling.** Any change of resolution goes through
  `spectral_resample`, which low-pass filters in the rFFT domain before
  decimating (never plain striding). Verified round-trip error 1.2 × 10⁻⁷.
* **No split leakage.** Train / val / test initial conditions come from seed
  ranges separated by 10 000 ([`data.py`](data.py) `build_data`).
* **Channel-balanced normalisation.** Fields are normalised per channel to unit
  RMS using train statistics only. On `wave1d` the raw channel scales are
  roughly `[u≈0.15, v≈1.14]` (~7.5×), reproducing the "q and p live on very
  different scales" imbalance flagged in `elm1.py`; without per-channel
  normalisation the MSE would be dominated by the larger channel.

## 5. Ablation design

Four configurations, all sharing the same total epoch budget and the same
train/val/test trajectories, so any difference is attributable to the recipe:

| Config | K-curriculum | teacher forcing | noise | energy penalty | stencil |
|---|---|---|---|---|---|
| `baseline` | K=1 only | off | 0 | 0 | 1 |
| `pushforward` | 1→2→4→8 | annealed 1→0 | 0 | 0 | 1 |
| `track2` | 1→2→4→8 | annealed 1→0 | 0.02 | 0.1 | 1 |
| `track2_2step` | 1→2→4→8 | annealed 1→0 | 0.02 | 0.1 | 2 |

The push-forward LR uses a **non-zero cosine floor** (`eta_min>0`); decaying the
LR fully to zero at the end of a high-K phase is what collapsed the K=8 seismic
run.

## 6. Evaluation protocol (item 3)

On the held-out test trajectories, from a truth seed, free-running rollout (no
noise, no teacher forcing):

* **relative RMS per step**  ‖pred−truth‖ / ‖truth‖  (normalised units).
* **relative energy error per step**  |E_pred−E_true| / |E_true|  (physical).
* **persistence reference** — freeze the seed state; the step where the model's
  RMS exceeds persistence is where it stops beating "nothing moves."
* **breakdown horizon** — first step at which relative RMS crosses 5/10/20/50/100 %.
  Larger = rolls out longer at the same accuracy.

## 7. Results

### 7.1 wave1d — 150-step rollout (24 epochs, 32 train trajectories, N=32)

| config | params | one-step val MSE | RMS@10 | RMS@150 | energy err@150 | breakdown (RMS>5%) | train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 21,110 | **1.8 × 10⁻⁸** | 0.002 | **0.014** | **0.003** | never (>150) | 102 s |
| `pushforward` | 21,110 | **1.45 × 10⁻⁸** | 0.001 | **0.010** | 0.003 | never (>150) | 271 s |
| `track2` (noise+energy) | 21,110 | 2.1 × 10⁻⁶ | 0.014 | 0.174 | 0.236 | **30** | 213 s |
| `track2_2step` | 21,322 | 6.9 × 10⁻⁷ | 0.006 | 0.061 | 0.069 | 109 | 175 s |

Plot: [`results/ablation_wave1d.png`](results/ablation_wave1d.png). Raw: [`results/ablation_wave1d.json`](results/ablation_wave1d.json).

Reading the table:
* **Plain `baseline` CKINO already rolls out the full 150 steps at 1.4 % relative
  RMS and 0.3 % energy error** — it never crosses even the 5 % breakdown line. On
  a clean smooth Hamiltonian PDE, CKINO's rollout is simply *not broken*.
* `pushforward` (items 5, 7) is a hair better on the one-step metric and RMS@150
  but essentially tied — at 2.7× the training cost. The drift it is designed to
  cure is nearly absent here.
* `track2` (adding noise + energy penalty, items 1, 4) is **markedly worse**: it
  lifts the one-step floor ~100× (1.8 × 10⁻⁸ → 2.1 × 10⁻⁶) and RMS@150 12×, and
  now breaks down at step 30.
* `track2_2step` (item 6) claws back most of that damage (val 6.9 × 10⁻⁷) but
  still does not beat `baseline`.

> **Caveat on the persistence metric.** All configs "beat persistence until
> step 100" identically — an artefact of wave *recurrence*: the wave equation is
> temporally periodic with period ≈ 2/c = 100 steps on this domain, so the field
> returns near its initial condition at step 100 and the frozen-IC reference
> coincidentally matches. The RMS-breakdown horizon is the reliable metric here.

### 7.2 kdv — 150-step rollout (24 epochs, 32 train trajectories, N=64)

| config | params | one-step val MSE | RMS@10 | RMS@150 | breakdown (RMS>5%) | rollout verdict |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 25,067 | **3.0 × 10⁻¹⁰** | 0.000 | **4.3 × 10¹²** | 41 | **diverges** |
| `pushforward` | 25,067 | 8.1 × 10⁻¹⁰ | 0.000 | **4.3 × 10¹¹** | 46 | **diverges** |
| `track2` (noise+energy) | 25,067 | 1.2 × 10⁻⁴ | 0.059 | **0.288** | 9 | bounded; beats persistence to step 149 |
| `track2_2step` | 25,173 | 1.3 × 10⁻⁴ | 0.040 | **0.270** | 13 | bounded; beats persistence to step 150 |

Plot: [`results/ablation_kdv.png`](results/ablation_kdv.png). Raw: [`results/ablation_kdv.json`](results/ablation_kdv.json).

**The wave1d picture inverts completely.** On nonlinear KdV the un-stabilised
configs (`baseline`, `pushforward`) achieve the *best one-step accuracy ever
seen here* (val MSE 3 × 10⁻¹⁰ — better even than on wave) yet their
autoregressive rollout **diverges to ~10¹²× the signal magnitude** (energy
overflows to NaN). Both stay accurate until ~step 40, then the nonlinear term
amplifies the tiny per-step error until numerical blow-up. The noise-injected
configs sacrifice ~6 orders of one-step accuracy but their rollout stays
**bounded** (~27 % RMS, never diverging) and beats trivial persistence for the
entire horizon.

## 8. Findings

**W1 — CKINO's rollout is not structurally broken.** On a smooth, linear,
well-resolved Hamiltonian PDE, plain one-step CKINO (21 k parameters) rolls out
150 steps at < 1.5 % relative RMS and 0.3 % energy error, decisively clearing any
absolute usefulness bar. The 3-D seismic rollout collapse was therefore a
property of *that hard problem* — heterogeneous velocity, under-resolution,
dissipation, only two training trajectories — not a "CKINO cannot roll out"
property. This is the single most important Phase-1 result.

**W2 — On an easy regime the push-forward curriculum barely helps.** `pushforward`
matches `baseline` within noise at 2.7× the cost. Multi-step training earns its
keep only when single-step error is large enough to accumulate; here it is not.

**W3 — Stabilizers must be calibrated to the problem's natural rollout error.**
Noise injection at 2 % of field RMS and an energy penalty of weight 0.1 *hurt* on
wave1d, because the model's intrinsic rollout drift is ~0.01 % — two orders of
magnitude smaller than the injected noise. The "medicine" (items 1, 4) is dosed
for a disease (large drift) that this clean problem does not have. Blind
application is harmful; the noise/energy strength should scale with the measured
single-step error, not be a fixed constant.

**W4 — The two-step stencil is a partial antidote to over-strong noise.** Feeding
(uₜ₋₁, uₜ) recovers most of the accuracy the noise floor destroyed (one-step val
6.9 × 10⁻⁷ vs 2.1 × 10⁻⁶), consistent with the extra state giving the network a
finite-difference velocity estimate — but it does not surpass the un-noised
baseline on this problem.

**Implication for the next experiments.** The stabilizers should prove their
worth where the single-step map is genuinely inaccurate: the *nonlinear* KdV
equation (§7.2) and the homogeneous seismic problem. The wave result tells us to
(a) treat `baseline`/`pushforward` as the references, and (b) reduce or adapt the
noise and energy weights rather than keep them fixed at 0.02 / 0.1.

**K1 — On nonlinear KdV the picture inverts: the stabilizers become essential.**
`baseline` and `pushforward` diverge to 10¹¹–10¹²× the signal despite the best
one-step accuracy measured anywhere in this study. Adding noise injection makes
the rollout bounded and persistence-beating. Exactly the regime the mentor's
noise-injection instruction was aimed at.

**K2 — One-step validation MSE does not predict rollout stability.** On KdV the
model with the *best* one-step MSE (`baseline`, 3 × 10⁻¹⁰) had the *worst*
rollout (divergence). A one-step metric can be six-plus orders of magnitude
"better" while hiding a catastrophic rollout instability. Rollout-based
evaluation (this module) is therefore not optional — it is the only metric that
sees the failure.

**K3 — The push-forward curriculum alone does not tame a violent nonlinear
instability; noise does.** `pushforward` (K up to 8, no noise) still diverged.
Multi-step training and noise injection are *not* interchangeable: the curriculum
lengthens the horizon the loss can see, but only the input noise regularises the
learned operator's off-manifold (expanding-Jacobian) behaviour that drives blow-up.

**Synthesis — the stabilizers are a regime-dependent remedy, not a universal
good.**

| | rollout WITHOUT stabilizers | effect of noise+energy |
|---|---|---|
| `wave1d` (linear, stable) | already excellent (1.4 % RMS @150) | **hurts** — adds a ~100× error floor |
| `kdv` (nonlinear, unstable) | **diverges** (10¹²×) | **rescues** — bounded, persistence-beating |

The single principle that fits both: **noise injection trades one-step accuracy
for rollout stability, and is worth it exactly when — and only when — the
un-stabilised rollout is unstable.** The dose must match the instability: 2 %
noise is far too much for the already-stable wave equation and about right (for
boundedness) on KdV. The calibration below (§7.3) isolates the stabilizer and
finds the minimum effective dose.

### 7.3 kdv noise calibration — isolating the stabilizer

Same K=[1,2,4,8] curriculum and data as §7.2; only (noise, energy λ) vary, with
`λ=0` on the noise ladder so noise is the sole stabilizer.

| config | noise | energy λ | one-step val | RMS@10 | RMS@150 | bounded? | breakdown (RMS>5%) | beats persist. |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|
| none (control) | 0 | 0 | 8.1 × 10⁻¹⁰ | 0.000 | 4.3 × 10¹¹ | ✗ | 46 | 53 |
| energy-only | 0 | 0.1 | 4.6 × 10⁻⁷ | 0.002 | 1.5 × 10¹² | ✗ | 27 | 35 |
| noise 0.005 | 0.005 | 0 | 4.6 × 10⁻⁷ | 0.002 | 5.6 × 10¹⁰ | ✗ | 28 | 37 |
| noise 0.010 | 0.010 | 0 | 1.2 × 10⁻⁶ | 0.004 | 1.8 × 10⁵ | ✗ | 41 | 54 |
| **noise 0.020** | 0.020 | 0 | 2.6 × 10⁻⁶ | 0.004 | **0.058** | **✓** | **127** | **151** |

Plot: [`results/calibrate_kdv.png`](results/calibrate_kdv.png). Raw: [`results/calibrate_kdv.json`](results/calibrate_kdv.json).

**C1 — Input noise is the stabilizer; the energy penalty is not.** With noise
off, the energy penalty alone diverged to 1.5 × 10¹² — *worse* than the
no-stabilizer control. Item 4 does not tame the instability.

**C2 — There is a noise-magnitude threshold for stability.** RMS@150 falls
monotonically as noise rises (4.3 × 10¹¹ → 5.6 × 10¹⁰ → 1.8 × 10⁵ → 0.058); only
noise = 0.02 gives a bounded rollout. Below that the blow-up is merely delayed.

**C3 — The energy penalty is counter-productive here; noise-only is best.**
Noise-only 0.02 (RMS@150 = 0.058, accurate to < 5 % until step 127) dramatically
beats the §7.2 `track2` config that *added* the energy penalty (RMS@150 = 0.288,
breakdown at step 9). Dropping item 4 turned a mediocre bounded rollout into a
genuinely good one — CKINO rolling out KdV usefully for ~127 of 150 steps.

> **Caveat on item 4.** The energy penalty matches an energy *trajectory*; on
> conservative wave/KdV that is a conservation prior, which hurts (CKINO already
> approximately conserves via structure, and the relative-energy term is
> ill-conditioned when the true energy passes near zero). Its intended use case
> — matching a **dissipative decay envelope** — is untested here because both
> testbeds are conservative. That is the one remaining experiment that would
> give item 4 a fair trial (a damped/heat-type PDE, or the real seismic field).

**Refined, evidence-based recipe for CKINO rollout on these smooth PDEs:**
1. **Noise injection (item 1) is the key stabilizer** — apply it only when the
   un-stabilised rollout is unstable, and dose it to the instability (≈0.02 for
   KdV; 0 for the already-stable wave).
2. **Drop the energy penalty (item 4)** on conservative problems.
3. Push-forward curriculum + teacher forcing (items 5, 7) buy horizon, not
   stability — cheap, keep them.
4. Two-step stencil (item 6) is a minor positive.
5. **Always evaluate on rollout (item 3); never trust one-step MSE alone.**


## 9. Reproducibility

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"          # Windows OpenMP guard
# reference-solver invariant check
python -m track2.pde_solvers
# full ablation (CPU-tractable settings used for the tables above)
python -m track2.run_experiments --problem wave1d --epochs 24 --n-traj 32 --horizon 150 --stride 2
python -m track2.run_experiments --problem kdv    --epochs 24 --n-traj 32 --horizon 150 --stride 2
```

Environment: Anaconda Python 3.13.5, PyTorch 2.12.0+cpu.
