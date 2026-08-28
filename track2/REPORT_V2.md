# Track-2 v2 — Large-N, Long-Horizon, Multi-Operator Study

**Prepared for:** Dr Gareth O'Brien
**Question:** how do SKINO **and other neural operators** behave at long autoregressive
rollout, at realistic sample counts, and does recursion itself cause the failure?

---

## 0. Executive summary

Ten model configurations were trained per equation (four operator families × two
recipes, plus two non-recursive predictors), on **512 training trajectories**
rolled out **500 steps**, on a linear and a nonlinear PDE.

The five results that matter:

1. **The non-recursive (direct) predictor failed — it was the worst approach
   tested.** It collapses to predicting a constant field. Recursion is therefore
   *not* the root cause of rollout error; removing it is decisively worse.
2. **A flat RMS curve does not mean a good model.** Several configurations with
   stable-looking RMS plateaus are, on visual inspection, **amplitude-collapsed
   or constant predictors**. Only the predicted-vs-real snapshots exposed this.
3. **SKINO beats FNO on all three problems, and its parameter advantage grows
   sharply with dimension** — 5.4–6.5× fewer parameters in 1-D, **83× fewer in
   2-D**, where it is simultaneously **11× more accurate**.
4. **The symplectic structure contributes nothing measurable.** The
   `skino_nosymp` ablation (identical kernel/lift/projection, symplectic block
   replaced by a plain residual block) matches or slightly *beats* full SKINO on
   both 1-D equations.
5. **Noise injection is equation-dependent, and its failure mode is amplitude
   collapse.** Essential on KdV (prevents divergence); actively harmful on both
   the 1-D and 2-D wave equations.

**Status against an absolute usefulness bar:** two configurations are genuinely
good and visually verified — **`skino_plain` on the 2-D wave (1.6 % error at
t=100, 3.0 % at t=200, visually indistinguishable from truth)** and
`skino_noise` on KdV (4 % at t=100, 16 % at t=500). Everything else degrades,
diverges, or collapses.

---

## 1. Requested items and where they are answered

| # | Item | Section |
|---|---|---|
| 1 | N > 500, rollout ≈ 500 steps | §2.2 |
| 2 | Predicted vs real at t = 100, 200, … | §4.3, §5.3 |
| 3 | RMS vs rollout time, validated against (2) | §4.2, §5.2, **§6.1** |
| 4 | Repeat for nonlinear equation | §5 (KdV) |
| 5 | Repeat for FNO / other operators | §3.2, §5.4 |
| 6 | Compare to non-recursive solution | §3.3, **§6.2** |
| 7 | Move to 2-D if working | §7 |

### 1.1 Interpretation note — please confirm

**"Increase N > 500" was interpreted as the number of training trajectories
(N = 512)**, not the number of spatial grid points, because it is paired with
"rollout ~500 timesteps" and follows the earlier "~10,000 samples" guidance.
Spatial resolution was held at 32 (wave) / 64 (KdV) points.

If *grid* resolution was meant, it is a single flag (`--grid 512`) and the study
can be re-run; SKINO is resolution-agnostic by construction so this is cheap.

---

## 2. Setup

### 2.1 Equations

| Problem | Equation | Type | Grid | Δt | Invariant |
|---|---|---|---:|---:|---|
| `wave1d` | u_tt = c² u_xx, c=1 | linear, conservative | 32 | 0.02 | H = ½∫(v²+c²u_x²) |
| `kdv` | u_t + 6u u_x − u_xxx = 0 | **nonlinear**, solitons | 64 | 0.001 | ∫(u_x²−2u³), mass |
| `wave2d` | u_tt = c²(u_xx+u_yy) | linear, conservative | 32×32 | 0.02 | H = ½∫(v²+c²\|∇u\|²) |

Reference integrators conserve their invariants to 0.023 % (wave) and 0.26 %
(KdV) over 200 steps, so rollout error is the operator's, not the solver's.

### 2.2 Data (item 1)

* **512 training trajectories**, 16 validation, 16 test — **disjoint IC seed
  ranges** (separated by 10 000) so no split leakage.
* **501 states per trajectory** → rollout evaluated over the full **500 steps**.
* Training windows sub-sampled with stride 25 → ≈10 000 windows per curriculum
  stage (matching the "~10 000 samples" guidance).
* Band-limited ICs (≤3 modes on wave, ≤4 on KdV) — nothing aliased at t=0.
* Per-channel unit-RMS normalisation from **train statistics only** (wave
  channels differ ~6.6×: u≈0.159, v≈1.052).

### 2.3 Training

Identical budget for every configuration: 12 epochs, K-curriculum [1,2,4] with
teacher forcing annealed 1→0, AdamW lr 3e-3 with a **non-zero cosine floor**,
gradient clip 1.0, batch 32, seed 0. `plain` = no input noise; `noise` = Gaussian
noise at 2 % of field RMS injected at every unroll step.

---

## 3. Models compared (item 5)

### 3.1 The candidate
* **`skino`** — SKINO_ND, pseudo-symplectic kernel-integral operator.

### 3.2 Other neural operators
* **`skino_nosymp`** — *ablation*: identical Chebyshev kernel, Lie lift and
  projection; the Störmer–Verlet block is replaced by a plain residual block.
  Isolates the contribution of the symplectic structure.
* **`fno`** — Fourier Neural Operator (Li et al. 2021), the standard baseline.
* **`transformer`** — encoder-only PDE-transformer over grid tokens.

### 3.3 Non-recursive predictor (item 6)
* **`*_direct`** — horizon-conditioned operator mapping **(u₀, T) → u_T in one
  shot**, with T supplied as a normalised constant input channel and trained on
  random (t₀, T) pairs. No feedback loop is ever formed, so it isolates how much
  rollout error is *recursion-induced*.

### 3.4 Parameter counts

| model | wave1d | kdv |
|---|---:|---:|
| skino | **21,110** | **25,067** |
| skino_nosymp | 35,222 | 37,131 |
| transformer | 39,698 | 41,137 |
| fno | 136,514 | 136,449 |

FNO uses **5.4–6.5× more parameters** than SKINO throughout.

---

## 4. Evaluation protocol

1. **§4.2 RMS vs rollout time (item 3)** — relative RMS ‖pred−truth‖/‖truth‖ at
   every step, free-running from a truth initial condition, plus a divergence
   guard that records the step at which a rollout becomes non-finite.
2. **§4.3 Snapshots (item 2)** — predicted and real fields overlaid at
   t = 100, 200, 300, 400, 500.
3. **Cross-validation (item 3 "validate against 2")** — every RMS number is
   checked against its snapshot. **This step overturned several conclusions
   that the RMS table alone would have supported** (§6.1).

---

## 5. Results

### 5.1 wave1d — relative RMS at checkpoints

| config | params | 1-step val MSE | t=100 | t=200 | t=300 | t=400 | t=500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `skino_plain` | 21,110 | 5.89 × 10⁻⁶ | **0.118** | **0.257** | **0.429** | 0.681 | 1.16 |
| `skino_noise` | 21,110 | 1.80 × 10⁻⁴ | 0.403 | 0.484 | 0.510 | 0.527 | 0.542 † |
| `skino_nosymp_plain` | 35,222 | 3.24 × 10⁻⁶ | **0.094** | **0.204** | **0.350** | **0.597** | 1.10 |
| `skino_nosymp_noise` | 35,222 | 2.63 × 10⁻⁴ | 0.424 | 0.485 | 0.509 | 0.530 | 0.550 † |
| `fno_plain` | 136,514 | 1.08 × 10⁻⁴ | 0.430 | 0.671 | 0.883 | 1.08 | 1.44 |
| `fno_noise` | 136,514 | 5.98 × 10⁻⁴ | 0.596 | 0.710 | 0.791 | 0.871 | 0.955 † |
| `transformer_plain` | 39,698 | 1.65 × 10⁻³ | 2.22 | 4.80 | 8.39 | 11.7 | 15.3 |
| `transformer_noise` | 39,698 | 6.46 × 10⁻⁴ | 0.905 | 1.19 | 1.22 | 1.10 | 1.15 |
| `skino_direct` | 21,216 | — | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 ‡ |
| `fno_direct` | 136,546 | — | 0.998 | 0.990 | 0.977 | 0.959 | 0.942 ‡ |

† **flat RMS is misleading — amplitude collapse**, see §5.3.
‡ **degenerate: predicts ≈ zero**, see §5.3.

### 5.2 kdv (nonlinear) — relative RMS at checkpoints

| config | params | 1-step val MSE | t=100 | t=200 | t=300 | t=400 | t=500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `skino_plain` | 25,067 | **2.41 × 10⁻¹¹** | 8.1 × 10⁵ | — | — | — | — **DIVERGED @86** |
| **`skino_noise`** | 25,067 | 1.39 × 10⁻⁶ | **0.041** | **0.077** | **0.110** | **0.139** | **0.165** |
| `skino_nosymp_plain` | 37,131 | 6.19 × 10⁻¹¹ | 8.0 × 10⁵ | — | — | — | — **DIVERGED @91** |
| `skino_nosymp_noise` | 37,131 | 1.00 × 10⁻⁶ | 0.044 | 0.081 | 0.114 | 0.143 | 0.169 |
| `fno_plain` | 136,449 | 1.17 × 10⁻⁵ | 0.059 | 0.129 | 0.217 | 0.339 | 0.462 |
| `fno_noise` | 136,449 | 2.52 × 10⁻⁵ | 0.098 | 0.193 | 0.282 | 0.352 | 0.404 |
| `transformer_plain` | 41,137 | 1.18 × 10⁻⁵ | 4.87 | 4.13 | 4.54 | 4.75 | 4.19 |
| `transformer_noise` | 41,137 | 4.32 × 10⁻⁵ | 0.085 | 0.152 | 0.206 | 0.250 | 0.286 |
| `skino_direct` | 25,173 | — | 0.300 | 0.311 | 0.300 | 0.312 | 0.299 ‡ |
| `fno_direct` | 136,481 | — | 0.270 | 0.274 | 0.270 | 0.271 | 0.275 ‡ |

‡ **degenerate: predicts the constant temporal mean**, see §5.3.

### 5.3 Snapshot validation — what the RMS numbers hide

Reading [`results_v2/snapshots_kdv.png`](results_v2/snapshots_kdv.png) and
[`results_v2/snapshots_wave1d.png`](results_v2/snapshots_wave1d.png):

| config | what the snapshot actually shows | RMS reading was… |
|---|---|---|
| kdv `skino_noise` | **tracks the true waveform closely at every checkpoint** | ✅ honest |
| kdv `skino_nosymp_noise` | tracks closely, visually indistinguishable from SKINO | ✅ honest |
| kdv `*_plain` | **grid-scale sawtooth** filling ±3 — numerical blow-up | ✅ honest (diverged) |
| kdv `fno_plain/noise` | right amplitude, **progressive phase drift** by t≥300 | ✅ honest |
| kdv `*_direct` | **flat line at the mean (≈1.0)** — no spatial structure at all | ❌ 0.27–0.30 looked "stable" |
| wave `skino_plain` | tracks the wave; high-frequency contamination grows after t≈300 | ✅ honest |
| wave `skino_noise` | **amplitude collapsed to ±0.4 vs truth ±2** — heavily over-damped | ❌ 0.54 plateau looked "stable" |
| wave `fno_noise` | partial amplitude collapse (±1 vs ±2) | ❌ partly |
| wave `*_direct` | **flat line at zero** | ❌ 0.94–0.98 |

This is the single most important methodological result of the study: **three
different configurations produced flat, benign-looking RMS curves while being
degenerate predictors.** Item 2 was necessary to catch it — item 3 alone would
have reported them as the "most stable" models.

### 5.4 Operator ranking (best recipe per family)

**wave1d** (by genuine waveform tracking at t=100):
`skino_nosymp_plain` (0.094) ≈ `skino_plain` (0.118) > `fno_plain` (0.430) ≫ `transformer` (2.22) ≫ direct (0.98)

**kdv:**
`skino_noise` (0.041) ≈ `skino_nosymp_noise` (0.044) > `fno_plain` (0.059) > `transformer_noise` (0.085) ≫ direct (0.27)

SKINO wins both — at 5.4–6.5× fewer parameters than FNO. One caveat in FNO's
favour: **FNO never diverged on KdV without noise, while SKINO did.** FNO is the
more robust operator when no stabiliser is used; SKINO is the more accurate one
when the stabiliser is correctly dosed.

---

## 6. Findings

### 6.1 A flat RMS curve is not evidence of a working model *(item 3 validated against item 2)*
Amplitude collapse and mean-prediction both produce stable, moderate RMS values
(0.27–0.55) that rank *above* honestly-degrading models in a table. Only the
predicted-vs-real overlay distinguishes them. **Every future rollout metric in
this project should be reported alongside snapshots.** This repeats — with a
different mechanism — the "frozen prediction" failure diagnosed in the seismic
study.

### 6.2 The non-recursive predictor is the worst approach tested *(item 6)*
Given the same architecture, data and budget, direct (u₀,T)→u_T prediction
collapsed to a constant field on **both** equations and **both** operator
families (RMS 0.27–0.98, no spatial structure). Learning a 500-step operator in
one shot is a much harder function-approximation problem than learning a
one-step operator, and with this budget it fits only the mean.
**Implication: recursion is not the villain.** Error accumulation is a real
problem, but removing the recursion is decisively worse than stabilising it.

### 6.3 SKINO is the most parameter-efficient accurate operator, but not the most robust
Best-recipe SKINO beats FNO on all three problems: 1-D wave (0.118 vs 0.430),
KdV (0.041 vs 0.059) and 2-D wave (0.016 vs 0.183). The parameter advantage is
5.4–6.5× in 1-D and **83× in 2-D**, because SKINO's separable rank-R Chebyshev
kernel scales linearly in the number of axes while FNO's spectral weights scale
as (modes)^d. **This is the finding with the clearest path to practical value**,
since the target problem is 3-D.

However SKINO diverged on KdV without noise where FNO did not — SKINO's accuracy
advantage comes with a stability liability that must be managed by the recipe.

### 6.4 The symplectic structure shows no measurable benefit
`skino_nosymp` — same kernel, same lift, same projection, symplectic block
replaced by a plain residual block — matched SKINO on KdV (0.044 vs 0.041) and
**beat** it on wave (0.094 vs 0.118), while also diverging in the same place
without noise. Combined with the earlier analysis that the architecture is only
*pseudo*-symplectic (volume-preserving, not ω-preserving, and non-symplectic
lift/projection), the reasonable conclusion is that **SKINO's advantage comes
from the Chebyshev kernel-integral parameterisation and its parameter
efficiency, not from the symplectic block.** The "symplectic" claim in the
project's documentation is not supported by this evidence.

### 6.5 Noise injection is equation-dependent, and over-dosing causes amplitude collapse
* KdV: **essential** — without it both SKINO variants diverge (step 86 / 91);
  with it they are the best models in the study.
* wave1d: **harmful** — it converts an honestly-degrading model into an
  over-damped one that under-predicts amplitude by ~5×.
The dose must scale with the problem's intrinsic instability. A fixed 2 % is not
transferable.

### 6.6 One-step validation MSE anti-correlates with rollout quality
On KdV the two models with the best one-step MSE ever recorded here
(2.4 × 10⁻¹¹ and 6.2 × 10⁻¹¹) both diverged, while the models 5 orders of
magnitude "worse" (1.4 × 10⁻⁶) were the only good ones. One-step loss must not
be used for model selection in this project.

### 6.7 The transformer baseline is the weakest operator
Worst on both equations in both recipes (2.2–15.3 on wave; 4.1–4.9 plain on
KdV), despite a competitive parameter count.

---

## 7. 2-D extension (item 7)

The "if working" gate was satisfied by `skino_noise` on KdV, so the 2-D wave
equation was run with the same driver: **128 trajectories, 200 steps, 32×32
grid**, 6 configurations, identical 12-epoch budget.

### 7.1 wave2d — relative RMS at checkpoints

| config | params | 1-step val MSE | t=100 | t=200 |
|---|---:|---:|---:|---:|
| **`skino_plain`** | **25,414** | 1.36 × 10⁻⁷ | **0.016** | **0.030** |
| `skino_noise` | 25,414 | 3.21 × 10⁻⁷ | 0.042 | 0.083 |
| `fno_plain` | 2,102,594 | 2.41 × 10⁻⁵ | 0.183 | 0.316 |
| `fno_noise` | 2,102,594 | 3.49 × 10⁻⁵ | 0.163 | 0.300 |
| `skino_direct` | 25,560 | — | 0.985 | 0.976 ‡ |
| `fno_direct` | 2,102,626 | — | 0.288 | 0.161 |

‡ degenerate — see §7.2.

### 7.2 Snapshot validation (2-D)

From [`results_v2/snapshots_wave2d.png`](results_v2/snapshots_wave2d.png):

* **`skino_plain` is visually near-indistinguishable from the ground truth** at
  both t=100 and t=200 — the full 2-D interference pattern, node lines and
  amplitudes are reproduced. The 1.6 % / 3.0 % RMS numbers are **honest**.
* `skino_noise` is very close but slightly smoothed (the familiar mild
  over-damping from noise injection).
* `fno_plain` / `fno_noise` reproduce the gross pattern but with visible
  blurring and distortion of the fine structure — consistent with 0.16–0.32.
* **`skino_direct` is nearly blank** — amplitude collapsed to ≈0, the same
  degenerate failure seen in 1-D. Its 0.98 RMS is *not* a near-miss.
* `fno_direct` is poor at t=100 (diagonal artefacts) but genuinely reasonable at
  t=200, which is why its RMS *decreases* with horizon. The direct approach is
  not uniformly degenerate for FNO in 2-D, but it is erratic.

### 7.3 The headline 2-D result — parameter efficiency scales with dimension

| | SKINO | FNO2D | ratio |
|---|---:|---:|---|
| parameters | 25,414 | 2,102,594 | **83× fewer** |
| RMS @ t=100 | 0.016 | 0.183 | **11× more accurate** |

The FNO parameter count explodes in 2-D because its spectral weights scale as
(modes)^d × hidden², whereas SKINO's **separable** rank-R Chebyshev kernel scales
linearly in the number of axes. **SKINO's efficiency advantage grows sharply with
dimension: 5.4–6.5× in 1-D → 83× in 2-D.** For a 3-D target problem this is the
most commercially relevant finding in the study.

### 7.4 Honest caveat on the 2-D numbers

The 2-D result is *not* evidence that "2-D is easier than 1-D". The 2-D initial
conditions use only K≤2 Fourier modes per axis (vs K=3 on wave1d) and the horizon
is 200 rather than 500 steps, so this problem is intrinsically smoother and
shorter. The **relative** comparison between operators is fair (identical data,
budget and protocol); the absolute error should not be compared across sections.

---

## 8. Caveats

1. **Single seed.** No error bars; differences under ~10 % relative should not be
   over-read.
2. **Modest training budget** (12 epochs). The wave1d numbers here are worse than
   the earlier 24-epoch/150-step study (RMS@150 0.014 there), so the models are
   under-converged; the *comparison* is fair (identical budget) but the absolute
   values are not the best achievable.
3. **Toy problems.** 1-D/2-D smooth periodic PDEs on small grids. None of this
   has yet been demonstrated on the heterogeneous 3-D seismic problem.
4. **Direct-model budget.** The non-recursive predictor was given the same epoch
   budget; a substantially larger budget or capacity might change §6.2. The
   conclusion "recursion is not the villain" is supported at equal budget.
5. **Noise dose not re-tuned per equation** in the main sweep (fixed 2 %); §6.5
   is therefore a statement about a fixed dose, not an optimum.

---

## 9. Recommendations

1. **Always report snapshots with RMS.** Add a degeneracy check (predicted field
   variance vs truth variance) to flag amplitude collapse automatically.
2. **Keep the recursive formulation.** Invest in stabilising it, not replacing it.
3. **Re-tune the noise level per problem** — scale it to the measured one-step
   error rather than fixing 2 %.
4. **Re-examine the symplectic claim.** Either enforce genuine symplecticity
   (symmetrise the kernel; make lift/projection symplectic) and re-test, or
   re-describe the architecture honestly as a volume-preserving Verlet-templated
   operator.
5. **Next milestone:** apply the best recipe to the *homogeneous* 3-D seismic
   case — the bridge from these toy results back to the real problem.

---

## 10. Reproducibility

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -m track2.experiments_v2 --problem wave1d --n-traj 512 --horizon 500 --epochs 12 --stride 25
python -m track2.experiments_v2 --problem kdv    --n-traj 512 --horizon 500 --epochs 12 --stride 25
python -m track2.experiments_v2 --problem wave2d --n-traj 128 --horizon 200 --epochs 12 --stride 10 --grid 32
```

Artefacts in [`results_v2/`](results_v2): `results_<problem>.json` (all metrics),
`rms_<problem>.png` (RMS vs time, linear + log), `snapshots_<problem>.png`
(predicted vs real), `<problem>_full.log` (training logs).

Code: [`models.py`](models.py) (operator zoo + horizon-conditioned wrapper),
[`experiments_v2.py`](experiments_v2.py) (driver),
[`pde_solvers.py`](pde_solvers.py), [`data.py`](data.py).

Environment: Anaconda Python 3.13.5, PyTorch 2.12.0+cpu.
