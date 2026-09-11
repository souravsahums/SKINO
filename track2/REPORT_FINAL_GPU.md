# CKINO — Final Report (GPU / Tesla T4)

**One-line result:** at genuinely matched ~25k parameters on a single hardware
(NVIDIA Tesla T4), across 9 PDEs × 3 seeds, the decisive finding is a **controlled
ablation of structure itself**. We built an operator (SA-Cheb) that is *exactly*
symplectic on a non-periodic domain — symplectic defect **2×10⁻¹⁶**, machine
precision — together with an otherwise-identical twin whose only difference is a
deliberately wrong adjoint (defect 0.65–0.78, i.e. not symplectic at all). **The
non-symplectic twin is more accurate on all six 1-D problems, by 1.1× to 4.3×.**
Meanwhile the *basis* choice does what the structure was supposed to do: on a
non-periodic Hamiltonian problem the Chebyshev operators beat every Fourier
operator by 3–5×. **Geometry of the discretisation matters; geometry of the flow
does not.**

Numbers below come from the main GPU run — the complete matrix of **31
configurations × 3 seeds × 9 problems (657 training runs)**, submitted to a
2-node `Standard_NC8as_T4_v3` cluster — augmented by **six standalone
experiments** (Darcy elliptic flow, spectral-truncation ablation, 6k→400k
parameter scaling, zero-shot super-resolution, **symplectic-defect measurement**,
and inference speedup). Figures and videos are in [`results_gpu/`](results_gpu/).

---

## 1. Executive summary

Best model per problem — relative RMS, mean ± std over 3 seeds (reported at the
last common rollout checkpoint: t=200 in 1-D/2-D, t=150 in 3-D):

| Problem | Best model | rel RMS | Winner | Mode |
|---|---|---:|---|---|
| advection | `ckino_plain` | **0.00109 ± 0.00057** | CKINO (tied SNO) | recursive |
| heat | `tfno_plain` | **0.00109 ± 0.00031** | T-FNO | recursive |
| wave1d | `sno_seq2seq` | **0.00130 ± 0.00016** | SNO | non-recursive |
| **wave1d_dir** (non-periodic) | `naive_seq2seq` | **0.0274 ± 0.0011** | **Chebyshev family** | non-recursive |
| burgers | `tfno_plain` | **0.00396 ± 0.00098** | T-FNO | recursive |
| kdv | `sno_seq2seq` | **0.000467 ± 0.000063** | SNO | non-recursive |
| wave2d | `ckino_plain` | **0.0289 ± 0.0083** | CKINO† | recursive |
| wave3d | `ckino_seq2seq` | **0.1386 ± 0.0373** | CKINO† | non-recursive |
| ns2d | `ckino_seq2seq` | **0.188 ± 0.002** | CKINO† | non-recursive |

† SNO, GENERIC-FNO and SA-Cheb are implemented in 1-D only, so the three
multi-dimensional rows are contested by FNO alone.

**Tally by basis: Chebyshev-family 5/9 · Fourier-family 4/9** — and the split is
not random: **Fourier wins 4 of the 5 periodic problems; Chebyshev wins the one
non-periodic problem and all three multi-dimensional ones.**

### The decisive experiment: does exact symplecticity help?

`sacheb` and `naive` are the **same architecture with the same parameter count**;
the only difference is whether the shear uses the correct weighted adjoint
$K^{*}=W^{-1}K^{\top}W$ (exactly symplectic) or the naive transpose (not
symplectic). Measured symplectic defect $\lVert WA-A^{\top}W\rVert/\lVert WA\rVert$:

| grid | `sacheb` | `naive` | `ckino` |
|---:|---:|---:|---:|
| 16 | **2.0e-16** | 0.780 | 1.367 |
| 32 | **2.4e-16** | 0.696 | 1.390 |
| 48 | **3.0e-16** | 0.653 | 1.379 |

So one model is symplectic to machine precision and the other is not symplectic
at all. Their accuracy (rel RMS, 3 seeds, `_seq2seq` arm):

| problem | `sacheb` (exact) | `naive` (not) | result |
|---|---:|---:|---|
| advection | 0.003556 | **0.001508** | non-symplectic 2.4× better |
| heat | 0.005565 | **0.003491** | non-symplectic 1.6× better |
| wave1d | 0.005176 | **0.001530** | non-symplectic 3.4× better |
| wave1d_dir | 0.03667 | **0.02739** | non-symplectic 1.3× better |
| burgers | 0.09196 | **0.08567** | non-symplectic 1.1× better |
| kdv | 0.002693 | **0.000627** | non-symplectic 4.3× better |

**The non-symplectic twin wins all six.** Worse for the structure-preserving
thesis, the promised benefit — long-rollout stability — also fails to appear: both
`sacheb_plain` and `naive_plain` diverge on wave1d_dir (rel RMS clipped at 1e3).
Exact symplecticity is, in this study, a measurable *cost*.

### What does help: matching the basis to the boundary conditions

On `wave1d_dir` (Hamiltonian **and** non-periodic — the setting SNO's authors
explicitly do not cover), every Chebyshev operator beats every Fourier operator:

| Chebyshev family | rel RMS | Fourier family | rel RMS |
|---|---:|---|---:|
| `naive_seq2seq` | **0.0274** | `fno_seq2seq` | 0.0829 |
| `ckino_seq2seq` | 0.0354 | `sno_seq2seq` | 0.135 |
| `sacheb_seq2seq` | 0.0367 | `tfno_seq2seq` | 0.147 |

Best Chebyshev vs best Fourier: **3.0×**; vs SNO specifically: **4.9×**. The
mirror image holds on the periodic rungs, where Fourier wins 4 of 5. The basis,
not the geometric prior, is what tracks performance.

**Where CKINO still leads, and against whom.** For completeness, CKINO's best
config vs the best *non*-CKINO config per problem:

| problem | CKINO best | best competitor | margin | character |
|---|---|---|---|---|
| Darcy | ckino 0.082 | fno 0.677 | **8.3× lower** (boundary **17×**) | substantial (vs FNO only) |
| wave2d | ckino_plain 0.029 | fno_seq2seq 0.154 | **5.3× lower**; FNO diverges (667) | substantial (vs FNO only) |
| wave3d | ckino_seq2seq 0.139 | fno_seq2seq 0.639 | **4.6× lower**; FNO unusable (UH 2 vs 150) | substantial (vs FNO only) |
| ns2d | ckino_seq2seq 0.188 | fno_seq2seq 0.266 | 1.4× lower (~30%, ~11σ) | moderate (vs FNO only) |
| wave1d_dir | ckino_seq2seq 0.0354 | naive_seq2seq 0.0274 | 1.3× **worse** | loss (to a Chebyshev peer) |
| advection | ckino_plain 0.00109 | sno_seq2seq 0.00110 | tie | no win |
| heat | ckino_plain 0.0028 | tfno_plain 0.0011 | 2.6× **worse** | loss |
| wave1d | ckino_seq2seq 0.0026 | sno_seq2seq 0.0013 | 2.0× **worse** | loss |
| kdv | ckino_seq2seq 0.0016 | sno_seq2seq 0.00047 | 3.5× **worse** | loss |
| burgers | ckino_seq2seq 0.42 | tfno_plain 0.0040 | ~100× **worse** | clear loss |

CKINO's surviving advantage is confined to the multi-dimensional problems and
Darcy, and there it has only been tested against FNO. Read together with the
ablation above, the pattern is that CKINO's wins are attributable to its
**non-periodic basis**, not to its (non-existent — defect 1.37) symplectic structure.

**Findings that the data support:**

1. **Exact symplecticity is a measurable cost, not a benefit.** In a perfectly
   controlled ablation (identical architecture and parameter count; symplectic
   defect 2e-16 vs 0.65), the non-symplectic twin is more accurate on all six 1-D
   problems and neither variant delivers the promised rollout stability (§4.7f, F13).
2. **Matching the basis to the boundary conditions is what works.** Chebyshev
   operators beat every Fourier operator by 3–5× on the non-periodic problem;
   Fourier operators win 4 of 5 periodic problems (§1, F14).
3. **CKINO's own symplectic claim is false.** Its kernel shear has symplectic
   defect ≈1.37 at every resolution — it was never a symplectic method (§4.7f).
2. **Non-recursive prediction wins exactly where autoregressive rollout is
   *unstable*.** Noise-matched, seq2seq beats its recursive twin on wave1d and
   KdV for *every* backbone, but a no-noise recursive model wins the stable
   advection/heat/burgers. It is not universal; it tracks rollout instability
   (§4.3, §5).
3. **Symplecticity buys nothing and can hurt.** The exactly-symplectic `strict`
   variant and the structure-removed `nosymp` variant are both indistinguishable
   from nominal CKINO on the easy problems, and `strict` *blows up on KdV*
   (§4.4).
4. **RMS alone is not a sufficient metric.** The verdict map (§4.4) shows
   configurations with moderate RMS that are physically dead or amplitude-
   collapsed; only the amplitude/correlation decomposition separates them.
5. **Non-periodic boundaries are where CKINO's advantage is largest.** On the
   Darcy elliptic problem (Dirichlet BC) CKINO reaches 0.082 rel-L2 vs FNO's 0.68,
   and **17× lower error at the boundary**, where the Fourier basis fails; it also
   holds accuracy **zero-shot** from a 64² to a 128² grid (§4.7).

---

## 2. Setup

**Equation ladder (simple → hard).** advection (linear transport) · heat (linear
dissipative) · wave1d (linear oscillatory) · **wave1d_dir (wave on [0,L] with
homogeneous DIRICHLET boundaries — Hamiltonian *and* non-periodic)** · burgers
(nonlinear, shocks) · kdv (nonlinear, dispersive) · wave2d (32², conservative) ·
wave3d (16³, conservative) · **ns2d (2-D Navier–Stokes, 64², vorticity form,
decaying turbulence)**. `wave1d_dir` reproduces the benchmark family used by the
concurrent SNO paper (uniform grid, centred finite differences, symplectic
leap-frog, initial conditions damped by the envelope x(L-x)); it is the rung that
separates a periodic basis from a non-periodic one. Reference solvers are
verified (energy/mass drift ≈ 1e-8; wave1d_dir drift 2.3e-4 over 500 steps with
boundaries held at exactly 0). Three **standalone gap experiments** sit outside the rollout matrix:
**Darcy** (elliptic coefficient→solution map with Dirichlet boundaries, plus a
zero-shot 64²→128² super-resolution test), a **spectral-truncation ablation**
(sweeping the Chebyshev/Fourier resolution), and an **inference-speedup**
measurement against the spectral solver (§4.7).

**Operator families, all parameter-matched to ~25k.** CKINO (Chebyshev
kernel-integral), `ckino_strict` (exactly symplectic), `ckino_nosymp` (symplectic
structure removed), **SA-Cheb (new: exactly-symplectic gradient shear on a
non-periodic Chebyshev grid)** and **`naive` (its ablation twin — identical but
with the weighted adjoint replaced by a plain transpose, so it is *not*
symplectic)**, **SNO (concurrent Symplectic Neural Operator, Makara–Yaguchi
2026, arXiv:2605.15881 — genuinely-symplectic Fourier shear blocks)**, **GENERIC-FNO
(concurrent metriplectic operator, Sulskis & Ravi 2026, arXiv:2606.08343 — learned
energy/entropy with projected Fourier multipliers)**, FNO, T-FNO (CP-factorised
FNO), U-FNO, U-Net, DeepONet, Transformer. **Configs** cross each family with prediction mode
{recursive, non-recursive `seq2seq`}, training noise {none `_plain`, 2% `_noise`}
and a physics-informed loss {`_pinn`}, for 31 configurations in 1-D (SNO,
GENERIC-FNO and SA-Cheb are 1-D only; the 2-D/3-D stages run the CKINO/FNO subset).
This yields a clean 2×2 factorial — {Fourier, Chebyshev} basis × {exactly
symplectic, not} — realised as `sno`/`sacheb`/`fno`/`ckino`, plus the `naive`
adjoint ablation that isolates symplecticity with everything else held fixed.

**Protocol.** 3 seeds; recursive models trained on 1-step windows and rolled out
autoregressively; `seq2seq` emits the whole trajectory in one pass at a matched
*gradient-step* budget. Metrics per checkpoint: relative RMS **plus** amplitude
ratio, pattern correlation, spectral error and invariant drift, combined into a
physical **verdict** and a **usable horizon** (last step with corr ≥ 0.9 and
0.7 ≤ amplitude ≤ 1.4).

**Hardware.** All runs on NVIDIA **Tesla T4** (`Standard_NC8as_T4_v3`), TF32
enabled (curated PyTorch environment) — the full 8-problem × 3-seed matrix (over
400 runs) plus the three gap experiments. Single hardware → no device confound.

---

## 3. Evaluation methodology — what we measure, how, and why

Five deliberate design choices separate this study from a single-number
leaderboard. Each is given as *what* we do, *how* we do it, and *why* it matters.

**1 — RMS is only one metric; we judge whether the model actually solves the physics.**
- *What.* Alongside relative RMS we compute amplitude ratio, pattern correlation,
  spectral error (low/high band) and invariant drift (energy/mass), and collapse
  them into a single physical **verdict** (good → degraded → decorrelated →
  amplitude-collapse → dead → blow-up) and a **usable horizon**. We also render
  the **actual-vs-predicted fields** for every model (§4.5).
- *How.* Metrics are recomputed from the saved full trajectories at each rollout
  checkpoint, not only at the final step; verdict thresholds are corr ≥ 0.9 and
  0.7 ≤ amplitude ≤ 1.4.
- *Why.* A model can post a middling RMS while being physically dead — right
  average error, wrong wave. RMS cannot tell "blurred but in phase" from "sharp
  but out of phase"; the verdict map (§4.4) and the eyeball test (§4.5) can. This
  is finding **F5**.

**2 — We score the whole rollout, not a single step.**
- *What.* Every model is evaluated as a function of rollout step (t = 10 … 500),
  never at one horizon.
- *How.* `fig_rms_time_*` (§4.2) plots RMS-vs-step for all configs;
  `fig_rollout_growth` (§4.3) reduces each curve to a growth factor (last ÷ first).
- *Why.* In practice you march the operator forward hundreds of steps, so a model
  that is excellent at t = 1 but compounds error is useless at t = 200. Only the
  time-resolved view exposes error accumulation and the exact step where a model
  destabilises.

**3 — We compare recursive rollout against a full non-recursive (seq2seq) solution.**
- *What.* Each backbone is trained two ways: **recursive** (learn a one-step map,
  apply it autoregressively) and **seq2seq** (train on all timesteps, emit the
  entire trajectory in one pass).
- *How.* Both are trained to the same *gradient-step* budget and scored on the
  identical rollout grid, and the comparison is kept **noise-matched** (choice 4)
  so the prediction mode is the only variable.
- *Why.* Recursion re-feeds the model its own output, creating a feedback loop
  that can amplify error; seq2seq has no such loop. The head-to-head (**F2**,
  §4.3) shows one-shot prediction wins *precisely* on the problems where the
  rollout is unstable (wave1d, KdV, multi-D wave) and recursion wins where it is
  stable — a result invisible if you only ever train one mode.

**4 — We compare models of the same size.**
- *What.* Every family is parameter-matched to ~25k trainable parameters.
- *How.* Width/depth are tuned per family to land within a few percent of 25k;
  the exact count is printed next to every model in the tables.
- *Why.* Otherwise "CKINO beats FNO" could just mean "the bigger network wins."
  Earlier in this project the families ranged from 25k to 2.1M parameters — not a
  fair fight. Matched capacity makes any accuracy gap attributable to the
  *operator design*, not the budget.

**5 — We do this across the whole equation ladder, simple → complex.**
- *What.* All eight problems: advection, heat, wave1d (simple linear) → burgers,
  KdV (nonlinear) → 2-D and 3-D wave and 2-D Navier–Stokes (multi-dimensional
  conservative) — plus the Darcy elliptic map as a non-periodic-BC gap test.
- *How.* The identical protocol (metrics, modes, budget, seeds) is applied to
  every equation and reported per problem, never averaged into one score.
- *Why.* Operator superiority is problem-class dependent — CKINO's Chebyshev
  basis shines on smooth transport/dispersion and fails on shocks (burgers),
  while Fourier operators do the opposite. A single equation would over- or
  under-sell any method; the ladder shows *where* and *why* each family wins,
  which is the actual scientific content (**F1**).

---

## 4. Results and figures

### 4.1 Master accuracy — `fig_multiseed.png`

![Multi-seed accuracy](results_gpu/fig_multiseed.png)

**What it shows.** One panel per problem; horizontal bars are the mean relative
RMS (log scale) of every configuration, with std error bars, sorted best-on-top.
**How to read it.** The top bar in each panel is the winner from §1. The huge
dynamic range within a panel (e.g. KdV spans 1e-3 to >1e2) is the point: the same
budget produces both excellent and diverging models depending on family and
mode. Note that the CKINO/T-FNO winners sit far below the weak baselines
(DeepONet, Transformer, U-Net), confirming the operator families dominate the
generic ones at matched capacity.

### 4.2 Rollout error vs step — `fig_rms_time_<problem>.png`

One figure per problem; each line is a configuration's relative RMS as the
rollout advances (log-y, mean over seeds).

![advection](results_gpu/fig_rms_time_advection.png)
![heat](results_gpu/fig_rms_time_heat.png)
![wave1d](results_gpu/fig_rms_time_wave1d.png)
![wave1d_dir](results_gpu/fig_rms_time_wave1d_dir.png)
![burgers](results_gpu/fig_rms_time_burgers.png)
![kdv](results_gpu/fig_rms_time_kdv.png)
![wave2d](results_gpu/fig_rms_time_wave2d.png)
![wave3d](results_gpu/fig_rms_time_wave3d.png)
![ns2d](results_gpu/fig_rms_time_ns2d.png)

**How to read them.** A flat line = an operator that does not accumulate error;
an upward line = error compounding with the rollout; a line that shoots vertical
= divergence. Key readings:
- **advection / heat:** almost every operator is stable; the `_plain` (no-noise)
  recursive models sit lowest, and the `seq2seq` lines are flat — this is the
  "stable problem" regime where recursion is fine.
- **wave1d / kdv:** the recursive lines climb steeply or diverge, while the
  `seq2seq` lines stay flat near the bottom — the "unstable rollout" regime where
  one-shot prediction wins by 1–2 orders of magnitude.
- **burgers:** all CKINO lines converge to the same ~0.4 plateau early (the basis
  ceiling), while T-FNO/U-FNO sit an order of magnitude lower.
- **wave2d / wave3d:** `ckino` variants stay bounded; the parameter-matched FNO
  climbs off the top of the panel (it blows up above 1-D).
- **ns2d (2-D Navier–Stokes):** the `seq2seq` arms stay flat and bounded with
  `ckino_seq2seq` lowest; the recursive `plain` arms drift up as the
  decaying-turbulence field loses small-scale coherence.

### 4.3 Error growth — `fig_rollout_growth.png`

![Rollout growth](results_gpu/fig_rollout_growth.png)

**What it shows.** The same rollout curves, but the story is the *growth factor*
— relative RMS at the last checkpoint ÷ the first. **How to read it.** Every
autoregressive configuration grows its error 5–20× over the rollout (and 10⁴–10⁷
when it destabilises), whereas the `seq2seq` arms grow only ~0.5–1.2× — they never
consume their own output, so there is no feedback loop to amplify. The single
exception is `ckino_seq2seq` on Burgers (5.5×), which is the basis ceiling, not a
recursion effect (the Fourier `seq2seq` arms stay flat even there). **This is the
mechanism behind finding 2:** one-shot prediction helps precisely and only when
the autoregressive feedback loop is what fails.

### 4.4 Physical verdicts — `fig_verdict_heatmap.png`

![Verdict heatmap](results_gpu/fig_verdict_heatmap.png)

**What it shows.** For each (config, problem) the *worst* physical verdict across
the 3 seeds at the final step. Green = good; yellow → orange → black = degraded,
decorrelated, amplitude-collapsed, dead, blow-up. White = not run (U-Net /
DeepONet / Transformer are 1-D only). **How to read it.**
- The `seq2seq` rows (`ckino_`, `tfno_`, `ufno_seq2seq`) are green across the
  board — the most *robust* configurations.
- `fno_*` go **black** (blow-up) on KdV and both multi-D waves — the
  parameter-matched FNO is not usable above 1-D.
- `strict_*` go black on KdV — enforcing exact symplecticity destabilises the
  dispersive problem.
- `tfno_plain` is black on wave1d and KdV yet is a *winner* on heat and burgers —
  the clean illustration that no-noise recursion is superb on stable problems and
  unstable on oscillatory/dispersive ones.
- **Why this matters:** several of these black/orange cells have unremarkable RMS
  values. Only the amplitude/correlation-based verdict exposes that they are
  physically useless — this is finding 4, "RMS alone is insufficient".

### 4.5 Actual vs predicted — every model, every equation — `fig_pred_vs_real_<problem>.png`

This is the direct "does it solve the problem?" view behind methodology choice 1,
and the qualitative face of findings F1–F4.

**What it shows.** One grid per equation. The **top row is ground truth** at
several rollout times (t = 100 … 500 in 1-D; evenly sampled instants in 2-D/3-D).
**Every row below is one model's prediction** — for 1-D, the prediction (red)
over the truth (faint grey); for 2-D/3-D, a seismic heat-map of the field (wave3d
shown as a mid-plane z-slice). Every model is parameter-matched to ~25k.

![advection](results_gpu/fig_pred_vs_real_advection.png)
![heat](results_gpu/fig_pred_vs_real_heat.png)
![wave1d](results_gpu/fig_pred_vs_real_wave1d.png)
![wave1d_dir](results_gpu/fig_pred_vs_real_wave1d_dir.png)
![burgers](results_gpu/fig_pred_vs_real_burgers.png)
![kdv](results_gpu/fig_pred_vs_real_kdv.png)
![wave2d](results_gpu/fig_pred_vs_real_wave2d.png)
![wave3d](results_gpu/fig_pred_vs_real_wave3d.png)
![ns2d](results_gpu/fig_pred_vs_real_ns2d.png)

**How to read them.** A model "solves" the equation when its red curve sits on
the grey truth at every column (or its heat-map matches the truth panel). Three
failure signatures are visible at a glance: **phase drift** (right shape, shifted
sideways), **amplitude collapse** (flattening toward a straight line), and
**divergence** (dense vertical red spikes / cells labelled *diverged*).

**What the panels reveal (and how it matches the metrics).**
- **advection / heat:** nearly every operator tracks the truth — the easy rungs.
- **wave1d / KdV:** the recursive no-noise arms (`ckino_plain`, `fno_noise`,
  `transformer_noise`) blow up or drift out of phase, while the `seq2seq` rows
  (`ckino_seq2seq`, `fno_seq2seq`) stay locked on the truth — the visual form of
  **F2**. `ckino_direct` collapses to a near-flat line (amplitude collapse), the
  exact case RMS alone would under-penalise (**F5**).
- **burgers:** the CKINO rows visibly round off the shock front (the Chebyshev
  basis ceiling) while T-FNO/U-FNO keep it sharp — the picture behind **F1**.
- **wave2d / wave3d:** the parameter-matched FNO heat-maps decorrelate into noise
  while `ckino` stays spatially coherent — **F1/F3** above 1-D.
- **ns2d:** `ckino_seq2seq` keeps the vortices spatially coherent and in phase,
  while the recursive and FNO arms smear the small scales first — the flagship
  turbulence case where CKINO's edge is real but modest (~30%).
- `nosymp_noise` is visually indistinguishable from `ckino_noise`, and `strict`
  destabilises on KdV — the eyeball confirmation of **F3** (symplecticity is
  neutral or harmful).

> **Provenance.** These eight panels are rendered from **genuine Tesla-T4
> prediction fields** — the full ~25k-matched matrix saved with `--save-fields`
> on a dedicated one-seed GPU run (126 field arrays across all eight problems).
> Extracting them took two fixes: a CUDA tensor hit `.numpy()` inside a swallowed
> `try` (now `.cpu().numpy()`), and training crashed at *import* because the
> cluster's matplotlib is ABI-incompatible with its numpy (the transitive
> `import matplotlib` is now optional). The fields are single-seed; the headline
> accuracy tables come from the full 3-seed GPU metric run.

### 4.6 Rollout videos — `results_gpu/videos/`

For each problem the truth, prediction and pointwise error are animated
side-by-side over the full rollout (`python -m track2.make_videos`). Motion makes
the failure signatures of §4.5 impossible to miss — phase drift, amplitude
collapse and divergence all read instantly.

| problem | winner video | contrast |
|---|---|---|
| advection | [ckino_plain](results_gpu/videos/vid_advection_ckino_plain.mp4) | — |
| heat | [tfno_plain](results_gpu/videos/vid_heat_tfno_plain.mp4) | — |
| wave1d | [tfno_seq2seq](results_gpu/videos/vid_wave1d_tfno_seq2seq.mp4) | — |
| burgers | [tfno_plain](results_gpu/videos/vid_burgers_tfno_plain.mp4) | — |
| kdv | [ckino_seq2seq](results_gpu/videos/vid_kdv_ckino_seq2seq.mp4) | — |
| wave2d | [ckino_plain](results_gpu/videos/vid_wave2d_ckino_plain.mp4) | — |
| wave3d | [ckino_seq2seq](results_gpu/videos/vid_wave3d_ckino_seq2seq.mp4) | — |
| ns2d | [ckino_plain](results_gpu/videos/vid_ns2d_ckino_plain.mp4) | [fno_plain](results_gpu/videos/vid_ns2d_fno_plain.mp4) |

The **ns2d pair is the most instructive**: side-by-side, CKINO holds the vortex
structure while the matched FNO loses the fine scales first. (Videos are `.mp4`,
generated locally and git-ignored; regenerate with `python -m track2.make_videos
--results-dir track2/results_gpu`.)

### 4.7 Neural-operator gap experiments (Darcy, mode-truncation, speedup)

Three experiments outside the rollout matrix probe the properties a neural
operator is *supposed* to have.

**(a) Darcy elliptic flow — non-periodic boundaries + zero-shot super-resolution.**
Learn the coefficient→solution map −∇·(a∇u)=f on a 64² grid with **Dirichlet
boundaries**, 3 seeds. Relative L2 (mean over seeds):

| model | rel-L2 | boundary L2 | interior L2 | 64²→128² super-res |
|---|---:|---:|---:|---:|
| **CKINO** | **0.084** | **0.22** | **0.081** | **0.10** |
| ckino_strict | 0.084 | 0.26 | 0.079 | 0.13 |
| FNO | 0.73 | 3.9 | 0.61 | 0.71 |

CKINO is **~8× more accurate overall and ~17× at the boundary** — the direct
consequence of a Chebyshev basis that respects a non-periodic domain where FNO's
Fourier basis cannot. CKINO also transfers **zero-shot to a 2× finer grid** (0.10
at 128², trained only at 64²); the FNO stays uniformly poor. This is the defining
neural-operator property (discretisation invariance) the main matrix did not test.

**(b) Spectral-truncation ablation — grid independence.** Sweeping the resolution
N ∈ {32, 48, 64, 96} at matched budget:
- **burgers:** CKINO is essentially **grid-independent** (rel-RMS ≈ 0.25 at every
  N), while FNO swings 0.51–1.06 with N — CKINO's Chebyshev truncation converges,
  FNO's mode count does not.
- **kdv:** CKINO is excellent at N=32 (rel-RMS ≈ 0.003–0.03) but **destabilises
  at N ≥ 64** (blows up), where the third-derivative dispersion turns stiff; FNO
  stays bounded (≈ 0.16–0.88). An honest failure mode — the advantage is not free
  at high resolution on stiff dispersive problems.

**(c) Inference speedup vs the spectral solver** (Tesla T4, batch 16, 100 steps;
value = solver time ÷ operator time, >1 means the operator is faster):

| problem | solver | CKINO | FNO |
|---|---:|---:|---:|
| advection | 0.034 s | 0.07× | 0.24× |
| heat | 0.052 s | 0.11× | 0.37× |
| wave1d | 0.047 s | 0.10× | 0.33× |
| burgers | 0.456 s | 0.95× | **3.2×** |
| kdv | 1.09 s | **2.3×** | **7.7×** |

The operators carry a fixed ~0.1–0.5 s overhead, so on **cheap** linear solves
the spectral reference wins; they only pay off once the reference is **expensive**
(nonlinear/stiff), where FNO reaches 3–8× and CKINO 2.3× on KdV. Speedup is a
property of the *problem's* solver cost, not of the operator alone.

**(d) Parameter-scaling sweep — is 25k enough?** Sweeping the budget 6k→400k
(fixed problem/grid, one-step training, 3 seeds) tests whether the matched-25k
point under-serves any family:
- **burgers:** CKINO's error is **flat at rel-RMS ≈ 0.25 across the entire 60×
  range** (0.26 at 6k → 0.25 at 400k) — the shock ceiling is a *representational*
  limit of the Chebyshev basis, not a capacity or training deficit. No budget
  rescues it. FNO's one-step error is higher and noisier (≈0.4–0.8) and does not
  cleanly converge at this horizon.
- **kdv:** recursive CKINO **diverges at every budget** (rel-RMS 10 → 8×10⁴, never
  below ≈9 even at 400k), while FNO stays bounded (0.14–0.48). The instability is
  structural (the hard q,p split on a dispersive bracket), not a capacity problem.

This is the budget sweep the caveats promised; it *strengthens* F1/F7 — CKINO's
two headline losses are both capacity-independent, so "scale it up" is not a fix.

**(e) Zero-shot super-resolution — the defining operator property.**
`track2.discretization` trains each family at N=64 and evaluates the *same weights*
at N=128, reporting the one-step operator error at each grid and the degradation
ratio (err@128 ÷ err@64; ≈1 = resolution-invariant), 3 seeds. The Fourier
validators pin the harness (`fno`, `tfno` ratio ≈ 1.00; the self-check passes), so
the CKINO numbers are trustworthy. Mean degradation ratio:

| problem | CKINO | FNO | T-FNO |
|---|---:|---:|---:|
| heat | **1.1×** (invariant) | 1.0× | 1.0× |
| kdv | 3.0× | 1.0× | 1.0× |
| advection | **57×** (0.0003 → 0.02) | 1.0× | 1.0× |

**The Fourier operators are exactly discretisation-invariant; CKINO is not.** On
heat CKINO transfers cleanly, but on advection it is the *most accurate model at
the training grid* (0.0003) and loses ~57× of that accuracy at 2× resolution — its
recursive kernel is effectively tied to the training node count, not to the
continuous operator. This is the sharpest honest limit in the study: **the
Chebyshev-kernel CKINO does not robustly possess the zero-shot super-resolution
property that defines a neural operator in the time-dependent setting.** (Contrast
§4.7a, where CKINO *does* super-resolve the *static* Darcy solve — that map is
grid-agnostic by construction; the time-stepping kernel is not.)

**Diagnosis and attempted fix — the leak is the lift, not the kernel.** CKINO_ND's
kernel-integral is *already* coefficient-parameterised (Chebyshev coefficients
re-evaluated at the query grid), hence resolution-agnostic. The leak is the
default **Lie-lifting layer — a fixed-stencil depthwise convolution** — which is
resolution-dependent exactly like the U-Net control. Replacing it with a
**spectral-derivative lift** (`--lift-kind spectral`) *does* restore exact
invariance, on every problem and seed:

| problem | CKINO conv | CKINO **spectral** | FNO | T-FNO | SNO |
|---|---:|---:|---:|---:|---:|
| advection | 57.4× | **1.00** | 1.02 | 1.00 | 0.99 |
| heat | 1.14× | **1.00** | 1.02 | 1.00 | 0.99 |
| kdv | 3.02× | **1.00** | 1.01 | 1.00 | 0.99 |

**But the fix costs accuracy, and that is the real finding.** At the study's
training protocol the spectral lift trains far worse at the *training* grid:

| problem | CKINO conv err@64 | CKINO spectral err@64 | T-FNO | SNO |
|---|---:|---:|---:|---:|
| advection | **0.00082** | 0.475 | 0.0030 | 0.0017 |
| heat | **0.00119** | 0.608 | 0.0014 | 0.0021 |
| kdv | 0.0424 | 0.625 | 0.00078 | **0.00052** |

So CKINO faces a genuine **invariance-vs-accuracy trade-off**: it can be accurate
(conv lift) *or* discretisation-invariant (spectral lift), but this study did not
achieve both at once. The Fourier operators — FNO, T-FNO and SNO — get **both
simultaneously and for free** (e.g. SNO on KdV: error 0.00052 *and* ratio 0.99).
A earlier single-seed CPU probe suggested the spectral lift was free; the 3-seed
GPU run does not support that, and the GPU numbers supersede it. Whether a better
normalised spectral lift closes the gap is open — our escalating `(ik)^g`
generators are a first cut and may simply be badly conditioned.

**Baselines — SNO and GENERIC-FNO.** Two concurrent structure-preserving
operators were re-implemented as families for a head-to-head, matched-capacity
comparison. **`sno`** (Makara–Yaguchi 2026) composes *genuinely* symplectic shear
blocks whose update is a gradient field `K*ρ(K·)` (self-adjoint Jacobian — the
property CKINO's low-rank kernel lacks); it is resolution-invariant, trains
efficiently, and **wins or ties every 1-D problem against CKINO** (§1).
**`generic`** (Sulskis & Ravi 2026, metriplectic) **failed to train in our
harness and diverged on every problem** — this is a defect of *our*
re-implementation, not a result about the published method (see §6).

**(f) Symplectic defect — measuring the structure directly.** Rather than assume
a model is symplectic because it is called symplectic, we measure it. A shear
$\Phi(q,p)=(q,p+F(q))$ with $A=DF$ satisfies
$(D\Phi)^{\top}\Omega D\Phi-\Omega = \mathrm{diag}(WA-A^{\top}W,\,0)$ under the
weighted symplectic form $\Omega=\begin{psmallmatrix}0&W\\-W&0\end{psmallmatrix}$,
so the relative defect $\lVert WA-A^{\top}W\rVert_F/\lVert WA\rVert_F$ computed
from the autograd Jacobian is zero **iff** the block is symplectic. On
Chebyshev–Gauss–Lobatto nodes the physical inner product carries Clenshaw–Curtis
weights $W$, so the correct adjoint is $K^{*}=W^{-1}K^{\top}W$ — on a
uniform/periodic grid $W\propto I$ and this collapses to the transpose, which is
why a Fourier operator never encounters the distinction. Results (3 seeds):

| grid | `sacheb` (weighted adjoint) | `naive` (transpose) | `ckino` kernel |
|---:|---:|---:|---:|
| 16 | **2.0e-16** | 0.780 | 1.367 |
| 24 | **2.2e-16** | 0.683 | 1.369 |
| 32 | **2.4e-16** | 0.696 | 1.390 |
| 48 | **3.0e-16** | 0.653 | 1.379 |

Three things follow. (i) The construction is **exactly symplectic at machine
precision, independently of resolution** — a non-periodic analogue of SNO's
Fourier result. (ii) The weighted adjoint is **necessary**: the naive transpose
leaves an O(1) defect. (iii) **CKINO is not a symplectic method** — defect ≈1.37
at every grid — so its "Chebyshev kernel-integral" label is not supported, and the earlier
finding F3 was testing a prior that was never present. Pairing (i) and (ii) as
trained models is what makes the ablation in §1 a clean test of symplecticity.

---

## 5. Findings (all GPU-backed)

**F1 — Winners split by basis, matched to the boundary conditions.** Across 9
problems: Chebyshev-family operators take 5 (advection, wave1d_dir, wave2d,
wave3d, ns2d), Fourier-family 4 (heat, wave1d, burgers, kdv). The split is
systematic — **Fourier wins 4 of the 5 periodic 1-D rungs; Chebyshev wins the
non-periodic rung and every multi-dimensional one.** Burgers remains the
diagnostic loss for the Chebyshev family: its basis cannot represent the shock at
any budget (§4.7d), so CKINO plateaus at ≈0.42 while T-FNO reaches 0.0040.

**F2 — Non-recursive prediction wins where rollout is unstable.** Noise-matched
(`_plain` recursive vs `_seq2seq`), the GPU data give:

| backbone | advection | heat | wave1d | burgers | kdv |
|---|---|---|---|---|---|
| CKINO | recursive | recursive | **seq2seq** | seq2seq | **seq2seq** |
| FNO | seq2seq | seq2seq | **seq2seq** | seq2seq | **seq2seq** |
| T-FNO | seq2seq | recursive | **seq2seq** | recursive | **seq2seq** |
| U-FNO | seq2seq | seq2seq | **seq2seq** | recursive | **seq2seq** |

The one universal column is instability: **wave1d and KdV go to `seq2seq` for
every backbone**; the stable problems are a mixed bag where a no-noise recursive
model frequently wins. The mechanism is measured in §4.3 (no error accumulation).

**F3 — Symplecticity does not help.** Removing the structure (`nosymp`) is
indistinguishable from nominal CKINO; enforcing it exactly (`strict`) is neutral
on the easy problems and **blows up on KdV** (black cells, §4.4). Note the
important correction supplied by §4.7f: **none of these variants was actually
symplectic** (measured defect ≈1.37), so F3 alone only shows that a *nominal*
structural label is inert. The genuine test is F13, which reaches the same
conclusion with a verified 2e-16 symplectic prior.

**F4 — Training noise is a stabiliser, not free accuracy.** `_plain` (no noise)
is the most accurate recursive model on stable problems but destabilises on
oscillatory/dispersive ones (e.g. `tfno_plain`, `ckino_plain` blow up on KdV);
`_noise` trades peak accuracy for rollout stability. Any comparison of prediction
modes must therefore be *noise-matched*, which is what F2 does.

**F5 — RMS is insufficient; the metric suite matters.** The verdict map (§4.4)
and the amplitude/correlation decomposition separate physically-useful models
from dead/collapsed ones that RMS alone would rank similarly.

**F6 — Non-periodic boundaries are where CKINO's design pays off most.** On the
Darcy elliptic problem (Dirichlet BC) CKINO is ~8× more accurate than FNO overall
and ~17× at the boundary, and transfers zero-shot to a 2× finer grid (§4.7a). The
Fourier basis's periodicity assumption is a real liability off the torus; the
Chebyshev basis is not. This is the single most substantial CKINO win in the study.

**F7 — CKINO's spectral truncation is grid-convergent, but not unconditionally.**
On burgers CKINO is grid-independent (rel-RMS ≈ 0.25 for N = 32…96) where FNO is
not (§4.7b) — the discretisation invariance a neural operator should have. But on
KdV the same one-step CKINO **destabilises at N ≥ 64** (dispersive stiffness),
which FNO does not; the property is real but conditional.

**F8 — Speedup is solver-dependent, not automatic.** Against the spectral
reference (§4.7c) the operators lose on cheap linear PDEs (fixed overhead) and win
only where the reference is expensive — FNO 3–8×, CKINO 2.3× on KdV. "Neural
operators are faster" holds only relative to a costly solver.

**F9 — CKINO's failures are capacity-independent.** A 6k→400k budget sweep (§4.7d)
leaves CKINO's burgers error flat at ≈0.25 (shock ceiling) and its KdV recursion
divergent at every budget. Neither loss is a capacity or training artefact; both
are structural, so scaling the model up does not fix them.

**F10 — CKINO trades discretisation-invariance against accuracy; Fourier operators
do not.** With the default fixed-stencil Lie-lift, CKINO loses accuracy under 2×
super-resolution (advection ratio 57×, KdV 3.0×) while FNO, T-FNO and SNO are
exactly invariant (§4.7e). The kernel is *not* the cause — it is
coefficient-parameterised and grid-agnostic; the leak is the convolutional lift.
A spectral-derivative lift restores exact invariance (all ratios → 1.00) but, at
this training protocol, costs 1–2 orders of magnitude of training-grid accuracy.
So CKINO achieves invariance *or* accuracy, not both, whereas the Fourier family
gets both for free. This is the sharpest structural argument against the
Chebyshev-kernel design in this study.

**F11 — A concurrent published operator (SNO) dominates CKINO where both were
tested.** At matched capacity on the five 1-D problems, SNO wins 3 (wave1d 2.0×,
KdV 3.5×, burgers 22×), ties 2 (advection, heat) and loses none (§1). SNO also
clears the burgers shock ceiling that bounds every CKINO variant, and is
resolution-invariant for free. Since SNO's shear blocks are *genuinely* symplectic
(gradient fields with self-adjoint Jacobians) while CKINO's low-rank kernel is
not, the comparison suggests the Fourier-parameterised symplectic design is the
stronger one on periodic domains.

**F12 — Our GENERIC-FNO re-implementation failed; this is our bug, not the
method's.** `generic_*` diverged on every 1-D problem (rel-RMS clipped at 1e3,
usable horizon 0). We traced the cause precisely: the PSD multiplier is
parameterised as `M = b(k)²` and initialised at `b = 0`, where `dM/db = 2b = 0` —
a saddle. The dissipative channel is therefore permanently dead (verified: `|b|`
stays at exactly 0 after training), leaving a purely skew update whose amplitude
grows under the explicit-Euler step — the exact failure mode the original paper
documents for its reversible channel. **No conclusion about the published
GENERIC-FNO should be drawn from these rows**; the fix (non-zero or unconstrained
`b` parameterisation) is straightforward and untested here.

**F13 — Exact symplecticity is a measurable cost (the controlled ablation).**
`sacheb` and `naive` share an architecture and a parameter count; only the adjoint
differs, giving symplectic defect 2e-16 versus 0.65–0.78 (§4.7f). The
**non-symplectic twin is more accurate on all six 1-D problems** — advection 2.4×,
heat 1.6×, wave1d 3.4×, wave1d_dir 1.3×, burgers 1.1×, kdv 4.3× — and the
structure's advertised payoff, long-rollout stability, does not appear either
(both recursive arms diverge on wave1d_dir). Because the earlier `strict`/`nosymp`
ablations (F3) used a kernel that was never symplectic (defect 1.37), this is the
first test in the study where the symplectic prior genuinely existed — and it
still fails to pay for itself.

**F14 — What actually helps is matching the basis to the boundary conditions.**
On `wave1d_dir` (Hamiltonian, non-periodic) every Chebyshev operator beats every
Fourier operator: best Chebyshev 0.0274 vs best Fourier 0.0829 (**3.0×**) and vs
SNO 0.135 (**4.9×**). On the five periodic rungs the ordering reverses and Fourier
wins 4 of 5. Discretisation geometry tracks performance; flow geometry does not.

---

## 6. Caveats and what is *not* in the GPU set

- **The §4.5 grids are genuine Tesla-T4 fields (one-seed field run).** They come
  from a dedicated `--save-fields` GPU run (126 arrays, all eight problems),
  fetched with `python -m track2.aml_fetch --with-fields` and rendered with
  `python -m track2.paper_analysis --results-dir track2/results_gpu`. Two bugs had
  to be fixed to persist them on GPU: `.numpy()` on a CUDA tensor (now
  `.cpu().numpy()`), and a training-import crash from a matplotlib/numpy ABI
  mismatch in the cluster env (the transitive `import matplotlib` is now
  optional). The field run is single-seed; the 3-seed metric figures are
  unaffected.
- **The SNO and GENERIC-FNO baselines are our own re-implementations, 1-D only.**
  They follow the published constructions (SNO: gradient-field symplectic shears;
  GENERIC-FNO: learned E/S with projected Fourier multipliers) but are not the
  authors' code, are not tuned to their protocols, and were not extended to 2-D/3-D.
  Consequently (i) CKINO's three multi-dimensional wins are contested by FNO only,
  and (ii) the GENERIC-FNO divergence is an artefact of our zero-init `b²` saddle
  (F12), not a statement about the published method. Both caveats cut *against*
  over-claiming for CKINO.
- **Parameter-scaling and super-resolution are now in the study (§4.7d–e).** The
  6k→400k budget sweep (F9) and the N=64→128 super-resolution test (F10) were the
  two previously-missing items. Both are negative for CKINO: capacity-independent
  failure on burgers/kdv, and an invariance-vs-accuracy trade-off the Fourier
  operators do not face.
- **KdV recursion is a genuine open problem.** The one-step CKINO destabilises on
  KdV at every resolution (§4.7b) and every budget (§4.7d); the reported KdV
  rollout winner is the *seq2seq* arm, which is unaffected, but the recursive
  high-resolution case is unsolved.
- **Toy scales.** Periodic domains (Darcy is the exception — Dirichlet), 32–64 pt
  in 1-D, 32² in 2-D, 16³ in 3-D, 64² for Navier–Stokes; three seeds. Enough to
  resolve the orderings above, not a deployment benchmark.

Complete per-config numbers are in
[`results_gpu/TABLES_MULTISEED.md`](results_gpu/TABLES_MULTISEED.md) and
[`results_gpu/TABLES_ROLLOUT.md`](results_gpu/TABLES_ROLLOUT.md).

---

## 7. What this study actually contributes

The contribution is a **controlled falsification of the structure-preserving
premise**, backed by an operator built specifically to give that premise its best
shot.

1. **A new construction, and a theorem.** On Chebyshev–Gauss–Lobatto nodes the
   physical inner product carries Clenshaw–Curtis weights, so the adjoint is
   K* = W⁻¹KᵀW, not Kᵀ. With that correction the gradient shear
   Φ(q,p) = (q, p + K*ρ(Kq)) is **exactly symplectic on a non-periodic domain**
   for any K and ρ — verified at 2e-16, resolution-independently (§4.7f). On a
   periodic grid W ∝ I and the distinction vanishes, which is why the concurrent
   Fourier-based SNO never had to confront it.
2. **The falsification that construction enables.** Because we can build two
   models that differ *only* in whether they are symplectic (defect 2e-16 vs
   0.65), the premise can be tested directly. **The non-symplectic twin wins all
   six 1-D problems** (F13). This is materially stronger than a conventional
   "we ablated the prior" argument, because both arms are otherwise identical and
   the prior is *measured*, not asserted.
3. **A positive result about what does matter.** Basis-boundary matching predicts
   the winner: Chebyshev 3–5× better on the non-periodic problem, Fourier better
   on 4 of 5 periodic ones (F14).
4. **A measurement methodology.** Symplectic defect from the autograd Jacobian
   turns "is this method structure-preserving?" into a number — and it shows one
   published-style claim (CKINO's own) to be false (defect 1.37).
5. **The surrounding benchmark:** 31 configurations × 9 PDEs × 3 seeds with
   prediction mode and training noise controlled (F2, F4), plus capacity (F9),
   super-resolution (F10), speedup (F8) and metric methodology (F5) — and honest
   reporting of our own GENERIC-FNO implementation failure (F12).

The claim to make is therefore not "a better operator" but: *structure-preserving
priors do not deliver what the 2026 literature claims, and we show this with an
operator that provably has the structure.* The constructive half — exact
symplecticity on non-periodic domains, plus the basis-boundary result — stands on
its own as the positive contribution.
