# Validation suite

A small, CPU-scale harness that exercises CKINO and baseline models on three tiers of conservative problems. It is separate from the study in the paper, which lives in [`track2/`](../track2/) (see the [main README](../README.md)). The models are small (194 to 102,625 parameters, not matched across models), the datasets are small (256 to 384 training states for the ODEs, 80 to 96 training pairs for the PDEs), each model is trained once with seed 0, and none of the paper's tables or figures is drawn from it. Treat the numbers as sanity checks, not as a benchmark.

## Levels and tiers

| Level | Concern | Measured by |
| ----- | ------- | ----------- |
| L1 | Pointwise prediction accuracy | one-step relative L² error on held-out data (`test_rel_L2`) |
| L2 | Long-rollout stability | relative L² error after a long autoregressive rollout (`long_horizon_rel_L2`) |
| L3 | Conservation | energy, mass, momentum and angular-momentum drift; the symplectic defect (tier 1) |
| L4 | Generalisation to unseen resolutions | `results/efficiency.json`, `resolution_errors` (N = 32, 64, 128; `NaN` for the fixed-grid DeepONet and Transformer) |
| L5 | Computational efficiency | parameters, training and inference time, parameter memory ([`efficiency/`](efficiency/)) |
| L6 | Ablation | CKINO against CKINO-NoSymp (tiers 2 and 3); the leap-frog ODE model against a generic MLP (tier 1) |

* **Tier 1**: canonical Hamiltonian ODEs (harmonic oscillator, pendulum, Kepler, double pendulum). CKINO-SympNet, a Störmer–Verlet-style step with learnable $U_q$ and $U_p$, against a non-symplectic MLP (NonSymp-MLP) and a generic residual MLP (ResidualMLP).
* **Tier 2**: Hamiltonian PDEs (1-D wave, KdV). CKINO, CKINO-NoSymp, FNO, DeepONet and a small Transformer.
* **Tier 3**: conservative reservoir transport (1-D porous-media flow), with the same five models.

CKINO-NoSymp is the CKINO architecture with the Störmer–Verlet-style block replaced by an ordinary residual block.

## Reading the results

* **Volume-preserving is not symplectic.** In the paper's terms (§4.1, Appendix A) the CKINO block, and the tier-1 leap-frog step built the same way, is volume-preserving but not symplectic. The two coincide only in one degree of freedom, where an area-preserving map is symplectic. This shows in the symplectic defect of `results/summary.csv`: about $3\times10^{-8}$ and $1\times10^{-8}$ for the harmonic oscillator and the pendulum, and $8\times10^{-4}$ and $6\times10^{-3}$ for the double pendulum and Kepler, which have two degrees of freedom.
* **KdV is linear as generated.** The reference solver in [`tier2_pde/kdv.py`](tier2_pde/kdv.py) multiplies its nonlinear term by an extra factor of the micro-step, the same defect as in the main benchmark's `kdv` problem (paper §9). The generated data are those of the linear dispersive equation: after 200 macro-steps they lie within $1.6\times10^{-4}$ (relative) of the exact linear flow, against $0.45$ for the same scheme with the factor removed. The KdV rows are results on a linear problem.
* **Several long rollouts diverge.** In the committed results the wave-equation rollouts of CKINO, CKINO-NoSymp and FNO (final errors of $2\times10^{5}$, $6\times10^{16}$ and non-finite), the KdV rollouts of CKINO and CKINO-NoSymp (non-finite) and the porous-flow rollout of FNO (infinite) do not stay bounded, so the L2 and L3 columns of those rows record divergence rather than accuracy.
* **Single runs.** Each model is trained once, with no seeds or intervals, so small differences between rows should not be read as differences between models.

## Layout

```
validation/
├── common/               # baselines, metrics, plotting, seeding
├── tier1_ode/            # harmonic oscillator, pendulum, Kepler, double pendulum
├── tier2_pde/            # 1-D wave, KdV
├── tier3_application/    # 1-D porous-media flow
├── efficiency/           # complexity, runtime, memory, resolution generalisation
├── run_all.py            # orchestrator (writes results/ and figures/)
├── _diag.py              # prints rollout diagnostics (finite steps, first non-finite step) from results/
├── results/              # JSON tables and summary.csv
├── figures/              # PNG plots
└── report/               # working notes, not kept in sync with the paper
```

## Quick start

From the repository root:

```powershell
python -m validation.run_all
```

The full suite takes about 17 minutes on an 8-thread laptop CPU (1,004 s measured here), most of it in tiers 2 and 3. It rewrites `results/` and `figures/`. To run only some tiers, call `validation.run_all.main(skip_tiers=(...))` with any of `"tier1"`, `"tier2"`, `"tier3"`, `"efficiency"`. Requires PyTorch, NumPy and Matplotlib.

The committed `results/` and `figures/` come from an earlier run of the suite in which CKINO was labelled `SKINO` (`SKINO`, `SKINO-NoSymp`, `SKINO-SympNet`). A full re-run with the current code reproduces all 27 rows of `results/summary.csv`, relabelled, apart from the DeepONet rows (test error within 3%, the drift columns by more) and the training times. It does not reproduce the efficiency stage: the L4 resolution errors changed by up to a factor of two (for example CKINO at $N=32$: 0.40 committed, 0.22 re-run), so treat them as indicative only. `_diag.py` expects the current labels.
