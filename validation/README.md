# SKINO Validation Suite

End-to-end validation of the **Symplectic Kernel-Integral Neural Operator** along
the six required levels:

| Level | Concern                                |
| ----- | -------------------------------------- |
| L1    | Pointwise prediction accuracy          |
| L2    | Long rollout stability                 |
| L3    | Conservation properties                |
| L4    | Out-of-distribution PDE generalization |
| L5    | Computational efficiency               |
| L6    | Ablation studies                       |

Three tiers of physics:

* **Tier 1** — canonical Hamiltonian ODEs (harmonic oscillator, pendulum, Kepler)
* **Tier 2** — Hamiltonian PDEs (wave, KdV)
* **Tier 3** — conservative reservoir transport (1-D porous-media flow)

## Layout

```
validation/
├── common/        # baselines, metrics, plotting, seeding
├── tier1_ode/     # ODE Hamiltonian experiments
├── tier2_pde/     # PDE Hamiltonian experiments
├── tier3_application/  # conservative reservoir transport
├── ablation/      # symplectic-on / symplectic-off comparison
├── efficiency/    # complexity, runtime, memory
├── run_all.py     # orchestrator (writes results/ and figures/)
├── results/       # JSON tables and CSV summaries (auto-generated)
├── figures/       # PNG plots (auto-generated)
└── report/        # comparative_study.md, research_paper.md, checklist.md
```

## Quick start

```powershell
cd /path/to/skino
python -m validation.run_all
```

The orchestrator is deliberately tuned to finish on CPU within a few minutes:
small models (hidden ≤ 32, depth ≤ 3, rank ≤ 8), small datasets (≤ 64 samples),
moderate epoch counts (100–300). The experimental *protocol* is what is
publication-grade, not the wall-clock; scaling the constants does not change
the qualitative conclusions.
