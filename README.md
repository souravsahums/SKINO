# The Weight of the Adjoint

Code, archived results and paper source for

**The Weight of the Adjoint: Which Symplectic Form a Neural Operator Preserves, and Whether It Survives the Lift**
Sourav Sahu and Gareth O'Brien, Microsoft, 2026

[Paper (PDF)](paper/weight_of_the_adjoint.pdf) · [LaTeX source](paper/weight_of_the_adjoint.tex) · [MIT licence](LICENSE)

> **Names.** The repository is called `SKINO` on GitHub and the Python package is `ckino`. In the paper, **SA-Cheb** is the symplectic construction and **CKINO** is the non-gradient Chebyshev kernel-integral operator that serves as a baseline. The older name survives in a few identifiers: the `SKINO_RESULTS_DIR` variable, the `--skino-*` options of `HamiltonianNN/`, and the `skino` label that `track2/symplectic_defect.py` prints for the CKINO row.

---

## The question

Structure-preserving neural operators rest on a question that is rarely stated explicitly: symplectic with respect to which inner product? On a non-uniform grid the discrete $L^2$ adjoint is $K^{*}=W^{-1}K^{\top}W$, where $W$ holds the quadrature weights, and the choice of $W$ selects which symplectic form a gradient shear $F=K^{*}\rho(Kq)$ preserves. The paper makes that choice explicit, measures it, and asks three questions.

| | Question | Paper |
| --- | --- | --- |
| RQ1 | Does each construction preserve the form it claims, and only that one? | §6.1 |
| RQ2 | Does matching the form to the discretisation of the data improve prediction on uniform, Chebyshev and mapped grids? | §6.2 |
| RQ3 | Does the guarantee survive the lift, projection and residual update that conventionally wrap the core, and what does end-to-end symplecticity buy over a long rollout? | §6.3 |

The mathematics is in §3 (Theorem 5 is the exactness result), the two operators in §4, the protocol in §5, and a secondary nine-equation benchmark that places the constructions among standard neural operators in §7.

## Results in brief

* **RQ1, which form.** The symplectic defect, computed from the autograd Jacobian, is at most $4.4\times10^{-16}$ in the form a shear was built from and $O(1)$ in the other, in 1-D, 2-D and 3-D. The non-gradient CKINO kernel has a defect of about 1.4 in both forms: its block has unit Jacobian determinant (it is volume-preserving), which is not symplecticity (Appendix A).
* **RQ2, matching the grid.** On uniform grids the matched form has the lower error in 17 of 22 paired comparisons over five seeds, and none favours the other form. On Chebyshev nodes the preference reverses in two of three comparisons and is tied in the third; on a mapped grid, where neither standard form is the quadrature, matching brings no consistent benefit. The weights also set the quadrature of the kernel integral, so the experiments cannot separate the form from the quadrature rule. The control that would is described in §9 and has not been run.
* **RQ3, end to end.** Over $2\times10^{4}$-step rollouts, end-to-end symplectic maps stayed bounded in 60 of 63 runs (the exceptions: one mismatched-form model in 3-D), while a lifted variant with a non-symplectic linear lift and projection was bounded in 3 of 12. Bounded is not accurate: an identity map is bounded too, the error is comparable to the signal, and the energy is off by more than 100%, against 0.03% for the reference solver.
* **Benchmark.** 783 runs over nine equations in 1-D, 2-D and 3-D at matched parameter count, plus 338 further training runs for the form and long-horizon experiments. No operator dominates (§8). SA-Cheb wins reliability under long rollout (the lift-free, matched-form variant is the only family bounded in all 12 runs, though not accurate) and accuracy on the non-periodic and multi-dimensional problems, clearly on three of four. SNO and T-FNO win discretisation invariance, and the Fourier family has the lowest mean error on every periodic 1-D problem. A workload that is periodic, or that must be evaluated at a resolution the model was not trained on, is better served by a Fourier operator.

## Repository layout

| Path | Contents |
| --- | --- |
| [`ckino/`](ckino/) | The PyTorch package. [`sacheb.py`](ckino/sacheb.py): SA-Cheb (`SAChebNO`, `SAChebShear`, `SelfAdjointChebKernel`). [`nd.py`](ckino/nd.py), [`kernel.py`](ckino/kernel.py), [`symplectic.py`](ckino/symplectic.py), [`equivariance.py`](ckino/equivariance.py), [`hypernet.py`](ckino/hypernet.py), [`model.py`](ckino/model.py): CKINO and its building blocks, in 1-D and N-D. [`train.py`](ckino/train.py): a small Burgers' demo. |
| [`track2/`](track2/) | Everything behind the paper's experiments: reference solvers ([`pde_solvers.py`](track2/pde_solvers.py)), the model registry and baselines ([`models.py`](track2/models.py), [`operators.py`](track2/operators.py)), experiment drivers ([`launcher.py`](track2/launcher.py), [`experiments_paper.py`](track2/experiments_paper.py), [`longhorizon.py`](track2/longhorizon.py), [`multires.py`](track2/multires.py), [`symplectic_defect.py`](track2/symplectic_defect.py)), analysis and figure scripts, cluster scripts (`aml_*.py`) and the archived per-run results (`results*/`). |
| [`paper/`](paper/) | LaTeX source and PDF of the paper, the arXiv package builder ([`make_arxiv.py`](paper/make_arxiv.py), [`make_arxiv_metadata.py`](paper/make_arxiv_metadata.py)) and the bibliography checker ([`check_references.py`](paper/check_references.py)). |
| [`arxiv_updated_2026-10-09/`](arxiv_updated_2026-10-09/), [`arxiv/`](arxiv/) | Metadata, abstract and submission steps for the arXiv packages. The packages themselves (`*.zip`) are regenerated by `paper/make_arxiv.py` and are not tracked. |
| [`validation/`](validation/) | A small CPU-scale harness that compares CKINO with baselines on Hamiltonian ODEs, two PDEs and a porous-flow problem. Separate from the paper; see its [README](validation/README.md). |
| [`HamiltonianNN/`](HamiltonianNN/) | A side study: FNO3D against CKINO3D on a 3-D elastic-lattice simulator. Separate from the paper; see its [README](HamiltonianNN/README.md). |
| `proofs.md`, `CKINO_BOOK.md`, `track2/*.md`, `validation/report/` | Working notes and intermediate reports. They are not kept in sync with the paper; where they differ, the paper is authoritative. |

## Quick start

Tested here with Python 3.13, PyTorch 2.12 (CPU), NumPy 2.1 and Matplotlib 3.10; the archived GPU runs used PyTorch 2.8 and 2.13. The operators in `ckino/` need only PyTorch.

```bash
pip install torch numpy matplotlib
```

### SA-Cheb

```python
import torch
from ckino.sacheb import SAChebNO

# 65 Chebyshev-Gauss-Lobatto nodes, one (q, p) pair. lift=False: the shears act directly on the
# PDE's own (q, p) channels, with no lift or projection, so x_{t+1} = model(x_t) is exactly
# symplectic in the Clenshaw-Curtis form W_cheb.
model = SAChebNO(n_train=64, in_channels=2, out_channels=2,
                 rank=8, depth=3, weighted=True, lift=False)
y = model(torch.randn(4, 2, 65))        # (batch, [q, p], nodes) -> same shape
```

| Option | Effect |
| --- | --- |
| `weighted=False` (or `weight_kind="unif"`) | Uniform form $W_{\mathrm{unif}}$: the adjoint is the plain transpose, the matched form on a uniform grid. |
| `weight_kind="kte"` | The form of the Kosloff-Tal-Ezer mapped grid. |
| `lift=True`, `hidden_channels=H` | Lifted variant: a pointwise linear lift and projection around the symplectic core. The shears are symplectic; the end-to-end map is not. |
| `canonical=True` (with `lift=False`) | The core inside a canonical symplectic wrapper. |
| `spatial_dims=2` or `3` | Tensor-product basis and quadrature in 2-D and 3-D. |
| `adjoint_eps > 0` | Perturbs the adjoint, dialling symplecticity down continuously. |

Measuring the defect of one shear against both inner products (the paper's instrument, in miniature):

```python
from ckino.sacheb import SAChebShear
from ckino.nd import clenshaw_curtis_weights

n = 33                                                  # CGL nodes
shear = SAChebShear(n_coeff=n, channels=1, hidden=8, rank=4, weighted=True).double()
shear.gain.data.fill_(0.7)                              # the gain starts at 0, i.e. the identity shear
q = torch.randn(n, dtype=torch.float64)
A = torch.autograd.functional.jacobian(lambda v: shear(v.view(1, 1, n)).view(-1), q)   # A = DF

for name, w in [("W_cheb", clenshaw_curtis_weights(n - 1, dtype=torch.float64)),
                ("W_unif", torch.ones(n, dtype=torch.float64))]:
    W = torch.diag(w)
    print(name, ((W @ A - A.T @ W).norm() / (W @ A).norm()).item())
# W_cheb ~ 2e-16 (the form the adjoint was built from); W_unif ~ 0.6 (O(1))
```

### CKINO

```python
from ckino import CKINO

model = CKINO(n_modes=32, in_channels=1, out_channels=1, hidden_channels=32, rank=8,
              depth=4, pde_param_dim=1, n_generators=2, dt=0.1)
u_T = model(torch.randn(8, 1, 33), torch.randn(8, 1))   # 33 CGL nodes; one PDE coefficient per sample
```

`CKINO2D` and `CKINO3D` take the same arguments, with `n_train` (the basis degree per axis) in place of `n_modes`. `pde_param_dim=0` disables the hypernetwork, which the paper's experiments do not use. The CKINO block is volume-preserving but not symplectic. Its kernel is stored as Chebyshev coefficients, so the same weights can be evaluated on a grid of any size, but accuracy does not transfer: trained at $N=64$ and evaluated at $N=128$, CKINO with a convolutional lift degrades by $57\times$ on advection while SNO and T-FNO change by less than 2% (§7.5).

`python -m ckino.train --n_train 64 --epochs 200` trains CKINO on a viscous-Burgers' family with a Chebyshev reference solver. It is a smoke test, not a benchmark, and takes a few minutes on a CPU, most of it spent generating the reference solutions.

## Reproducing the paper

Run everything from the repository root. The drivers write to `track2/results_paper` unless `--results-dir` or the `SKINO_RESULTS_DIR` variable says otherwise, and the launcher skips any run whose result file already exists, so use an empty directory for a clean run.

```bash
# Reference solvers: self-tests (about 20 s on a CPU)
python -m track2.pde_solvers

# RQ1: the defect against each inner product (dense Jacobian; Hutchinson probes for the largest 3-D grid)
python -m track2.symplectic_defect --dims 1 2 3 --forms W_cheb W_unif W_kte
python -m track2.symplectic_defect --probe-study          # probe estimator against the dense value

# RQ2 and RQ3 and the benchmark (GPU): list the jobs, or print the exact cluster command lines
python -m track2.launcher --plan --stage 1d --configs full
python -m track2.aml_trigger --dry-run                    # the main matrix
python -m track2.aml_trigger --review --dry-run           # the additional runs of §5.5
python -m track2.longhorizon --problem wave1d --steps 20000 --device cuda    # one long-horizon run

# Tables and figures from the archived results (CPU; about 1.5 min)
python -m track2.review_analysis --results-dir track2/results_review --baseline-dir track2/results_gpu_v3
python -m track2.review_tables   --results-dir track2/results_review --baseline-dir track2/results_gpu_v3
python -m track2.seed_analysis   --results-dir track2/results_gpu_v3
```

The three analysis scripts write into the results directory (`review_analysis` also accepts `--out`): `review_summary.md` (every table of the additional runs, with verdicts) and the `fig_review_*.png` figures; `review_tables.tex` (the table bodies used in the paper); `TABLES_MULTISEED.md`, `TABLES_ROLLOUT.md` and their figures. Run on the archived data, they regenerate the committed `review_summary.md`, `review_tables.tex`, `TABLES_MULTISEED.md` and `TABLES_ROLLOUT.md` exactly.

To build the paper and the arXiv package:

```bash
cd paper && latexmk -pdf weight_of_the_adjoint.tex && cd ..          # the bibliography is inline
python paper/make_arxiv.py --out arxiv_updated_2026-10-09            # clean package, checked to build standalone
python paper/check_references.py                                     # checks each reference against Crossref/arXiv (needs network)
```

## Results and data

| Directory | Contents |
| --- | --- |
| `track2/results_gpu_v2`, `track2/results_gpu_v3` | Per-run JSON results and summary tables of the main matrix: execution A and execution B (§7.4). |
| `track2/results_review` | The additional runs (§5.5): seeds 3 and 4 of the headline pairs, the two non-uniform grids, long-horizon controls and the multi-resolution test, the defect measurements, and `review_summary.md`. |
| `track2/results_cpu` | The probe-against-dense study of the defect estimator (§6.1). |
| other `track2/results*/` | Earlier result sets, kept for provenance. |

Per-run result files, the analysis scripts and the figure generators are archived here. Model checkpoints (`*.pt`) and stored field snapshots (`*.npz`, seed 0 only) are not, so the field figures cannot be regenerated from the repository alone. The archived runs used a single accelerator type (Tesla T4, TF32 disabled).

## Known limitations

The paper's §9 lists them all; those that matter most to anyone reusing the code or the results:

* **Form and quadrature are not separated.** The weight $W$ defines the quadrature of the kernel integral as well as the adjoint, so the matched and mismatched variants differ in the forward operator too (§4.2). The experiments show that using the grid's quadrature weights throughout helps where the grid has a standard quadrature, not that the symplectic form alone is responsible.
* **The `kdv` problem is linear as generated.** The exponential integrator behind it multiplies its nonlinear term by an extra factor of the micro-step, so the stored trajectories are those of the linear dispersive equation $u_t=u_{xxx}$ (see the `KdVProblem` docstring in [`track2/pde_solvers.py`](track2/pde_solvers.py)). Every result labelled `kdv` is a result on a linear problem. A corrected solver is registered as `kdv_nl` (it conserves $\int(u_x^2+2u^3)\,dx$ to $4\times10^{-5}$ over 100 stored steps); the benchmark has not been re-run with it.
* **Baselines are re-implementations.** SNO, GENERIC-FNO, FNO, T-FNO, U-FNO, U-Net, DeepONet and the Transformer were written from the published descriptions, not taken from the authors' code, and run here at 25k to 262k parameters (SNO's authors train about 2.3M), so the numbers are not reproductions of theirs. Our GENERIC-FNO diverged in all 12 entries, a defect of that implementation rather than evidence about the published method.
* **Repetition is partial.** The two executions of the main matrix, and the additional runs on a newer PyTorch, cover part of the study (§7.4); some claims rest on a single execution.

## Azure ML

The cluster scripts in `track2/` (`aml_trigger.py`, `aml_fetch.py`, `azureml_submit.py`) detect the workspace from the compute instance they run on (the `AZUREML_ARM_*` variables Azure ML sets, falling back to `MLClient.from_config()`), so they need no configuration there. To target a different workspace, or to use `aml_submit_window.ps1`, set

```bash
export AML_SUBSCRIPTION_ID="<your-subscription-id>"
export AML_RESOURCE_GROUP="<your-resource-group>"
export AML_WORKSPACE="<your-workspace-name>"
export AML_STORAGE_ACCOUNT="<your-storage-account>"   # aml_submit_window.ps1 only
```

(PowerShell: `$env:AML_SUBSCRIPTION_ID = "<your-subscription-id>"`, and so on.) Authentication uses `DefaultAzureCredential`, for example after `az login`; no keys or secrets are stored in the repository. These scripts additionally need `azure-ai-ml` and `azure-identity`.

## Licence and citation

The code is released under the MIT licence ([`LICENSE`](LICENSE)). If you use it, please cite the paper:

```bibtex
@misc{sahu2026weight,
  title  = {The Weight of the Adjoint: Which Symplectic Form a Neural Operator Preserves,
            and Whether It Survives the Lift},
  author = {Sahu, Sourav and O'Brien, Gareth},
  year   = {2026},
  note   = {Preprint. Code and data: https://github.com/souravsahums/SKINO}
}
```
