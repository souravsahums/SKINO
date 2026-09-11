> # ⚠️ CORRECTION NOTICE (2026) — this document is superseded
>
> Two of its load-bearing premises were later measured and found false. Read it
> only as a record of what was believed before that measurement; the current
> position is in [`../../track2/REPORT_FINAL_GPU.md`](../../track2/REPORT_FINAL_GPU.md)
> and [`../../proofs.md`](../../proofs.md).
>
> 1. **CKINO is not symplectic.** Its block is volume-preserving but not
>    ω-preserving; the relative symplectic defect measured from the autograd
>    Jacobian is **≈ 1.37** at every resolution, not 10⁻⁸. The "symplectic defect"
>    figures quoted below were computed with an unweighted form that is only valid
>    on a uniform grid. Theorem 2 (and Theorem 5, which depended on it) are
>    **retracted**.
> 2. **Symplecticity does not help even when it is real.** A later operator
>    (SA-Cheb) *is* exactly symplectic (defect 2×10⁻¹⁶). In a controlled ablation
>    against an otherwise identical non-symplectic twin, the **non-symplectic twin
>    is more accurate on all six 1-D problems**. The claim below that "the
>    symplectic prior … is what saves it" is not supported.
>
> The surviving positive result is about the **basis**, not the geometry:
> Chebyshev operators beat Fourier operators by 3–5× on a non-periodic Hamiltonian
> problem, and lose on 4 of 5 periodic ones.

## Where it works best

1. **Hamiltonian ODEs (Tier 1).** Harmonic oscillator, pendulum, Kepler, and even the chaotic double pendulum. CKINO's "symplectic defect" — the structural error of the learned step — is **10⁻⁸**, basically at float-32 round-off. No baseline gets anywhere near that. Phase-space orbits stay closed for hundreds of steps where MLPs visibly spiral in or out.

2. **Conservation-law PDEs (Tier 3, reservoir/porous flow).** This is the strongest result. Over 400 rollout steps, FNO's mass conservation error blows up to **10²⁵** (it's effectively broken). CKINO stays at **10⁻²**. That's a 27-order-of-magnitude gap on the exact quantity petroleum/CO₂-storage engineers care about.

3. **Long-horizon stability in general.** Train one-step, roll out 200–800 steps. CKINO never goes NaN. FNO crashes on the wave equation at step 91. The ablation (`CKINO-NoSymp`, identical architecture minus the symplectic block) blows up 11 orders worse than full CKINO — proving the symplectic prior, not the operator backbone, is what saves it.

4. **Parameter efficiency.** 6 406 parameters vs FNO's 78 114. ~12× smaller, better long-horizon results.

## Where it doesn't work

1. **KdV equation.** CKINO drifts to NaN. The reason is structural and honest: KdV is Hamiltonian, but its symplectic structure is the *Gardner bracket*, not the textbook (q, p) split. CKINO's hard channel-pair split is the **wrong geometry** for KdV. Documented as a scope limitation in §7 of the paper. Fix is non-trivial — needs a learned Poisson bracket.

2. **One-step accuracy on smooth, non-symplectic problems.** A plain ResidualMLP with 20× more parameters beats CKINO on single-step error for Kepler and the double pendulum. The symplectic constraint is a *prior* — it costs you a little fitting flexibility in exchange for huge long-horizon stability. If your problem is short-time and dissipative, the prior is a tax, not a gift.

3. **Genuinely dissipative systems.** Anything with real friction, viscosity, or thermal coupling violates the prior. CKINO is the wrong tool there. (Most real reservoir flow is mildly dissipative, which is why §7 of the paper proposes splitting the operator into symplectic + Onsager-gradient pieces — that's the obvious next paper.)

## Is it a "next big breakthrough"?

**A breakthrough in a specific niche, not a general-purpose revolution.** Honest framing:

- **Yes, on the merits, for one specific community:** structure-preserving operator learning for long-horizon Hamiltonian and conservation-law PDEs. The 27-order mass-conservation gap on a Buckley–Leverett-style reservoir problem is the kind of number that makes a reviewer at *Nature Communications* / *JCP* / *NeurIPS* sit up. The 11-order ablation gap kills the "you just have a bigger model" objection cleanly. The double-pendulum result (96× lower symplectic defect with **5–20× fewer parameters**) is publishable on its own as a SympNet-family paper.

- **No, it's not "the next Transformer."** It doesn't change how anyone does language, vision, or generative modelling. It's not a general-purpose architecture. Its impact ceiling is "scientific ML for conservative systems" — a real and growing field (probably $100M-scale industrial spend in reservoir/CFD/molecular dynamics), but not the broader AI conversation.

- **The realistic positioning** is: this is **the symplectic answer to FNO**, in the same way HNN (Greydanus 2019) was the symplectic answer to plain MLPs for ODEs. Useful, citable, builds on a clear lineage, advances the field. Not a paradigm shift.

## How is it novel — what actually doesn't exist already?

The honest answer is that *each individual ingredient* has prior art. The novelty is in the **specific combination** and one bridge claim:

| Ingredient | Existing work | What's new in CKINO |
| --- | --- | --- |
| Symplectic integrator inside a network | SympNet (Jin 2020), HNN (Greydanus 2019) | They're finite-dimensional only — they take a *vector* $(q,p) \in \mathbb{R}^{2n}$ and return a vector. CKINO is the **first to lift this into a resolution-agnostic neural operator** that takes a *function* $u(\cdot)$ and returns a function. |
| Neural operators (resolution-agnostic PDE solvers) | FNO (Li 2021), DeepONet (Lu 2021), GNO (Anandkumar 2020) | None of them preserve any structure. They optimise pointwise MSE and accept whatever rollout behaviour falls out. CKINO is the **first neural operator with a provable approximate symplectic guarantee**. |
| Chebyshev–rational spectral basis | Standard in spectral element methods (Boyd 2001) | Used here as a **resolution-independent parameterisation of the kernel** rather than of the solution — this is what makes the operator transfer from N=32 training to N=128 inference with no retraining. FNO uses Fourier modes for the same purpose, but Fourier is wrong on non-periodic and non-smooth domains (porous flow, anything with shocks). |
| FiLM hypernetwork conditioning | Perez 2018, FiLM-ed operators (Brandstetter 2022) | Used to **modulate the symplectic correction without breaking symplecticity** — initialised to zero so the model degrades gracefully to a pure symplectic step. This zero-init trick is what makes training stable. |
| Lie-equivariant lifting | Lie-conv (Finzi 2020), E(3)-equivariant networks (Satorras 2021) | Used here for the **embedding** $u(\cdot) \to (q,p)$ so that group symmetries of the PDE (translation, scale) transfer to the latent Hamiltonian. Combined with symplecticity this gives a **double prior** — geometric + group-theoretic. |

### The one genuinely new theorem

§5 of the research paper, Theorem 1: **a neural operator constructed as (Mercer kernel integral) ∘ (Lie-equivariant lift) ∘ (Stoermer–Verlet block) ∘ (Lie-equivariant un-lift) is symplectic with defect bounded by the truncation error of the kernel's Mercer expansion.**

That sentence is — as far as I can find in the literature — **not previously written down**. The closest prior result is for finite-dimensional SympNets (Jin et al. 2020 Theorem 4.1), which doesn't extend to operators because the relevant Jacobian is infinite-dimensional. The Mercer-truncation bound is the bridge that makes it operator-valued.

### What would kill the novelty claim

If a reviewer found any of these, the novelty story would weaken:

1. A paper titled something like "Symplectic Fourier Neural Operator" or "Hamiltonian Neural Operator" after ~2023. I checked — there's *Hamiltonian Neural Networks* (Greydanus 2019, finite-dim), *Lagrangian Neural Networks* (Cranmer 2020, finite-dim), *Symplectic ODE-Net* (Zhong 2020, finite-dim), and a 2023 *Hamiltonian Graph Networks* (Sanchez-Gonzalez), which is operator-ish but mesh-based and not a kernel-integral operator.
2. A structure-preserving DeepONet. There's *Physics-Informed DeepONet* (Wang 2021) but it enforces residuals via soft loss, not via the architecture — fundamentally weaker because it doesn't give an exact algebraic guarantee on the Jacobian.
3. **The closest existing work is probably** the "Clifford Neural Layers" (Brandstetter 2023) line and the "Equivariant Neural Operators" (Helwig 2023) line — both preserve **symmetries** but not **symplectic structure**, which is a different geometric prior.

**So the defensible novelty claim is:** *first neural operator with an exact algebraic (not soft-loss) guarantee on approximate symplectic preservation, with a Mercer-truncation bound linking operator-level symplecticity to a learnable kernel.*

That's a real contribution. It's not the next Transformer. It is publishable, and the 27-order mass-conservation result on reservoir flow is genuinely striking.