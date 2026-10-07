"""Long-horizon rollout: does the symplectic form you preserve matter eventually?

The matched-capacity matrix rolls out a few hundred steps. The classical
argument for symplectic integrators is not about accuracy at short times -- it is
that the energy error stays *bounded* over exponentially long horizons while a
non-symplectic scheme drifts secularly. That only becomes visible at 10^4-10^5
steps, so this module runs the pair out that far.

Two things have to be right for the measurement to mean anything.

1. The deployed map must actually be symplectic. The lifted SA-Cheb families
   wrap their symplectic shears in pointwise lift/projection layers and are then
   driven residually as x + model(x); none of that is symplectic, so an energy
   curve for them describes the wrapper. The ``*_pure`` families drop the lift
   and step non-residually, so the trained one-step map carries the guarantee.

2. The form must match the grid. ``sacheb`` preserves the Clenshaw-Curtis form
   W_cheb; ``sacheb_naive`` preserves the uniform form W_unif. Both are exact
   (see track2.symplectic_defect) -- they differ in *which* structure they keep.
   Every problem here is discretised on a uniform grid, so W_unif is the
   physically relevant form and W_cheb is not.

Two quantities are tracked:

    rel_rms          accuracy against the reference solver. Saturates near
                     sqrt(2) for everything once phases decorrelate -- this is
                     NOT the discriminating measurement.
    energy_drift     |E(t) - E(0)| / E(0) on the *model's own* trajectory in
                     physical units. This is the classical signature: bounded
                     oscillation vs secular growth.

Truth and prediction are advanced in lockstep one step at a time, so peak memory
is independent of the horizon and 10^5 steps costs what 10^3 does.

Run:
    python -m track2.longhorizon --problem wave1d --steps 20000 --device cuda
    python -m track2.longhorizon --families sacheb_pure sacheb_pure_naive --steps 50000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch

from .data import build_data
from .experiments_paper import TrainConfig, hardware_info, pin_numerics, train_recursive_pinn
from .models import (FAMILIES_1D, FAMILIES_2D, FAMILIES_3D, PHASE_SPACE_FAMILIES,
                     build_matched, is_residual)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SKINO_RESULTS_DIR") or os.path.join(HERE, "results_paper")

# Long rollout only means something for models that step autoregressively.
# seq2seq emits a fixed-length block and cannot be extended past its head.
#
# The lift-free families drop the lift/projection layers, so the trained one-step
# map is itself symplectic rather than a symplectic core wrapped in two linear
# layers that are not.  Only for those is an energy-drift curve a test of the
# classical symplectic argument; for the lifted families it is a test of the
# wrapper.  Residual vs non-residual stepping is decided in models.is_residual.
PURE = PHASE_SPACE_FAMILIES
DTYPES = {"float64": torch.float64, "float32": torch.float32}

FAMILY_LABEL = {
    "sacheb": "SA-Cheb, W_cheb form (lifted: end-to-end map not symplectic)",
    "sacheb_naive": "SA-Cheb, W_unif form (lifted: end-to-end map not symplectic)",
    "sacheb_kte": "SA-Cheb, W_kte form (lifted: end-to-end map not symplectic)",
    "sacheb_pure": "SA-Cheb, W_cheb form (symplectic end to end)",
    "sacheb_pure_naive": "SA-Cheb, W_unif form (symplectic end to end)",
    "sacheb_pure_kte": "SA-Cheb, W_kte form (symplectic end to end)",
    "sacheb_nores_naive": "SA-Cheb, W_unif, lift/projection but NO residual update",
    "sacheb_canon_naive": "SA-Cheb, W_unif, canonical (symplectic) wrapper",
    "persistence": "persistence x_{t+1} = x_t (exactly symplectic, no dynamics)",
    "skino": "CKINO kernel (volume-preserving only)",
    "sno": "SNO (symplectic, Fourier)",
    "fno": "FNO (unstructured)",
}


class Persistence(torch.nn.Module):
    """x -> x.  Exactly symplectic, zero energy drift, and no dynamics at all."""

    def forward(self, x):
        return x


def parse_spec(spec: str):
    """'family', 'persistence', or 'family@random' | '@short' | '@eps=<float>'.

    @random  exactly-symplectic family left untrained with fixed nonzero gains
    @short   the same family trained for a single K=1 epoch
    @eps=x   adjoint perturbed by relative size x (symplectic in no form)
    """
    base, _, mod = spec.partition("@")
    if not mod:
        return base, {}
    if mod in ("random", "short"):
        return base, {"control": mod}
    if mod.startswith("eps="):
        return base, {"adjoint_eps": float(mod[4:])}
    raise ValueError(f"unknown modifier in {spec!r}")


def spec_label(spec: str) -> str:
    base, mods = parse_spec(spec)
    lab = FAMILY_LABEL.get(base, base)
    if mods.get("control") == "random":
        lab += " [untrained, gains fixed at 0.3]"
    elif mods.get("control") == "short":
        lab += " [trained one epoch only]"
    elif "adjoint_eps" in mods:
        lab += f" [adjoint perturbed, eps={mods['adjoint_eps']:g}]"
    return lab


def log_checkpoints(n_steps: int, per_decade: int = 12) -> list:
    """Log-spaced step indices, so a 10^5 rollout costs the same reporting as 10^3."""
    hi = math.log10(n_steps)
    raw = [int(round(10 ** e)) for e in np.linspace(0, hi, int(per_decade * hi) + 1)]
    return sorted({c for c in raw if 1 <= c <= n_steps})


@torch.no_grad()
def stream_rollout(model, data, cfg, n_steps: int, checkpoints, n_test: int = 8,
                   snap_at=(), snap_traj: int = 2, per_traj_at=()):
    """Advance model and reference solver in lockstep; return per-checkpoint metrics.

    Memory is O(1) in ``n_steps`` -- nothing but the current state is retained,
    except the few snapshots requested in ``snap_at`` (kept so the long rollout
    can be shown as fields, not just curves).  At ``per_traj_at`` steps the
    row also carries each trajectory's own error and drift, for intervals.
    """
    prob = data.problem
    model.eval()
    ckpt = set(checkpoints)
    want = set(snap_at)
    keep = set(per_traj_at)

    phys = data.test_traj[:n_test, 0].contiguous()          # (B, C, *grid) physical
    truth = data.normalize(phys, channel_dim=1)
    x = truth.clone()
    red = tuple(range(1, x.dim()))                          # every non-batch axis

    e0_model = prob.energy(data.denormalize(x, channel_dim=1))
    e0_true = prob.energy(phys)
    rows, diverged_at, snaps = [], None, {}

    for t in range(1, n_steps + 1):
        phys = prob.true_step(phys)
        x = x + model(x) if cfg.residual else model(x)
        if not torch.isfinite(x).all():
            diverged_at = diverged_at or t
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

        if t in want:
            snaps[t] = (x[:snap_traj].cpu().numpy(),
                        data.normalize(phys, channel_dim=1)[:snap_traj].cpu().numpy())

        if t in ckpt:
            ref = data.normalize(phys, channel_dim=1)
            num = (x - ref).pow(2).mean(dim=red).sqrt()
            den = ref.pow(2).mean(dim=red).sqrt() + 1e-12
            e_m = prob.energy(data.denormalize(x, channel_dim=1))
            e_t = prob.energy(phys)
            a = (x - x.mean(dim=red, keepdim=True)).flatten(1)
            b = (ref - ref.mean(dim=red, keepdim=True)).flatten(1)
            corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)
            drift = (e_m - e0_model).abs() / (e0_model.abs() + 1e-12)
            rows.append({
                "step": t,
                "rel_rms": float(num.div(den).mean()),
                "energy_drift": float(drift.mean()),
                "energy_drift_true": float(((e_t - e0_true).abs() / (e0_true.abs() + 1e-12)).mean()),
                "amp_ratio": float((x.flatten(1).std(1) / (ref.flatten(1).std(1) + 1e-12)).mean()),
                "pattern_corr": float(corr.mean()),
            })
            if t in keep:
                rows[-1]["per_traj_rel_rms"] = [float(f"{v:.6g}") for v in num.div(den).tolist()]
                rows[-1]["per_traj_energy_drift"] = [float(f"{v:.6g}") for v in drift.tolist()]
                rows[-1]["per_traj_pattern_corr"] = [float(f"{v:.6g}") for v in corr.tolist()]
    return rows, diverged_at, snaps


def secular_slope(rows, lo_frac: float = 0.1) -> float:
    """Slope of log10(energy drift) against log10(step) over the final decade.

    ~0 means the drift is bounded (the symplectic signature); ~1 means it grows
    linearly with step count (the classical non-symplectic failure).
    """
    pts = [(r["step"], r["energy_drift"]) for r in rows
           if r["step"] >= lo_frac * rows[-1]["step"] and r["energy_drift"] > 0]
    if len(pts) < 3:
        return float("nan")
    xs = np.log10([p[0] for p in pts])
    ys = np.log10([p[1] for p in pts])
    return float(np.polyfit(xs, ys, 1)[0])


def _training_tracker(data, prob, residual, n_probe: int, roll_n: int = 4,
                      max_states: int = 64, dtype=torch.float64):
    """Per-epoch: one-step val error, end-to-end defect, and a short-rollout check.

    Answers whether the defect of the deployed map tracks rollout stability as
    training proceeds, without a second training run.
    """
    from .jacobian_diag import end_to_end_defect, float64_copy, physical_weight, step_map

    hist = []
    val = data.normalize(data.val_traj, channel_dim=2)
    xv = val[:, :-1:10].flatten(0, 1)[:max_states]
    yv = val[:, 1::10].flatten(0, 1)[:max_states]
    test = data.normalize(data.test_traj[:roll_n], channel_dim=2).transpose(0, 1)
    x0 = test[0, :1]
    W = physical_weight(prob, tuple(x0.shape[2:]), device=x0.device, dtype=dtype)
    red = tuple(range(1, test.dim() - 1))

    def callback(ep, model, train_loss):
        model.eval()
        f = step_map(model, residual)
        with torch.no_grad():
            one = float(((f(xv) - yv) ** 2).sum() / ((yv ** 2).sum() + 1e-12))
            x = test[0]
            for _ in range(1, test.shape[0]):
                x = torch.nan_to_num(f(x), nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
            ref = test[-1]
            rms = float((((x - ref) ** 2).sum(red).sqrt()
                         / ((ref ** 2).sum(red).sqrt() + 1e-12)).mean())
            growth = float((x.flatten(1).norm(dim=1)
                            / (test[0].flatten(1).norm(dim=1) + 1e-12)).max())
        rec = {"epoch": ep + 1, "train_loss": train_loss, "val_onestep_rel_mse": one,
               "rollout_steps": test.shape[0] - 1, "rollout_rel_rms": rms,
               "rollout_norm_ratio_max": growth}
        if x0.shape[1] % 2 == 0:
            g = step_map(float64_copy(model, dtype), residual)
            rec["defect_end_to_end"] = end_to_end_defect(g, x0.to(dtype), W, n_probe)
        hist.append(rec)

    return callback, hist


def run_family(spec, a, data, prob, sd, dev, ckpts, snaps_at, per_traj_at):
    """Build, train (or not, for controls), diagnose and roll out one family spec."""
    from ckino.sacheb import SAChebShear
    from .jacobian_diag import diagnose

    base, mods = parse_spec(spec)
    control = mods.get("control")
    model_kw = {k: v for k, v in mods.items() if k == "adjoint_eps"}
    residual = False if base == "persistence" else is_residual(base)
    cfg = TrainConfig(k_schedule=[1, 2, 4], epochs_per_k=max(round(a.epochs / 3), 1),
                      stride=a.stride, noise_std=0.0, lambda_energy=0.0, stencil=1,
                      tf_start=1.0, tf_end=0.0, batch=a.batch,
                      residual=residual, seed=a.seed)
    # Re-seeded per family, so a family's initialisation does not depend on which
    # other families share the job.
    torch.manual_seed(a.seed); random.seed(a.seed)
    if base == "persistence":
        model, npar, wr = Persistence(), 0, (0, 0)
    elif a.width:
        from .models import build_at_width
        model, npar = build_at_width(base, sd, prob.n_channels, prob.grid_n, prob.dt,
                                     a.width[0], a.width[1], model_kw=model_kw)
        wr = tuple(a.width)
    else:
        model, npar, wr = build_matched(base, sd, prob.n_channels, prob.grid_n,
                                        prob.dt, a.budget, model_kw=model_kw)
    model = model.to(dev)
    print(f"\n=== {spec} === params={npar:,} width={wr} residual={residual}")
    print(f"    {spec_label(spec)}")

    t0 = time.time()
    hist = None
    if control == "random":
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, SAChebShear):
                    mod.gain.fill_(0.3)
    elif base != "persistence":
        if control == "short":
            cfg.k_schedule, cfg.epochs_per_k = [1], 1
        cb = None
        if a.track_training:
            cb, hist = _training_tracker(data, prob, residual, a.diag_probes // 4 or 8,
                                         dtype=DTYPES[a.diag_dtype])
        train_recursive_pinn(model, data, cfg, 0.0, tag=spec, on_epoch_end=cb)
    train_s = time.time() - t0

    rec = {"family": base, "spec": spec, "params": npar, "width": list(wr),
           "label": spec_label(spec), "residual_update": residual,
           "symplectic_end_to_end": (base in PURE or base == "persistence")
           and "adjoint_eps" not in mods,
           "adjoint_eps": mods.get("adjoint_eps", 0.0), "control": control,
           "train_time_s": train_s}
    if hist is not None:
        rec["training"] = hist
    if a.diagnostics:
        t1 = time.time()
        x0 = data.normalize(data.test_traj[:1, 0], channel_dim=1)
        try:
            rec["diagnostics"] = diagnose(model, residual, prob, x0, n_probe=a.diag_probes,
                                          lyap_steps=a.lyap_steps, seed=a.seed,
                                          dtype=DTYPES[a.diag_dtype])
        except Exception as exc:
            rec["diagnostics"] = {"error": f"{type(exc).__name__}: {exc}"}
        rec["diagnostics_time_s"] = time.time() - t1
        d = rec["diagnostics"]
        print(f"  diag: defect={d.get('defect_end_to_end', float('nan')):.3e}  "
              f"rho={d.get('spectrum', {}).get('spectral_radius', float('nan')):.4f}  "
              f"lyap={d.get('lyapunov', {}).get('lambda_max', float('nan')):.3e}")
    if a.save_ckpt and npar > 0:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", spec)
        torch.save(model.state_dict(), os.path.join(
            RES, f"ckpt_lh_{a.problem}_{safe}_s{a.seed}.pt"))

    t2 = time.time()
    rows, div, snaps = stream_rollout(model, data, cfg, a.steps, ckpts, a.n_test,
                                      snap_at=snaps_at, per_traj_at=per_traj_at)
    rec.update({"diverged_at": div, "rollout_time_s": time.time() - t2,
                "energy_slope": secular_slope(rows), "curve": rows})
    print(f"{'step':>8} {'rel_rms':>10} {'E-drift':>11} {'corr':>7}")
    for r in rows:
        if r["step"] in (ckpts[0], *ckpts[len(ckpts) // 4::len(ckpts) // 4]):
            print(f"{r['step']:>8} {r['rel_rms']:>10.4f} "
                  f"{r['energy_drift']:>11.3e} {r['pattern_corr']:>7.3f}")
    print(f"  energy-drift slope over final decade = {rec['energy_slope']:.3f} "
          f"(0 = bounded, 1 = secular)" + (f"   [diverged at step {div}]" if div else ""))
    return rec, snaps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="wave1d",
                    help="Hamiltonian problem with a conserved energy")
    ap.add_argument("--families", nargs="*",
                    default=["sacheb_pure", "sacheb_pure_naive", "sacheb", "sacheb_naive"],
                    help="family names, 'persistence', or family@random|@short|@eps=<x>")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--budget", type=int, default=25000,
                    help="matched parameter budget; 0 selects each family's best width")
    ap.add_argument("--width", type=int, nargs=2, default=None,
                    metavar=("W", "R"), help="override with an explicit (width, rank)")
    ap.add_argument("--n-traj", type=int, default=384)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=400, help="training trajectory length")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap-at", type=int, nargs="*",
                    default=[10, 100, 1000, 5000, 20000],
                    help="steps at which to keep the field itself, for figures")
    ap.add_argument("--per-traj-at", type=int, nargs="*", default=[1000, 5000, 20000],
                    help="steps at which each test trajectory's own error is kept")
    ap.add_argument("--diagnostics", action="store_true",
                    help="end-to-end defect, Jacobian spectrum and Lyapunov growth per family")
    ap.add_argument("--diag-probes", type=int, default=32)
    ap.add_argument("--diag-dtype", default="float64", choices=list(DTYPES),
                    help="float64 resolves exact symplecticity to ~1e-15; float32 is ~30x "
                         "cheaper on a T4 and still separates 1e-7 from O(1e-2)")
    ap.add_argument("--lyap-steps", type=int, default=1000)
    ap.add_argument("--track-training", action="store_true",
                    help="log one-step error, defect and short-rollout stability every epoch")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    pin_numerics()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    data = build_data(a.problem, n_train=a.n_traj, n_val=12, n_test=max(12, a.n_test),
                      horizon=a.horizon, device=dev)
    prob = data.problem
    sd = getattr(prob, "spatial_dims", 1)
    if not hasattr(prob, "energy"):
        raise SystemExit(f"{a.problem} has no energy invariant; long-horizon drift is undefined")
    ckpts = log_checkpoints(a.steps)
    snaps_at = [s for s in a.snap_at if s <= a.steps]
    per_traj_at = sorted({min(s, a.steps) for s in a.per_traj_at} & set(ckpts))
    allowed = set({1: FAMILIES_1D, 2: FAMILIES_2D, 3: FAMILIES_3D}[sd]) | {"persistence"}
    specs = [s for s in a.families if parse_spec(s)[0] in allowed]
    skipped = [s for s in a.families if s not in specs]
    # A lift-free symplectic map needs a canonical (q, p) pair to act on.
    if prob.n_channels != 2:
        scalar = [s for s in specs if parse_spec(s)[0] in PURE]
        specs = [s for s in specs if s not in scalar]
        if scalar:
            print(f"[skip] {' '.join(scalar)}: {a.problem} is a scalar field, "
                  f"so it has no canonical (q, p) split")
    print(f"[{a.problem}] d={sd} steps={a.steps} checkpoints={len(ckpts)} device={dev}")
    print(f"families: {' '.join(specs)}"
          + (f"   (skipped, no {sd}-D impl: {' '.join(skipped)})" if skipped else ""))

    suffix = f"_{a.tag}" if a.tag else ""
    tag = f"longhorizon_{a.problem}_s{a.seed}{suffix}"
    out = {"_meta": {"problem": f"longhorizon_{a.problem}", "steps": a.steps,
                     "budget": a.budget, "seed": a.seed, "n_test": a.n_test,
                     "n_traj": a.n_traj, "horizon": a.horizon, "epochs": a.epochs,
                     "stride": a.stride, "checkpoints": ckpts, "snap_at": snaps_at,
                     "per_traj_at": per_traj_at, "families": specs,
                     "hardware": hardware_info(dev)}}
    field_store = {}
    for spec in specs:
        try:
            out[spec], snaps = run_family(spec, a, data, prob, sd, dev, ckpts,
                                          snaps_at, per_traj_at)
            for t, (pred, ref) in snaps.items():
                field_store[f"{spec}__pred__{t}"] = pred
                field_store[f"{spec}__truth__{t}"] = ref
        except Exception as exc:
            out[spec] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  !! {spec} FAILED: {type(exc).__name__}: {exc}")
        # written after every family, so a job cut short keeps what it finished
        with open(os.path.join(RES, f"{tag}.json"), "w") as f:
            json.dump(out, f, indent=2)

    print(f"\n[saved] {tag}.json")
    if field_store:
        fp = os.path.join(RES, f"lhfields_{a.problem}_s{a.seed}{suffix}.npz")
        np.savez_compressed(fp, **field_store)
        print(f"[saved] {os.path.basename(fp)} "
              f"({os.path.getsize(fp) // 1024} KB, {len(field_store) // 2} snapshots)")
    return out


if __name__ == "__main__":
    main()
