"""Fire-and-forget trigger for the CKINO study on an Azure ML compute CLUSTER.

Run this FROM AN IN-AZURE MACHINE (a small CPU compute *instance* in the same
workspace). The workspace storage has public network access disabled, so a
laptop cannot upload the code snapshot - but a compute instance reaches storage
over the trusted Azure backbone and can.

What it does:
  * uploads the code snapshot (this repo, minus .amlignore), and
  * submits ONE command job per shard to the GPU cluster, each running
    ``track2.launcher`` over its slice with ``--device cuda``, then
  * RETURNS IMMEDIATELY. It never waits for or streams the jobs. The cluster
    runs them independently; watch them in the Studio.

Typical use, on the compute-instance terminal::

    conda env create -f track2/aml_trigger_env.yml     # first time only
    conda activate skino-trigger
    python -m track2.aml_trigger                        # trigger 1d+2d+3d, then exit

Only Azure ML jobs are triggered here; the heavy training happens on the
cluster's own curated PyTorch environment, not in this conda env.
"""
from __future__ import annotations

import argparse
import os

# --- workspace / cluster this repo targets ---
# Workspace/subscription/resource-group are auto-resolved from the compute
# instance's own context (track2/aml_config); set AML_* env vars to override.
from .aml_config import get_ml_client, workspace_label

CLUSTER = "gpu-t4"
# curated GPU environment used by the cluster jobs (not by this trigger)
ENVIRONMENT = "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/labels/latest"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # repo root -> uploaded as job code
STAGES = ["1d", "2d", "3d", "cgl"]

# Problems swept by the unconstrained-capacity study, with the spatial dim each
# one lives in (used only to size the sweep, not passed to the runner).
SWEEP_PROBLEMS = ["advection", "heat", "wave1d", "wave1d_dir", "burgers", "kdv",
                  "wave2d", "wave3d", "ns2d"]
# Every equation carries an energy functional, so drift is defined for all of
# them -- conserved for the Hamiltonian ones, decaying for the dissipative ones.
# The model's drift is always reported alongside the reference solver's own
# (energy_drift_true), so the dissipative rows stay interpretable.
LONGHORIZON_PROBLEMS = ["advection", "heat", "wave1d", "wave1d_dir", "burgers",
                        "kdv", "wave2d", "wave3d", "ns2d"]

_COLLECT = (" ; mkdir -p ./outputs && cp -r track2/results_paper/* ./outputs/ 2>/dev/null; "
            "ls ./outputs | head -50")


def make_command(stage, shard, shards, seeds, configs, save_fields):
    seed_s = " ".join(str(s) for s in seeds)
    cmd = (f"python -m track2.launcher --stage {stage} --seeds {seed_s} "
           f"--shard {shard} --num-shards {shards} --device cuda --jobs 1 --gpus 1")
    if configs:
        cmd += " --configs " + " ".join(configs)
    if save_fields:
        cmd += " --save-fields"
    # surface per-config results as AML job outputs for later download
    cmd += (" ; mkdir -p ./outputs && cp -r track2/results_paper/* ./outputs/ 2>/dev/null; "
            "ls ./outputs | head -50")
    return cmd


def make_extra_command(seed):
    """Standalone gap experiments: Darcy (non-periodic BC + zero-shot super-res),
    spectral-mode-truncation ablation, parameter-scaling sweep, zero-shot
    super-resolution on the rollout problems, and solver-speedup. Each writes its
    JSON into results_paper, which is copied to the job outputs."""
    cmds = [
        f"python -m track2.darcy --device cuda --budget 25000 --seed {seed} --epochs 60",
        f"python -m track2.mode_ablation --problem burgers --device cuda --seed {seed}",
        f"python -m track2.mode_ablation --problem kdv --device cuda --seed {seed}",
        f"python -m track2.scaling --problem burgers --device cuda --seed {seed}",
        f"python -m track2.scaling --problem kdv --device cuda --seed {seed}",
        f"python -m track2.discretization --device cuda --seed {seed}",
        f"python -m track2.discretization --device cuda --seed {seed} --lift-kind spectral",
        f"python -m track2.symplectic_defect --dims 1 2 3 --device cuda --seed {seed}",
    ]
    if seed == 0:
        cmds.append("python -m track2.speedup --device cuda --seed 0")
    return " ; ".join(cmds) + _COLLECT


def make_longhorizon_command(seed, steps):
    """Energy drift out to 10^4-10^5 steps for the symplectic-form comparison.

    Only the lift-free families carry the guarantee end to end, so both they and
    their lifted counterparts are run -- the difference between the two is the
    measurement of what the lift costs.  Families with no >1-D implementation are
    dropped per problem by the runner itself.
    """
    fams = ("sacheb_pure sacheb_pure_naive sacheb sacheb_naive "
            "skino skino_strict sno fno tfno")
    cmds = [f"python -m track2.longhorizon --problem {p} --families {fams} "
            f"--steps {steps} --device cuda --seed {seed}"
            for p in LONGHORIZON_PROBLEMS]
    return " ; ".join(cmds) + _COLLECT


def make_bestwidth_command(problem, seed):
    """Per-family capacity sweep with the matched-budget constraint removed."""
    cmds = [f"python -m track2.best_width --problem {problem} --mode {m} "
            f"--device cuda --seed {seed}"
            for m in ("recursive", "seq2seq")]
    return " ; ".join(cmds) + _COLLECT


# ---------------------------------------------------------------------------
# Review suite: the experiments requested in review, and ONLY those.  Each job is
# small and independent so the cluster queue packs them onto its nodes and a
# failure costs one job, not the suite.  Submitted to its own experiment
# (skino-review) so aml_fetch can gather exactly these results.
# ---------------------------------------------------------------------------
GRID_CONFIGS = {
    "cgl": ["sacheb_plain", "naive_plain", "sacheb_seq2seq", "naive_seq2seq",
            "purecheb_plain", "pureunif_plain", "sno_seq2seq", "tfno_seq2seq", "fno_plain"],
    "kte": ["sacheb_plain", "naive_plain", "kte_plain", "sacheb_seq2seq", "naive_seq2seq",
            "kte_seq2seq", "purecheb_plain", "pureunif_plain", "purekte_plain",
            "sno_seq2seq", "fno_plain"],
}
HEADLINE_CONFIGS = ["sacheb_plain", "naive_plain", "sacheb_seq2seq", "naive_seq2seq",
                    "purecheb_plain", "pureunif_plain"]
_STAGED = ["sacheb_pure_naive", "sacheb_canon_naive", "sacheb_nores_naive", "sacheb_naive",
           "persistence"]
LH_FAMILIES = {
    "wave1d": _STAGED + ["sacheb_pure_naive@random", "sacheb_pure_naive@short", "sno", "fno",
                         "sacheb_pure_naive@eps=0.01", "sacheb_pure_naive@eps=0.1",
                         "sacheb_pure_naive@eps=1.0"],
    "wave1d_dir": _STAGED + ["sacheb_pure_naive@random", "sacheb_pure_naive@short", "sno",
                             "fno", "sacheb_pure_naive@eps=0.01", "sacheb_pure_naive@eps=0.1",
                             "sacheb_pure_naive@eps=1.0"],
    "wave2d": _STAGED,
    "wave3d": _STAGED,
    "wave1d_cgl": ["sacheb_pure", "sacheb_pure_naive", "sacheb", "sacheb_naive", "persistence"],
    "wave1d_kte": ["sacheb_pure_kte", "sacheb_pure", "sacheb_pure_naive", "sacheb_kte",
                   "sacheb", "sacheb_naive", "persistence"],
}
MULTIRES_PROBLEMS = ["wave1d_cgl", "wave1d_dir"]
REVIEW_PARTS = ("grids", "seeds", "longhorizon", "multires")

# Cost model, seconds on one Tesla T4, from the train_time_s recorded in
# results_gpu_v3 (matrix: 512 traj x 600 steps x 15 epochs in 1-D).  CGL/KTE runs on
# 65 nodes are costed as the 64-point 1-D problems.
_MATRIX_S = {
    "plain": {"1d": 255, "wave2d": 81, "ns2d": 103, "wave3d": 73},
    "seq2seq": {"1d": 120, "wave2d": 37, "ns2d": 49, "wave3d": 32},
    "pure": {"wave1d": 458, "wave1d_dir": 342, "1d": 400, "wave2d": 115, "wave3d": 98},
    "sno_seq2seq": {"1d": 112}, "tfno_seq2seq": {"1d": 63}, "fno_plain": {"1d": 77},
}
_PER_RUN_OVERHEAD_S = 40          # process start, data generation, evaluation
# Long-horizon totals (training + 2e4-step rollout), measured per family class.
_LH_S = {
    "pure": {"wave1d": 447, "wave1d_dir": 347, "wave2d": 601, "wave3d": 1004, "1d": 350},
    "lifted": {"wave1d": 272, "wave1d_dir": 279, "wave2d": 468, "wave3d": 750, "1d": 280},
    "sno": {"wave1d": 286, "wave1d_dir": 293},
    "fno": {"wave1d": 97, "wave1d_dir": 103},
}
_ROLLOUT_ONLY = {"1d": 0.30, "wave2d": 0.6, "wave3d": 0.7}   # share of a pure LH run
_DIAG_OVERHEAD = {"1d": 1.15, "wave2d": 1.30, "wave3d": 1.25}
_JOB_STARTUP_S = 180              # image already cached on the node; ~600 s on a cold node


def _dimkey(problem):
    return problem if problem in ("wave2d", "wave3d", "ns2d") else "1d"


def _matrix_cost(problem, config):
    d = _dimkey(problem)
    if config.startswith("pure"):
        s = _MATRIX_S["pure"].get(problem, _MATRIX_S["pure"].get(d))
    elif config.endswith("seq2seq") and config in _MATRIX_S:
        s = _MATRIX_S[config][d]
    elif config.endswith("seq2seq"):
        s = _MATRIX_S["seq2seq"][d]
    elif config in _MATRIX_S:
        s = _MATRIX_S[config][d]
    else:
        s = _MATRIX_S["plain"][d]
    return s + _PER_RUN_OVERHEAD_S


def _lh_cost(problem, spec):
    d = _dimkey(problem)
    base = spec.split("@")[0]
    key = problem if problem in ("wave1d", "wave1d_dir", "wave2d", "wave3d") else d
    pure = _LH_S["pure"].get(key, _LH_S["pure"]["1d"])
    if base == "persistence":
        s = 0.25 * _ROLLOUT_ONLY[d] * pure
    elif spec.endswith("@random") or spec.endswith("@short"):
        s = _ROLLOUT_ONLY[d] * pure * (1.1 if spec.endswith("@short") else 1.0)
    elif base in ("sacheb_pure", "sacheb_pure_naive", "sacheb_pure_kte", "sacheb_canon_naive"):
        s = pure
    elif base in ("sno", "fno"):
        s = _LH_S[base].get(key, 300)
    else:
        s = _LH_S["lifted"].get(key, _LH_S["lifted"]["1d"])
    return s * _DIAG_OVERHEAD[d]


def review_jobs(parts=REVIEW_PARTS, grid_seeds=(0, 1, 2, 3, 4), extra_seeds=(3, 4),
                lh_seeds=(0, 1, 2), mr_seeds=(0, 1, 2), lh_steps=20000):
    """[(display_name, command, estimated_seconds)] for the review experiments."""
    L = "python -m track2.launcher --device cuda --jobs 1 --gpus 1 --save-ckpt --n-test 32"
    jobs = []
    if "grids" in parts:
        for stage in ("cgl", "kte"):
            prob = {"cgl": "wave1d_cgl", "kte": "wave1d_kte"}[stage]
            for s in grid_seeds:
                cfgs = GRID_CONFIGS[stage]
                jobs.append((f"review-{stage}-s{s}",
                             f"{L} --stage {stage} --seeds {s} --save-fields "
                             f"--configs {' '.join(cfgs)}",
                             sum(_matrix_cost(prob, c) for c in cfgs)))
    if "seeds" in parts:
        halves = {"a": ["advection", "heat", "wave1d"], "b": ["wave1d_dir", "burgers", "kdv"]}
        cfgs = " ".join(HEADLINE_CONFIGS)
        two_ch = {"wave1d", "wave1d_dir", "wave2d", "wave3d"}
        for s in extra_seeds:
            for h, probs in halves.items():
                est = sum(_matrix_cost(p, c) for p in probs for c in HEADLINE_CONFIGS
                          if not c.startswith("pure") or p in two_ch)
                jobs.append((f"review-seeds-1d{h}-s{s}",
                             f"{L} --stage 1d --problems {' '.join(probs)} --seeds {s} "
                             f"--configs {cfgs}", est))
            est = sum(_matrix_cost(p, c) for p in ("wave2d", "ns2d", "wave3d")
                      for c in HEADLINE_CONFIGS if not c.startswith("pure") or p in two_ch)
            jobs.append((f"review-seeds-hid-s{s}",
                         f"{L} --stage 2d --seeds {s} --configs {cfgs} ; "
                         f"{L} --stage 3d --seeds {s} --configs {cfgs}", est))
    if "longhorizon" in parts:
        for p, fams in LH_FAMILIES.items():
            hi = p in ("wave2d", "wave3d")
            flags = ("--diag-dtype float32 --lyap-steps 300" if hi
                     else "--diag-dtype float64 --lyap-steps 1000")
            for s in lh_seeds:
                jobs.append((f"review-lh-{p}-s{s}",
                             f"python -m track2.longhorizon --problem {p} --steps {lh_steps} "
                             f"--device cuda --seed {s} --diagnostics --track-training "
                             f"--save-ckpt --tag review {flags} --families "
                             + " ".join(fams),
                             sum(_lh_cost(p, f) for f in fams)))
    if "multires" in parts:
        fam_s = (_MATRIX_S["plain"]["1d"] * 2 + 400 * 2 + 235 + 77) * 0.83
        for s in mr_seeds:
            cmd = " ; ".join(f"python -m track2.multires --problem {p} --device cuda "
                             f"--seed {s} --save-ckpt" for p in MULTIRES_PROBLEMS)
            est = fam_s * len(MULTIRES_PROBLEMS) + 120
            if s == mr_seeds[0]:
                cmd += (" ; python -m track2.symplectic_defect --dims 1 2 3 --forms W_cheb "
                        "W_unif W_kte --out-tag forms3 --device cuda --seed 0")
                est += 120
            jobs.append((f"review-multires-s{s}", cmd, est))
    return [(n, c + _COLLECT, e + _JOB_STARTUP_S) for n, c, e in jobs]


def makespan(seconds, nodes):
    """Longest-processing-time-first packing: wall-clock with ``nodes`` in parallel."""
    load = [0.0] * max(nodes, 1)
    for s in sorted(seconds, reverse=True):
        load[load.index(min(load))] += s
    return max(load)


def print_review_plan(jobs, nodes):
    print(f"\n{'job':30s} {'est. min':>9}")
    for name, _, est in jobs:
        print(f"{name:30s} {est / 60:9.0f}")
    total = sum(e for _, _, e in jobs)
    wall = makespan([e for _, _, e in jobs], nodes)
    print(f"\n{len(jobs)} jobs   total {total / 3600:.1f} GPU-h   "
          f"wall-clock on {nodes} node(s) ~{wall / 3600:.1f} h   "
          f"(+/-30%: rollout and diagnostic shares are estimated, training is measured)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trigger CKINO cluster jobs and exit.")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES,
                    help="which stages to launch (default: all three)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shards", type=int, default=2,
                    help="parallel cluster jobs per stage (keep <= cluster max nodes)")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="launcher configs to run (default: the launcher's core set; "
                         "use 'full' for the whole matrix)")
    ap.add_argument("--cluster", default=CLUSTER)
    ap.add_argument("--experiment", default=None,
                    help="AML experiment name (default: skino-study, or skino-review with --review)")
    ap.add_argument("--no-save-fields", action="store_true",
                    help="skip saving prediction fields (smaller/faster)")
    ap.add_argument("--extras", action="store_true",
                    help="also submit the standalone gap experiments (Darcy, mode ablation, speedup)")
    ap.add_argument("--longhorizon", action="store_true",
                    help="submit the long-horizon energy-drift runs (one job per seed)")
    ap.add_argument("--lh-steps", type=int, default=20000,
                    help="rollout length for --longhorizon")
    ap.add_argument("--best-width", action="store_true",
                    help="submit the unconstrained-capacity sweeps (one job per problem)")
    ap.add_argument("--sweep-problems", nargs="*", default=SWEEP_PROBLEMS,
                    help="problems for --best-width")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be submitted and exit")
    ap.add_argument("--review", action="store_true",
                    help="submit ONLY the review experiments (nothing from the main matrix)")
    ap.add_argument("--review-parts", nargs="+", default=list(REVIEW_PARTS),
                    choices=REVIEW_PARTS, help="subset of the review suite")
    ap.add_argument("--nodes", type=int, default=2,
                    help="cluster nodes, used only for the wall-clock estimate")
    args = ap.parse_args(argv)
    save_fields = not args.no_save_fields
    args.experiment = args.experiment or ("skino-review" if args.review else "skino-study")

    if args.review:
        jobs = review_jobs(tuple(args.review_parts), lh_steps=args.lh_steps)
        print(f"workspace : {workspace_label()}")
        print(f"cluster   : {args.cluster}   experiment: {args.experiment}")
        print(f"review    : {' '.join(args.review_parts)}")
        print_review_plan(jobs, args.nodes)
        if args.dry_run:
            for name, cmd, _ in jobs:
                print(f"\n[{name}]\n  {cmd}")
            print("\n--dry-run: nothing submitted.")
            return
        from azure.ai.ml import command
        from azure.identity import DefaultAzureCredential

        ml = get_ml_client(DefaultAzureCredential())
        print(f"\nconnected to {ml.workspace_name} (rg={ml.resource_group_name})\n")
        for name, cmd, _ in jobs:
            created = ml.jobs.create_or_update(command(
                code=ROOT, command=cmd, environment=ENVIRONMENT, compute=args.cluster,
                display_name=name, experiment_name=args.experiment,
                environment_variables={"KMP_DUPLICATE_LIB_OK": "TRUE"}))
            print(f"submitted {created.name}  {name}")
        print(f"\n{len(jobs)} review job(s) queued on '{args.cluster}' in experiment "
              f"'{args.experiment}'. Fetch with:\n"
              f"  python -m track2.aml_fetch --experiment {args.experiment} "
              f"--out results_review --with-fields --with-ckpt")
        return

    print(f"workspace : {workspace_label()}")
    print(f"cluster   : {args.cluster}")
    print(f"stages    : {args.stages}   seeds: {args.seeds}   shards/stage: {args.shards}")
    print(f"configs   : {args.configs or 'core (launcher default)'}")

    if args.dry_run:
        for stage in args.stages:
            for k in range(args.shards):
                print(f"\n[{stage} shard {k}]\n  " +
                      make_command(stage, k, args.shards, args.seeds, args.configs, save_fields))
        if args.extras:
            for s in args.seeds:
                print(f"\n[extras seed {s}]\n  " + make_extra_command(s))
        if args.longhorizon:
            for s in args.seeds:
                print(f"\n[longhorizon seed {s}]\n  "
                      + make_longhorizon_command(s, args.lh_steps))
        if args.best_width:
            for p in args.sweep_problems:
                for s in args.seeds:
                    print(f"\n[best-width {p} seed {s}]\n  " + make_bestwidth_command(p, s))
        print("\n--dry-run: nothing submitted.")
        return

    from azure.ai.ml import command
    from azure.identity import DefaultAzureCredential

    ml = get_ml_client(DefaultAzureCredential())
    print(f"connected to {ml.workspace_name} (rg={ml.resource_group_name})\n")

    submitted = []
    for stage in args.stages:
        for k in range(args.shards):
            job = command(
                code=ROOT,                               # uploaded over the backbone
                command=make_command(stage, k, args.shards, args.seeds,
                                     args.configs, save_fields),
                environment=ENVIRONMENT,
                compute=args.cluster,
                display_name=f"skino-{stage}-shard{k}of{args.shards}",
                experiment_name=args.experiment,
                environment_variables={"KMP_DUPLICATE_LIB_OK": "TRUE"},
            )
            created = ml.jobs.create_or_update(job)       # submits and returns; no wait/stream
            submitted.append(created.name)
            print(f"submitted {created.name}  ->  {created.studio_url}")

    if args.extras:
        for s in args.seeds:
            job = command(
                code=ROOT,
                command=make_extra_command(s),
                environment=ENVIRONMENT,
                compute=args.cluster,
                display_name=f"skino-extras-s{s}",
                experiment_name=args.experiment,
                environment_variables={"KMP_DUPLICATE_LIB_OK": "TRUE"},
            )
            created = ml.jobs.create_or_update(job)
            submitted.append(created.name)
            print(f"submitted {created.name}  ->  {created.studio_url}")

    if args.longhorizon:
        for s in args.seeds:
            job = command(
                code=ROOT,
                command=make_longhorizon_command(s, args.lh_steps),
                environment=ENVIRONMENT,
                compute=args.cluster,
                display_name=f"skino-longhorizon-s{s}",
                experiment_name=args.experiment,
                environment_variables={"KMP_DUPLICATE_LIB_OK": "TRUE"},
            )
            created = ml.jobs.create_or_update(job)
            submitted.append(created.name)
            print(f"submitted {created.name}  ->  {created.studio_url}")

    if args.best_width:
        for p in args.sweep_problems:
            for s in args.seeds:
                job = command(
                    code=ROOT,
                    command=make_bestwidth_command(p, s),
                    environment=ENVIRONMENT,
                    compute=args.cluster,
                    display_name=f"skino-bestwidth-{p}-s{s}",
                    experiment_name=args.experiment,
                    environment_variables={"KMP_DUPLICATE_LIB_OK": "TRUE"},
                )
                created = ml.jobs.create_or_update(job)
                submitted.append(created.name)
                print(f"submitted {created.name}  ->  {created.studio_url}")

    print(f"\n{len(submitted)} job(s) queued on '{args.cluster}'. Trigger-and-forget: done. "
          f"Track them in the Studio (Jobs -> experiment '{args.experiment}'). "
          f"You can stop this compute instance now - the cluster keeps running.")


if __name__ == "__main__":
    main()
