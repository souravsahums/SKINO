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
STAGES = ["1d", "2d", "3d"]

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
    ap.add_argument("--experiment", default="skino-study")
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
    args = ap.parse_args(argv)
    save_fields = not args.no_save_fields

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
