"""Submit the SKINO study to Azure ML.

The experiment matrix is embarrassingly parallel, so this submits one command
job per shard; each job runs :mod:`track2.launcher` over its slice and writes
into ``./outputs``, which Azure ML captures automatically.

Nothing here provisions infrastructure - it targets compute clusters that
already exist in the workspace and that scale to zero when idle.

Typical use::

    # inspect the plan and the cost estimate, submit nothing
    python -m track2.azureml_submit --stage 1d --dry-run

    # one cheap validation job first
    python -m track2.azureml_submit --stage 1d --shards 1 --smoke

    # then the full matrix
    python -m track2.azureml_submit --stage 1d --shards 4

    # collect results when the jobs finish
    python -m track2.azureml_submit --download
"""
from __future__ import annotations

import argparse
import os

SUBSCRIPTION = os.environ.get("AML_SUBSCRIPTION_ID", "<your-subscription-id>")
RESOURCE_GROUP = os.environ.get("AML_RESOURCE_GROUP", "<your-resource-group>")
WORKSPACE = os.environ.get("AML_WORKSPACE", "<your-workspace>")

# name -> (GPUs per node, approx USD/hour) for the clusters in this workspace
CLUSTERS = {
    "gpu-t4": (1, 0.75),              # Standard_NC8as_T4_v3  1 x Tesla T4 16GB
}
# Every stage must run on ONE cluster so the comparison is like-for-like: mixing
# GPU generations changes the arithmetic (TF32 on Ampere+) and the hardware
# becomes a confounding variable alongside the model.
STAGE_CLUSTER = "gpu-t4"

ENVIRONMENT = "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/labels/latest"

# jobs in the matrix per stage, used only for the runtime/cost estimate
STAGE_JOBS = {"1d": 120, "2d": 15, "3d": 15}
MINUTES_PER_JOB_GPU = 1.5


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_command(stage, shard, shards, gpus, seeds, smoke, save_fields):
    seed_s = " ".join(str(s) for s in seeds)
    if smoke:
        # one tiny job end-to-end: proves data gen, training, metrics and output capture
        return ("python -m track2.experiments_paper --problem heat "
                "--n-traj 16 --horizon 60 --t-out 30 --epochs 2 --stride 20 "
                "--configs skino_noise --out-tag smoke --device cuda && "
                "mkdir -p ./outputs && cp -r track2/results_paper/* ./outputs/ && "
                "echo SMOKE_OK && ls -la ./outputs")
    return (f"python -m track2.launcher --stage {stage} --seeds {seed_s} "
            f"--shard {shard} --num-shards {shards} "
            f"--device cuda --jobs {gpus} --gpus {gpus}"
            + (" --save-fields" if save_fields else "")
            + " ; mkdir -p ./outputs && cp -r track2/results_paper/* ./outputs/ 2>/dev/null; "
              "ls ./outputs | head -50")


def estimate(stage, shards, gpus, rate):
    jobs = STAGE_JOBS.get(stage, 0)
    per_shard = jobs / max(shards, 1)
    minutes = per_shard * MINUTES_PER_JOB_GPU / max(gpus, 1)
    hours = minutes / 60.0
    return jobs, minutes, hours * rate * shards


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="1d", choices=["1d", "2d", "3d"])
    ap.add_argument("--cluster", default="gpu-t4", choices=list(CLUSTERS))
    ap.add_argument("--shards", type=int, default=1, help="parallel jobs (= nodes)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--save-fields", action="store_true", default=True)
    ap.add_argument("--smoke", action="store_true", help="submit one tiny validation job")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, submit nothing")
    ap.add_argument("--download", action="store_true", help="download completed job outputs")
    ap.add_argument("--experiment", default="skino-study")
    args = ap.parse_args(argv)

    gpus, rate = CLUSTERS[args.cluster]
    jobs, minutes, cost = estimate(args.stage, args.shards, gpus, rate)

    print(f"workspace   : {WORKSPACE}  (rg={RESOURCE_GROUP})")
    print(f"cluster     : {args.cluster}  {gpus} GPU/node  ~${rate:.2f}/node-hour")
    print(f"stage       : {args.stage}   matrix jobs: {jobs}")
    print(f"shards      : {args.shards}  -> {jobs/max(args.shards,1):.0f} jobs each, "
          f"{gpus} concurrent per node")
    print(f"est. runtime: ~{minutes:.0f} min   est. cost: ~${cost:.2f}"
          + ("   [SMOKE: a few minutes, <$1]" if args.smoke else ""))
    print(f"environment : {ENVIRONMENT}")

    if args.dry_run:
        print("\n--dry-run: nothing submitted. Commands that would run:\n")
        for k in range(args.shards):
            print(f"  [shard {k}] " + make_command(args.stage, k, args.shards, gpus,
                                                   args.seeds, args.smoke, args.save_fields))
        return

    from azure.ai.ml import MLClient, command
    from azure.identity import DefaultAzureCredential

    ml = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    print(f"\nconnected to {ml.workspace_name}")

    if args.download:
        for j in ml.jobs.list(max_results=50):
            if j.experiment_name == args.experiment and j.status == "Completed":
                dest = os.path.join(HERE, "results_azureml", j.name)
                os.makedirs(dest, exist_ok=True)
                ml.jobs.download(name=j.name, download_path=dest, output_name="default")
                print(f"downloaded {j.name} -> {dest}")
        return

    n = 1 if args.smoke else args.shards
    submitted = []
    for k in range(n):
        job = command(
            code=ROOT,                       # uploads track2/ skino/ validation/
            command=make_command(args.stage, k, n, gpus, args.seeds,
                                 args.smoke, args.save_fields),
            environment=ENVIRONMENT,
            compute=args.cluster,
            display_name=f"skino-{args.stage}-{'smoke' if args.smoke else f'shard{k}of{n}'}",
            experiment_name=args.experiment,
            environment_variables={"KMP_DUPLICATE_LIB_OK": "TRUE"},
        )
        created = ml.jobs.create_or_update(job)
        submitted.append(created)
        print(f"submitted {created.name}  ->  {created.studio_url}")

    print(f"\n{len(submitted)} job(s) submitted. Monitor in the studio, then run "
          f"`python -m track2.azureml_submit --download` to fetch results.")


if __name__ == "__main__":
    main()
