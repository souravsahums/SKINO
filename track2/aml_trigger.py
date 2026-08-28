"""Fire-and-forget trigger for the SKINO study on an Azure ML compute CLUSTER.

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

# --- workspace / cluster this repo targets (override on the CLI if needed) ---
SUBSCRIPTION = os.environ.get("AML_SUBSCRIPTION_ID", "<your-subscription-id>")
RESOURCE_GROUP = os.environ.get("AML_RESOURCE_GROUP", "<your-resource-group>")
WORKSPACE = os.environ.get("AML_WORKSPACE", "<your-workspace>")
CLUSTER = "gpu-t4"
# curated GPU environment used by the cluster jobs (not by this trigger)
ENVIRONMENT = "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/labels/latest"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # repo root -> uploaded as job code
STAGES = ["1d", "2d", "3d"]


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trigger SKINO cluster jobs and exit.")
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
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be submitted and exit")
    args = ap.parse_args(argv)
    save_fields = not args.no_save_fields

    print(f"workspace : {WORKSPACE}  (rg={RESOURCE_GROUP})")
    print(f"cluster   : {args.cluster}")
    print(f"stages    : {args.stages}   seeds: {args.seeds}   shards/stage: {args.shards}")
    print(f"configs   : {args.configs or 'core (launcher default)'}")

    if args.dry_run:
        for stage in args.stages:
            for k in range(args.shards):
                print(f"\n[{stage} shard {k}]\n  " +
                      make_command(stage, k, args.shards, args.seeds, args.configs, save_fields))
        print("\n--dry-run: nothing submitted.")
        return

    from azure.ai.ml import MLClient, command
    from azure.identity import DefaultAzureCredential

    ml = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    print(f"connected to {ml.workspace_name}\n")

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

    print(f"\n{len(submitted)} job(s) queued on '{args.cluster}'. Trigger-and-forget: done. "
          f"Track them in the Studio (Jobs -> experiment '{args.experiment}'). "
          f"You can stop this compute instance now - the cluster keeps running.")


if __name__ == "__main__":
    main()
