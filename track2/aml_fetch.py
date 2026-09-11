"""Download completed CKINO cluster-job results ON THE COMPUTE INSTANCE.

Run this on `skino-trigger` (inside the VNet, so it can reach the workspace
storage; your laptop cannot). It gathers the per-config result JSONs into
``results_gpu/`` and zips them for you to pull to your laptop via Jupyter.

Why not ``az ml job download`` / ``MLClient.jobs.download``? Those helpers fetch a
storage *account key* and download with it - but this storage account has
``allowSharedKeyAccess=false`` (AAD only), so they fail with
``KeyBasedAuthenticationNotPermitted``. This script instead reads the artifact
blobs directly with the AAD credential (the same one that made job submission
work), which the account permits.

    python -m track2.aml_fetch
    # download results_gpu.zip via Jupyter, unzip into track2/results_gpu/ locally
"""
from __future__ import annotations

import argparse
import glob
import os
import zipfile

from .aml_config import get_ml_client


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="skino-study")
    ap.add_argument("--out", default="results_gpu")
    ap.add_argument("--datastore", default="workspaceartifactstore")
    ap.add_argument("--with-fields", action="store_true",
                    help="also download fields_*.npz (large ~200 MB) into a separate zip")
    args = ap.parse_args(argv)

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    cred = DefaultAzureCredential()
    ml = get_ml_client(cred)
    print(f"workspace: {ml.workspace_name} (rg={ml.resource_group_name})")

    ds = ml.datastores.get(args.datastore)
    account = getattr(ds, "account_name", None)
    container = getattr(ds, "container_name", None)
    if not account or not container:
        raise SystemExit(f"could not read account/container from datastore {args.datastore}: {ds}")
    print(f"artifact store: account={account} container={container}")

    # AAD blob access - the SDK's own downloader uses account keys, forbidden here.
    bsc = BlobServiceClient(f"https://{account}.blob.core.windows.net", credential=cred)
    cc = bsc.get_container_client(container)

    done = [j for j in ml.jobs.list()
            if getattr(j, "experiment_name", None) == args.experiment and j.status == "Completed"]
    print(f"{len(done)} completed jobs in '{args.experiment}'")
    os.makedirs(args.out, exist_ok=True)

    total = 0
    for j in done:
        # job artifacts live under ExperimentRun/dcid.<job>/ (outputs/, user_logs/, ...)
        prefix = f"ExperimentRun/dcid.{j.name}/"
        got = 0
        gotf = 0
        for blob in cc.list_blobs(name_starts_with=prefix):
            base = os.path.basename(blob.name)
            if base.endswith(".json") and (
                    (base.startswith("paper_") and "_sub_" in base)
                    or base.startswith(("darcy_", "modes_", "speedup_",
                                        "scaling_", "discretization_",
                                        "symplectic_defect_", "longhorizon_",
                                        "best_width_"))):
                data = cc.download_blob(blob.name).readall()
                with open(os.path.join(args.out, base), "wb") as fh:
                    fh.write(data)
                got += 1
                total += 1
            elif args.with_fields and base.startswith("fields_") and base.endswith(".npz"):
                dst = os.path.join(args.out, base)
                if not os.path.exists(dst):  # one file per (problem,config); first job wins
                    data = cc.download_blob(blob.name).readall()
                    with open(dst, "wb") as fh:
                        fh.write(data)
                    gotf += 1
        print(f"  {j.name}: {got} json(s)" + (f", {gotf} field(s)" if args.with_fields else ""))

    jsons = (glob.glob(os.path.join(args.out, "paper_*_sub_*.json"))
             + glob.glob(os.path.join(args.out, "darcy_*.json"))
             + glob.glob(os.path.join(args.out, "modes_*.json"))
             + glob.glob(os.path.join(args.out, "speedup_*.json"))
             + glob.glob(os.path.join(args.out, "scaling_*.json"))
             + glob.glob(os.path.join(args.out, "discretization_*.json"))
             + glob.glob(os.path.join(args.out, "symplectic_defect_*.json"))
             + glob.glob(os.path.join(args.out, "longhorizon_*.json"))
             + glob.glob(os.path.join(args.out, "best_width_*.json")))
    print(f"\ngathered {len(jsons)} per-config result JSONs into {args.out}/")
    if not jsons:
        raise SystemExit("no result JSONs found - check the artifact prefix/container printed above")

    zpath = args.out + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in jsons:
            z.write(f, os.path.join(os.path.basename(args.out), os.path.basename(f)))
    print(f"wrote {zpath} ({os.path.getsize(zpath) // 1024} KB) -> download via Jupyter, "
          f"unzip into track2/{args.out}/ on your laptop, then:\n"
          f"  python -m track2.launcher --merge --results-dir track2/{args.out}\n"
          f"  python -m track2.seed_analysis --results-dir track2/{args.out}")

    if args.with_fields:
        npzs = glob.glob(os.path.join(args.out, "fields_*.npz"))
        fzip = args.out + "_fields.zip"
        with zipfile.ZipFile(fzip, "w", zipfile.ZIP_STORED) as z:  # npz already compressed
            for f in npzs:
                z.write(f, os.path.join(os.path.basename(args.out), os.path.basename(f)))
        print(f"wrote {fzip} ({os.path.getsize(fzip) // (1024 * 1024)} MB, {len(npzs)} field files)"
              f" -> download via Jupyter, unzip into track2/{args.out}/, then locally:\n"
              f"  python -m track2.paper_analysis --results-dir track2/{args.out} --only pred")


if __name__ == "__main__":
    main()
