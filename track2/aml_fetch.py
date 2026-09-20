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

    python -m track2.aml_fetch --since 2026-09-11 --with-fields
    # download results_gpu.zip and results_gpu_fields.zip via Jupyter,
    # unzip both into track2/results_gpu/ locally
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
                    help="also download fields_*.npz / lhfields_*.npz (large) into a separate zip")
    ap.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                    help="only fetch jobs CREATED on or after this date (UTC). Artefact "
                         "names repeat across runs, so without this a twice-run "
                         "experiment is silently merged. Accepts YYYY-MM-DD or ISO.")
    ap.add_argument("--since-hours", type=float, default=None,
                    help="alternative to --since: keep jobs within this many hours of "
                         "the newest one")
    ap.add_argument("--all-runs", action="store_true",
                    help="fetch every completed job regardless of age (may mix runs)")
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

    # Artefact names are identical across runs (paper_kdv_..._sub_fno_plain.json is
    # the same name every time), so fetching an experiment that has been run twice
    # silently merges them. Filter by job creation date, and process oldest-first so
    # that when names do collide the NEWEST run deterministically wins.
    import datetime as _dt

    def _started(j):
        for attr in ("creation_context", "properties"):
            ctx = getattr(j, attr, None)
            ts = getattr(ctx, "created_at", None) if ctx else None
            if ts:
                return ts
        return None

    stamped = [(j, _started(j)) for j in done]
    undated = [j for j, t in stamped if t is None]

    if args.all_runs:
        kept = stamped
    elif args.since:
        try:
            cutoff = _dt.datetime.fromisoformat(args.since)
        except ValueError:
            raise SystemExit(f"--since: could not parse {args.since!r}; use YYYY-MM-DD")
        if cutoff.tzinfo is None:                      # AML stamps are tz-aware UTC
            cutoff = cutoff.replace(tzinfo=_dt.timezone.utc)
        kept = [(j, t) for j, t in stamped if t and t >= cutoff]
        print(f"[filter] keeping jobs created on or after {cutoff:%Y-%m-%d %H:%M %Z}")
    elif args.since_hours:
        known = [t for _, t in stamped if t]
        if known:
            newest = max(known)
            cutoff = newest - _dt.timedelta(hours=args.since_hours)
            kept = [(j, t) for j, t in stamped if t and t >= cutoff]
            print(f"[filter] newest job at {newest:%Y-%m-%d %H:%M}; keeping jobs "
                  f"within {args.since_hours}h of it")
        else:
            kept = stamped
    else:
        kept = stamped
        print("[filter] no date filter -- fetching every completed job. "
              "Pass --since YYYY-MM-DD if this experiment has been run more than once.")

    dropped = len(stamped) - len(kept)
    if dropped:
        print(f"         skipped {dropped} older job(s); use --all-runs to include them")
    if undated and not args.all_runs and (args.since or args.since_hours):
        print(f"         note: {len(undated)} job(s) had no readable timestamp and were skipped")

    # oldest first => later writes win, so a re-run supersedes the run before it
    kept.sort(key=lambda jt: jt[1] or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc))
    done = [j for j, _ in kept]
    job_time = {j.name: t for j, t in kept}
    if kept:
        lo, hi = kept[0][1], kept[-1][1]
        print(f"{len(done)} completed jobs in '{args.experiment}'"
              + (f"  ({lo:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M} UTC)" if lo and hi else ""))
    else:
        raise SystemExit("no jobs matched the filter - widen --since or use --all-runs")
    os.makedirs(args.out, exist_ok=True)

    total = 0
    total_collisions = []
    owner = {}                      # basename -> job that wrote it, for collision reporting
    for j in done:
        # job artifacts live under ExperimentRun/dcid.<job>/ (outputs/, user_logs/, ...)
        prefix = f"ExperimentRun/dcid.{j.name}/"
        got = 0
        gotf = 0
        for blob in cc.list_blobs(name_starts_with=prefix):
            base = os.path.basename(blob.name)
            is_json = base.endswith(".json") and (
                (base.startswith("paper_") and "_sub_" in base)
                or base.startswith(("darcy_", "modes_", "speedup_",
                                    "scaling_", "discretization_",
                                    "symplectic_defect_", "longhorizon_",
                                    "best_width_")))
            is_field = (args.with_fields and base.endswith(".npz")
                        and base.startswith(("fields_", "lhfields_")))
            if not (is_json or is_field):
                continue
            # jobs are processed oldest-first, so an unconditional write means the
            # newest run wins any name collision
            data = cc.download_blob(blob.name).readall()
            with open(os.path.join(args.out, base), "wb") as fh:
                fh.write(data)
            if base in owner and owner[base] != j.name:
                total_collisions.append(base)
            owner[base] = j.name
            if is_json:
                got += 1
                total += 1
            else:
                gotf += 1
        stamp = job_time.get(j.name)
        print(f"  {j.name}: {got} json(s)" + (f", {gotf} field(s)" if args.with_fields else "")
              + (f"   [{stamp:%m-%d %H:%M}]" if stamp else ""))

    if total_collisions:
        uniq = sorted(set(total_collisions))
        print(f"\n[note] {len(uniq)} artefact name(s) were produced by more than one job "
              f"in this window; the newest job's copy was kept. e.g. {uniq[:3]}")

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
        npzs = (glob.glob(os.path.join(args.out, "fields_*.npz"))
                + glob.glob(os.path.join(args.out, "lhfields_*.npz")))
        fzip = args.out + "_fields.zip"
        with zipfile.ZipFile(fzip, "w", zipfile.ZIP_STORED) as z:  # npz already compressed
            for f in npzs:
                z.write(f, os.path.join(os.path.basename(args.out), os.path.basename(f)))
        mb = os.path.getsize(fzip) / (1024 * 1024)
        print(f"wrote {fzip} ({mb:.0f} MB, {len(npzs)} field files)"
              f" -> download via Jupyter, unzip into track2/{args.out}/, then locally:\n"
              f"  python -m track2.make_field_figs   # rebuilds the paper's field panels")
        print("\n[!] verify the download before trusting it -- a truncated transfer looks\n"
              "    like a valid file until you open it:\n"
              f"      python -c \"import zipfile;print(len(zipfile.ZipFile('{os.path.basename(fzip)}').namelist()),'entries')\"")


if __name__ == "__main__":
    main()
