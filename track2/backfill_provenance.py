"""One-off: backfill hardware provenance into results written before stamping existed.

Seed 0 of the 1-D ladder was produced by the local CPU run on this machine, but
predates ``hardware_info``. Without a stamp the merge/analysis guards flag it as
a different device and refuse a like-for-like comparison.

We only fill entries that have NO hardware block, and we mark them
``inferred: true`` so a reconstructed stamp is never mistaken for a measured one.
Anything that already carries provenance is left untouched.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")

STAMP = {"device": "cpu", "gpu": "cpu", "torch": "2.12.0+cpu",
         "host": "RbParadox", "platform": "Windows-11",
         "inferred": True,
         "note": "local CPU run predating hardware stamping; device known from run context"}


def main(apply: bool):
    changed, skipped = [], []
    for f in sorted(glob.glob(os.path.join(RES, "paper_*_sub.json"))):
        d = json.load(open(f))
        meta = d.get("_meta")
        if not meta:
            continue
        if meta.get("hardware"):
            skipped.append((os.path.basename(f), meta["hardware"].get("gpu")))
            continue
        changed.append(os.path.basename(f))
        if apply:
            meta["hardware"] = STAMP
            with open(f, "w") as fh:
                json.dump(d, fh, indent=2)

    print(f"already stamped ({len(skipped)}):")
    for n, g in skipped:
        print(f"   {n}  [{g}]")
    print(f"\n{'backfilled' if apply else 'WOULD backfill'} ({len(changed)}):")
    for n in changed:
        print(f"   {n}")
    if not apply:
        print("\nre-run with --apply to write.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
