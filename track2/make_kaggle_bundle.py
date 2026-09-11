"""Build the Kaggle upload bundle: source packages only, no result data.

Kaggle datasets are immutable per version and slow to upload, so this copies
just the code that the jobs need (track2, skino, validation) and strips result
archives, simulator output, caches and checkpoints.

    python -m track2.make_kaggle_bundle
"""
from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "kaggle_bundle")

PACKAGES = ["track2", "ckino", "validation"]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "results", "results_v2",
             "results_paper", "results_gpu", "results_azureml", "cache", "figures",
             "output2", "models", "video"}
SKIP_EXT = {".npz", ".pt", ".mp4", ".zip", ".png", ".log", ".csv"}


def _filter(_dir, names):
    drop = []
    for n in names:
        full = os.path.join(_dir, n)
        if (n in SKIP_DIRS
                or (os.path.isdir(full) and n.startswith("results"))
                or os.path.splitext(n)[1].lower() in SKIP_EXT):
            drop.append(n)
    return drop


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for pkg in PACKAGES:
        src = os.path.join(ROOT, pkg)
        if not os.path.isdir(src):
            raise SystemExit(f"missing package: {src}")
        shutil.copytree(src, os.path.join(OUT, pkg), ignore=_filter)

    n_files = total = 0
    for r, _, fs in os.walk(OUT):
        for f in fs:
            n_files += 1
            total += os.path.getsize(os.path.join(r, f))
    archive = shutil.make_archive(os.path.join(ROOT, "ckino_code"), "zip", OUT)

    print(f"bundle : {OUT}")
    print(f"files  : {n_files}   size: {total/1e6:.2f} MB")
    print(f"zip    : {archive}  ({os.path.getsize(archive)/1e6:.2f} MB)")
    for pkg in PACKAGES:
        c = sum(len(fs) for _, _, fs in os.walk(os.path.join(OUT, pkg)))
        print(f"   {pkg:12s} {c:3d} files")
    print("\nUpload ckino_code.zip to Kaggle -> Datasets -> New Dataset.")


if __name__ == "__main__":
    main()
