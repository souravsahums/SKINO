"""Distributed launcher for the SKINO study.

The experiment matrix (problem x config x seed x budget) is embarrassingly
parallel: no job needs to talk to any other. This launcher enumerates the full
matrix deterministically, hands each worker a disjoint slice, and lets every job
write its own result file, so the same command works for

    * one machine, many cores      --jobs 4
    * several Kaggle/Colab sessions  --shard k --num-shards N
    * an Azure ML / SLURM array job  --shard $RANK --num-shards $WORLD

Determinism: the job list is sorted, so shard k on any machine selects exactly
the same subset. Nothing is lost or duplicated if a worker dies - just re-run
that shard.

Examples
--------
    # what would run
    python -m track2.launcher --plan

    # single GPU box, 4 concurrent jobs
    python -m track2.launcher --jobs 4 --device cuda

    # session 0 of 3 (run the same line with --shard 1 and 2 elsewhere)
    python -m track2.launcher --shard 0 --num-shards 3 --device cuda

    # merge per-job files into the combined results the analysis expects
    python -m track2.launcher --merge
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_paper")

LADDER = ["advection", "heat", "wave1d", "burgers", "kdv"]
STAGE = {"1d": LADDER, "2d": ["wave2d"], "3d": ["wave3d"]}

CORE_CONFIGS = ["skino_noise", "strict_noise", "nosymp_noise",
                "fno_noise", "tfno_noise", "skino_seq2seq",
                "fno_seq2seq", "tfno_seq2seq"]
FULL_CONFIGS = ["skino_plain", "skino_noise", "skino_pinn", "strict_plain",
                "strict_noise", "fno_plain", "fno_noise", "fno_pinn",
                "ufno_noise", "ufno_plain", "tfno_noise", "tfno_plain",
                "unet_noise", "deeponet_noise",
                "transformer_noise", "nosymp_noise", "skino_seq2seq",
                "fno_seq2seq", "tfno_seq2seq", "ufno_seq2seq", "skino_direct"]

# per-stage data settings
SETTINGS = {
    "1d": dict(n_traj=512, horizon=600, t_out=500, stride=25, epochs=15),
    "2d": dict(n_traj=96, horizon=300, t_out=250, stride=12, epochs=15),
    "3d": dict(n_traj=48, horizon=200, t_out=150, stride=8, epochs=12),
}


# Only skino / skino_strict / fno have >1-D implementations, so higher-dimensional
# stages drop the 1-D-only baselines instead of scheduling no-op jobs.
HI_D_CONFIGS = {"skino_plain", "skino_noise", "skino_pinn", "strict_plain",
                "strict_noise", "fno_plain", "fno_noise", "fno_pinn",
                "skino_seq2seq", "fno_seq2seq", "skino_direct"}


def build_jobs(stage: str, seeds, configs, budget: int, problems=None):
    """Deterministic, sorted job list. One job == one (problem, config, seed)."""
    if stage != "1d":
        configs = [c for c in configs if c in HI_D_CONFIGS]
    probs = STAGE[stage] if not problems else [p for p in STAGE[stage] if p in problems]
    jobs = []
    for prob, cfg, seed in itertools.product(sorted(probs), sorted(configs), sorted(seeds)):
        jobs.append({"problem": prob, "config": cfg, "seed": seed,
                     "budget": budget, "stage": stage})
    return jobs


def job_cmd(job, device, save_fields=False):
    s = SETTINGS[job["stage"]]
    cmd = [sys.executable, "-m", "track2.experiments_paper",
           "--problem", job["problem"], "--budget", str(job["budget"]),
           "--seed", str(job["seed"]), "--configs", job["config"],
           "--out-tag", job["config"], "--device", device,
           "--n-traj", str(s["n_traj"]), "--horizon", str(s["horizon"]),
           "--t-out", str(s["t_out"]), "--stride", str(s["stride"]),
           "--epochs", str(s["epochs"])]
    if save_fields:
        cmd.append("--save-fields")
    return cmd


def done_path(job):
    return os.path.join(
        RES, f"paper_{job['problem']}_b{job['budget']}_s{job['seed']}_sub_{job['config']}.json")


def run_job(job, device, save_fields, dry, gpu_id=None, threads=0):
    if os.path.isfile(done_path(job)):
        return job, "skipped (already done)"
    cmd = job_cmd(job, device, save_fields)
    if dry:
        return job, " ".join(cmd)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    if gpu_id is not None:
        # each worker sees exactly one GPU, so torch always picks cuda:0 inside
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if threads > 0:
        # concurrent CPU jobs would otherwise each grab every core and thrash
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
            env[v] = str(threads)
    p = subprocess.run(cmd, cwd=os.path.dirname(HERE), env=env,
                       capture_output=True, text=True)
    return job, ("ok" if p.returncode == 0 else f"FAILED\n{p.stdout[-1500:]}{p.stderr[-1500:]}")


def merge(strict: bool = True, results_dir: str | None = None):
    """Fold per-config shard files into one combined file per (problem, budget, seed).

    An existing combined file is used as the base layer so that configurations
    produced by an earlier monolithic run are not dropped when only a few
    configurations are later back-filled as shards. Shards win on conflict.

    Refuses by default to mix results produced on different hardware: a table
    that silently combined CPU-trained and GPU-trained models would not be a
    like-for-like comparison.
    """
    res = results_dir or RES
    groups, hw_seen = {}, {}
    for f in sorted(glob.glob(os.path.join(res, "paper_*_sub_*.json"))):
        if "_smoke" in os.path.basename(f):
            continue                       # validation artefact, not comparison data
        d = json.load(open(f))
        m = d.get("_meta", {})
        key = (m.get("problem"), m.get("budget"), m.get("seed"))
        hw = m.get("hardware", {}).get("gpu", "unknown")
        hw_seen.setdefault(key, {}).setdefault(hw, []).append(os.path.basename(f))
        if key not in groups:
            groups[key] = {"_meta": m}
            base = os.path.join(res, f"paper_{key[0]}_b{key[1]}_s{key[2]}_sub.json")
            if os.path.isfile(base):
                bd = json.load(open(base))
                bhw = bd.get("_meta", {}).get("hardware", {}).get("gpu", "unknown")
                hw_seen[key].setdefault(bhw, []).append(os.path.basename(base))
                groups[key].update({k: v for k, v in bd.items() if k != "_meta"})
        groups[key].update({k: v for k, v in d.items() if k != "_meta"})

    bad = {k: v for k, v in hw_seen.items() if len(v) > 1}
    if bad:
        print("!! MIXED HARDWARE - these groups were produced on more than one device:")
        for k, v in bad.items():
            print(f"   {k}: " + "; ".join(f"{hw} x{len(fs)}" for hw, fs in v.items()))
        if strict:
            print("   refusing to merge. Re-run the odd ones on the same hardware, "
                  "or pass --allow-mixed-hardware if you accept the caveat.")
            return

    for (prob, budget, seed), merged in groups.items():
        if prob is None:
            continue
        out = os.path.join(res, f"paper_{prob}_b{budget}_s{seed}_sub.json")
        with open(out, "w") as fh:
            json.dump(merged, fh, indent=2)
        hw = merged["_meta"].get("hardware", {}).get("gpu", "?")
        print(f"merged {len(merged)-1:2d} configs [{hw}] -> {os.path.basename(out)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="1d", choices=["1d", "2d", "3d"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="*", default=None,
                    help="default: the 6 core configs; use 'full' for all 17")
    ap.add_argument("--problems", nargs="*", default=None,
                    help="restrict to these equations within the stage (default: all)")
    ap.add_argument("--budget", type=int, default=25000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--jobs", type=int, default=1, help="concurrent jobs on this machine")
    ap.add_argument("--gpus", type=int, default=0,
                    help="number of GPUs to spread jobs over (0 = don't pin)")
    ap.add_argument("--threads-per-job", type=int, default=0,
                    help="CPU threads each job may use; set to cores//jobs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--plan", action="store_true", help="list jobs and exit")
    ap.add_argument("--merge", action="store_true", help="merge shard outputs")
    ap.add_argument("--allow-mixed-hardware", action="store_true",
                    help="merge even if shards ran on different devices (not like-for-like)")
    ap.add_argument("--results-dir", default=None,
                    help="where results live; keep CPU and GPU runs in separate dirs")
    args = ap.parse_args(argv)

    if args.merge:
        merge(strict=not args.allow_mixed_hardware, results_dir=args.results_dir)
        return

    cfgs = CORE_CONFIGS if args.configs is None else (
        FULL_CONFIGS if args.configs == ["full"] else args.configs)
    jobs = build_jobs(args.stage, args.seeds, cfgs, args.budget, args.problems)
    mine = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(f"[launcher] stage={args.stage} total={len(jobs)} "
          f"shard={args.shard}/{args.num_shards} -> {len(mine)} jobs, "
          f"jobs-in-parallel={args.jobs}")

    if args.plan:
        for j in mine:
            mark = "DONE" if os.path.isfile(done_path(j)) else "    "
            print(f"  {mark} {j['problem']:10s} {j['config']:18s} seed={j['seed']}")
        return

    os.makedirs(RES, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_job, j, args.device, args.save_fields, False,
                          (i % args.gpus) if args.gpus > 0 else None,
                          args.threads_per_job)
                for i, j in enumerate(mine)]
        for k, fu in enumerate(futs, 1):
            job, status = fu.result()
            print(f"[{k}/{len(mine)}] {job['problem']}/{job['config']}/s{job['seed']}: "
                  f"{status.splitlines()[0] if status else 'ok'}", flush=True)
            if status.startswith("FAILED"):
                print(status)
    print("[launcher] done. Run with --merge to combine shard outputs.")


if __name__ == "__main__":
    main()
