"""Generate simulator trajectories used to train FNO and SKINO.

Wraps elm1.py: imports its CFG, overrides a few fields (no live plotting,
deterministic seeds, output path inside HamiltonianNN/output2), and
calls run_simulation for run_id = 0..N-1.

Usage
-----
python generate_data.py                        # default 3 runs
python generate_data.py --n-runs 5             # 5 runs
python generate_data.py --n-steps 500          # shorter trajectories
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=600)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--output-dir", default=os.path.join(HERE, "output2"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    # Import after argparse so user gets fast --help.
    import elm1  # noqa: E402

    base_cfg = copy.deepcopy(elm1.CFG)
    base_cfg["n_steps"] = args.n_steps
    base_cfg["save_every"] = args.save_every
    base_cfg["enable_live_plot"] = False
    base_cfg["plot_velocity_model_once"] = False
    base_cfg["output_npz"] = os.path.join(args.output_dir, "small_run")
    base_cfg["output_metadata_json"] = os.path.join(args.output_dir, "small_run")

    # --- More visible source so the wavefield is obvious in the comparison video. ---
    # Theory unchanged (still a Ricker moment source on the same elastic lattice);
    # only the *parameters* are tuned for visibility:
    #   - place the source exactly on the mid-XY slice (z = nz//2) used by the video,
    #     so the slice cuts right through the radiation pattern;
    #   - raise the peak frequency 10 Hz -> 18 Hz so ~6 wavelengths fit in the domain
    #     (still safely above 6 points/wavelength at vp_min=2200 m/s, dx=20 m);
    #   - increase the moment scale 4x for larger displacement (still in linear bond regime);
    #   - shorter time shift so the wavelet peaks earlier and we observe more propagation.
    base_cfg["use_random_source"] = False
    nx, ny, nz = base_cfg["nx"], base_cfg["ny"], base_cfg["nz"]
    base_cfg["source_xyz"] = (nx // 2, ny // 2, nz // 2)
    base_cfg["source_peak_frequency"] = 18.0
    base_cfg["source_scale"] = 4.0e12
    base_cfg["source_time_shift"] = 0.05

    t0_all = time.time()
    for run_id in range(args.n_runs):
        cfg = copy.deepcopy(base_cfg)
        cfg["run_id"] = run_id
        # Deterministic but varied seeds for each run.
        cfg["model_seed"] = 1000 + run_id
        cfg["topo_seed"] = 2000 + run_id
        cfg["source_seed"] = 3000 + run_id
        print(f"\n=== Generating run_id={run_id} ===", flush=True)
        t0 = time.time()
        elm1.run_simulation(cfg)
        print(f"=== run_id={run_id} done in {time.time() - t0:.1f}s ===", flush=True)
    print(f"\nAll {args.n_runs} runs done in {time.time() - t0_all:.1f}s")
    print(f"Trajectories saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
