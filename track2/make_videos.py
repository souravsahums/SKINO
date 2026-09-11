"""Truth-vs-predicted rollout videos from the saved GPU field arrays.

For each (problem, config) this renders a side-by-side animation of the ground
truth and the model prediction (plus their difference) evolving over the whole
rollout, in the style of the neural-operator demo clips.

    python -m track2.make_videos                                  # winners, mp4
    python -m track2.make_videos --configs skino_seq2seq fno_seq2seq
    python -m track2.make_videos --problems kdv --configs skino_plain skino_seq2seq --format gif

Fields come from a `--save-fields` run (see track2/results_gpu/fields_*.npz).
Output goes to <results-dir>/videos/vid_<problem>_<config>.<mp4|gif>.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RES = os.path.join(HERE, "results_gpu")
ORDER = ["advection", "heat", "wave1d", "burgers", "kdv", "wave2d", "wave3d"]
# best model per problem from the matched-capacity GPU study
WINNERS = {"advection": "skino_plain", "heat": "tfno_plain", "wave1d": "tfno_seq2seq",
           "burgers": "tfno_plain", "kdv": "skino_seq2seq", "wave2d": "skino_plain",
           "wave3d": "skino_seq2seq"}
CLIP = 1e3


def load(res, problem, config):
    f = os.path.join(res, f"fields_{problem}_{config}.npz")
    if not os.path.isfile(f):
        return None, None
    try:
        z = np.load(f)
        return z["pred"], z["truth"]
    except Exception as e:  # truncated/corrupt npz
        print(f"  warn: unreadable {os.path.basename(f)}: {e}")
        return None, None


def _clip(a):
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)


def _rel_rms(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / (np.sqrt(np.mean(truth ** 2)) + 1e-9))


def _save(ani, out, fps):
    if out.endswith(".mp4"):
        ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    else:
        ani.save(out, writer=animation.PillowWriter(fps=fps))
    print(f"wrote {out}  ({os.path.getsize(out) // 1024} KB)")


def video_1d(pred, truth, problem, config, out, fps, frames, sample=0):
    tr = truth[:, sample, 0]                 # [T, X]
    pr = pred[:, sample, 0]
    vm = float(np.abs(tr).max()) * 1.4 + 1e-9
    x = np.arange(tr.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lt, = ax.plot(x, tr[0], "k", lw=2.2, label="truth")
    lp, = ax.plot(x, _clip(pr[0]), "r", lw=1.6, label="prediction")
    ax.set_ylim(-vm, vm); ax.set_xlim(0, tr.shape[1] - 1)
    ax.legend(loc="upper right", fontsize=9); ax.set_xlabel("x")
    ttl = ax.set_title("")

    def upd(t):
        p = _clip(pr[t]); lt.set_ydata(tr[t]); lp.set_ydata(p)
        ttl.set_text(f"{problem} — {config}    t={t}    rel-RMS={_rel_rms(p, tr[t]):.3g}")
        return lt, lp, ttl

    fig.tight_layout()
    _save(animation.FuncAnimation(fig, upd, frames=frames, blit=False), out, fps)
    plt.close(fig)


def video_grid(pred, truth, problem, config, out, fps, frames, sample=0, is3d=False):
    def plane(arr, t):
        a = arr[t, sample, 0]
        return a[:, :, a.shape[-1] // 2] if is3d else a

    vm = float(np.abs(plane(truth, 0)).max()) + 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, name in zip(axes, ["truth", "prediction", "error"]):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=11)
    kw = dict(origin="lower", cmap="seismic", vmin=-vm, vmax=vm)
    t0 = plane(truth, 0); p0 = _clip(plane(pred, 0))
    im0 = axes[0].imshow(t0.T, **kw)
    im1 = axes[1].imshow(p0.T, **kw)
    im2 = axes[2].imshow((p0 - t0).T, **kw)
    sup = fig.suptitle("")

    def upd(t):
        tt = plane(truth, t); pp = _clip(plane(pred, t))
        im0.set_data(tt.T); im1.set_data(pp.T); im2.set_data((pp - tt).T)
        extra = "   (mid-z slice)" if is3d else ""
        sup.set_text(f"{problem} — {config}    t={t}    rel-RMS={_rel_rms(pp, tt):.3g}{extra}")
        return im0, im1, im2, sup

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(animation.FuncAnimation(fig, upd, frames=frames, blit=False), out, fps)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RES)
    ap.add_argument("--problems", nargs="*", default=ORDER)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="models to render (default: the winning model per problem)")
    ap.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=160)
    a = ap.parse_args(argv)

    outdir = os.path.join(a.results_dir, "videos")
    os.makedirs(outdir, exist_ok=True)
    for prob in a.problems:
        configs = a.configs or [WINNERS.get(prob)]
        for cfg in configs:
            if cfg is None:
                continue
            pred, truth = load(a.results_dir, prob, cfg)
            if pred is None:
                print(f"  skip {prob}/{cfg}: no field file")
                continue
            T = min(pred.shape[0], truth.shape[0])
            stride = max(1, T // a.max_frames)
            frames = range(0, T, stride)
            out = os.path.join(outdir, f"vid_{prob}_{cfg}.{a.format}")
            if pred.ndim == 4:       # [T, batch, ch, X]
                video_1d(pred, truth, prob, cfg, out, a.fps, frames)
            elif pred.ndim == 5:     # [T, batch, ch, X, Y]
                video_grid(pred, truth, prob, cfg, out, a.fps, frames, is3d=False)
            elif pred.ndim == 6:     # [T, batch, ch, X, Y, Z]
                video_grid(pred, truth, prob, cfg, out, a.fps, frames, is3d=True)
            else:
                print(f"  skip {prob}/{cfg}: unexpected shape {pred.shape}")
    print("done")


if __name__ == "__main__":
    main()
