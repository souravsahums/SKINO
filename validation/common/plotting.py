"""Matplotlib helpers used by the validation suite.

All plots are deterministic, headless (Agg backend) and saved as PNG so
they can be embedded in the comparative-study report.
"""
from __future__ import annotations

import os
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_curves(
    curves: Mapping[str, Sequence[float]],
    x: Sequence[float] | None,
    title: str,
    xlabel: str,
    ylabel: str,
    savepath: str,
    log_y: bool = False,
    log_x: bool = False,
) -> None:
    _ensure_dir(savepath)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, y in curves.items():
        ax.plot(x if x is not None else np.arange(len(y)), y, label=name, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120)
    plt.close(fig)


def plot_phase_space(
    trajectories: Mapping[str, np.ndarray],
    title: str,
    savepath: str,
) -> None:
    _ensure_dir(savepath)
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, qp in trajectories.items():
        ax.plot(qp[:, 0], qp[:, 1], label=name, linewidth=1.0, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120)
    plt.close(fig)


def plot_bar(
    values: Mapping[str, float],
    title: str,
    ylabel: str,
    savepath: str,
    log_y: bool = False,
) -> None:
    _ensure_dir(savepath)
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(values.keys())
    vals = [values[k] for k in names]
    ax.bar(range(len(names)), vals, color="tab:blue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120)
    plt.close(fig)


def plot_fields(
    fields: Mapping[str, np.ndarray],
    x: Sequence[float],
    title: str,
    savepath: str,
) -> None:
    _ensure_dir(savepath)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, y in fields.items():
        ax.plot(x, y, label=name, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120)
    plt.close(fig)
