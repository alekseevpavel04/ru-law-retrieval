"""Plot style and figures (matplotlib, PNG). Raw data of every figure is saved next to it as CSV/JSON
by the code that calls these functions, so figures can be redrawn."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# categorical palette in fixed order (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": TEXT_2,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def ema(values: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    acc = values[0]
    for i, v in enumerate(values):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def plot_run(summary: dict, path: Path) -> None:
    """Three panels: training loss, learning rate, dev nDCG@10 (overall + by slice)."""
    setup_style()
    log = pd.DataFrame(summary["train_log"])
    dev = pd.DataFrame(summary["dev_history"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.2, 0.8, 1.2]})

    ax = axes[0]
    if not log.empty:
        ax.plot(log["step"], log["loss"], color=SERIES[0], alpha=0.25, linewidth=1.2, label="loss (raw)")
        ax.plot(log["step"], ema(log["loss"].to_numpy()), color=SERIES[0], label="loss (EMA)")
    ax.set_title("Train loss (CachedMNRL)")
    ax.set_xlabel("step")
    ax.legend(loc="upper right")

    ax = axes[1]
    if not log.empty:
        ax.plot(log["step"], log["lr"], color=SERIES[6])
    ax.set_title("Learning rate")
    ax.set_xlabel("step")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    ax = axes[2]
    slice_cols = [c for c in dev.columns if c.startswith("ndcg@10_")]
    for i, col in enumerate(slice_cols):
        ax.plot(dev["step"], dev[col], color=SERIES[1 + i], linewidth=1.5, marker="o", markersize=4,
                label=col.replace("ndcg@10_", ""))  # fmt: skip
    ax.plot(dev["step"], dev["ndcg@10"], color=SERIES[0], marker="o", markersize=6, label="dev (all)")
    best = dev.loc[dev["ndcg@10"].idxmax()]
    ax.scatter([best["step"]], [best["ndcg@10"]], s=140, facecolor="none", edgecolor=TEXT, linewidth=1.5, zorder=5)
    ax.annotate(f"best {best['ndcg@10']:.3f}", (best["step"], best["ndcg@10"]), textcoords="offset points",
                xytext=(6, 8), color=TEXT, fontsize=9)  # fmt: skip
    ax.axhline(dev["ndcg@10"].iloc[0], color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("base model", (dev["step"].iloc[-1], dev["ndcg@10"].iloc[0]), textcoords="offset points",
                xytext=(-4, -12), ha="right", color=MUTED, fontsize=8)  # fmt: skip
    ax.set_title("Dev nDCG@10 (chunk protocol)")
    ax.set_xlabel("step")
    ax.legend(loc="lower right")

    fig.suptitle(f"{summary['name']}  ·  {summary['base_model'].split('/')[-1]}  ·  "
                 f"{summary['train_rows']} pairs  ·  {summary['train_seconds'] / 60:.1f} min  ·  "
                 f"peak {summary['peak_mem_gb']} GB", x=0.01, ha="left", color=TEXT_2, fontsize=10)  # fmt: skip
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_runs_overlay(summaries: list[dict], path: Path, title: str) -> None:
    """Dev nDCG@10 vs fraction of the epoch for several runs, plus smoothed loss."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, s in enumerate(summaries):
        color = SERIES[i % len(SERIES)]
        dev = pd.DataFrame(s["dev_history"])
        x = dev["step"] / s["steps_per_epoch"]
        axes[0].plot(x, dev["ndcg@10"], color=color, marker="o", markersize=4, label=s["name"])
        log = pd.DataFrame(s["train_log"])
        if not log.empty:
            axes[1].plot(log["step"] / s["steps_per_epoch"], ema(log["loss"].to_numpy()), color=color,
                         label=s["name"])  # fmt: skip
    axes[0].set_title("Dev nDCG@10")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("Train loss (EMA)")
    axes[1].set_xlabel("epoch")
    axes[0].legend(loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", color=TEXT_2, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
