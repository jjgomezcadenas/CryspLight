#!/usr/bin/env python3
"""Training-convergence figure from a saved history.json (tracked figure source).

Two panels: training loss per epoch (logarithmic axis) and mean validation
RMSE [mm] per epoch, CNN and MLP overlaid, best CNN epoch marked. Writes
convergence.png next to the history file.

Usage: python3 analysis/cnn/plot_history.py [tag ...]   (default: both crystals)
Also called at the end of train.py so every training emits its own figure.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results" / "cnn"


def plot_history(outdir):
    outdir = Path(outdir)
    tag = outdir.name
    h = json.loads((outdir / "history.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, color in (("cnn", "C0"), ("mlp", "C1")):
        if model not in h:
            continue
        ep = [e["epoch"] for e in h[model]]
        loss = [e["train_loss"] for e in h[model]]
        rmse = np.array([e["val_rmse_mm"] for e in h[model]]).mean(axis=1)
        axs[0].plot(ep, loss, "o-", ms=3, color=color, label=model)
        axs[1].plot(ep, rmse, "o-", ms=3, color=color, label=model)
        if model == "cnn":
            b = int(np.argmin(rmse))
            axs[1].plot(ep[b], rmse[b], "*", color="red", ms=14, zorder=5,
                        label=f"best cnn: epoch {ep[b]}, {rmse[b]:.2f} mm")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("training loss (Huber, normalized coords)")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("validation RMSE, mean of x/y/z [mm]")
    for ax in axs:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{tag}: training convergence (cosine schedule over "
                 f"{len(h['cnn'])} epochs)")
    fig.tight_layout()
    fig.savefig(outdir / "convergence.png", dpi=150)
    plt.close(fig)
    print(f"{tag}: wrote {outdir / 'convergence.png'}")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["cnn_csitl", "cnn_bgo"]
    for t in tags:
        plot_history(RESULTS / t)
