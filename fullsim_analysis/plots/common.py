"""Shared readers and style of the full-simulation plots.

The Julia scripts under fullsim_analysis/scripts/ write, for a study of a campaign,
fullsim_analysis/results/<campaign>/<study>/<study>.h5 (arrays, histograms, maps) and
CSV tables; the plotters here read them and write PNG files beside them. Nothing is
computed here beyond what a figure needs.

Usage from a plotter:  from common import results_dir, read_hist, read_table, save
"""
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.titlesize": 10, "legend.fontsize": 8, "legend.framealpha": 0.9,
})


def results_dir(campaign: str, study: str) -> Path:
    d = RESULTS_ROOT / campaign / study
    d.mkdir(parents=True, exist_ok=True)
    return d


def open_results(campaign: str, study: str) -> h5py.File:
    return h5py.File(results_dir(campaign, study) / f"{study}.h5", "r")


def read_hist(g, name):
    """Edges and counts of a histogram written by write_hist!."""
    return g[f"{name}_edges"][...], g[f"{name}_counts"][...]


def read_table(campaign: str, study: str, name: str) -> pd.DataFrame:
    return pd.read_csv(results_dir(campaign, study) / f"{name}.csv")


def step_hist(ax, edges, counts, **kw):
    """Draw a histogram from edges and counts as steps."""
    ax.stairs(counts, edges, **kw)


def channel_map(ax, m, title, x=None, y=None, **kw):
    """An nx x ny map (Julia column-major: m[i, j], i along x) with the SiPM geometry."""
    m = np.asarray(m)
    im = ax.imshow(m.T, origin="lower", aspect="equal", **kw)
    ax.set_title(title)
    ax.set_xlabel("channel i (x)")
    ax.set_ylabel("channel j (y)")
    ax.grid(False)
    return im


def save(fig, campaign: str, study: str, name: str) -> Path:
    path = results_dir(campaign, study) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("written", path)
    return path
