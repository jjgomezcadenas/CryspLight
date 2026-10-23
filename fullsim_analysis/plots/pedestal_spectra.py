"""Pedestal spectra of the high gain for the daily report: one figure per board, and the
demonstration that grouping the data by node alone mixes boards. Reads
fullsim_analysis/results/bench/<day>/pedestal_spectra.csv (pedestal_spectra.jl) and runs.csv
(inspect_run.jl).

Writes fullsim_analysis/results/bench/<day>/:
    pedestal_spectra_board<B>.png  8 x 8 panels, one per channel: spectrum of the high gain
                                   in each run of the csv (one colour per HG shaping), bins
                                   of 2 ADC counts; channels that read 0 in every event are
                                   marked "not read";
    node_mixing.png                for the first run, some channels of the boards sharing a
                                   node number: each board alone and the sum, which is what
                                   a plot grouped by node shows.

    python3 fullsim_analysis/plots/pedestal_spectra.py <YYYY-MM-DD>
"""
import sys

import numpy as np
import pandas as pd

from common import plt, results_dir, save

BIN = 2
# (node, channel) panels of node_mixing.png
MIXED = [(0, 0), (0, 6), (0, 27), (0, 47), (0, 1), (0, 57), (1, 10), (1, 45)]


def spectrum(x, lo, hi):
    edges = np.arange(lo, hi + BIN, BIN)
    h, _ = np.histogram(x.adc, edges, weights=x["count"])
    return edges, h


def main(day: str):
    d = results_dir("bench", day)
    sp = pd.read_csv(d / "pedestal_spectra.csv")
    runs = pd.read_csv(d / "runs.csv")
    topo = runs[runs.run == sp.run.min()].set_index("board")[["chain", "node"]]
    shaping = runs.groupby("run").HG_ShapingTime.first()

    for b in sorted(sp.board.unique()):
        fig, axes = plt.subplots(8, 8, figsize=(20, 15))
        x = sp[sp.board == b]
        for c, ax in enumerate(axes.flat):
            y = x[(x.channel == c) & (x.adc > 0)]
            ax.set_title(f"ch {c}", fontsize=8, pad=2)
            ax.tick_params(labelsize=6)
            if y.empty:
                ax.text(0.5, 0.5, "not read", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            lo, hi = np.percentile(np.repeat(y.adc, y["count"]), [0.1, 99.9])
            lo, hi = BIN * int(lo // BIN) - 30, BIN * int(hi // BIN) + 30
            for i, (r, z) in enumerate(y.groupby("run")):
                e, h = spectrum(z, lo, hi)
                ax.step(e[:-1], h, where="post", lw=0.8, c=f"C{i}", label=shaping[r])
        handles, labels = next((a.get_legend_handles_labels() for a in axes.flat
                                if a.get_legend_handles_labels()[0]), ([], []))
        fig.legend(handles, labels, loc="upper right", fontsize=10)
        ch, nd = topo.loc[b]
        fig.suptitle(f"{day}, board {b} (chain {ch}, node {nd}): high gain, HoldDelay 0, runs "
                     f"{', '.join(str(r) for r in sorted(x.run.unique()))}", fontsize=13)
        fig.supxlabel("high gain (ADC counts)"); fig.supylabel("events per 2 counts")
        fig.tight_layout(rect=(0, 0, 0.93, 0.97))
        save(fig, "bench", day, f"pedestal_spectra_board{b}")

    r0 = sp.run.min()
    x = sp[(sp.run == r0) & (sp.adc > 0)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, (nd, c) in zip(axes.flat, MIXED):
        boards = topo[topo.node == nd].index.tolist()
        y = x[x.channel == c]
        lo, hi = BIN * int(y.adc.min() // BIN) - 20, BIN * int(y.adc.max() // BIN) + 20
        total = 0
        for b in boards:
            e, h = spectrum(y[y.board == b], lo, hi)
            total = total + h
            if h.sum():
                ax.step(e[:-1], h, where="post", lw=1, label=f"board {b} alone")
        ax.step(e[:-1], total, where="post", c="k", lw=0.8, ls="--",
                label=f"node {nd} (boards {', '.join(map(str, boards))})")
        ax.set_title(f"node {nd}, channel {c}", fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel("high gain (ADC counts)")
    axes[0, 0].set_ylabel("events per 2 counts"); axes[1, 0].set_ylabel("events per 2 counts")
    fig.suptitle(f"{day}, run {r0}: grouping by node mixes the boards of the two chains "
                 "(board 1, channels 0-31, reads 0 and is not drawn)", fontsize=11)
    fig.tight_layout()
    save(fig, "bench", day, "node_mixing")


if __name__ == "__main__":
    main(sys.argv[1])
