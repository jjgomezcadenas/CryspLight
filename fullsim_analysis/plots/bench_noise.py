"""Noise of the high gain, channel by channel, for a day of bench data, to see which channels
do not see their SiPM: a SiPM at the input adds capacitance and raises the noise of the
preamplifier, so a channel with a bad contact is quieter than its neighbours. Reads
fullsim_analysis/results/bench/<day>/channels.csv and runs.csv (written by
fullsim_analysis/scripts/bench/inspect_run.jl) and uses the runs with HoldDelay 0, one per
HG shaping. Channels at 0 in every event (not read) are left out.

Writes fullsim_analysis/results/bench/<day>/:
    bench_noise.png  histograms of the high-gain rms divided by the median of its board, one
                     row per shaping, one column per board and a last column with all boards;
                     the channels confirmed silent by the staircases (CONFIRMED) are drawn
                     apart, in red.

    python3 fullsim_analysis/plots/bench_noise.py <YYYY-MM-DD>
"""
import sys

import numpy as np
import pandas as pd

from common import plt, results_dir, save

# channels without avalanches in the staircases (board -> channels): scans 10 and 11 of
# 21 Oct 2026, board 0 (findings F4)
CONFIRMED = {0: [15, 55, 57, 59, 61, 63]}
BINS = np.linspace(0.0, 1.5, 61)


def main(day: str):
    d = results_dir("bench", day)
    ch = pd.read_csv(d / "channels.csv")
    runs = pd.read_csv(d / "runs.csv")[["run", "board", "HG_ShapingTime", "HoldDelay"]]
    df = ch.merge(runs, on=["run", "board"])
    df = df[df.HoldDelay.astype(str).str.startswith("0") & (df.hg_zero < 1)].copy()
    df["shaping"] = df.HG_ShapingTime.astype(str).str.split().str[0].astype(float)
    # one run per shaping: the first one
    first = df.groupby("shaping").run.min()
    df = df[df.run == df.shaping.map(first)]
    df["ratio"] = df.hg_rms / df.groupby(["run", "board"]).hg_rms.transform("median")
    df["confirmed"] = [c in CONFIRMED.get(b, []) for b, c in zip(df.board, df.channel)]

    shapings, boards = sorted(df.shaping.unique()), sorted(df.board.unique())
    fig, axes = plt.subplots(len(shapings), len(boards) + 1, sharex=True,
                             figsize=(2.6 * (len(boards) + 1), 2.2 * len(shapings)), squeeze=False)
    for i, tp in enumerate(shapings):
        for j, b in enumerate(boards + ["all"]):
            ax = axes[i, j]
            x = df[df.shaping == tp] if b == "all" else df[(df.shaping == tp) & (df.board == b)]
            ax.hist(x[~x.confirmed].ratio, BINS, histtype="stepfilled", alpha=0.6, label="other")
            if x.confirmed.any():
                ax.hist(x[x.confirmed].ratio, BINS, histtype="stepfilled", color="tab:red",
                        alpha=0.8, label="silent in staircase")
            ax.set_title(f"{tp:g} ns, " + ("all boards" if b == "all" else f"board {b}") +
                         f" (run {int(x.run.iloc[0])})" if len(x) else "")
            if i == len(shapings) - 1:
                ax.set_xlabel("HG rms / board median")
            if j == 0:
                ax.set_ylabel("channels")
    axes[0, 0].legend()
    save(fig, "bench", day, "bench_noise")
    for tp in shapings:
        x = df[df.shaping == tp]
        print(f"{tp:g} ns: ratio of the confirmed channels "
              f"{np.round(np.sort(x[x.confirmed].ratio.values), 2)}; lowest of the others "
              f"{np.round(np.sort(x[~x.confirmed].ratio.values)[:5], 2)}")


if __name__ == "__main__":
    main(sys.argv[1])
