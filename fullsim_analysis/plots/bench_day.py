"""Pedestals of a day of bench data, channel by channel and board by board, for the daily
report. Reads fullsim_analysis/results/bench/<day>/channels.csv and runs.csv (written by
fullsim_analysis/scripts/bench/inspect_run.jl).

Writes fullsim_analysis/results/bench/<day>/:
    bench_pedestals.png  one panel per board: mean of the high gain against the channel, with
                         the rms as error bar (the robust width, interquartile range / 1.349,
                         as a second, narrower bar), for one run (the first of the day unless
                         given); channels that read 0 in every event are marked at the bottom
                         and the channels silent in the staircases (CONFIRMED) are drawn in
                         red. The two Citirocs of the A5202 are channels 0-31 and 32-63.

    python3 fullsim_analysis/plots/bench_day.py <YYYY-MM-DD> [run]
"""
import sys

import pandas as pd

from bench_noise import CONFIRMED
from common import plt, results_dir, save


def main(day: str, run: int | None = None):
    d = results_dir("bench", day)
    ch = pd.read_csv(d / "channels.csv")
    runs = pd.read_csv(d / "runs.csv")
    run = int(ch.run.min()) if run is None else run
    ch, info = ch[ch.run == run], runs[runs.run == run]
    boards = sorted(ch.board.unique())
    fig, axes = plt.subplots(len(boards), 1, sharex=True, figsize=(10, 1.9 * len(boards)),
                             squeeze=False)
    for ax, b in zip(axes[:, 0], boards):
        x = ch[ch.board == b]
        live, dead = x[x.hg_zero < 1], x[x.hg_zero == 1]
        red = live.channel.isin(CONFIRMED.get(b, []))
        for sel, c in ((~red, "C0"), (red, "tab:red")):
            y = live[sel]
            ax.errorbar(y.channel, y.hg_mean, y.hg_rms, fmt="o", ms=2.5, c=c, lw=0.8, capsize=0)
            ax.errorbar(y.channel, y.hg_mean, y.hg_sigr, fmt="none", c=c, lw=2.5, alpha=0.5)
        if len(dead):
            ax.plot(dead.channel, [0] * len(dead), "x", c="k", ms=4, label="0 in every event")
            ax.legend(loc="upper right", fontsize=7)
        ax.axvline(31.5, c="gray", lw=0.6, ls=":")
        r = info[info.board == b].iloc[0]
        ax.set_title(f"board {b} (chain {r.chain}, node {r.node}), T detector "
                     f"{r.T_detector:.1f} C", fontsize=9)
        ax.set_ylabel("HG (ADC)")
        ax.set_ylim(-20, max(350, x.hg_mean.max() + 60))
    r = info.iloc[0]
    axes[-1, 0].set_xlabel("channel")
    fig.suptitle(f"{day}, run {run}: HG {r.HG_ShapingTime}, HoldDelay {r.HoldDelay}, "
                 f"HV_Vbias {r.HV_Vbias} V; bar = rms, thick bar = robust width", fontsize=10)
    fig.tight_layout()
    save(fig, "bench", day, "bench_pedestals")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
