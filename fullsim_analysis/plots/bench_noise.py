"""Noise of every channel for a day of bench data, and the channels that do not see their
SiPM. Reads fullsim_analysis/results/bench/<day>/channel_noise.csv (noise_components.jl) and
runs.csv (inspect_run.jl), and uses the runs with HoldDelay 0, one per HG shaping. Chips not
read are absent from channel_noise.csv.

Writes fullsim_analysis/results/bench/<day>/:
    bench_rms.png    rms of each channel (without jumps and outliers): one panel per HG
                     shaping and one for the low gain, a box per board (quartiles, whiskers to
                     the extremes) of the channels that see their SiPM, and the channels whose
                     SiPM is not seen (median slope against the common mode below
                     NO_SIPM_SLOPE) as red circles;
    bench_slopes.png slope b_i of each channel against the common mode of its chip, median
                     over the runs with HoldDelay 0, against the channel, one panel per board,
                     with the bar from the lowest to the highest run, the cut NO_SIPM_SLOPE and
                     the channels silent in the staircases (CONFIRMED) in red. A channel whose
                     SiPM is not connected does not receive the common mode, which enters
                     through the SiPM.

    python3 fullsim_analysis/plots/bench_noise.py <YYYY-MM-DD>
"""
import sys

import numpy as np
import pandas as pd

from common import plt, results_dir, save

# channels without avalanches in the staircases (board -> channels): scans 10 and 11 of
# 21 Oct 2026, board 0 (findings F4)
CONFIRMED = {0: [15, 55, 57, 59, 61, 63]}
NO_SIPM_SLOPE = 0.75   # as in noise_components.jl


def channel_noise(day: str) -> pd.DataFrame:
    """channel_noise.csv of the runs with HoldDelay 0, with the shaping of each run and the
    flag of the channels whose SiPM is not seen (median slope below NO_SIPM_SLOPE)."""
    d = results_dir("bench", day)
    s = pd.read_csv(d / "channel_noise.csv")
    runs = pd.read_csv(d / "runs.csv")[["run", "board", "HG_ShapingTime", "HoldDelay"]]
    s = s.merge(runs, on=["run", "board"])
    s = s[s.HoldDelay.astype(str).str.startswith("0")].copy()
    s["shaping"] = s.HG_ShapingTime.astype(str).str.split().str[0].astype(float)
    med = s.groupby(["board", "channel"]).slope.transform("median")
    s["no_sipm"] = med < NO_SIPM_SLOPE
    return s


def main(day: str):
    s = channel_noise(day)
    s = s[s.run == s.shaping.map(s.groupby("shaping").run.min())]   # one run per shaping
    shapings, boards = sorted(s.shaping.unique()), sorted(s.board.unique())
    panels = [(tp, "hg_rms", f"high gain, {tp:g} ns") for tp in shapings]
    panels.append((shapings[0], "lg_rms", "low gain, 87.5 ns"))
    fs = 18
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flat
    for ax, (tp, g, title) in zip(axes, panels):
        x = s[s.shaping == tp]
        ok = x[~x.no_sipm]
        ax.boxplot([ok[ok.board == b][g] for b in boards], positions=boards, widths=0.55,
                   whis=(0, 100), medianprops={"color": "k", "lw": 2},
                   boxprops={"lw": 1.6}, whiskerprops={"lw": 1.6}, capprops={"lw": 1.6})
        red = x[x.no_sipm]
        ax.plot(red.board, red[g], "o", ms=9, mfc="none", mew=2, c="tab:red",
                label="SiPM not seen")
        ax.set_xticks(boards, [str(b) for b in boards], fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_xlabel("board", fontsize=fs)
        ax.set_ylabel("rms (ADC counts)", fontsize=fs)
        ax.set_title(f"{title} (run {int(x.run.iloc[0])})", fontsize=fs)
        ax.set_ylim((9.5, 14.5) if g == "hg_rms" else (4.75, 6.75))
        for b in boards:
            print(f"{title}, board {b}: median {ok[ok.board == b][g].median():.2f}")
    axes[0].legend(fontsize=fs - 2, loc="lower left")
    fig.tight_layout()
    save(fig, "bench", day, "bench_rms")


def slopes(day: str):
    d = results_dir("bench", day)
    s = channel_noise(day)
    g = s.groupby(["board", "channel"]).slope.agg(["median", "min", "max"]).reset_index()
    boards = sorted(g.board.unique())
    fig, axes = plt.subplots(len(boards), 1, sharex=True, figsize=(10, 1.8 * len(boards)),
                             squeeze=False)
    for ax, b in zip(axes[:, 0], boards):
        x = g[g.board == b]
        red = x.channel.isin(CONFIRMED.get(b, []))
        for sel, c in ((~red, "C0"), (red, "tab:red")):
            y = x[sel]
            ax.errorbar(y.channel, y["median"], [y["median"] - y["min"], y["max"] - y["median"]],
                        fmt="o", ms=3, c=c, lw=0.8, capsize=0)
        ax.axhline(NO_SIPM_SLOPE, c="k", ls="--", lw=0.8)
        ax.axvline(31.5, c="gray", lw=0.6, ls=":")
        ax.set_xlim(-1, 64); ax.set_ylim(-0.1, 1.3)
        ax.set_ylabel("slope b$_i$")
        low = x[x["median"] < NO_SIPM_SLOPE].channel.tolist()
        ax.set_title(f"board {b}: below {NO_SIPM_SLOPE} in channels {low}" if low else
                     f"board {b}: no channel below {NO_SIPM_SLOPE}", fontsize=9)
        print(f"board {b}: slope below {NO_SIPM_SLOPE} in {low}")
    axes[-1, 0].set_xlabel("channel")
    fig.suptitle(f"{day}: common mode taken by each channel (median over the runs with "
                 "HoldDelay 0); red: silent in the staircases of 21 Oct", fontsize=10)
    fig.tight_layout()
    save(fig, "bench", day, "bench_slopes")


if __name__ == "__main__":
    main(sys.argv[1])
    slopes(sys.argv[1])
