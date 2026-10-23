"""Noise of every channel for a day of bench data, and the channels that do not see their
SiPM. Reads fullsim_analysis/results/bench/<day>/channel_noise.csv (noise_components.jl) and
runs.csv (inspect_run.jl), and uses the runs with HoldDelay 0, one per HG shaping. Chips not
read are absent from channel_noise.csv.

Writes fullsim_analysis/results/bench/<day>/:
    bench_rms.png    rms of each channel (without jumps and outliers) in the high gain (left)
                     and the low gain (right), one column of points per board and shaping;
                     the channels whose SiPM is not seen (median slope against the common
                     mode below NO_SIPM_SLOPE) in red;
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
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, g, lab in ((axes[0], "hg_rms", "high gain"), (axes[1], "lg_rms", "low gain")):
        for k, tp in enumerate(shapings):
            x = s[s.shaping == tp]
            pos = x.board + (k - 1) * 0.27 + rng.uniform(-0.08, 0.08, len(x))
            ok = ~x.no_sipm
            ax.plot(pos[ok], x[g][ok], ".", ms=3, c=f"C{k}",
                    label=f"HG {tp:g} ns (run {int(x.run.iloc[0])})")
            ax.plot(pos[~ok], x[g][~ok], "o", ms=4, mfc="none", c="tab:red",
                    label="SiPM not seen" if k == 0 else None)
            print(f"{lab}, HG {tp:g} ns: median rms {x[g][ok].median():.2f}, "
                  f"5-95% {x[g][ok].quantile(0.05):.2f}-{x[g][ok].quantile(0.95):.2f}")
        ax.set_xticks(boards, [f"board {b}" for b in boards])
        ax.set_ylabel("rms (ADC counts)")
        ax.set_title(f"{lab}: rms of each channel" +
                     (" (low gain at 87.5 ns in all runs)" if g == "lg_rms" else ""),
                     fontsize=10)
        ax.set_ylim((9, 15) if g == "hg_rms" else (4.5, 7))
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{day}: pedestal noise of every channel, HoldDelay 0, without jumps and "
                 "outliers", fontsize=10)
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
