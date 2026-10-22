"""Staircases of the PET CAEN DAQ for a day of bench data: the count rate of the timing
discriminator of each channel against its coarse threshold (TD_CoarseThreshold, DAC
units). The day folder is benchdata/<YYYY-MM-DD>/ with one subdirectory scan-N per
staircase; each holds points.jsonl (one JSON line per threshold, written by the DAQ during
the scan: threshold, elapsed in ns, counts and rates of the 64 channels, counts and rates
of the T-OR and Q-OR). The staircase.h5 artifact carries the same numbers; it is not read
yet, since none has been downloaded.

Writes fullsim_analysis/results/bench/<day>/:
    staircase_N.csv  one row per threshold and channel: threshold, channel, counts,
                     elapsed_s, rate_cps, and the T-OR and Q-OR rates of that threshold;
    staircase_N.png  rate against threshold, one curve per channel, the T-OR on top;
                     channels with zero rate over the whole scan are listed, not drawn.
and prints, per channel, the threshold below which the rate exceeds 1 kHz, scanning down
from the highest threshold: a first measure of where the baseline noise of the fast
shaper sits in each channel.

    python3 fullsim_analysis/plots/staircase.py benchdata/<YYYY-MM-DD>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import plt, results_dir, save

RATE_EDGE = 1e3  # Hz


def read_points(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        for ch, (n, r) in enumerate(zip(p["channel_counts"], p["channel_rates_cps"])):
            rows.append({"threshold": p["threshold"], "channel": ch, "counts": n,
                         "elapsed_s": p["elapsed"] * 1e-9, "rate_cps": r,
                         "t_or_rate_cps": p["t_or_rate_cps"], "q_or_rate_cps": p["q_or_rate_cps"]})
    return pd.DataFrame(rows).sort_values(["channel", "threshold"]).reset_index(drop=True)


def edge_threshold(df_ch: pd.DataFrame) -> float:
    """Scanning down from the highest threshold, the first one with rate >= RATE_EDGE."""
    d = df_ch.sort_values("threshold", ascending=False)
    hit = d[d.rate_cps >= RATE_EDGE]
    return float(hit.threshold.iloc[0]) if len(hit) else np.nan


def plot(df: pd.DataFrame, title: str, day: str, name: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dead = []
    for ch, d in df.groupby("channel"):
        if (d.rate_cps == 0).all():
            dead.append(ch)
            continue
        ax.plot(d.threshold, d.rate_cps.where(d.rate_cps > 0), lw=0.6, alpha=0.6)
    tor = df.drop_duplicates("threshold").sort_values("threshold")
    ax.plot(tor.threshold, tor.t_or_rate_cps.where(tor.t_or_rate_cps > 0), "k", lw=1.5, label="T-OR")
    ax.set_yscale("log")
    ax.set_xlabel("TD coarse threshold (DAC)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(title + (f"; zero rate in channels {dead}" if dead else ""))
    ax.legend()
    save(fig, "bench", day, name)
    return dead


def main(day_dir: Path):
    day = day_dir.name
    out = results_dir("bench", day)
    scans = sorted((d for d in day_dir.glob("scan-*") if (d / "points.jsonl").exists()),
                   key=lambda d: int(d.name.split("-")[-1]))
    for d in scans:
        n = d.name.split("-")[-1]
        df = read_points(d / "points.jsonl")
        csv = out / f"staircase_{n}.csv"
        df.to_csv(csv, index=False)
        print(csv)
        th = df.threshold
        dead = plot(df, f"{day} scan {n}: thresholds {th.min()}-{th.max()}", day, f"staircase_{n}")
        edges = df.groupby("channel").apply(edge_threshold, include_groups=False)
        e = edges.dropna()
        print(f"scan {n}: {df.threshold.nunique()} thresholds {th.min()}-{th.max()}, "
              f"zero rate in channels {dead}")
        if len(e):
            print(f"  rate >= 1 kHz below threshold: median {e.median():.0f}, "
                  f"range {e.min():.0f}-{e.max():.0f}, in {len(e)} channels")
            print("  per channel: " + " ".join(f"{c}:{v:.0f}" for c, v in e.items()))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
