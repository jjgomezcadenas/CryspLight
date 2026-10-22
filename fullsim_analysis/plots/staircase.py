"""Staircases of the PET CAEN DAQ for a day of bench data: the count rate of the timing
discriminator of each channel against its coarse threshold (TD_CoarseThreshold, DAC
units). The day folder is benchdata/<YYYY-MM-DD>/ with one subdirectory scan-N per
staircase; each holds points.jsonl (one JSON line per threshold, written by the DAQ during
the scan: threshold, elapsed in ns, counts and rates of the 64 channels, counts and rates
of the T-OR and Q-OR). Points with a channel rate above MAX_HZ are not measurements (the
DAQ has written 0xDEADBEEF as a count): they are listed, kept in the CSV with valid = 0, and
left out of the figure and the summary. The staircase.h5 artifact carries the same numbers; it is not read
yet, since none has been downloaded.

Writes fullsim_analysis/results/bench/<day>/:
    staircase_N.csv  one row per threshold and channel: threshold, channel, counts,
                     elapsed_s, rate_cps, valid, and the T-OR and Q-OR rates of that threshold;
    staircase_N.png  rate against threshold, one curve per channel, the T-OR on top;
                     channels with zero rate over the whole scan are listed, not drawn.
    staircase_N_channels.csv  one row per channel: the plateau rate (the largest rate of the
                     scan, set by the baseline noise of the fast shaper), the edge of the
                     plateau (scanning down from the highest threshold, the first one where
                     the rate reaches half the plateau rate) and the rate EDGE_STEP DAC
                     above the edge, where a channel that sees avalanches still counts
                     dark pulses and one that sees only noise has fallen to near zero;
and prints the edges and the channels without avalanches.

    python3 fullsim_analysis/plots/staircase.py benchdata/<YYYY-MM-DD>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import plt, results_dir, save

EDGE_STEP = 30  # DAC above the plateau edge where the dark rate is read
QUIET_HZ = 1e3   # below this rate there, the channel is listed as seeing no avalanches
MAX_HZ = 1e8     # a channel rate above this is not a measurement (e.g. counts 0xDEADBEEF)


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
    df = pd.DataFrame(rows).sort_values(["channel", "threshold"]).reset_index(drop=True)
    df["valid"] = (df.rate_cps <= MAX_HZ).astype(int)
    return df


def channel_summary(df_ch: pd.DataFrame) -> pd.Series:
    """Plateau rate, plateau edge and the rate EDGE_STEP DAC above the edge of one channel."""
    d = df_ch.sort_values("threshold", ascending=False)
    plateau = d.rate_cps.max()
    hit = d[d.rate_cps >= plateau / 2]
    edge = float(hit.threshold.iloc[0]) if plateau > 0 else np.nan
    above = d[d.threshold >= edge + EDGE_STEP]
    r = float(above.sort_values("threshold").rate_cps.iloc[0]) if len(above) else np.nan
    return pd.Series({"plateau_rate_cps": plateau, "edge": edge, f"rate_edge_plus_{EDGE_STEP}": r})


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
        bad = df[df.valid == 0]
        for _, b in bad.iterrows():
            print(f"scan {n}: invalid point, channel {b.channel} threshold {b.threshold}: "
                  f"counts {b.counts} (0x{int(b.counts):X})")
        csv = out / f"staircase_{n}.csv"
        df.to_csv(csv, index=False)
        print(csv)
        df = df[df.valid == 1]
        th = df.threshold
        dead = plot(df, f"{day} scan {n}: thresholds {th.min()}-{th.max()}", day, f"staircase_{n}")
        summ = df.groupby("channel").apply(channel_summary, include_groups=False)
        csv = out / f"staircase_{n}_channels.csv"
        summ.to_csv(csv)
        print(csv)
        e, rq = summ.edge.dropna(), summ[f"rate_edge_plus_{EDGE_STEP}"]
        quiet = list(summ.index[rq < QUIET_HZ])
        print(f"scan {n}: {df.threshold.nunique()} thresholds {th.min()}-{th.max()}, "
              f"zero rate in channels {dead}")
        print(f"  plateau edge: median {e.median():.0f}, range {e.min():.0f}-{e.max():.0f}")
        print(f"  rate {EDGE_STEP} DAC above the edge: median {rq.median():.3g} Hz; "
              f"below {QUIET_HZ:.0f} Hz (no avalanches) in channels {quiet}")
        low = summ[(rq >= QUIET_HZ) & (rq < rq.median() / 3)]
        print(f"  below a third of the median there: channels {list(low.index)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
