"""Hold-delay scans of a day of bench data (menu 2): the response of the peak detector to a
dark pulse of the shooting channel, and the stored value of the other channels against the
length of the window. Reads fullsim_analysis/results/bench/<day>/hold_scan_points.csv and
hold_scan_spectra.csv (hold_scan.jl) and channels.csv (inspect_run.jl, for the pedestal of each
channel below breakdown at the same shaping, runs PEDESTAL_RUNS).

The 1-pe peak of the shooting channel is the most populated bin of its spectrum, refined by a
parabola through the three bins around it; the gain is the peak minus the pedestal. The knee of
the response is the first delay at which the peak reaches KNEE of its plateau, the plateau being
the median of the peak over the last PLATEAU_POINTS points. The model of the front end
(bench note 2) puts the maximum of the shaped pulse MODEL_KNEE ns after the crossing of the
fast-shaper threshold; the window opens after the latency of the board, so the knee in the data
precedes it by that latency.

Writes fullsim_analysis/results/bench/<day>/:
    hold_scan_response.png  mean of the shooting channel (left) and position of its 1-pe peak
                            above the pedestal (right) against HoldDelay, one curve per scan,
                            25 ns on top and 50 ns below; stale points (the DAQ still at the
                            delay of the previous scan) hollow; the model knee dotted;
    hold_scan_spectra.png   spectra of the shooting channel of scan SPECTRA_SCAN at a few
                            delays, and of the reference channel at the same delays;
    hold_scan_window.png    mean of the reference channel against HoldDelay, 25 and 50 ns:
                            the stored value in a window at a random time;
    hold_scan_knees.csv     one row per scan: shooting channel, shaping, pedestal, plateau of
                            the 1-pe peak (gain in counts per pe), knee delay, mean of the
                            shooting channel at the plateau; hold_scan_knees.tex the same as a
                            LaTeX table;
and prints the table.

    python3 fullsim_analysis/plots/hold_scan.py <YYYY-MM-DD>
"""
import sys

import numpy as np
import pandas as pd

from common import plt, results_dir, save

PEDESTAL_RUNS = {25.0: 91, 50.0: 92, 87.5: 93}   # HoldDelay 0, SiPMs below breakdown
BOARD = 0                                          # DAQ number of the board scanned
REFERENCE = 32
MODEL_KNEE = {25.0: 31.0, 50.0: 58.0, 87.5: 90.0}
KNEE = 0.95
PLATEAU_POINTS = 8
SPECTRA_SCAN = 101
SPECTRA_DELAYS = [0, 16, 32, 48, 64, 96, 248]
PEAK_ABOVE = (48.0, 400.0)   # the 1-pe peak is searched this far above the pedestal (ADC)
FS = 16


def peak_position(bins, counts, low, high):
    """Refined position of the most populated bin between low and high (ADC counts)."""
    sel = (bins >= low) & (bins <= high)
    k = int(np.argmax(np.where(sel, counts, -1)))
    if 0 < k < len(counts) - 1:
        a, b, c = counts[k - 1], counts[k], counts[k + 1]
        d = (a - c) / (2.0 * (a - 2 * b + c)) if (a - 2 * b + c) != 0 else 0.0
        return bins[k] + d * (bins[1] - bins[0])
    return bins[k]


def main(day):
    d = results_dir("bench", day)
    pts = pd.read_csv(d / "hold_scan_points.csv")
    sp = pd.read_csv(d / "hold_scan_spectra.csv")
    ch = pd.read_csv(d / "channels.csv")

    def pedestal(channel, shaping):
        x = ch[(ch.run == PEDESTAL_RUNS[shaping]) & (ch.board == BOARD) & (ch.channel == channel)]
        return float(x.hg_mean.iloc[0])

    # 1-pe peak of the shooting channel at every point
    pts["pedestal"] = [pedestal(c, t) for c, t in zip(pts.channel, pts.shaping)]
    rows = []
    for (s, p), g in sp.groupby(["scan", "point"]):
        info = pts[(pts.scan == s) & (pts.point == p)].iloc[0]
        x = g[g.channel == info.channel].sort_values("bin_low")
        bins = np.arange(0, 8192, 16) + 8.0
        counts = np.zeros(len(bins)); counts[(x.bin_low.values // 16)] = x["count"].values
        rows.append((s, p, peak_position(bins, counts, info.pedestal + PEAK_ABOVE[0],
                                         info.pedestal + PEAK_ABOVE[1])))
    peaks = pd.DataFrame(rows, columns=["scan", "point", "peak"])
    pts = pts.merge(peaks, on=["scan", "point"])
    pts["gain"] = pts.peak - pts.pedestal

    # knees and plateaus
    knees = []
    for s, g in pts.groupby("scan"):
        g = g.sort_values("delay_ns")
        ok = g[g.stale == 0]
        plateau = ok.gain.iloc[-PLATEAU_POINTS:].median()
        above = ok[ok.gain >= KNEE * plateau]
        knee = above.delay_ns.iloc[0] if len(above) else np.nan
        knees.append((s, int(g.channel.iloc[0]), g.shaping.iloc[0], g.pedestal.iloc[0], plateau,
                      knee, ok.hg_mean.iloc[-PLATEAU_POINTS:].median()))
    kn = pd.DataFrame(knees, columns=["scan", "channel", "shaping", "pedestal", "gain_plateau",
                                      "knee_ns", "mean_plateau"])
    kn.to_csv(d / "hold_scan_knees.csv", index=False)
    print(kn.round(1).to_string(index=False))
    lines = [r"\begin{tabular}{cccccc}", r"\toprule",
             r"scan & channel & shaping & pedestal & knee & 1-pe peak above the pedestal \\",
             r"\midrule"]
    for r in kn.itertuples():
        lines.append(f"{r.scan} & {r.channel} & {r.shaping:g} ns & {r.pedestal:.0f} & "
                     f"{r.knee_ns:.0f} ns & {r.gain_plateau:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (d / "hold_scan_knees.tex").write_text("\n".join(lines) + "\n")

    # response of the shooting channel
    fig, ax = plt.subplots(2, 2, figsize=(16, 11), sharex="col")
    for i, tp in enumerate((25.0, 50.0)):
        for s, g in pts[pts.shaping == tp].groupby("scan"):
            g = g.sort_values("delay_ns")
            for j, col in enumerate(("hg_mean", "gain")):
                ok, st = g[g.stale == 0], g[g.stale == 1]
                line, = ax[i, j].plot(ok.delay_ns, ok[col], "o-", ms=7, lw=1.8,
                                      label=f"scan {s}, channel {int(g.channel.iloc[0])}")
                ax[i, j].plot(st.delay_ns, st[col], "o", ms=7, mfc="none", c=line.get_color())
        for j in range(2):
            ax[i, j].axvline(MODEL_KNEE[tp], ls=":", c="k", lw=1.5)
            ax[i, j].tick_params(labelsize=FS - 2)
            ax[i, j].legend(fontsize=FS - 4, loc="lower right")
        ax[i, 0].set_ylabel(f"mean of the shooting channel (ADC), {tp:g} ns", fontsize=FS)
        ax[i, 1].set_ylabel(f"1-pe peak above the pedestal (ADC), {tp:g} ns", fontsize=FS)
    for j in range(2):
        ax[1, j].set_xlabel("HoldDelay (ns)", fontsize=FS)
    ax[0, 0].set_title("channel that triggers; hollow: DAQ still at the previous delay",
                       fontsize=FS - 2)
    ax[0, 1].set_title("dotted: maximum of the shaped pulse in the model", fontsize=FS - 2)
    for i in range(2):
        ax[i, 1].set_ylim(0, 250)
    fig.tight_layout()
    save(fig, "bench", day, "hold_scan_response")

    # spectra of the shooting channel at a few delays
    g = sp[sp.scan == SPECTRA_SCAN]
    info = pts[pts.scan == SPECTRA_SCAN].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    for k, (c, title) in enumerate(((int(info.channel), "shooting channel"),
                                    (REFERENCE, "reference channel"))):
        for dl in SPECTRA_DELAYS:
            x = g[(g.channel == c) & (g.delay_ns == dl)].sort_values("bin_low")
            ax[k].step(x.bin_low + 8, x["count"], where="mid", lw=1.6, label=f"{dl} ns")
        ax[k].set_xlim(0, 1200); ax[k].set_yscale("log")
        ax[k].set_xlabel("high gain (ADC counts)", fontsize=FS)
        ax[k].set_ylabel("events per 16 counts", fontsize=FS)
        ax[k].set_title(f"scan {SPECTRA_SCAN}, {info.shaping:g} ns: {title} {c}", fontsize=FS)
        ax[k].tick_params(labelsize=FS - 2)
        ax[k].legend(fontsize=FS - 4, title="HoldDelay", title_fontsize=FS - 4)
    fig.tight_layout()
    save(fig, "bench", day, "hold_scan_spectra")

    # the reference channel against the window
    fig, ax = plt.subplots(figsize=(12, 7))
    for s, g in pts.groupby("scan"):
        x = sp[(sp.scan == s) & (sp.channel == REFERENCE)]
        m = x.groupby("delay_ns").apply(
            lambda y: np.average(y.bin_low + 8, weights=y["count"]), include_groups=False)
        g = g.sort_values("delay_ns").set_index("delay_ns")
        ok = g.stale == 0
        ped = pedestal(REFERENCE, g.shaping.iloc[0])
        ax.plot(m.index[ok.values], m.values[ok.values] - ped, "o-", ms=7, lw=1.8,
                label=f"scan {s}, {g.shaping.iloc[0]:g} ns")
    ax.set_xlabel("HoldDelay (ns)", fontsize=FS)
    ax.set_ylabel(f"mean of channel {REFERENCE} above its pedestal (ADC)", fontsize=FS)
    ax.set_title("a channel that does not trigger: stored value against the window, SiPMs at "
                 "5 V of overvoltage", fontsize=FS - 2)
    ax.tick_params(labelsize=FS - 2)
    ax.legend(fontsize=FS - 4)
    fig.tight_layout()
    save(fig, "bench", day, "hold_scan_window")


if __name__ == "__main__":
    main(sys.argv[1])
