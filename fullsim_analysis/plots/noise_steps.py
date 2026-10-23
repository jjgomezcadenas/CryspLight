"""The steps of the common-mode analysis of the pedestal noise, for the daily report and the
noise note (docs_fullsim/model_data/noise.tex). Reads, in fullsim_analysis/results/bench/<day>/,
noise_steps_matrix.csv, noise_steps_hist.csv, noise_steps_events.csv (noise_steps.jl) and
channel_noise.csv, runs.csv (noise_components.jl, inspect_run.jl). Definitions in
fullsim_analysis/lib/chip_noise.jl.

Writes fullsim_analysis/results/bench/<day>/:
    noise_steps_A.png  example chip: deviations d_i(e) of its 32 channels in 100 consecutive
                       events (colour), with the common mode CM(e) of each event above; the
                       events with |CM| beyond 1.5 rms framed by vertical lines;
    noise_steps_B.png  common mode CM(e) of every event along the run for the example chip
                       and the chip with jumps, with the band outside which an event is a jump
                       (noise_steps_cuts.csv) and the jumps in red; the map of the deviations
                       of the chip with jumps around its first jump;
    noise_steps_C.png  example chip: distribution of CM(e); d_i(e) of one channel against
                       CM(e); distribution of the slopes b_i of all channels and chips in the
                       runs with HoldDelay 0;
    noise_steps_D.png  example chip: distribution of d_i and of d_i - CM over all channels;
    noise_steps_E.png  CM(e) of two chips of the same board, and of chips of two boards.

    python3 fullsim_analysis/plots/noise_steps.py <YYYY-MM-DD>
"""
import sys

import numpy as np
import pandas as pd

from common import plt, results_dir, save

VMAX = 40     # colour range of the maps (counts)


def heat(ax, m, title):
    z = m.pivot(index="channel", columns="event", values="d")
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-VMAX, vmax=VMAX,
                   extent=(z.columns[0] - 0.5, z.columns[-1] + 0.5, z.index[-1] + 0.5,
                           z.index[0] - 0.5))
    ax.set_xlabel("event"); ax.set_ylabel("channel"); ax.set_title(title, fontsize=10)
    return im, z


def main(day: str):
    d = results_dir("bench", day)
    mat = pd.read_csv(d / "noise_steps_matrix.csv")
    hist = pd.read_csv(d / "noise_steps_hist.csv")
    ev = pd.read_csv(d / "noise_steps_events.csv")
    slopes = pd.read_csv(d / "channel_noise.csv")
    runs = pd.read_csv(d / "runs.csv")[["run", "board", "HoldDelay"]]
    cms = [c for c in ev.columns if c.startswith("cm_b")]
    dch = [c for c in ev.columns if c.startswith("d_ch")][0]
    ex_cm = cms[1]

    # A: deviations and common mode of the example chip (first NA events); the events whose
    # common mode is beyond STRIPE rms (the stripes of the map) are framed by vertical lines
    NA, STRIPE, fs = 100, 1.5, 16
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.3], width_ratios=[40, 1], hspace=0.08,
                          wspace=0.03)
    a0, a1, ac = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ex = mat[(mat.panel == "example") & (mat.event < mat[mat.panel == "example"].event.min() + NA)]
    im, z = heat(a1, ex, "")
    cmv = z.mean(axis=0)
    stripes = cmv.index[cmv.abs() > STRIPE * ev[ex_cm].std()]
    a0.plot(cmv.index, cmv.values, "k.-", lw=1, ms=5)
    a0.plot(stripes, cmv.loc[stripes].values, "o", ms=8, mfc="none", mew=1.8, c="tab:green")
    a0.axhline(0, c="gray", lw=0.5); a0.set_xlim(a1.get_xlim()); a0.set_xticklabels([])
    for e in stripes:
        for ax in (a0, a1):
            for xv in (e - 0.5, e + 0.5):
                ax.axvline(xv, c="tab:green", lw=1.2)
    a0.set_ylabel("CM(e)\n(counts)", fontsize=fs)
    a1.set_xlabel("event", fontsize=fs); a1.set_ylabel("channel", fontsize=fs)
    for ax in (a0, a1, ac):
        ax.tick_params(labelsize=fs - 2)
    a0.set_title("example chip (board 2, chip 1): deviation of each channel from its pedestal "
                 "(below)\nand their mean (above); framed in green, events with |CM| > "
                 f"{STRIPE:g} rms", fontsize=fs - 1)
    fig.colorbar(im, cax=ac).set_label("d$_i$ (counts)", fontsize=fs)
    save(fig, "bench", day, "noise_steps_A")
    print(f"stripes (|CM| > {STRIPE:g} rms) at events", list(stripes))

    # B: jumps, seen in the common mode of every event along the run
    cuts = pd.read_csv(d / "noise_steps_cuts.csv").set_index("chip")
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.35, wspace=0.18)
    for row, (key, lab) in enumerate((("example", "example chip (board 2, chip 1)"),
                                      ("jumps", "board 0, chip 1"))):
        a = fig.add_subplot(gs[row, 0])
        x = ev[["event", f"cm_all_{key}", f"jump_{key}"]].dropna()
        j = x[f"jump_{key}"] == 1
        a.plot(x.event[~j], x[f"cm_all_{key}"][~j], ".", ms=1.5, c="k")
        a.plot(x.event[j], x[f"cm_all_{key}"][j], "o", ms=4, mfc="none", c="tab:red",
               label=f"jumps: {int(j.sum())}")
        c0, w = cuts.loc[key, "centre"], cuts.loc[key, "half_width"]
        for y in (c0 - w, c0 + w):
            a.axhline(y, c="tab:red", ls="--", lw=0.8)
        a.set_ylim(-250, 250); a.set_ylabel("CM(e) (counts)")
        a.set_title(f"{lab}: common mode of every event, cut at ±{w:.0f} counts", fontsize=10)
        if j.any():
            a.legend(fontsize=8, loc="upper right")
    a.set_xlabel("event")
    jm = mat[mat.panel == "jump"]
    e0 = jm.groupby("event").d.mean().abs().idxmax()
    a = fig.add_subplot(gs[:, 1])
    im, _ = heat(a, jm[(jm.event - e0).abs() <= 40], f"board 0, chip 1, around event {e0}")
    fig.colorbar(im, ax=a, label="d$_i$ (counts), saturated at ±40", shrink=0.8)
    save(fig, "bench", day, "noise_steps_B")

    # C: common mode, one channel against it, slopes
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cm = ev[ex_cm].dropna()
    ax[0].hist(cm, np.linspace(-30, 30, 121), histtype="step", lw=1.2, color="k")
    ax[0].set_xlabel("CM(e) (counts)"); ax[0].set_ylabel("events")
    ax[0].set_title(f"example chip: rms of CM {cm.std():.2f} counts", fontsize=10)
    x = ev[[ex_cm, dch]].dropna()
    ax[1].hexbin(x[ex_cm], x[dch], gridsize=60, bins="log", cmap="Greys",
                 extent=(-25, 25, -60, 60))
    b = np.cov(x[dch], x[ex_cm])[0, 1] / x[ex_cm].var()
    t = np.linspace(-25, 25, 2)
    ax[1].plot(t, t, "r--", lw=1, label="slope 1")
    ax[1].plot(t, b * t, "b", lw=1, label=f"fit: slope {b:.2f}")
    ax[1].set_xlabel("CM(e) (counts)"); ax[1].set_ylabel(f"d$_i$(e), channel {dch[4:]} (counts)")
    ax[1].legend(fontsize=8)
    s = slopes.merge(runs, on=["run", "board"])
    s = s[s.HoldDelay.astype(str).str.startswith("0")]
    ax[2].hist(s.slope, np.linspace(0.0, 1.4, 71), histtype="stepfilled", alpha=0.7)
    ax[2].axvline(0.75, c="k", ls="--", lw=1, label="cut 0.75 (SiPM not seen below)")
    ax[2].set_yscale("log"); ax[2].legend(fontsize=8)
    ax[2].set_xlabel("slope b$_i$ of d$_i$ against CM"); ax[2].set_ylabel("channel-runs")
    ax[2].set_title(f"all chips, runs with HoldDelay 0 ({len(s)} channel-runs)", fontsize=10)
    fig.tight_layout()
    save(fig, "bench", day, "noise_steps_C")
    print(f"CM rms {cm.std():.2f}; slope of channel {dch[4:]} {b:.2f}; slopes "
          f"{s.slope.quantile(0.01):.2f}-{s.slope.quantile(0.99):.2f} (1-99%), "
          f"median {s.slope.median():.3f}")

    # D: d and d - CM
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col, lab in (("count_d", "d$_i$"), ("count_d_minus_cm", "d$_i$ − CM")):
        w = hist[col].values
        m = np.average(hist.d, weights=w)
        sd = np.sqrt(np.average((hist.d - m) ** 2, weights=w))
        ax.step(hist.d, w, where="mid", lw=1.2, label=f"{lab}: rms {sd:.2f} counts")
        print(f"{lab}: rms {sd:.2f}")
    ax.set_yscale("log"); ax.set_ylim(1, None)
    ax.set_xlabel("counts"); ax.set_ylabel("entries (all channels, all events)")
    ax.set_title("example chip: before and after subtracting the common mode", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "bench", day, "noise_steps_D")

    # E: correlation of the common modes
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, (u, v) in zip(ax, ((cms[0], cms[1]), (cms[1], cms[2]))):
        x = ev[[u, v]].dropna()
        rho = np.corrcoef(x[u], x[v])[0, 1]
        a.hexbin(x[u], x[v], gridsize=50, bins="log", cmap="Greys", extent=(-30, 30, -30, 30))
        lab = lambda c: c[3:].replace("b", "board ").replace("c", ", chip ")
        a.set_xlabel(f"CM, {lab(u)} (counts)"); a.set_ylabel(f"CM, {lab(v)} (counts)")
        a.set_title(f"correlation {rho:.2f}", fontsize=10)
        print(f"{u} vs {v}: correlation {rho:.2f}")
    fig.tight_layout()
    save(fig, "bench", day, "noise_steps_E")


if __name__ == "__main__":
    main(sys.argv[1])
