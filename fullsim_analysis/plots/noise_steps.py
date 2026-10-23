"""The steps of the common-mode analysis of the pedestal noise, for the daily report and the
noise note (docs_fullsim/model_data/noise.tex). Reads, in fullsim_analysis/results/bench/<day>/,
noise_steps_matrix.csv, noise_steps_hist.csv, noise_steps_events.csv (noise_steps.jl) and
channel_noise.csv, runs.csv (noise_components.jl, inspect_run.jl). Definitions in
fullsim_analysis/lib/chip_noise.jl.

Writes fullsim_analysis/results/bench/<day>/:
    noise_steps_A.png  example chip, data against the same data with each channel shuffled
                       independently (noise_steps.jl): the mean of the 32 channels in the
                       first 100 rows and its distribution, and the correlation matrix of the
                       channels in both;
    noise_steps_B.png  common mode CM(e) of every event along the run for the example chip
                       and the chip with jumps, with the band outside which an event is a jump
                       (noise_steps_cuts.csv) and the jumps in red; the map of the deviations
                       of the chip with jumps around its first jump;
    noise_steps_C.png  example chip: distribution of CM(e); d_i(e) of one channel against
                       CM(e); distribution of the slopes b_i of all channels and chips in the
                       runs with HoldDelay 0;
    noise_steps_D.png  example chip: distribution of d_i and of d_i - CM over all channels;
    noise_steps_E.png  CM(e) of two chips of the same board, and of chips of two boards;
    noise_steps_outliers.png  distribution of the pull z = (d_i - r) / s_i of every channel
                       (outlier_pulls.csv, noise_components.jl) in the first run with HoldDelay
                       0 and in the run with the longest HoldDelay, log scale, with the
                       Gaussian of unit width and the cut at |z| = 5.

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

    # A: the common mode against its reference, the same data with the channels shuffled
    fs = 18
    corr = pd.read_csv(d / "noise_steps_corr.csv")
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    x = ev[["event", ex_cm, "cm_shuffled_example"]].dropna().reset_index(drop=True)
    rd, rs = x[ex_cm].std(), x.cm_shuffled_example.std()
    w = x.iloc[:100]
    ax[0, 0].plot(w.index, w[ex_cm], "k.-", lw=1.2, ms=6, label="data")
    ax[0, 0].plot(w.index, w.cm_shuffled_example, ".-", c="tab:orange", lw=1.2, ms=6,
                  label="channels shuffled")
    ax[0, 0].set_xlabel("row (first 100)", fontsize=fs)
    ax[0, 0].set_ylabel("mean of the 32 channels (counts)", fontsize=fs)
    ax[0, 0].set_ylim(-25, 25)
    bins = np.linspace(-30, 30, 121)
    ax[0, 1].hist(x[ex_cm], bins, histtype="step", lw=2, color="k", label=f"data: rms {rd:.1f}")
    ax[0, 1].hist(x.cm_shuffled_example, bins, histtype="step", lw=2, color="tab:orange",
                  label=f"shuffled: rms {rs:.1f}")
    ax[0, 1].set_xlabel("mean of the 32 channels (counts)", fontsize=fs)
    ax[0, 1].set_ylabel("rows", fontsize=fs)
    for k, (t, a_) in enumerate((("data", ax[1, 0]), ("shuffled", ax[1, 1]))):
        c = corr[corr.table == t].pivot(index="channel_a", columns="channel_b", values="rho")
        m = c.values.copy(); np.fill_diagonal(m, np.nan)
        im = a_.imshow(m, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                       extent=(c.columns[0] - 0.5, c.columns[-1] + 0.5, c.index[-1] + 0.5,
                               c.index[0] - 0.5))
        off = m[~np.isnan(m)]
        a_.set_title(f"{t}: correlation of channels\n(mean off diagonal {off.mean():.2f})",
                     fontsize=fs)
        a_.set_xlabel("channel", fontsize=fs); a_.set_ylabel("channel", fontsize=fs)
        print(f"{t}: mean correlation {off.mean():.3f}, rms {off.std():.3f}")
        cb = fig.colorbar(im, ax=a_, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fs - 4)
    for a_ in ax.flat:
        a_.tick_params(labelsize=fs - 2)
    for a_ in (ax[0, 0], ax[0, 1]):
        a_.legend(fontsize=fs - 3)
    ax[0, 0].set_title("mean of each row", fontsize=fs)
    ax[0, 1].set_title("its distribution", fontsize=fs)
    save(fig, "bench", day, "noise_steps_A")
    print(f"mean of the 32 channels: rms {rd:.2f} (data), {rs:.2f} (shuffled)")

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


def outliers(day: str, cut: float = 5.0, fs: int = 18):
    d = results_dir("bench", day)
    pulls = pd.read_csv(d / "outlier_pulls.csv")
    runs = pd.read_csv(d / "runs.csv").drop_duplicates("run").set_index("run")
    hold = runs.HoldDelay.astype(str).str.split().str[0].astype(float)
    shown = [int(hold[hold == 0].index.min()), int(hold.idxmax())]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, r in zip(axes, shown):
        x = pulls[pulls.run == r]
        w = 0.1
        n = x["count"].sum()
        ax.step(x.z_low + w / 2, x["count"], where="mid", c="k", lw=1.5, label="data")
        zz = np.linspace(-10, 10, 801)
        ax.plot(zz, n * w * np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi), c="tab:blue", lw=1.5,
                label="Gaussian, width 1")
        for c in (-cut, cut):
            ax.axvline(c, c="tab:red", ls="--", lw=1.5)
        beyond = x[(x.z_low + w / 2).abs() > cut]["count"].sum()
        expected = n * 5.7e-7
        ax.set_yscale("log"); ax.set_ylim(0.5, None); ax.set_xlim(-10, 10)
        ax.set_title(f"run {r}, HoldDelay {hold[r]:g} ns: {beyond} beyond $\\pm${cut:g}\n"
                     f"({expected:.1f} expected for Gaussian noise)", fontsize=fs)
        ax.set_xlabel("z = (d$_i$ - r) / s$_i$", fontsize=fs)
        ax.tick_params(labelsize=fs - 2)
        print(f"run {r}: {n} entries, {beyond} beyond {cut}, {expected:.2f} expected")
    axes[0].set_ylabel("entries per 0.1", fontsize=fs)
    axes[0].legend(fontsize=fs - 3, loc="upper left")
    fig.tight_layout()
    save(fig, "bench", day, "noise_steps_outliers")


if __name__ == "__main__":
    main(sys.argv[1])
    outliers(sys.argv[1])
