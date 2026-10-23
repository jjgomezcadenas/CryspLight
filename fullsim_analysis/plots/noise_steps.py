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
                       (top) and the chip with jumps (bottom), with the band outside which an
                       event is a jump (noise_steps_cuts.csv) and the jumps in red;
    noise_steps_C.png  example chip: d_i(e) of one channel against CM(e), with the fitted
                       slope; distribution of the slopes b_i of all channels and chips in the
                       runs with HoldDelay 0;
    noise_steps_D.png  total, coherent and incoherent noise of every chip in the first run with
                       HoldDelay 0 (noise_components.csv), and the total against
                       sqrt(coherent^2 + incoherent^2) for every chip of every run with
                       HoldDelay 0;
    noise_steps_E.png  correlation of the common modes of every pair of chips read in the first
                       run (common_mode.csv, noise_components.jl), as a matrix;
    noise_steps_outliers.png  distribution of the pull z = (d_i - r) / s_i of every channel
                       (outlier_pulls.csv, noise_components.jl) in the first run with HoldDelay
                       0, log scale, with the Gaussian of unit width and the cut at |z| = 5
                       (the cut is applied only with HoldDelay 0).

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
    fs = 18
    cuts = pd.read_csv(d / "noise_steps_cuts.csv").set_index("chip")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for a, (key, lab) in zip(axes, (("example", "example chip (board 2, chip 1)"),
                                    ("jumps", "board 0, chip 1"))):
        x = ev[["event", f"cm_all_{key}", f"jump_{key}"]].dropna()
        j = x[f"jump_{key}"] == 1
        a.plot(x.event[~j], x[f"cm_all_{key}"][~j], ".", ms=2, c="k")
        a.plot(x.event[j], x[f"cm_all_{key}"][j], "o", ms=8, mfc="none", mew=2, c="tab:red")
        c0, w = cuts.loc[key, "centre"], cuts.loc[key, "half_width"]
        for y in (c0 - w, c0 + w):
            a.axhline(y, c="tab:red", ls="--", lw=1.5)
        a.set_ylim(-300, 300); a.set_ylabel("CM(e) (counts)", fontsize=fs)
        a.set_title(f"{lab}: cut at $\\pm${w:.0f} counts, {int(j.sum())} jumps (red)",
                    fontsize=fs)
        a.tick_params(labelsize=fs - 2)
    axes[-1].set_xlabel("event", fontsize=fs)
    fig.tight_layout()
    save(fig, "bench", day, "noise_steps_B")

    # C: one channel against the common mode with its fitted slope; slopes of all channels
    fs = 18
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    cm = ev[ex_cm].dropna()
    x = ev[[ex_cm, dch]].dropna()
    ax[0].hexbin(x[ex_cm], x[dch], gridsize=50, bins="log", cmap="Greys",
                 extent=(-25, 25, -60, 60))
    b = np.cov(x[dch], x[ex_cm])[0, 1] / x[ex_cm].var()
    t = np.linspace(-25, 25, 2)
    ax[0].plot(t, t, "r--", lw=2, label="slope 1")
    ax[0].plot(t, b * t, "b", lw=2, label=f"fit: slope {b:.2f}")
    ax[0].set_xlabel("CM(e) (counts)", fontsize=fs)
    ax[0].set_ylabel(f"d$_i$(e), channel {dch[4:]} (counts)", fontsize=fs)
    ax[0].set_title("example chip, one point per event", fontsize=fs)
    ax[0].legend(fontsize=fs - 3, loc="upper left")
    s_ = slopes.merge(runs, on=["run", "board"])
    s_ = s_[s_.HoldDelay.astype(str).str.startswith("0")]
    ax[1].hist(s_.slope, np.linspace(0.0, 1.4, 71), histtype="stepfilled", alpha=0.7)
    ax[1].axvline(0.75, c="k", ls="--", lw=2, label="0.75: blind channels below")
    ax[1].set_yscale("log"); ax[1].legend(fontsize=fs - 3, loc="upper left")
    ax[1].set_xlabel("slope b$_i$", fontsize=fs); ax[1].set_ylabel("channels x runs", fontsize=fs)
    ax[1].set_title("all chips, runs with HoldDelay 0", fontsize=fs)
    for a_ in ax:
        a_.tick_params(labelsize=fs - 2)
    fig.tight_layout()
    save(fig, "bench", day, "noise_steps_C")
    print(f"CM rms {cm.std():.2f}; slope of channel {dch[4:]} {b:.2f}; slopes "
          f"{s_.slope.quantile(0.01):.2f}-{s_.slope.quantile(0.99):.2f} (1-99%), "
          f"median {s_.slope.median():.3f}")

    # D: total, coherent and incoherent noise of every chip, and the check of the decomposition
    fs = 18
    nc = pd.read_csv(d / "noise_components.csv")
    rr = pd.read_csv(d / "runs.csv")[["run", "board", "HG_ShapingTime", "HoldDelay"]]
    nc = nc.merge(rr, on=["run", "board"])
    nc = nc[nc.HoldDelay.astype(str).str.startswith("0")].copy()
    nc["quad"] = np.sqrt(nc.hg_coherent**2 + nc.hg_incoherent**2)
    first = int(nc.run.min())
    x = nc[nc.run == first].sort_values(["board", "chip"]).reset_index(drop=True)
    fig, ax = plt.subplots(2, 1, figsize=(13, 12))
    k = np.arange(len(x))
    ax[0].plot(k, x.hg_total, "o", ms=11, c="k", label="$\\sigma_\\mathrm{tot}$")
    ax[0].plot(k, x.quad, "o", ms=17, mfc="none", mew=2, c="tab:purple",
               label="$\\sqrt{\\sigma_\\mathrm{coh}^2 + \\sigma_\\mathrm{inc}^2}$")
    ax[0].plot(k, x.hg_incoherent, "s", ms=10, c="tab:green", label="$\\sigma_\\mathrm{inc}$")
    ax[0].plot(k, x.hg_coherent, "^", ms=11, c="tab:orange", label="$\\sigma_\\mathrm{coh}$")
    ax[0].set_xticks(k, [f"{b}/{32 * c}-{32 * c + 31}" for b, c in zip(x.board, x.chip)],
                     fontsize=fs - 4)
    ax[0].set_xlabel("board / channels", fontsize=fs)
    ax[0].set_ylabel("noise (ADC counts)", fontsize=fs)
    ax[0].set_ylim(0, 15)
    ax[0].set_title(f"run {first}, {x.HG_ShapingTime.iloc[0]}: every chip read", fontsize=fs)
    ax[0].legend(fontsize=fs - 4, ncol=4, loc="lower center")
    for tp, g in nc.groupby("HG_ShapingTime"):
        ax[1].plot(g.quad, g.hg_total, "o", ms=10, label=f"HG {tp}")
    t = np.array([10.5, 14.5])
    ax[1].plot(t, t, "k--", lw=1.5, label="equal")
    ax[1].set_xlabel("$\\sqrt{\\sigma_\\mathrm{coh}^2 + \\sigma_\\mathrm{inc}^2}$ (ADC counts)",
                     fontsize=fs)
    ax[1].set_ylabel("$\\sigma_\\mathrm{tot}$ (ADC counts)", fontsize=fs)
    ax[1].set_title("every chip, runs with HoldDelay 0", fontsize=fs)
    ax[1].legend(fontsize=fs - 4, loc="upper left")
    for a_ in ax:
        a_.tick_params(labelsize=fs - 2)
    fig.tight_layout()
    save(fig, "bench", day, "noise_steps_D")
    dev = (nc.quad / nc.hg_total - 1).abs()
    print(f"check of the decomposition: |quad/total - 1| at most {dev.max():.4f}")

    # E: correlation of the common modes of every pair of chips in the first run
    fs = 18
    cmc = pd.read_csv(d / "common_mode.csv")
    r0 = int(cmc.run.min())
    c = cmc[cmc.run == r0]
    chips = sorted(set(zip(c.board_a, c.chip_a)) | set(zip(c.board_b, c.chip_b)))
    idx = {k: i for i, k in enumerate(chips)}
    mat_ = np.eye(len(chips))
    for r in c.itertuples():
        i, j = idx[(r.board_a, r.chip_a)], idx[(r.board_b, r.chip_b)]
        mat_[i, j] = mat_[j, i] = r.correlation
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(mat_, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(chips)):
        for j in range(len(chips)):
            ax.text(j, i, f"{mat_[i, j]:.2f}", ha="center", va="center", fontsize=fs - 5,
                    color="white" if mat_[i, j] > 0.5 else "k")
    labels = [f"{b}/{32 * ch}-{32 * ch + 31}" for b, ch in chips]
    ax.set_xticks(range(len(chips)), labels, rotation=45, ha="right", fontsize=fs - 4)
    ax.set_yticks(range(len(chips)), labels, fontsize=fs - 4)
    ax.set_xlabel("board / channels", fontsize=fs); ax.set_ylabel("board / channels", fontsize=fs)
    ax.set_title(f"run {r0}: correlation of the common modes of two chips", fontsize=fs)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=fs - 4); cb.set_label("correlation", fontsize=fs)
    fig.tight_layout()
    save(fig, "bench", day, "noise_steps_E")


def outliers(day: str, cut: float = 5.0, fs: int = 18):
    d = results_dir("bench", day)
    pulls = pd.read_csv(d / "outlier_pulls.csv")
    runs = pd.read_csv(d / "runs.csv").drop_duplicates("run").set_index("run")
    hold = runs.HoldDelay.astype(str).str.split().str[0].astype(float)
    shown = [int(hold[hold == 0].index.min())]
    fig, axes = plt.subplots(1, 1, figsize=(10, 7), squeeze=False)
    axes = axes[0]
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
