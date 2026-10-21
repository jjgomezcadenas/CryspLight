"""Figures of the Citiroc calibration with the periodic trigger and the peak detector
(docs_fullsim/calib/citiroc_calib.tex) from results/calib/citiroc/, written by
fullsim_analysis/scripts/citiroc_calib.jl:
    citiroc_ladders.png   spectra of the windows at 25, 50 and 87.5 ns and 3, 4, 5 V, in
                          units of the true gain from the pedestal, with the fitted peaks;
    citiroc_noise.png     the spectrum at 25 ns and 5 V for three noise levels;
    citiroc_k.png         the correction factor k (fitted spacing over true gain) against
                          the noise, the overvoltage and the window;
    citiroc_channels.png  k channel by channel (64 channels, 5% gain spread);
    citiroc_decomposition.png  the spectrum at 25 and 87.5 ns, 5 V, without and with noise,
                          split by the arrival of the avalanches relative to the window;
    citiroc_cold.png      spectra at 20, -3 and -23 C, 5 V of overvoltage, 25, 50 and 87.5 ns,
                          with their split;
    citiroc_cold_k.png    k (ladder and one-photoelectron peak) and the fraction of windows
                          without an avalanche against the temperature;
    citiroc_scenarios.png the three operating scenarios with 64 channels (20 C at 25 ns, -3 C
                          and -23 C at 87.5 ns): the spectrum of all channels in true gains
                          and k channel by channel.

    python3 fullsim_analysis/plots/citiroc_calib.py
"""
import matplotlib.pyplot as plt
import numpy as np

from common import open_results, read_hist, read_table, save, step_hist

CAMP, STUDY = "calib", "citiroc"
TPS, VOVS = (25.0, 50.0, 87.5), (3.0, 4.0, 5.0)


def gname(tp, v, n, w):
    return f"A_tp{tp:g}_v{v:g}_n{n:g}_w{w:g}"


def ladders(h5):
    fig, axs = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for i, tp in enumerate(TPS):
        for j, v in enumerate(VOVS):
            ax = axs[i, j]
            g = h5[gname(tp, v, 1, 200)]
            e, c = read_hist(g, "spectrum_pe")
            step_hist(ax, e, c, color="C0")
            for p in g["peaks_pe"][...]:
                ax.axvline(p, color="C3", lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_title(f"{tp:g} ns, {v:g} V")
            if i == 2:
                ax.set_xlabel("stored value - pedestal (true gains)")
            if j == 0:
                ax.set_ylabel("windows per bin")
    save(fig, CAMP, STUDY, "citiroc_ladders")


def noise(h5):
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, col in ((0, "k"), (1, "C0"), (2, "C3")):
        g = h5[gname(25, 5, n, 200)]
        e, c = read_hist(g, "spectrum_pe")
        step_hist(ax, e, c, color=col, label=f"noise {n:g} x model")
        for p in g["peaks_pe"][...]:
            ax.axvline(p, color=col, lw=0.7, ls="--")
    for k in range(1, 4):
        ax.axvline(k, color="0.5", lw=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("stored value - pedestal (true gains)")
    ax.set_ylabel("windows per bin")
    ax.set_title("25 ns, 5 V, window 200 ns; dashed: fitted peaks")
    ax.legend()
    save(fig, CAMP, STUDY, "citiroc_noise")


def kfactor(df):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    d200 = df[df.window_ns == 200]
    for tp, col in zip(TPS, ("C0", "C1", "C2")):
        d = d200[(d200.tp_ns == tp) & (d200.vov == 5)].sort_values("noise_scale")
        axs[0].errorbar(d.noise_pe, d.k, d.k_err, fmt="o-", color=col, label=f"{tp:g} ns")
        d = d200[(d200.tp_ns == tp) & (d200.noise_scale == 1)].sort_values("vov")
        axs[1].errorbar(d.vov, d.k, d.k_err, fmt="o-", color=col, label=f"{tp:g} ns")
    axs[0].set_xlabel("rms noise of the high gain at its peaking time (pe)")
    axs[0].set_title("5 V, window 200 ns")
    axs[1].set_xlabel("overvoltage (V)")
    axs[1].set_title("noise of the model, window 200 ns")
    d = df[(df.tp_ns == 25) & (df.vov == 5) & (df.noise_scale == 1)].sort_values("window_ns")
    axs[2].errorbar(d.window_ns, d.k, d.k_err, fmt="o-", color="C0", label="k")
    ax2 = axs[2].twinx()
    ax2.plot(d.window_ns, d.f_signal, "s--", color="C3")
    ax2.set_ylabel("fraction of windows above half a gain", color="C3")
    ax2.grid(False)
    axs[2].set_xlabel("window = hold delay (ns)")
    axs[2].set_title("25 ns, 5 V, noise of the model")
    for ax in axs:
        ax.axhline(1, color="k", lw=0.6)
        ax.set_ylabel("k = fitted spacing / true gain")
    axs[0].legend()
    axs[1].legend()
    save(fig, CAMP, STUDY, "citiroc_k")


def channels(h5):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for v, col in ((3, "C0"), (5, "C3")):
        g = h5[f"B_tp25_v{v}_n1_w200"]
        k = g["k"][...]
        tg = g["true_gain"][...]
        ok = np.isfinite(k)
        axs[0].hist(k[ok], bins=25, histtype="step", color=col,
                    label=f"{v} V: {ok.sum()} channels, median {np.median(k[ok]):.3f}, rms {np.std(k[ok]):.3f}")
        axs[1].plot(tg[ok] / 10, k[ok], "o", color=col, ms=4, label=f"{v} V")
    axs[0].set_xlabel("k = fitted spacing / true gain")
    axs[0].set_ylabel("channels")
    axs[0].legend()
    axs[1].set_xlabel("true gain / nominal")
    axs[1].set_ylabel("k")
    axs[1].legend()
    fig.suptitle("64 channels, 5% gain spread, 25 ns, window 200 ns, noise of the model")
    save(fig, CAMP, STUDY, "citiroc_channels")


LABELS = {"none": "no avalanche (400 ns before or in the window)", "before": "only before the opening",
          "inside": "only inside, reaching the maximum", "late": "only inside, too late for the maximum",
          "mixed": "mixture"}


def decomposition(h5):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for i, tp in enumerate((25.0, 87.5)):
        for j, n in enumerate((0, 1)):
            ax = axs[i, j]
            g = h5[f"D_tp{tp:g}_v5_n{n:g}_w200"]
            tot = None
            for c, col in zip(LABELS, ("0.5", "C0", "C3", "C2", "C1")):
                e, h = read_hist(g, f"spectrum_{c}")
                tot = h if tot is None else tot + h
                step_hist(ax, e, h, color=col, label=LABELS[c])
            step_hist(ax, e, tot, color="k", lw=1.2, label="all windows")
            for k in range(1, 4):
                ax.axvline(k, color="0.6", lw=0.5)
            ax.set_yscale("log")
            ax.set_ylim(1, None)
            ax.set_title(f"{tp:g} ns, 5 V, window 200 ns, noise {n:g} x model "
                         f"(maximum {g.attrs['t_peak_ns']:.0f} ns after the avalanche)")
            if i == 1:
                ax.set_xlabel("stored value - pedestal (true gains)")
            if j == 0:
                ax.set_ylabel("windows per bin")
    axs[0, 0].legend(loc="upper right")
    save(fig, CAMP, STUDY, "citiroc_decomposition")


TEMPS = (293, 270, 250)


def cold(h5):
    fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for i, tk in enumerate(TEMPS):
        for j, tp in enumerate(TPS):
            ax = axs[i, j]
            g = h5[f"E_T{tk}_tp{tp:g}"]
            for c, col in zip(LABELS, ("0.5", "C0", "C3", "C2", "C1")):
                e, h = read_hist(g, f"spectrum_{c}")
                step_hist(ax, e, h, color=col, lw=0.8, label=LABELS[c])
            e, h = read_hist(g, "spectrum_pe")
            step_hist(ax, e, h, color="k", lw=1.2, label="all windows")
            for k in range(1, 4):
                ax.axvline(k, color="0.6", lw=0.5)
            ax.set_yscale("log")
            ax.set_ylim(1, None)
            ax.set_title(f"{tk - 273.15:.0f} C, {tp:g} ns, 5 V: dark {g.attrs['dark_rate_hz'] / 1e3:.0f} kHz, "
                         f"{g.attrs['records']} records")
            if i == 2:
                ax.set_xlabel("stored value - pedestal (true gains)")
            if j == 0:
                ax.set_ylabel("windows per bin")
    axs[0, 0].legend(loc="upper right", fontsize=7)
    save(fig, CAMP, STUDY, "citiroc_cold")


def cold_k(df):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    tc = df.t_k - 273.15
    for tp, col in zip(TPS, ("C0", "C1", "C2")):
        d = df[df.tp_ns == tp].sort_values("t_k")
        x = d.t_k - 273.15
        axs[0].errorbar(x, d.k, d.k_err, fmt="o-", color=col, label=f"{tp:g} ns")
        axs[1].errorbar(x, d.k1, d.k1_err, fmt="o-", color=col, label=f"{tp:g} ns")
    d = df[df.tp_ns == 25].sort_values("t_k")
    for c, col in zip(LABELS, ("0.5", "C0", "C3", "C2", "C1")):
        axs[2].plot(d.t_k - 273.15, d["f_" + c], "o-", color=col, label=LABELS[c])
    axs[0].set_ylabel("k from the ladder")
    axs[1].set_ylabel("k from the one-photoelectron peak")
    axs[2].set_ylabel("fraction of the windows (25 ns)")
    axs[2].set_yscale("log")
    for ax in axs[:2]:
        ax.axhline(1, color="k", lw=0.6)
        ax.legend()
    axs[2].legend(fontsize=7)
    for ax in axs:
        ax.set_xlabel("temperature (C)")
    fig.suptitle("5 V of overvoltage at each temperature, window 200 ns, noise of the model")
    save(fig, CAMP, STUDY, "citiroc_cold_k")


SCEN = (("F_T293_tp25", "20 C, 25 ns"), ("F_T270_tp87.5", "-3 C, 87.5 ns"), ("F_T250_tp87.5", "-23 C, 87.5 ns"))


def scenarios(h5):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for (gn, lab), col in zip(SCEN, ("C3", "C0", "C2")):
        g = h5[gn]
        e, c = read_hist(g, "spectrum_pe_all")
        step_hist(axs[0], e, c / c.sum(), color=col, label=lab)
        for key, ax in (("k", axs[1]), ("k1", axs[2])):
            k = g[key][...]
            ok = np.isfinite(k)
            ax.hist(k[ok], bins=np.linspace(0.6, 1.2, 61), histtype="step", color=col,
                    label=f"{lab}: {ok.sum()}/64, median {np.median(k[ok]):.3f}, rms {np.std(k[ok]):.3f}")
    for k in range(1, 5):
        axs[0].axvline(k, color="0.6", lw=0.5)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("stored value - pedestal (true gains of each channel)")
    axs[0].set_ylabel("fraction of the windows per bin")
    axs[0].legend()
    axs[1].set_xlabel("k from the ladder")
    axs[2].set_xlabel("k from the one-photoelectron peak")
    for ax in axs[1:]:
        ax.set_ylabel("channels")
        ax.axvline(1, color="k", lw=0.6)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("64 channels, 5% gain spread, 5 V of overvoltage, window 200 ns, noise of the model")
    save(fig, CAMP, STUDY, "citiroc_scenarios")


def main():
    df = read_table(CAMP, STUDY, "citiroc_A")
    with open_results(CAMP, STUDY) as h5:
        ladders(h5)
        noise(h5)
        channels(h5)
        decomposition(h5)
        cold(h5)
        scenarios(h5)
    cold_k(read_table(CAMP, STUDY, "citiroc_E"))
    kfactor(df)


if __name__ == "__main__":
    main()
