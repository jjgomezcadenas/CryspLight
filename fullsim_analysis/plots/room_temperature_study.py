"""Figures of the study at room temperature (docs_fullsim/room_temperature_study.tex) from
results/<campaign>/room_temperature/ (room_temperature.csv, room_temperature.h5), .../trigger_scan/trigger_scan.csv and
.../calibration/ (calibration.csv, calibration.h5):

  case_<case>.png    one per case: trigger efficiency and dark trigger rate against the
                     threshold, single-photoelectron spectrum of the calibration, energy
                     spectrum with the fit of the 511 keV peak, and the chain from emitted
                     photons to recorded charge;
  compare_resolution.png   FWHM at 511 keV of the six cases, and its decomposition;
  compare_light.png        photoelectrons, fired cells and recorded charge, and the
                           efficiencies of each step;
  compare_channels.png     brightest and least illuminated channel distributions.

    python3 fullsim_analysis/plots/room_temperature_study.py <campaign>
"""
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

from common import read_hist, read_table, results_dir, save, step_hist

LABEL = {"citiroc": "Citiroc 1A", "digitizer": "DT2745", "radioroc": "Radioroc2"}
CRYSTAL = {"bgo_ambient": "BGO", "csitl_ambient": "CsI(Tl)"}
MODE = {"citiroc": "single", "digitizer": "majority2", "radioroc": "tq"}
COLOR = {"citiroc": "#1f77b4", "digitizer": "#2ca02c", "radioroc": "#d62728"}


def case_label(r):
    return f"{CRYSTAL[r.crystal]}, {LABEL[r.frontend]}"


def trigger_panel(ax, campaign, r):
    t = read_table(campaign, "trigger_scan", "trigger_scan")
    d = t[(t.tag == r.tag) & (t["mode"] == MODE[r.frontend])].sort_values("threshold")
    if d.empty:
        ax.text(0.5, 0.5, "no scan", ha="center", transform=ax.transAxes)
        return
    x = d["threshold_pe"] if "threshold_pe" in d and d["threshold_pe"].notna().all() else d["threshold"]
    ax.plot(x, d.efficiency_511, "o-", label="511 keV photopeak")
    ax.plot(x, d.efficiency_200_300, "s-", label="deposits of 200-300 keV")
    ax.set_xlabel("threshold (photoelectron equivalents)")
    ax.set_ylabel("trigger efficiency")
    ax.set_ylim(0, 1.05)
    a2 = ax.twinx()
    a2.semilogy(x, np.maximum(d.dark_khz, 1e-3), "k^--", label="dark trigger rate")
    a2.set_ylabel("dark trigger rate (kHz)")
    a2.axhline(1.0, color="k", lw=0.6, ls=":")
    a2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title("trigger")


def calibration_panel(ax, campaign, r):
    c = read_table(campaign, "calibration", "calibration")
    key = r.tag
    row = c[c.tag == key]
    with h5py.File(results_dir(campaign, "calibration") / "calibration.h5", "r") as f:
        if key not in f:
            ax.text(0.5, 0.5, "no calibration", ha="center", transform=ax.transAxes)
            return
        e, n = read_hist(f[key], "spectrum_cal")
        tp = f[key].attrs.get("tp_ns", float("nan"))
    step_hist(ax, e, n, color="k")
    ax.set_xlabel("(amplitude - fitted zero) / fitted gain")
    ax.set_ylabel("entries")
    ttl = "dark-count spectrum"
    if r.frontend != "digitizer" and np.isfinite(tp):
        ttl += f", high gain, {tp:g} ns shaping"
    if not row.empty:
        ax.text(0.97, 0.95, f"fitted / true gain\n{row.fitted_over_true_median.iloc[0]:.3f} "
                f"$\\pm$ {row.fitted_over_true_mad.iloc[0]:.3f}", ha="right", va="top", transform=ax.transAxes)
    ax.set_title(ttl)
    ax.set_xlim(-0.5, 4.5)


def spectrum_panel(ax, h5, r):
    g = h5[r.case]
    e, n = read_hist(g, "spectrum_kev")
    step_hist(ax, e, n, color="k")
    mu, sg = g.attrs["fit511_mean"], g.attrs["fit511_sigma"]
    xx = np.linspace(mu - 2.5 * sg, mu + 2.5 * sg, 100)
    w = e[1] - e[0]
    sel = (e[:-1] > mu - sg) & (e[:-1] < mu + sg)
    amp = n[sel].max() if sel.any() else 1
    ax.plot(xx, amp * np.exp(-0.5 * ((xx - mu) / sg) ** 2), "r-",
            label=f"511 keV: FWHM {100 * r.fwhm511:.1f}%")
    ax.set_xlim(0, 2000)
    ax.set_xlabel("energy (keV)")
    ax.set_ylabel(f"events / {w:.0f} keV")
    ax.legend(loc="upper right")
    ax.set_title("energy spectrum")


def chain_panel(ax, r):
    names = ["emitted\nphotons", "photons at\nthe SiPMs", "photo-\nelectrons", "fired\ncells", "recorded"]
    vals = [r.photons_emitted, r.photons_at_sipm, r.photoelectrons, r.fired_cells, r.recorded_pe]
    b = ax.bar(names, vals, color=["0.7", "0.55", COLOR[r.frontend], COLOR[r.frontend], "k"])
    for bb, v in zip(b, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("per 511 keV photopeak event")
    ax.set_title("from photons to recorded charge")


def main():
    campaign = sys.argv[1]
    df = read_table(campaign, "room_temperature", "room_temperature")
    with h5py.File(results_dir(campaign, "room_temperature") / "room_temperature.h5", "r") as h5:
        for _, r in df.iterrows():
            fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
            trigger_panel(axs[0, 0], campaign, r)
            calibration_panel(axs[0, 1], campaign, r)
            spectrum_panel(axs[1, 0], h5, r)
            chain_panel(axs[1, 1], r)
            fig.suptitle(case_label(r))
            save(fig, campaign, "room_temperature", f"case_{r.case}")

        # resolution
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
        xs = np.arange(len(df))
        labs = [case_label(r).replace(", ", "\n") for _, r in df.iterrows()]
        a.bar(xs, 100 * df.fwhm511, yerr=100 * df.fwhm511_err, color=[COLOR[f] for f in df.frontend])
        a.axhline(100 * df.intrinsic.iloc[0], color="k", ls="--", lw=1, label="intrinsic resolution of the crystals")
        a.set_xticks(xs, labs, fontsize=8)
        a.set_ylabel("FWHM at 511 keV (%)")
        a.legend()
        terms = [("intrinsic", "intrinsic", "0.6"), ("term_statistics", "photoelectron statistics", "#9ecae1"),
                 ("term_sipm", "crosstalk and afterpulses", "#fdae6b"), ("term_readout", "readout", "#d62728")]
        bottom = np.zeros(len(df))
        for col, lab, colr in terms:
            v = (100 * df[col]) ** 2
            b.bar(xs, v, bottom=bottom, color=colr, label=lab)
            bottom += v
        b.plot(xs, (100 * df.width_recorded) ** 2, "k_", ms=25, mew=2, label="total (photopeak events)")
        b.set_xticks(xs, labs, fontsize=8)
        b.set_ylabel("(FWHM, %)$^2$")
        b.legend(fontsize=7)
        a.set_title("energy resolution")
        b.set_title("contributions, added in quadrature")
        save(fig, campaign, "room_temperature", "compare_resolution")

        # light
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
        w = 0.27
        a.bar(xs - w, df.photoelectrons, w, label="photoelectrons", color="#9ecae1")
        a.bar(xs, df.fired_cells, w, label="fired cells", color="#fdae6b")
        a.bar(xs + w, df.recorded_pe, w, label="recorded (photoelectron equivalents)", color="k")
        a.set_yscale("log")
        a.set_xticks(xs, labs, fontsize=8)
        a.set_ylabel("per 511 keV photopeak event")
        a.legend(fontsize=7)
        a.set_title("photoelectrons and recorded charge")
        b.bar(xs - w, df.collection, w, label="photons at the SiPMs / emitted", color="0.6")
        b.bar(xs, df.detection, w, label="photoelectrons / photons at the SiPMs", color="#9ecae1")
        b.bar(xs + w, df.recorded_per_pe, w, label="recorded / photoelectrons", color="k")
        b.set_xticks(xs, labs, fontsize=8)
        b.set_ylabel("ratio")
        b.legend(fontsize=7)
        b.set_title("efficiency of each step")
        save(fig, campaign, "room_temperature", "compare_light")

        # channels
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
        for _, r in df.iterrows():
            ls = "-" if r.crystal == "bgo_ambient" else "--"
            e, n = read_hist(h5[r.case], "brightest")
            step_hist(a, e, n / n.sum(), color=COLOR[r.frontend], ls=ls, label=case_label(r))
            e, n = read_hist(h5[r.case], "least")
            step_hist(b, e, n / n.sum(), color=COLOR[r.frontend], ls=ls, label=case_label(r))
        a.set_xscale("log")
        a.set_xlabel("brightest channel (photoelectron equivalents)")
        b.set_xlabel("least illuminated channel (photoelectron equivalents)")
        for ax in (a, b):
            ax.set_ylabel("fraction of events")
            ax.legend(fontsize=7)
        save(fig, campaign, "room_temperature", "compare_channels")


if __name__ == "__main__":
    main()
