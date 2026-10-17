"""Figures of the Radioroc2 shaping-time scan from results/<campaign>/shaping_scan/: for
every tag, the resolution, the stored fraction, the 1275/511 ratio and the pedestal against
the peaking time, and the spectra at each peaking time.

    python3 fullsim_analysis/plots/shaping_scan.py <campaign>
"""
import sys

import matplotlib.pyplot as plt

from common import open_results, read_hist, read_table, save, step_hist


def main():
    campaign = sys.argv[1]
    df = read_table(campaign, "shaping_scan", "shaping_scan")
    with open_results(campaign, "shaping_scan") as h5:
        for tag, d in df.groupby("tag"):
            d = d.sort_values("tp_ns")
            fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
            axs[0].plot(d.tp_ns, 100 * d.fwhm511, "o-", label="fit")
            axs[0].plot(d.tp_ns, 100 * d.fwhm511_mc, "s--", label="tagged decays")
            axs[0].set_ylabel("FWHM at 511 keV (%)")
            axs[0].legend()
            axs[1].plot(d.tp_ns, d.stored_fraction, "o-")
            axs[1].set_ylabel("stored fraction of the photons at the SiPMs")
            axs[2].plot(d.tp_ns, d.ratio_1275_511_mc, "o-", label="tagged decays")
            axs[2].plot(d.tp_ns, d.ratio_1275_511_fit, "s--", label="fit")
            axs[2].axhline(2.494, color="k", lw=0.6)
            axs[2].set_ylabel("1275 / 511 peak ratio")
            axs[2].legend()
            axs[3].errorbar(d.tp_ns, d.pedestal, yerr=d.pedestal_rms, fmt="o-")
            axs[3].set_ylabel("pedestal of the 64 peak detectors (avalanches)")
            for ax in axs:
                ax.set_xlabel("peaking time (ns)")
            fig.suptitle(tag)
            save(fig, campaign, "shaping_scan", f"shaping_scan_{tag}")
            if tag in h5:
                g = h5[tag]
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                for name in sorted(g.keys(), key=lambda s: int(s.split("_")[1])):
                    e, c = read_hist(g[name], "sum")
                    step_hist(ax, e, c, label=f"{name.split('_')[1]} ns")
                ax.set_xlabel("sum of the channels (avalanches)")
                ax.set_ylabel("events")
                ax.set_title(f"{tag}: spectra against the peaking time")
                ax.legend()
                save(fig, campaign, "shaping_scan", f"shaping_spectra_{tag}")


if __name__ == "__main__":
    main()
