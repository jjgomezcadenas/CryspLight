"""Figures of the first-look summary: for every tag in results/<campaign>/summary/summary.h5
the spectrum of the equalised sum with the fitted peaks, the brightest channel, the mean
and seed-channel maps. One PNG per tag, summary_<tag>.png.

    python3 fullsim_analysis/plots/summary.py <campaign>
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import channel_map, open_results, read_hist, save, step_hist


def plot_tag(campaign, h5, tag):
    g = h5[tag]
    a = dict(g.attrs)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ax = axs[0, 0]
    e, c = read_hist(g, "sum")
    step_hist(ax, e, c)
    for key, label in (("peak511", "511 keV"), ("peak1275", "1275 keV")):
        if np.isfinite(a.get(key, np.nan)):
            ax.axvline(a[key], color="r", ls="--", lw=0.8, label=f"{label}: {a[key]:.0f}")
    ax.set_xlabel("sum of the channels (avalanches)")
    ax.set_ylabel("events")
    ax.set_title(f"{tag}: {a['frontend']}, {int(a['events'])} events, "
                 f"FWHM {100 * a['fwhm511']:.1f}%, 1275/511 = {a['ratio_1275_511']:.2f}")
    ax.legend()
    ax = axs[0, 1]
    e, c = read_hist(g, "brightest")
    step_hist(ax, e, c)
    ax.set_xlabel("brightest channel (avalanches), 511 keV window")
    ax.set_ylabel("events")
    ax.set_title(f"brightest channel: median {a['brightest_median']:.0f}, 1% above {a['brightest_p99']:.0f}")
    im = channel_map(axs[1, 0], g["mean_map"][...], "mean charge per channel, 511 keV window")
    fig.colorbar(im, ax=axs[1, 0])
    im = channel_map(axs[1, 1], g["seed_map"][...], "seed-channel photopeak / median",
                     vmin=0.9, vmax=1.1, cmap="coolwarm")
    fig.colorbar(im, ax=axs[1, 1])
    save(fig, campaign, "summary", f"summary_{tag}")


def main():
    campaign = sys.argv[1]
    with open_results(campaign, "summary") as h5:
        tags = sys.argv[2:] or list(h5.keys())
        for tag in tags:
            plot_tag(campaign, h5, tag)


if __name__ == "__main__":
    main()
