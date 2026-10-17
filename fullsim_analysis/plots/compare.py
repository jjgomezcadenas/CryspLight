"""Figure of the front-end comparison from results/<campaign>/compare/: for every crystal,
the 511 keV photopeak sum of each front end (normalised to its mean), the seed-channel and
dimmest-channel distributions, and a table of the main numbers.

    python3 fullsim_analysis/plots/compare.py <campaign>
"""
import sys

import matplotlib.pyplot as plt

from common import open_results, read_hist, read_table, save, step_hist


def main():
    campaign = sys.argv[1]
    df = read_table(campaign, "compare", "compare")
    with open_results(campaign, "compare") as h5:
        for crystal, d in df.groupby("crystal"):
            fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
            for _, r in d.iterrows():
                g = h5[r.tag]
                lab = f"{r.frontend} {r.overvoltage_v:.1f} V"
                e, c = read_hist(g, "photopeak_sum")
                step_hist(axs[0, 0], e, c / c.sum(), label=f"{lab}: {100 * r.fwhm511_tagged:.1f}%")
                e, c = read_hist(g, "seed")
                step_hist(axs[0, 1], e, c / c.sum(), label=lab)
                e, c = read_hist(g, "dimmest")
                step_hist(axs[1, 0], e, c / c.sum(), label=lab)
            axs[0, 0].set_xlabel("photopeak sum / its mean (tagged 511 keV decays)")
            axs[0, 1].set_xlabel("seed channel (avalanches stored)")
            axs[1, 0].set_xlabel("dimmest channel (avalanches stored)")
            for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
                ax.set_ylabel("fraction of events")
                ax.legend()
            axs[1, 1].axis("off")
            cols = ["frontend", "overvoltage_v", "fwhm511_tagged", "stored_per_emitted",
                    "charge_per_channel_mean", "seed_median", "seed_above_ceiling", "spe_snr"]
            heads = ["front end", "Vov", "FWHM", "stored /\nemitted", "per\nchannel", "seed",
                     "seed above\nceiling", "SPE\nS/N"]
            cells = [[r.frontend, f"{r.overvoltage_v:.1f}", f"{100 * r.fwhm511_tagged:.1f}%",
                      f"{r.stored_per_emitted:.3f}", f"{r.charge_per_channel_mean:.1f}",
                      f"{r.seed_median:.0f}", f"{100 * r.seed_above_ceiling:.1f}%", f"{r.spe_snr:.0f}"]
                     for _, r in d[cols].iterrows()]
            t = axs[1, 1].table(cellText=cells, colLabels=heads, loc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            t.scale(1, 2.2)
            fig.suptitle(f"{crystal}: front ends compared on the same decays")
            save(fig, campaign, "compare", f"compare_{crystal}")


if __name__ == "__main__":
    main()
