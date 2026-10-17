"""Figures of the dynamic-range study from results/<campaign>/saturation/: for every tag
and case (peaking time or gain), the brightest channel of the 511 and 1275 keV events with
the ceilings, the dimmest channel and the pedestal per channel.

    python3 fullsim_analysis/plots/saturation.py <campaign>
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import open_results, read_hist, read_table, save, step_hist


def main():
    campaign = sys.argv[1]
    df = read_table(campaign, "saturation", "saturation")
    with open_results(campaign, "saturation") as h5:
        for tag, d in df.groupby("tag"):
            g = h5[tag]
            cases = list(d.case.astype(str))
            fig, axs = plt.subplots(len(cases), 3, figsize=(14, 3.4 * len(cases)), squeeze=False)
            for row, case in enumerate(cases):
                gc = g[case]
                r = d[d.case.astype(str) == case].iloc[0]
                ax = axs[row, 0]
                for name, lab in (("brightest_511", "511 keV"), ("brightest_1275", "1275 keV")):
                    e, c = read_hist(gc, name)
                    step_hist(ax, e, c, label=lab)
                for key, lab in (("ceiling_hg", "high-gain ceiling"), ("ceiling_lg", "low-gain ceiling")):
                    if np.isfinite(r[key]):
                        ax.axvline(r[key], color="r" if key == "ceiling_hg" else "k", ls="--", lw=0.8, label=lab)
                ax.set_xlabel("brightest channel (avalanches)")
                ax.set_ylabel("events")
                ax.set_title(f"{tag} {case}: above HG ceiling {100 * r.above_hg_511:.1f}% (511), "
                             f"{100 * r.above_hg_1275:.1f}% (1275)")
                ax.legend()
                ax = axs[row, 1]
                e, c = read_hist(gc, "dimmest_511")
                step_hist(ax, e, c)
                ax.set_xlabel("dimmest channel of 511 keV events (avalanches)")
                ax.set_title(f"dimmest channel: median {r.dimmest511_median:.1f}, 10% below {r.dimmest511_p10:.1f}")
                ax = axs[row, 2]
                e, c = read_hist(gc, "pedestal_channel")
                step_hist(ax, e, c)
                ax.set_xlabel("stored value of a channel without light (avalanches)")
                ax.set_title(f"pedestal per channel {r.pedestal_channel:.1f} ± {r.pedestal_channel_rms:.1f}; "
                             f"noise HG {r.noise_hg_pe:.3f}, LG {r.noise_lg_pe:.2f}")
            save(fig, campaign, "saturation", f"saturation_{tag}")


if __name__ == "__main__":
    main()
