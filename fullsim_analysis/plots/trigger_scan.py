"""Figure of the trigger threshold scan: for every tag in
results/<campaign>/trigger_scan/trigger_scan.csv, the 511 keV and Compton efficiencies and
the dark trigger probability per window against the threshold, one line style per mode.

    python3 fullsim_analysis/plots/trigger_scan.py <campaign>
"""
import sys

import matplotlib.pyplot as plt

from common import read_table, save

UNITS = {"citiroc": "threshold (photoelectron-equivalents, fast shaper)",
         "digitizer": "threshold (counts)",
         "radioroc": "threshold (avalanches: TQ on the low-gain shaper, T1/T2 on the time preamplifier)"}


def main():
    campaign = sys.argv[1]
    df = read_table(campaign, "trigger_scan", "trigger_scan")
    for tag, d in df.groupby("tag"):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        styles = ["-", "--", ":", "-."]
        for (mode, m), ls in zip(d.groupby("mode"), styles):
            m = m.sort_values("threshold")
            ax.plot(m.threshold, m.efficiency_511, ls, color="C0", label=f"511 keV efficiency, {mode}")
            ax.plot(m.threshold, m.efficiency_200_300, ls, color="C1", label=f"200 to 300 keV, {mode}")
            ax.plot(m.threshold, m.dark_per_window, ls, color="C2",
                    label=f"dark trigger per {m.window_ns.iloc[0] / 1000:.1f} us window, {mode}")
        ax.set_xlabel(UNITS.get(d.frontend.iloc[0], "threshold"))
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{tag}: {d.frontend.iloc[0]}")
        ax.legend(loc="center right")
        save(fig, campaign, "trigger_scan", f"trigger_scan_{tag}")


if __name__ == "__main__":
    main()
