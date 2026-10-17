"""Figures of the digitizer gate scan from results/<campaign>/gate_scan/: for every tag,
the resolution and the stored fraction against the gate start (at the configured length)
and against the gate length (at the chosen start), with the charge of an empty gate.

    python3 fullsim_analysis/plots/gate_scan.py <campaign>
"""
import sys

import matplotlib.pyplot as plt

from common import read_table, save


def main():
    campaign = sys.argv[1]
    df = read_table(campaign, "gate_scan", "gate_scan")
    for tag, d in df.groupby("tag"):
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        glen = d.gate_ns.mode().iloc[0]
        a = d[d.gate_ns == glen].sort_values("gate_start_ns")
        axs[0].plot(a.gate_start_ns, 100 * a.fwhm511, "o-", label="fit")
        axs[0].plot(a.gate_start_ns, 100 * a.fwhm511_mc, "s--", label="tagged decays")
        axs[0].set_xlabel(f"gate start relative to the trigger (ns), gate {glen:.0f} ns")
        axs[0].set_ylabel("FWHM at 511 keV (%)")
        axs[0].legend()
        gstart = d.gate_start_ns.mode().iloc[0] if len(d.gate_ns.unique()) > 1 else None
        b = d[d.gate_start_ns == (gstart if gstart is not None else a.gate_start_ns.iloc[0])].sort_values("gate_ns")
        axs[1].plot(b.gate_ns / 1000, 100 * b.fwhm511, "o-", label="fit")
        axs[1].plot(b.gate_ns / 1000, 100 * b.fwhm511_mc, "s--", label="tagged decays")
        axs[1].set_xlabel(f"gate length (us), start {b.gate_start_ns.iloc[0]:.0f} ns")
        axs[1].set_ylabel("FWHM at 511 keV (%)")
        axs[1].legend()
        ax2 = axs[1].twinx()
        ax2.plot(b.gate_ns / 1000, b.stored_fraction, "^:", color="C2")
        ax2.set_ylabel("stored fraction of the photons", color="C2")
        ax2.grid(False)
        axs[2].errorbar(b.gate_ns / 1000, b.empty_charge_mean, yerr=b.empty_charge_rms, fmt="o-")
        axs[2].set_xlabel("gate length (us)")
        axs[2].set_ylabel("charge of an empty gate, 64 channels (avalanches)")
        fig.suptitle(tag)
        save(fig, campaign, "gate_scan", f"gate_scan_{tag}")


if __name__ == "__main__":
    main()
