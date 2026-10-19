"""Figure of the simulated geometry, drawn from the optical configurations and the channel
centres of an event file: side view of the crystal (BGO and CsI(Tl) depths), wrapping,
optical coupling, SiPM tiles and source; face view of the readout (64 active areas, four
tiles, crystal outline). The coupling layers are drawn thicker than they are.

    python3 fullsim_analysis/plots/geometry.py <campaign> <optical bgo toml> <optical csitl toml> <events.h5>
"""
import sys
import tomllib

import h5py
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

from common import save


def main():
    campaign, fb, fc, ev = sys.argv[1:5]
    cb = tomllib.load(open(fb, "rb"))
    cc = tomllib.load(open(fc, "rb"))
    lx, ly, zb = cb["crystal"]["size_mm"]
    zc = cc["crystal"]["size_mm"][2]
    ro = cb["readout"]
    act = ro["active_mm"]
    src = cb["source"]["position_mm"]
    with h5py.File(ev, "r") as f:
        x = f["channels/x_mm"][...]
        y = f["channels/y_mm"][...]
        tile = f["channels/tile"][...]

    fig, (a1, a2, b) = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ux = np.unique(np.round(x, 2))
    t0, t1 = ux[0] - ro["pitch_mm"] / 2 - 0.1, ux[-1] + ro["pitch_mm"] / 2 + 0.1
    wrap, layer = 1.2, 1.2              # drawn thicknesses (not to scale)

    def side(a, depth, name, legend):
        """Side view: x horizontal, z vertical, source below, SiPMs on top."""
        a.add_patch(mp.Rectangle((0, 0), lx, depth, fc="#cfe3f5", ec="k", lw=1.2, label="crystal"))
        a.add_patch(mp.Rectangle((-wrap, -wrap), lx + 2 * wrap, depth + wrap, fill=False, ec="0.5", lw=2.5,
                                 label="PTFE wrapping (air gap)"))
        a.add_patch(mp.Rectangle((t0, depth), t1 - t0, layer, fc="#f6e7a8", ec="k", lw=0.8,
                                 label=f"grease, n = {cb['coupling']['n']}"))
        a.add_patch(mp.Rectangle((t0, depth + layer), t1 - t0, layer, fc="#e6c76a", ec="k", lw=0.8,
                                 label=f"tile window, n = {cb['coupling']['window_n']}"))
        for xc in ux:
            a.add_patch(mp.Rectangle((xc - act / 2, depth + 2 * layer), act, 1.6, fc="#3b6ea5", ec="k", lw=0.5))
        a.add_patch(mp.Rectangle((t0, depth + 2 * layer + 1.6), t1 - t0, 2.5, fc="0.85", ec="k", lw=0.8,
                                 label="SiPM tiles"))
        a.plot([], [], "s", color="#3b6ea5", label=f"SiPM active areas, {act:.0f} mm")
        a.plot(src[0], src[2], "r*", ms=14, label=f"$^{{22}}$Na source, {-src[2]:.0f} mm from the crystal")
        for ang in (-0.6, 0.0, 0.6):
            a.annotate("", xy=(src[0] + 20 * np.sin(ang), src[2] + 20 * np.cos(ang)), xytext=(src[0], src[2]),
                       arrowprops=dict(arrowstyle="->", color="r", lw=0.8))
        a.set_xlim(-6, lx + 6)
        a.set_ylim(src[2] - 5, zc + 10)
        a.set_aspect("equal")
        a.set_xlabel("x (mm)")
        a.set_ylabel("z (mm)")
        a.set_title(f"{name}: side view, {depth:.1f} mm deep")
        a.grid(False)
        if legend:
            a.legend(loc="upper left", fontsize=6.5, bbox_to_anchor=(0.0, -0.13), ncol=2)

    side(a1, zb, "BGO", True)
    side(a2, zc, "CsI(Tl)", False)
    # face view of the readout
    for t in np.unique(tile):
        m = tile == t
        bx0, bx1 = x[m].min() - ro["pitch_mm"] / 2 - 0.1, x[m].max() + ro["pitch_mm"] / 2 + 0.1
        by0, by1 = y[m].min() - ro["pitch_mm"] / 2 - 0.1, y[m].max() + ro["pitch_mm"] / 2 + 0.1
        b.add_patch(mp.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fc="0.88", ec="k", lw=1.0))
    for xc, yc in zip(x, y):
        b.add_patch(mp.Rectangle((xc - act / 2, yc - act / 2), act, act, fc="#3b6ea5", ec="k", lw=0.4))
    b.add_patch(mp.Rectangle((0, 0), lx, ly, fill=False, ec="r", ls="--", lw=1.4, label="crystal face"))
    b.set_xlim(-4, lx + 4)
    b.set_ylim(-4, ly + 4)
    b.set_aspect("equal")
    b.set_xlabel("x (mm)")
    b.set_ylabel("y (mm)")
    b.set_title(f"readout face: 4 tiles of 4 x 4 channels, pitch {ro['pitch_mm']} mm")
    b.legend(loc="upper right", fontsize=7)
    b.grid(False)
    fig.tight_layout()
    save(fig, campaign, "room_temperature", "geometry")


if __name__ == "__main__":
    main()
