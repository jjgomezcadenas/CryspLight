#!/usr/bin/env python3
"""Control plots of the interaction truth in events.h5 (tracked figure source).

Filters to fully contained events on the fly (edep > 510.5 keV -- NOT er == 0,
which would drop every contained C3+ event: er = 511 - E1 - E2 holds the energy
of deposits beyond the second). Writes, per input file, into results/true_info:
  <tag>_positions.png   x1 vs y1, x1 vs z1, x2 vs y2, x2 vs z2 (n_int >= 2)
  <tag>_energies.png    E1, E2, Er histograms and E1 vs E2
  <tag>_topology.png    interaction-type frequencies (photo, C1, C2, ...) and npe
  <tag>_stats.json/.txt total shot / interacted / contained + type breakdown

Usage: python3 analysis/true_vars.py [events.h5 ...]
       (default: the two 50k runs in ../output)
"""
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "results" / "true_info"
DEFAULTS = [HERE.parent / "output" / t / "events.h5"
            for t in ("gammas_csitl_50k", "gammas_bgo_50k")]
E0_KEV = 511.0
CONTAINED_KEV = 510.5


def load(path):
    with h5py.File(path, "r") as f:
        d = {k: f[k][()] for k in ("edep_kev", "npe", "e1_kev", "e2_kev",
                                   "er_kev", "xyz1_mm", "xyz2_mm",
                                   "int_type", "n_int")}
        def s(a):  # h5py may hand attrs back as bytes
            return a.decode() if isinstance(a, bytes) else str(a)

        d["tag"] = s(f.attrs["tag"])
        d["n_shot"] = int(f.attrs["n_events"])
        d["crystal"] = s(f.attrs["crystal"])
    return d


def type_label(code):
    return "photo" if code == 0 else f"C{code}"


def hist2d(ax, x, y, xlabel, ylabel, title, bins=60):
    h = ax.hist2d(x, y, bins=bins, cmap="viridis", cmin=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    return h


def main(paths):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        d = load(path)
        tag = d["tag"]
        contained = d["edep_kev"] > CONTAINED_KEV
        # the event axis is FIRST in h5py (Julia column-major (3,n) reads as (n,3))
        c = {k: d[k][contained]
             for k in ("e1_kev", "e2_kev", "er_kev", "xyz1_mm", "xyz2_mm",
                       "int_type", "n_int", "npe")}
        multi = c["n_int"] >= 2  # xyz2/e2 are zero-filled below this

        # ---- positions ----
        fig, axs = plt.subplots(2, 2, figsize=(10, 9))
        x1, y1, z1 = c["xyz1_mm"].T
        x2, y2, z2 = c["xyz2_mm"][multi].T
        hist2d(axs[0, 0], x1, y1, "x1 [mm]", "y1 [mm]", "first interaction: x1 vs y1")
        hist2d(axs[0, 1], x1, z1, "x1 [mm]", "z1 [mm]", "first interaction: x1 vs z1")
        hist2d(axs[1, 0], x2, y2, "x2 [mm]", "y2 [mm]",
               "second interaction: x2 vs y2 (n_int >= 2)")
        hist2d(axs[1, 1], x2, z2, "x2 [mm]", "z2 [mm]",
               "second interaction: x2 vs z2 (n_int >= 2)")
        fig.suptitle(f"{tag}: interaction positions, fully contained events")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{tag}_positions.png", dpi=150)
        plt.close(fig)

        # ---- energies ----
        fig, axs = plt.subplots(2, 2, figsize=(10, 9))
        axs[0, 0].hist(c["e1_kev"], bins=100)
        axs[0, 0].set_xlabel("E1 [keV]")
        axs[0, 0].set_title("first-interaction energy", fontsize=10)
        axs[0, 1].hist(c["e2_kev"][multi], bins=100)
        axs[0, 1].set_xlabel("E2 [keV] (n_int >= 2)")
        axs[0, 1].set_title("second-interaction energy", fontsize=10)
        axs[1, 0].hist(c["er_kev"], bins=100)
        axs[1, 0].set_xlabel("Er = 511 - E1 - E2 [keV]")
        axs[1, 0].set_title("energy beyond the second interaction (C3+)", fontsize=10)
        axs[1, 0].set_yscale("log")
        hist2d(axs[1, 1], c["e1_kev"][multi], c["e2_kev"][multi],
               "E1 [keV]", "E2 [keV]", "E1 vs E2 (n_int >= 2)")
        for ax in axs.flat[:3]:
            ax.set_ylabel("events")
        fig.suptitle(f"{tag}: interaction energies, fully contained events")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{tag}_energies.png", dpi=150)
        plt.close(fig)

        # ---- topology + photoelectron spectrum ----
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        codes, counts = np.unique(c["int_type"], return_counts=True)
        axs[0].bar([type_label(k) for k in codes], counts)
        axs[0].set_ylabel("events")
        axs[0].set_title("interaction type (number of Compton scatters)", fontsize=10)
        axs[1].hist(c["npe"], bins=100)
        axs[1].set_xlabel("photoelectrons")
        axs[1].set_ylabel("events")
        axs[1].set_title("npe spectrum, contained events", fontsize=10)
        fig.suptitle(f"{tag}: event topology, fully contained events")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{tag}_topology.png", dpi=150)
        plt.close(fig)

        # ---- stats / metadata ----
        n_int_evts = int(np.count_nonzero(d["int_type"] >= 0))
        n_cont = int(np.count_nonzero(contained))
        stats = {
            "tag": str(tag),
            "crystal": str(d["crystal"]),
            "events_shot": d["n_shot"],
            "events_interacted": n_int_evts,
            "events_fully_contained": n_cont,
            "contained_over_shot": round(n_cont / d["n_shot"], 4),
            "contained_over_interacted": round(n_cont / n_int_evts, 4),
            "contained_kev_cut": CONTAINED_KEV,
            "int_type_contained": {type_label(k): int(n)
                                   for k, n in zip(codes, counts)},
            "npe_mean_contained": round(float(np.mean(c["npe"])), 1),
        }
        with open(OUTDIR / f"{tag}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        lines = [f"{k:26s} {v}" for k, v in stats.items()]
        (OUTDIR / f"{tag}_stats.txt").write_text("\n".join(lines) + "\n")
        print(f"{tag}: shot {d['n_shot']}, interacted {n_int_evts}, "
              f"contained {n_cont} ({100 * n_cont / d['n_shot']:.1f}%)")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]] or DEFAULTS)
