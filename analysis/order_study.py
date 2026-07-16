#!/usr/bin/env python3
"""Ordering study for the two-site reconstruction (tracked figure source).

Question: in multi-site window events, is the FIRST interaction the one closer
to the entry face (z1 < z2)? If so, depth ordering gives the two-site targets
a well-posed convention, with a measured misidentification rate.

Measures, per crystal, on the energy-window sample (mask shared with training
via cnn/dataset.window_mask):
  - P(z1 < z2) overall and per interaction class (C1, C2+)
  - P(z1 < z2) vs E1 (the Compton-kinematics correlation: soft recoil =
    forward scatter) and vs the 3D site separation
  - the position cost of misordering: separation distribution split by order
  - with the phase-1 checkpoint: for multi-site TAIL events (>5 mm from site
    1), the fraction whose prediction lies closer to site 2

Writes <tag>_order.png + <tag>_order.json to results/order_study, and reads
the selection parameters from the phase-1 training configs.

Usage: python3 analysis/order_study.py
"""
import json
import sys
import tomllib
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cnn"))
from dataset import window_mask                                # noqa: E402
from model import CryspNet                                     # noqa: E402

OUTDIR = HERE / "results" / "order_study"
CONFIGS = ["cnn/runs/cnn_csitl_win.toml", "cnn/runs/cnn_bgo_win.toml"]


def binned_prob(x, ok, edges):
    """P(ok) and its binomial error in bins of x."""
    centres, prob, err = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        n = m.sum()
        if n < 20:
            continue
        p = ok[m].mean()
        centres.append(0.5 * (lo + hi))
        prob.append(p)
        err.append(np.sqrt(p * (1 - p) / n))
    return np.array(centres), np.array(prob), np.array(err)


def tail_site_check(tag, cfg, sel, d, size):
    """Phase-1 checkpoint on its own test split: where do tail events point?"""
    from train import make_tensors, to_mm
    n = int(sel.sum())
    rng = np.random.default_rng(cfg["train"]["seed"])
    idx = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    tr, te = idx[:n_tr], idx[n_tr + n_va:]
    s_stats = (np.log(d["npe"][tr]).mean(), np.log(d["npe"][tr]).std())
    m, s, _ = make_tensors(d["maps"][te], d["npe"][te], d["xyz1"][te],
                           size, s_stats)
    model = CryspNet()
    model.load_state_dict(torch.load(HERE / "results" / "cnn" / tag /
                                     "cnn_best.pt", map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(m), 4096):
            preds.append(model(m[k:k + 4096], s[k:k + 4096]))
    pred = to_mm(torch.cat(preds).numpy(), size)
    d1 = np.linalg.norm(pred - d["xyz1"][te], axis=1)
    d2 = np.linalg.norm(pred - d["xyz2"][te], axis=1)
    multi = d["n_int"][te] >= 2
    tail = multi & (d1 > 5.0)
    return {"test_multi": int(multi.sum()), "tail_multi": int(tail.sum()),
            "tail_closer_to_site2": round(float((d2[tail] < d1[tail]).mean()), 4)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for cfgpath in CONFIGS:
        cfg = tomllib.loads((HERE / cfgpath).read_text())
        tag = Path(cfgpath).stem
        size = np.array(cfg["data"]["size_mm"], np.float32)
        path = HERE.parent / cfg["data"]["events"]
        selcfg = cfg["selection"]
        sel = window_mask(path, selcfg["fwhm"], nsigma=selcfg["nsigma"],
                          seed=selcfg["smear_seed"])
        with h5py.File(path, "r") as f:
            d = {k: f[v][()][sel] for k, v in
                 (("xyz1", "xyz1_mm"), ("xyz2", "xyz2_mm"), ("e1", "e1_kev"),
                  ("n_int", "n_int"), ("itype", "int_type"),
                  ("npe", "npe"), ("maps", "maps"))}
        d["maps"] = d["maps"].astype(np.float32)
        d["npe"] = d["npe"].astype(np.float32)

        multi = d["n_int"] >= 2
        z1, z2 = d["xyz1"][multi, 2], d["xyz2"][multi, 2]
        e1 = d["e1"][multi]
        it = d["itype"][multi]
        sep = np.linalg.norm(d["xyz2"][multi] - d["xyz1"][multi], axis=1)
        ordered = z1 < z2

        stats = {
            "tag": tag, "multi_site_events": int(multi.sum()),
            "P_z1_lt_z2": round(float(ordered.mean()), 4),
            "P_C1": round(float(ordered[it == 1].mean()), 4),
            "P_C2plus": round(float(ordered[it >= 2].mean()), 4),
            "median_sep_ordered_mm": round(float(np.median(sep[ordered])), 2),
            "median_sep_misordered_mm": round(float(np.median(sep[~ordered])), 2),
            "misordered_frac_sep_gt_5mm":
                round(float((sep[~ordered] > 5.0).mean()), 4),
        }
        stats["phase1_tail"] = tail_site_check(tag, cfg, sel, d, size)

        fig, axs = plt.subplots(2, 2, figsize=(11, 9))
        axs[0, 0].hist(z2 - z1, bins=100)
        axs[0, 0].set_xlabel("z2 - z1 [mm]")
        axs[0, 0].set_ylabel("events")
        axs[0, 0].set_title(f"P(z1 < z2) = {stats['P_z1_lt_z2']:.3f}",
                            fontsize=10)
        c, p, e = binned_prob(e1, ordered, np.arange(0, 345, 15))
        axs[0, 1].errorbar(c, p, yerr=e, fmt="o", ms=3)
        axs[0, 1].set_xlabel("E1 [keV]")
        axs[0, 1].set_ylabel("P(z1 < z2)")
        axs[0, 1].set_title("ordering vs recoil energy (Compton kinematics)",
                            fontsize=10)
        c, p, e = binned_prob(sep, ordered, np.linspace(0, 40, 21))
        axs[1, 0].errorbar(c, p, yerr=e, fmt="o", ms=3)
        axs[1, 0].set_xlabel("3D site separation [mm]")
        axs[1, 0].set_ylabel("P(z1 < z2)")
        axs[1, 1].hist(sep[ordered], bins=80, range=(0, 40), histtype="step",
                       label=f"ordered (med {stats['median_sep_ordered_mm']} mm)")
        axs[1, 1].hist(sep[~ordered], bins=80, range=(0, 40), histtype="step",
                       label=f"misordered (med {stats['median_sep_misordered_mm']} mm)")
        axs[1, 1].set_xlabel("3D site separation [mm]")
        axs[1, 1].set_ylabel("events")
        axs[1, 1].legend(fontsize=8)
        fig.suptitle(f"{tag}: is the first interaction the shallower one? "
                     f"(multi-site window events)")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{tag}_order.png", dpi=150)
        plt.close(fig)

        (OUTDIR / f"{tag}_order.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
