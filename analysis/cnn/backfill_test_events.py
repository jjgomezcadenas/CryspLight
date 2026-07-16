#!/usr/bin/env python3
"""Backfill test_events.h5 for runs trained before train.py wrote it.

Rebuilds the test-split predictions from the saved CNN checkpoint (selection
and split are seeded, so the reconstruction is exact) and refits the Anger
baseline on the training split where the run used it. MLP checkpoints were
never saved, so mlp columns exist only for runs trained after the dataframe
layer landed.

Usage: python3 analysis/cnn/backfill_test_events.py [tag ...]
       (default: every run directory with a cnn_best.pt and no test_events.h5)
"""
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import (load_photo, load_window, save_test_events,   # noqa: E402
                     two_site_targets)
from model import CryspNet, anger_moments, fit_anger_z            # noqa: E402
from train import delta_to_slots, make_tensors, to_mm             # noqa: E402

RESULTS = HERE.parent / "results" / "cnn"


def backfill(tag):
    cfg = tomllib.loads((HERE / "runs" / f"{tag}.toml").read_text())
    size = np.array(cfg["data"]["size_mm"], np.float32)
    selcfg = cfg.get("selection", {"mode": "photo"})
    path = HERE.parent.parent / cfg["data"]["events"]
    if selcfg["mode"] == "window":
        d = load_window(path, selcfg["fwhm"], nsigma=selcfg.get("nsigma", 2.0),
                        seed=selcfg.get("smear_seed", 2026))
    else:
        d = load_photo(path)
    two_site = cfg["train"].get("targets", "first") == "two_site"
    n = len(d["npe"])
    rng = np.random.default_rng(cfg["train"]["seed"])
    idx = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    tr, te = idx[:n_tr], idx[n_tr + n_va:]
    s_stats = (np.log(d["npe"][tr]).mean(), np.log(d["npe"][tr]).std())

    n_out = 6 if two_site else 3
    model = CryspNet(n_out=n_out)
    model.load_state_dict(torch.load(RESULTS / tag / "cnn_best.pt",
                                     map_location="cpu"))
    model.eval()
    m, s, _ = make_tensors(d["maps"][te], d["npe"][te], d["xyz1"][te],
                           size, s_stats)
    preds = []
    with torch.no_grad():
        for k in range(0, len(m), 4096):
            preds.append(model(m[k:k + 4096], s[k:k + 4096]))
    raw = torch.cat(preds).numpy()
    if two_site and cfg["train"].get("delta_loss", False):
        raw = delta_to_slots(raw)
    pred = to_mm(raw, np.tile(size, n_out // 3))

    truth = d["xyz1"][te]
    cols = {"row": d["row"][te],
            "x1": truth[:, 0], "y1": truth[:, 1], "z1": truth[:, 2],
            "e1": d["e1"][te], "edep": d["edep"][te],
            "int_type": d["itype"][te], "n_int": d["n_int"][te],
            "npe": d["npe"][te]}
    if two_site:
        targ = two_site_targets(d)[te]
        for j, c in enumerate(("shallow_x", "shallow_y", "shallow_z",
                               "deep_x", "deep_y", "deep_z")):
            cols[c] = targ[:, j]
    for j in range(pred.shape[1]):
        cols[f"cnn_{'xyz'[j % 3]}{j // 3 + 1}"] = pred[:, j]
    if not two_site:   # runs that carried the classical baseline
        xc, yc, rr = anger_moments(d["maps"][tr])
        zfit = fit_anger_z(rr, d["xyz1"][tr][:, 2])
        xc_t, yc_t, rr_t = anger_moments(d["maps"][te])
        cols["anger_x1"], cols["anger_y1"] = xc_t, yc_t
        cols["anger_z1"] = zfit(rr_t)
    save_test_events(RESULTS / tag / "test_events.h5", cols)
    print(f"{tag}: wrote test_events.h5 ({len(te)} events, "
          f"{len(cols)} columns)")


if __name__ == "__main__":
    tags = sys.argv[1:] or sorted(
        p.parent.name for p in RESULTS.glob("*/cnn_best.pt")
        if not (p.parent / "test_events.h5").exists())
    for t in tags:
        backfill(t)
