#!/usr/bin/env python3
"""Single-site vs multi-site event classifier (photo-like / compton-like).

Trains CryspNet with one sigmoid output on the energy-window sample, label =
multi-site truth (n_int >= 2), same selection, split and D4 augmentation as
the reconstruction runs. Evaluation joins the other runs' per-event
dataframes on the events.h5 row key:
  - ROC/AUC, compared with the two-site network's predicted separation
  - the operational curve: reconstruction tail of the ACCEPTED (photo-like)
    sample vs acceptance, using the phase-1 predictions
Writes test_events.h5 (row, label, int_type, score), metrics.json and figures
into results/cnn/<tag>/.

Usage: python3 analysis/cnn/train_classifier.py analysis/cnn/runs/cnn_csitl_cls.toml
"""
import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import (d4_expand, load_test_events, load_window,  # noqa: E402
                     save_test_events)
from model import CryspNet                                      # noqa: E402
from train import make_tensors                                  # noqa: E402


def auc_score(label, score):
    """Rank-based AUC (probability a multi-site event scores above a
    single-site one)."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label > 0.5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc(label, score, n=200):
    """(efficiency for multi, false-positive rate on single) vs threshold."""
    ts = np.quantile(score, np.linspace(0, 1, n))
    pos = label > 0.5
    eff = [(score[pos] > t).mean() for t in ts]
    fpr = [(score[~pos] > t).mean() for t in ts]
    return np.array(fpr), np.array(eff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    tag = Path(args.config).stem
    size = np.array(cfg["data"]["size_mm"], np.float32)
    epochs = args.epochs or cfg["train"]["epochs"]
    outdir = HERE.parent / "results" / "cnn" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(cfg["train"]["seed"])

    selcfg = cfg["selection"]
    d = load_window(HERE.parent.parent / cfg["data"]["events"], selcfg["fwhm"],
                    nsigma=selcfg.get("nsigma", 2.0),
                    seed=selcfg.get("smear_seed", 2026))
    lab = (d["n_int"] >= 2).astype(np.float32)
    n = len(lab)
    rng = np.random.default_rng(cfg["train"]["seed"])
    idx = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    m_tr, p_tr, _ = d4_expand(d["maps"][tr], d["npe"][tr], d["xyz1"][tr],
                              w_mm=size[0])
    l_tr = np.tile(lab[tr], 8)
    s_stats = (np.log(p_tr).mean(), np.log(p_tr).std())
    print(f"[{tag}] {n} events, multi fraction {lab.mean():.3f}; "
          f"train {len(l_tr)} (x8), device {device.type}", flush=True)

    def loader(m, p, x, y, shuffle):
        mt, st, _ = make_tensors(m, p, x, size, s_stats)
        ds = torch.utils.data.TensorDataset(mt, st,
                                            torch.from_numpy(y)[:, None].float())
        return torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch"],
                                           shuffle=shuffle)

    tr_loader = loader(m_tr, p_tr, np.tile(d["xyz1"][tr], (8, 1)), l_tr, True)
    va_loader = loader(d["maps"][va], d["npe"][va], d["xyz1"][va], lab[va], False)
    te_loader = loader(d["maps"][te], d["npe"][te], d["xyz1"][te], lab[te], False)

    model = CryspNet(n_out=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    def predict(dl):
        outs = []
        model.eval()
        with torch.no_grad():
            for m, s, _ in dl:
                outs.append(torch.sigmoid(model(m.to(device), s.to(device))).cpu())
        return torch.cat(outs).numpy().ravel()

    t0 = time.time()
    best_auc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for m, s, y in tr_loader:
            loss = loss_fn(model(m.to(device), s.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(m)
        sched.step()
        a = auc_score(lab[va], predict(va_loader))
        if a > best_auc:
            best_auc = a
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] epoch {ep:3d}  loss {tot / len(l_tr):.4f}  val AUC {a:.4f}",
              flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), outdir / "cnn_best.pt")

    # ---- test evaluation, joined with the reconstruction runs ----
    score = predict(te_loader)
    l_te, it_te = lab[te], d["itype"][te]
    save_test_events(outdir / "test_events.h5",
                     {"row": d["row"][te], "label": l_te,
                      "int_type": it_te, "score": score})
    crystal = "csitl" if "csitl" in tag else "bgo"
    win = load_test_events(f"cnn_{crystal}_win").set_index("row")
    two = load_test_events(f"cnn_{crystal}_2site").set_index("row")
    rows = d["row"][te]
    w = win.loc[rows]
    d3 = np.sqrt((w["cnn_x1"] - w["x1"])**2 + (w["cnn_y1"] - w["y1"])**2
                 + (w["cnn_z1"] - w["z1"])**2).to_numpy()
    t2 = two.loc[rows]
    sep = np.sqrt((t2["cnn_x2"] - t2["cnn_x1"])**2 + (t2["cnn_y2"] - t2["cnn_y1"])**2
                  + (t2["cnn_z2"] - t2["cnn_z1"])**2).to_numpy()

    auc_cls = auc_score(l_te, score)
    auc_sep = auc_score(l_te, sep)
    # acceptance sweep: keep events the classifier calls photo-like
    ts = np.quantile(score, np.linspace(0.02, 0.98, 97))
    sweep = [{"acceptance": round(float(acc.mean()), 4),
              "single_site_purity": round(float((l_te[acc] < 0.5).mean()), 4),
              "tail_frac_5mm_accepted": round(float((d3[acc] > 5).mean()), 4)}
             for t in ts for acc in [score < t] if acc.mean() > 0.01]
    metrics = {"tag": tag, "test_events": int(len(l_te)),
               "multi_fraction": round(float(l_te.mean()), 4),
               "auc_classifier": round(float(auc_cls), 4),
               "auc_separation_flag": round(float(auc_sep), 4),
               "train_seconds": round(time.time() - t0, 1),
               "acceptance_sweep": sweep[::8]}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for code, label_ in ((0, "photo"), (1, "C1"), (2, "C2plus")):
        m = (it_te >= 2) if code == 2 else (it_te == code)
        axs[0].hist(score[m], bins=60, range=(0, 1), histtype="step",
                    density=True, label=label_)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("classifier score P(multi-site)")
    axs[0].set_ylabel("density")
    axs[0].legend(fontsize=8)
    f1, e1 = roc(l_te, score)
    f2, e2 = roc(l_te, sep)
    axs[1].plot(f1, e1, label=f"classifier (AUC {auc_cls:.3f})")
    axs[1].plot(f2, e2, "--", label=f"two-site separation (AUC {auc_sep:.3f})")
    axs[1].plot([0, 1], [0, 1], "k:", lw=0.7)
    axs[1].set_xlabel("single-site events mistagged")
    axs[1].set_ylabel("multi-site events tagged")
    axs[1].legend(fontsize=8)
    accs = [s["acceptance"] for s in sweep]
    tails = [s["tail_frac_5mm_accepted"] for s in sweep]
    purs = [s["single_site_purity"] for s in sweep]
    axs[2].plot(accs, tails, label="tail (>5 mm) of accepted")
    axs[2].plot(accs, purs, "--", label="single-site purity")
    axs[2].axhline((d3 > 5).mean(), color="k", ls=":", lw=0.8,
                   label="no selection tail")
    axs[2].set_xlabel("acceptance (photo-like selection)")
    axs[2].set_ylabel("fraction")
    axs[2].legend(fontsize=8)
    fig.suptitle(f"{tag}: single-site vs multi-site classification")
    fig.tight_layout()
    fig.savefig(outdir / "classification.png", dpi=150)
    plt.close(fig)
    print(f"[{tag}] AUC {auc_cls:.4f} (separation flag {auc_sep:.4f}); "
          f"figures + metrics in {outdir}")


if __name__ == "__main__":
    main()
