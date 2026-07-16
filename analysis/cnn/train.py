#!/usr/bin/env python3
"""Train CryspNet + baselines to reconstruct (x,y,z) of photoelectric events.

Usage: python3 analysis/cnn/train.py analysis/cnn/runs/cnn_csitl.toml [--epochs N]

Reads a TOML config (tag = filename), trains the CNN and the MLP baseline on
contained photoelectric events (D4-augmented train split), fits the classical
Anger baseline on the same split, and writes to results/cnn/<tag>/:
metrics.json, residual/resolution plots, and the best CNN checkpoint.
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
from dataset import d4_expand, load_photo, load_window        # noqa: E402
from model import MLP, CryspNet, anger_moments, fit_anger_z   # noqa: E402
from plot_history import plot_history                         # noqa: E402


def to_norm(xyz, size):
    return 2.0 * xyz / size - 1.0


def to_mm(t, size):
    return (t + 1.0) * size / 2.0


def make_tensors(maps, npe, xyz, size, s_stats):
    m = torch.from_numpy(maps / npe[:, None, None]).unsqueeze(1)
    s = torch.from_numpy((np.log(npe) - s_stats[0]) / s_stats[1])[:, None].float()
    y = torch.from_numpy(to_norm(xyz, size))
    return m, s, y


def run_model(model, loader, device, opt=None, loss_fn=None):
    total, n = 0.0, 0
    preds = []
    for m, s, y in loader:
        m, s, y = m.to(device), s.to(device), y.to(device)
        out = model(m, s)
        if opt is not None:
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(m)
            n += len(m)
        else:
            preds.append(out.cpu())
    return total / max(n, 1) if opt is not None else torch.cat(preds).numpy()


def train_net(model, tr_loader, va_loader, y_va, size, device, epochs, lr, wd, tag):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    best_rmse, best_state, history = np.inf, None, []
    for ep in range(epochs):
        model.train()
        tr_loss = run_model(model, tr_loader, device, opt, loss_fn)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = run_model(model, va_loader, device)
        res = to_mm(pv, size) - to_mm(y_va, size)
        rmse = np.sqrt((res**2).mean(axis=0))
        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_rmse_mm": [round(float(v), 3) for v in rmse]})
        if rmse.mean() < best_rmse:
            best_rmse = rmse.mean()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] epoch {ep:3d}  loss {tr_loss:.5f}  "
              f"val RMSE x/y/z = {rmse[0]:.2f}/{rmse[1]:.2f}/{rmse[2]:.2f} mm",
              flush=True)
    model.load_state_dict(best_state)
    return model, history


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

    # ---- data: truth (photo) or energy-window selection, split, augment train ----
    root = HERE.parent.parent
    selcfg = cfg.get("selection", {"mode": "photo"})
    if selcfg["mode"] == "window":
        maps, npe, xyz, itype = load_window(root / cfg["data"]["events"],
                                            selcfg["fwhm"],
                                            nsigma=selcfg.get("nsigma", 2.0),
                                            seed=selcfg.get("smear_seed", 2026))
    else:
        maps, npe, xyz, itype = load_photo(root / cfg["data"]["events"])
    n = len(npe)
    rng = np.random.default_rng(cfg["train"]["seed"])
    idx = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    m_tr, p_tr, x_tr = d4_expand(maps[tr], npe[tr], xyz[tr], w_mm=size[0])
    s_stats = (np.log(p_tr).mean(), np.log(p_tr).std())
    comp = {f"C{k}" if k else "photo": int(c)
            for k, c in zip(*np.unique(itype, return_counts=True))}
    print(f"[{tag}] {n} events ({selcfg['mode']} selection, {comp}): "
          f"train {n_tr} (x8 D4 = {len(p_tr)}), val {n_va}, test {len(te)}; "
          f"device {device.type}", flush=True)

    def loader(m, p, x, shuffle):
        ds = torch.utils.data.TensorDataset(*make_tensors(m, p, x, size, s_stats))
        return torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch"],
                                           shuffle=shuffle)

    tr_loader = loader(m_tr, p_tr, x_tr, True)
    va_loader = loader(maps[va], npe[va], xyz[va], False)
    te_loader = loader(maps[te], npe[te], xyz[te], False)
    y_va = to_norm(xyz[va], size)

    # ---- train CNN and MLP, fit Anger on the same training events ----
    t0 = time.time()
    results = {}
    nets = [("cnn", CryspNet())]
    if cfg["train"].get("train_mlp", True):   # comparison made; off for iteration work
        nets.append(("mlp", MLP()))
    for name, net in nets:
        model, history = train_net(net, tr_loader, va_loader, y_va, size, device,
                                   epochs, cfg["train"]["lr"],
                                   cfg["train"]["weight_decay"], f"{tag}:{name}")
        model.eval()
        with torch.no_grad():
            pt = run_model(model, te_loader, device)
        results[name] = {"pred_mm": to_mm(pt, size), "history": history}
        if name == "cnn":
            torch.save(model.state_dict(), outdir / "cnn_best.pt")

    xc, yc, rr = anger_moments(maps[tr])
    zfit = fit_anger_z(rr, xyz[tr][:, 2])
    xc_t, yc_t, rr_t = anger_moments(maps[te])
    results["anger"] = {"pred_mm": np.stack([xc_t, yc_t, zfit(rr_t)], axis=1)}

    # ---- evaluate on the test split ----
    truth = xyz[te]
    it_te = itype[te]
    classes = [("photo", it_te == 0), ("C1", it_te == 1), ("C2plus", it_te >= 2)]
    metrics = {"tag": tag, "selection": selcfg,
               "events": {"selected": n, "test": len(te),
                          "composition": comp},
               "epochs": epochs, "train_seconds": round(time.time() - t0, 1)}

    def summarize(res):
        d3 = np.sqrt((res**2).sum(axis=1))
        return {
            "rmse_mm": [round(float(v), 3) for v in np.sqrt((res**2).mean(axis=0))],
            "p68_mm": [round(float(np.percentile(np.abs(res[:, k]), 68)), 3)
                       for k in range(3)],
            "tail_frac_5mm": round(float((d3 > 5.0).mean()), 4)}

    for name, r in results.items():
        res = r["pred_mm"] - truth
        metrics[name] = summarize(res)
        metrics[name]["by_type"] = {lab: summarize(res[m]) for lab, m in classes
                                    if m.any()}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(outdir / "history.json", "w") as f:
        json.dump({name: results[name]["history"] for name, _ in nets}, f, indent=2)
    plot_history(outdir)

    # ---- plots ----
    labels = ("x", "y", "z")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for k, ax in enumerate(axs):
        for name, color in (("cnn", "C0"), ("mlp", "C1"), ("anger", "C2")):
            if name not in results:
                continue
            res = results[name]["pred_mm"][:, k] - truth[:, k]
            ax.hist(res, bins=100, range=(-15, 15), histtype="step", color=color,
                    label=f"{name}  RMSE {metrics[name]['rmse_mm'][k]:.2f} mm")
        ax.set_xlabel(f"{labels[k]} residual [mm]")
        ax.legend(fontsize=8)
    axs[0].set_ylabel("test events")
    fig.suptitle(f"{tag}: reconstruction residuals, photoelectric test events")
    fig.tight_layout()
    fig.savefig(outdir / "residuals.png", dpi=150)
    plt.close(fig)

    zb = np.linspace(0, size[2], 9)
    zc = 0.5 * (zb[1:] + zb[:-1])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for k, ax in enumerate(axs):
        for name, color in (("cnn", "C0"), ("mlp", "C1"), ("anger", "C2")):
            if name not in results:
                continue
            res = results[name]["pred_mm"][:, k] - truth[:, k]
            rms = [np.sqrt(np.mean(res[(truth[:, 2] >= zb[b])
                                       & (truth[:, 2] < zb[b + 1])]**2))
                   for b in range(len(zc))]
            ax.plot(zc, rms, "o-", color=color, label=name)
        ax.set_xlabel("true z1 [mm]")
        ax.set_ylabel(f"{labels[k]} RMSE [mm]")
        ax.legend(fontsize=8)
    fig.suptitle(f"{tag}: resolution vs depth (SiPMs at z = {size[2]:.1f} mm)")
    fig.tight_layout()
    fig.savefig(outdir / "rmse_vs_z.png", dpi=150)
    plt.close(fig)

    if (it_te > 0).any():   # mixture sample: core-plus-tail per interaction type
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        d3 = np.sqrt(((results["cnn"]["pred_mm"] - truth)**2).sum(axis=1))
        for lab, m in classes:
            if m.any():
                ax.hist(d3[m], bins=100, range=(0, 30), histtype="step",
                        label=f"{lab} ({m.sum()} ev, "
                              f"{100 * (d3[m] > 5).mean():.1f}% beyond 5 mm)")
        ax.set_yscale("log")
        ax.set_xlabel("CNN 3D distance to first interaction [mm]")
        ax.set_ylabel("test events")
        ax.set_title(f"{tag}: core and tail by interaction type")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "distance_by_type.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    h = ax.hist2d(truth[:, 2], results["cnn"]["pred_mm"][:, 2], bins=60,
                  cmap="viridis", cmin=1)
    ax.plot([0, size[2]], [0, size[2]], "r--", lw=1)
    ax.set_xlabel("true z1 [mm]")
    ax.set_ylabel("CNN z [mm]")
    ax.set_title(f"{tag}: CNN depth reconstruction")
    fig.colorbar(h[3], ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "z_true_vs_pred.png", dpi=150)
    plt.close(fig)

    for name in results:
        print(f"[{tag}] {name:5s} test RMSE x/y/z = "
              + "/".join(f"{v:.2f}" for v in metrics[name]["rmse_mm"]) + " mm")


if __name__ == "__main__":
    main()
