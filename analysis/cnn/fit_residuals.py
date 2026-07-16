#!/usr/bin/env python3
"""Two-Gaussian description of the reconstruction residuals (tracked source).

Rebuilds the test-split predictions from a saved checkpoint (selection and
split are seeded, so the reconstruction is exact), then fits the residual
distribution of each coordinate to
    N [ (1 - f_t) G(mu, sigma_core) + f_t G(mu, sigma_tail) ],
the standard descriptive decomposition of a core-plus-tail response: a narrow
Gaussian for the well-reconstructed population and a broad one for the
multi-site tail. Writes residual_fit.png + residual_fit.json into the run's
results directory.

Usage: python3 analysis/cnn/fit_residuals.py [config.toml ...]
       (default: cnn_csitl_win, cnn_bgo_win, cnn_bgo_2site)
"""
import json
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import load_photo, load_window                    # noqa: E402
from model import CryspNet                                     # noqa: E402
from train import make_tensors, to_mm                          # noqa: E402

DEFAULTS = ["cnn_csitl_win", "cnn_bgo_win", "cnn_bgo_2site"]


def rebuild_test_predictions(cfgpath):
    """Test-split truth (first interaction) and CNN first-site prediction."""
    cfg = tomllib.loads(Path(cfgpath).read_text())
    size = np.array(cfg["data"]["size_mm"], np.float32)
    path = HERE.parent.parent / cfg["data"]["events"]
    selcfg = cfg.get("selection", {"mode": "photo"})
    if selcfg["mode"] == "window":
        d = load_window(path, selcfg["fwhm"], nsigma=selcfg.get("nsigma", 2.0),
                        seed=selcfg.get("smear_seed", 2026))
    else:
        d = load_photo(path)
    n = len(d["npe"])
    rng = np.random.default_rng(cfg["train"]["seed"])
    idx = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    tr, te = idx[:n_tr], idx[n_tr + n_va:]
    s_stats = (np.log(d["npe"][tr]).mean(), np.log(d["npe"][tr]).std())
    m, s, _ = make_tensors(d["maps"][te], d["npe"][te], d["xyz1"][te],
                           size, s_stats)
    n_out = 6 if cfg["train"].get("targets", "first") == "two_site" else 3
    tag = Path(cfgpath).stem
    model = CryspNet(n_out=n_out)
    model.load_state_dict(torch.load(HERE.parent / "results" / "cnn" / tag /
                                     "cnn_best.pt", map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(m), 4096):
            preds.append(model(m[k:k + 4096], s[k:k + 4096]))
    pred = to_mm(torch.cat(preds).numpy(), np.tile(size, n_out // 3))
    return tag, d["xyz1"][te], pred[:, :3]


def two_gauss(r, amp, mu, s_core, f_tail, s_tail):
    g = lambda s: np.exp(-0.5 * ((r - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return amp * ((1 - f_tail) * g(s_core) + f_tail * g(s_tail))


def fit_coord(res, rng_mm=15.0, nbins=150):
    counts, edges = np.histogram(res, bins=nbins, range=(-rng_mm, rng_mm))
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    p0 = [len(res) * width, 0.0, 1.0, 0.2, 4.0]
    bounds = ([0, -3, 0.1, 0.0, 0.5], [np.inf, 3, 5.0, 1.0, rng_mm])
    popt, _ = curve_fit(two_gauss, centres, counts, p0=p0, bounds=bounds,
                        sigma=np.sqrt(np.maximum(counts, 1)))
    # enforce the core = the narrower component
    if popt[2] > popt[4]:
        popt = [popt[0], popt[1], popt[4], 1 - popt[3], popt[2]]
    return popt, centres, counts, width


def main(tags):
    for tag in tags:
        cfgpath = HERE / "runs" / f"{tag}.toml"
        tag, truth, pred = rebuild_test_predictions(cfgpath)
        outdir = HERE.parent / "results" / "cnn" / tag
        fitres = {}
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        for k, (lab, ax) in enumerate(zip("xyz", axs)):
            res = pred[:, k] - truth[:, k]
            (amp, mu, sc, ft, st), centres, counts, width = fit_coord(res)
            fitres[lab] = {"sigma_core_mm": round(float(sc), 3),
                           "fwhm_core_mm": round(float(2.355 * sc), 3),
                           "sigma_tail_mm": round(float(st), 3),
                           "core_fraction": round(float(1 - ft), 4),
                           "mu_mm": round(float(mu), 3)}
            ax.semilogy(centres, np.maximum(counts, 0.5), drawstyle="steps-mid",
                        lw=1, label="residuals")
            ax.semilogy(centres, two_gauss(centres, amp, mu, sc, ft, st),
                        "r-", lw=1.2, label="two-Gaussian fit")
            ax.semilogy(centres, amp * (1 - ft) * np.exp(
                -0.5 * ((centres - mu) / sc) ** 2) / (sc * np.sqrt(2 * np.pi)),
                "r--", lw=0.8,
                label=f"core: sigma {sc:.2f} mm ({100 * (1 - ft):.0f}%)")
            ax.semilogy(centres, amp * ft * np.exp(
                -0.5 * ((centres - mu) / st) ** 2) / (st * np.sqrt(2 * np.pi)),
                "r:", lw=0.8, label=f"tail: sigma {st:.2f} mm")
            ax.set_xlabel(f"{lab} residual [mm]")
            ax.set_ylim(bottom=0.5)
            ax.legend(fontsize=7)
        axs[0].set_ylabel("test events")
        fig.suptitle(f"{tag}: two-Gaussian decomposition of the residuals")
        fig.tight_layout()
        fig.savefig(outdir / "residual_fit.png", dpi=150)
        plt.close(fig)
        (outdir / "residual_fit.json").write_text(json.dumps(fitres, indent=2))
        print(f"{tag}:")
        for lab, f in fitres.items():
            print(f"  {lab}: core sigma {f['sigma_core_mm']:.2f} mm "
                  f"(FWHM {f['fwhm_core_mm']:.2f}, {100 * f['core_fraction']:.0f}% "
                  f"of events), tail sigma {f['sigma_tail_mm']:.2f} mm")


if __name__ == "__main__":
    main([Path(a).stem for a in sys.argv[1:]] or DEFAULTS)
