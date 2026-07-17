#!/usr/bin/env python3
"""Two-Gaussian residual fits with and without the reconstructability
selection (tracked figure source).

For each crystal, joins the phase-1 reconstruction dataframe with the
reconstructability-classifier scores (row key), applies the photo-like
selection at several global acceptances, and fits each coordinate's residuals
to core + tail Gaussians. Amplitudes are reported as EVENT COUNTS in each
component (peak heights also stored in the json).

Writes selection_fits.json, selection_fits.txt and selection_fits.png to
results/cnn/selection_fits/.

Usage: python3 analysis/cnn/selection_fits.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import load_test_events                           # noqa: E402
from fit_residuals import fit_coord, two_gauss                 # noqa: E402

OUTDIR = HERE.parent / "results" / "cnn" / "selection_fits"
ACCEPTANCES = [1.0, 0.8, 0.6, 0.5]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    allres = {}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    lines = []
    for ic, crystal in enumerate(("csitl", "bgo")):
        win = load_test_events(f"cnn_{crystal}_win").set_index("row")
        cls = load_test_events(f"cnn_{crystal}_rcls").set_index("row")
        score = cls.loc[win.index, "score"].to_numpy()
        allres[crystal] = {}
        for acc in ACCEPTANCES:
            m = score <= np.quantile(score, acc) if acc < 1.0 \
                else np.ones(len(score), bool)
            sel = win[m]
            name = "none" if acc == 1.0 else f"{int(100 * acc)}%"
            allres[crystal][name] = {"events": int(m.sum())}
            for k, lab in enumerate("xyz"):
                res = (sel[f"cnn_{lab}1"] - sel[f"{lab}1"]).to_numpy()
                (amp, mu, sc, ft, st), centres, counts, w = fit_coord(res)
                n1, n2 = amp * (1 - ft) / w, amp * ft / w
                allres[crystal][name][lab] = {
                    "A1_events": round(float(n1)), "sigma1_mm": round(float(sc), 3),
                    "A2_events": round(float(n2)), "sigma2_mm": round(float(st), 3),
                    "A2_over_A1": round(float(n2 / n1), 4),
                    "peak1": round(float(amp * (1 - ft) / (sc * np.sqrt(2 * np.pi))), 1),
                    "peak2": round(float(amp * ft / (st * np.sqrt(2 * np.pi))), 1),
                    "mu_mm": round(float(mu), 3)}
                # apples-to-apples: component shapes FROZEN to the no-selection
                # fit; only the two amplitudes float. Avoids the re-partitioning
                # degeneracy of free two-Gaussian fits on tail-depleted samples.
                if acc == 1.0:
                    allres[crystal][name][lab]["_shape"] = (mu, sc, st)
                else:
                    mu0, sc0, st0 = allres[crystal]["none"][lab]["_shape"]
                    model = lambda r, a, f: two_gauss(r, a, mu0, sc0, f, st0)
                    (a_f, f_f), _ = curve_fit(
                        model, centres, counts, p0=[m.sum() * w, 0.2],
                        bounds=([0, 0], [np.inf, 1]),
                        sigma=np.sqrt(np.maximum(counts, 1)))
                    allres[crystal][name][lab]["fixed_shape"] = {
                        "A1_events": round(float(a_f * (1 - f_f) / w)),
                        "A2_events": round(float(a_f * f_f / w)),
                        "A2_over_A1": round(float(f_f / (1 - f_f)), 4)}
                ax = axs[ic, k]
                ax.semilogy(centres, np.maximum(counts, 0.5), drawstyle="steps-mid",
                            lw=1, label=f"acc {name}" if k == 0 else None)
                ax.semilogy(centres, two_gauss(centres, amp, mu, sc, ft, st),
                            "--", lw=0.8, color=ax.lines[-1].get_color())
                if k == 0:
                    lines.append((crystal, name))
        for k, lab in enumerate("xyz"):
            axs[ic, k].set_xlabel(f"{lab} residual [mm]")
            axs[ic, k].set_title(f"{'CsI(Tl)' if crystal == 'csitl' else 'BGO'}: "
                                 f"{lab}", fontsize=10)
            axs[ic, k].set_ylim(bottom=0.5)
        axs[ic, 0].set_ylabel("test events")
        axs[ic, 0].legend(fontsize=8)
    fig.suptitle("first-site residuals and two-Gaussian fits vs "
                 "reconstructability-selection acceptance")
    fig.tight_layout()
    fig.savefig(OUTDIR / "selection_fits.png", dpi=150)
    plt.close(fig)

    for sels in allres.values():           # drop the internal shape cache
        for v in sels.values():
            for lab in "xyz":
                v[lab].pop("_shape", None)
    (OUTDIR / "selection_fits.json").write_text(json.dumps(allres, indent=2))
    rows = []
    for crystal, sels in allres.items():
        rows.append(f"\n{'CsI(Tl)' if crystal == 'csitl' else 'BGO'} "
                    "(fixed-shape columns: sigmas frozen to the no-selection fit)")
        rows.append(f"{'sel':>6s} {'coord':>5s} {'A1':>8s} {'sigma1':>7s} "
                    f"{'A2':>8s} {'sigma2':>7s} {'A2/A1':>7s}"
                    f" | {'A1fix':>8s} {'A2fix':>8s} {'A2/A1fix':>9s}")
        for name, v in sels.items():
            for lab in "xyz":
                f = v[lab]
                line = (f"{name:>6s} {lab:>5s} {f['A1_events']:8d} "
                        f"{f['sigma1_mm']:7.2f} {f['A2_events']:8d} "
                        f"{f['sigma2_mm']:7.2f} {f['A2_over_A1']:7.3f}")
                if "fixed_shape" in f:
                    x = f["fixed_shape"]
                    line += (f" | {x['A1_events']:8d} {x['A2_events']:8d} "
                             f"{x['A2_over_A1']:9.3f}")
                rows.append(line)
    table = "\n".join(rows)
    (OUTDIR / "selection_fits.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
