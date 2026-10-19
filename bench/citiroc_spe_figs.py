#!/usr/bin/env python3
"""citiroc_spe_figs.py: figures of bench/3_citiroc_spe_calib.tex (SPE calibration of the Citiroc).

    python3 bench/citiroc_spe_figs.py

Uses the single-avalanche model and the toy of hold_scan_figs.py (same directory): track &
hold at the maximum of the 25 ns slow shaper, dark pulses with branching crosstalk, noise.
Overvoltage dependence from data/sipm_s14160.toml as used by the full simulation: gain
0.93e6 per volt, crosstalk 0.088 / 0.145 / 0.195 and dark rate 4.3 / 4.35 / 4.4 MHz at
3 / 4 / 5 V. The breakdown voltage of the example is 38.0 V.
"""
from math import lgamma

import numpy as np
import matplotlib.pyplot as plt

import hold_scan_figs as h

OUT = h.OUT
VOV = {3.0: (0.088, 4.3), 4.0: (0.145, 4.35), 5.0: (0.195, 4.4)}
VBR = 38.0


def spectra():
    d25 = h.model_delay(25.0)
    bins = np.linspace(-0.5, 3.5, 201)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    pos = {}
    for (v, (xt, rate)), c in zip(VOV.items(), ("C0", "C1", "C2")):
        h.XT, h.DARK_MHZ = xt, rate
        a = h.stored(25.0, [d25], "th", n_ev=20000)[:, 0] * v / 5.0
        ax[0].hist(a, bins, histtype="step", lw=1.3, color=c,
                   label=f"{v:.0f} V: crosstalk {xt:.0%}, DP {rate:.1f} MHz")
        pos[v] = v / 5.0
    ax[0].set_xlabel("valor guardado (pe de 5 V)"); ax[0].set_ylabel("eventos")
    ax[0].set_title("Espectros SPE, alta ganancia a 25 ns (toy)"); ax[0].legend(fontsize=8)
    vb = np.array([VBR + v for v in VOV]); g = np.array([pos[v] for v in VOV])
    k, b = np.polyfit(vb, g, 1)
    xs = np.linspace(VBR - 0.5, VBR + 5.5, 50)
    ax[1].plot(vb, g, "ko"); ax[1].plot(xs, k * xs + b, "k--", lw=1)
    ax[1].axhline(0, c="gray", lw=0.5); ax[1].axvline(-b / k, c="C3", ls=":")
    ax[1].text(-b / k + 0.1, 0.9, f"BV = {-b / k:.1f} V", c="C3")
    ax[1].set_xlabel("tensión de bias (V)"); ax[1].set_ylabel("posición del pico de 1 pe (pe de 5 V)")
    ax[1].set_title("Ganancia frente a la tensión (ejemplo)")
    fig.tight_layout(); fig.savefig(OUT / "spe_tension.png", dpi=150); plt.close(fig)


def borel_tail(p, nmax=40):
    """P(n >= k) for the number of fired cells per dark pulse, branching crosstalk."""
    lam = -np.log(1 - p); n = np.arange(1, nmax + 1)
    pn = np.exp(-lam * n + (n - 1) * np.log(lam * n) - np.array([lgamma(x + 1) for x in n]))
    return np.cumsum(pn[::-1])[::-1]


def geometric_tail(p, nmax=40):
    return p ** np.arange(0, nmax)


def staircase_xt():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    k = np.arange(1, 11)
    for v, c in ((3.0, "C0"), (5.0, "C2")):
        p, rate = VOV[v]
        bt, gt = borel_tail(p)[:10], geometric_tail(p)[:10]
        ax[0].semilogy(k, rate * 1e3 * bt, "o-", c=c, label=f"{v:.0f} V, ramificación (modelo)")
        ax[0].semilogy(k, rate * 1e3 * gt, "s--", c=c, mfc="none", label=f"{v:.0f} V, una celda como mucho")
        ax[1].plot(k[1:], bt[1:] / bt[:-1], "o-", c=c, label=f"{v:.0f} V, ramificación (modelo)")
        ax[1].plot(k[1:], gt[1:] / gt[:-1], "s--", c=c, mfc="none", label=f"{v:.0f} V, una celda como mucho")
    ax[0].set_xlabel("meseta (pe)"); ax[0].set_ylabel("tasa de trigger (kHz)")
    ax[0].set_title("Altura de las mesetas de la staircase"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("meseta k (pe)"); ax[1].set_ylabel("tasa(k) / tasa(k-1)")
    ax[1].set_title("Cociente entre mesetas consecutivas"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "staircase_crosstalk.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    spectra()
    staircase_xt()
    for v in (3.0, 5.0):
        bt = borel_tail(VOV[v][0])[:6]
        print(v, "V plateau ratios (branching):", np.round(bt[1:] / bt[:-1], 3),
              "rate at 10 pe (Hz):", round(VOV[v][1] * 1e6 * borel_tail(VOV[v][0])[9]))
    print("figures in", OUT)
