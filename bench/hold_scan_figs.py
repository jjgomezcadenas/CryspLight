#!/usr/bin/env python3
"""hold_scan_figs.py: figures of the Citiroc bench notes (v1/2_citiroc_hold_track.tex, hold delay
scan; 1_citiroc_noise_v2.tex, pedestal and noise in peak stretch: ventana.png, mecanismo_ps.png;
2_citiroc_hold_scan_v2.tex, hold delay scan in peak stretch: scan_ps.png).

    python3 bench/hold_scan_figs.py

Reproduces the single-avalanche model of the CryspLight front end (src/frontend/shaping.jl):
SiPM pulse exp(-t/100) - exp(-t/50) ns (Ct = 2000 pF on 50 ohm, recovery 50 ns), Citiroc
voltage preamplifier followed by CR-RC^2 (one CR and two RC with theta = tp/2), fast shaper
15 ns with a threshold of 0.5 pe, same discrete filters on a 1 ns grid. The toy spectra add
dark pulses (4.4 MHz, branching crosstalk 0.195), Gaussian noise and a 1 ns jitter of the
trigger time; they illustrate the concepts and are not the full simulation.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figs"
DT = 1.0
N = 1200
TF, T2 = 100.0, 50.0           # SiPM pulse: fall and second exponential (ns)
DARK_MHZ = 4.4                 # dark rate per channel at 5 V
XT = 0.195                     # crosstalk probability at 5 V
THR = 0.5                      # discriminator threshold (pe)
RNG = np.random.default_rng(1)


def rc_lowpass(y, tau):
    a = np.exp(-DT / tau); out = np.empty_like(y); prev = 0.0
    for k, x in enumerate(y):
        prev = a * prev + (1 - a) * x; out[k] = prev
    return out


def cr_highpass(y, tau):
    a = np.exp(-DT / tau); out = np.empty_like(y); px = py = 0.0
    for k, x in enumerate(y):
        py = a * (py + x - px); px = x; out[k] = py
    return out


def sipm_pulse(y):
    s1 = 1 / (1 - np.exp(-DT / TF)); s2 = 1 / (1 - np.exp(-DT / T2))
    return rc_lowpass(y, TF) * s1 - rc_lowpass(y, T2) * s2


def shaped(tp, n=N):
    """Response to one avalanche at t = 0, normalised to a peak of one."""
    x = np.zeros(n); x[0] = 1.0
    y = cr_highpass(sipm_pulse(x), tp / 2)
    for _ in range(2):
        y = rc_lowpass(y, tp / 2)
    return y / y.max()


def crossing(y, thr):
    return int(np.argmax(y >= thr)) * DT


T = np.arange(N) * DT
DEV = sipm_pulse(np.r_[1.0, np.zeros(N - 1)]); DEV /= DEV.max()
FAST = shaped(15.0)


def model_delay(tp):
    return shaped(tp).argmax() * DT - crossing(FAST, THR)


def borel(n):
    """Number of fired cells per primary with branching crosstalk."""
    out = np.ones(n, dtype=int); todo = np.ones(n, dtype=int)
    while todo.any():
        new = np.where(todo > 0, RNG.poisson(-np.log(1 - XT) * todo), 0)
        out += new; todo = new
    return out


def stored(tp, delays, mode, n_ev=20000, jitter=1.0, noise=None, win_before=600):
    """Stored value (pe) per event and delay: the avalanche that triggers at t = 0 (with its
    crosstalk), dark pulses before and after, noise, hold at t_cross + delay."""
    sh = shaped(tp)
    noise = 0.1 * np.sqrt(87.5 / tp) if noise is None else noise
    tc = crossing(FAST, THR)
    n_trig = borel(n_ev)
    res = np.zeros((n_ev, len(delays)))
    span = win_before + 200
    for i in range(n_ev):
        k = RNG.poisson(DARK_MHZ * 1e-3 * span)
        td = RNG.uniform(-win_before, 200, k)
        ad = borel(k) if k else np.zeros(0)
        th = tc + np.asarray(delays) + RNG.normal(0, jitter)
        def sig(t):
            v = n_trig[i] * np.interp(t, T, sh, left=0.0)
            for t0, a in zip(td, ad):
                v = v + a * np.interp(t - t0, T, sh, left=0.0, right=0.0)
            return v
        if mode == "th":
            res[i] = sig(th)
        else:
            grid = np.arange(tc, tc + max(delays) + 1, 1.0)
            run = np.maximum.accumulate(sig(grid))
            res[i] = np.interp(th, grid, run)
    return res + RNG.normal(0, noise, res.shape)


def fig_mechanism(peak_stretch=False):
    """peak_stretch=True writes mecanismo_ps.png, with the arrows labelled as the minimum
    HoldDelay (bench note 1 v2); False writes mecanismo.png as in the v1 notes."""
    d25, d87 = model_delay(25.0), model_delay(87.5)
    tc = crossing(FAST, THR)
    fig, ax = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)
    ax[0].plot(T, DEV, "k"); ax[0].set_ylabel("pulso del modelo del SiPM")
    ax[0].set_title("Una avalancha de 1 pe en t = 0 (modelo)")
    ax[1].plot(T, FAST, "C1"); ax[1].axhline(THR, ls="--", c="gray")
    ax[1].axvline(tc, c="C3"); ax[1].text(tc + 3, 0.8, f"trigger: cruce de 0,5 pe\n(t = {tc:.0f} ns)", c="C3")
    ax[1].set_ylabel("fast shaper 15 ns")
    for tp, c, d in ((25.0, "C0", d25), (50.0, "C1", model_delay(50.0)), (87.5, "C2", d87)):
        y = shaped(tp); tm = y.argmax() * DT
        ax[2].plot(T, y, c, label=f"slow shaper {tp:g} ns: máximo a {tm:.0f} ns")
        yl = {25.0: 1.12, 50.0: 1.32, 87.5: 1.52}[tp]
        ax[2].annotate("", xy=(tm, yl), xytext=(tc, yl), arrowprops=dict(arrowstyle="<->", color=c))
        lab = f"HoldDelay mínimo {d:.0f} ns" if peak_stretch else f"hold delay {d:.0f} ns"
        ax[2].text(tm + 4, yl - 0.04, lab, c=c)
    ax[2].axvline(tc, c="C3")
    ax[2].set_ylim(-0.35, 1.65); ax[2].axhline(0, c="k", lw=0.5)
    ax[2].set_ylabel("shaped pulse (pe)"); ax[2].set_xlabel("tiempo (ns)"); ax[2].legend(loc="lower right")
    ax[2].set_xlim(0, 300)
    name = "mecanismo_ps.png" if peak_stretch else "mecanismo.png"
    fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)
    return d25, d87


def fig_modes(d25):
    tc = crossing(FAST, THR)
    y = shaped(25.0)
    delays = np.arange(0, 121, 10)
    th = np.interp(tc + delays, T, y)
    pd = np.interp(tc + delays, T, np.maximum.accumulate(np.where(T >= tc, y, 0)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(T - tc, y, "k", lw=1, label="shaped pulse, 25 ns")
    ax[0].plot(delays, th, "o", c="C0", label="track & hold: valor en el hold")
    ax[0].plot(delays, pd, "s", mfc="none", c="C3", label="peak detector: máximo hasta el hold")
    ax[0].axvline(d25, c="gray", ls=":")
    ax[0].set_xlim(-20, 130); ax[0].set_ylim(-0.4, 1.15); ax[0].axhline(0, c="k", lw=0.5)
    ax[0].set_xlabel("hold delay desde el trigger (ns)"); ax[0].set_ylabel("valor guardado (pe)")
    ax[0].set_title("Sin ruido ni dark pulses"); ax[0].legend(fontsize=8, loc="lower left")
    for mode, c, lab in (("th", "C0", "track & hold"), ("pd", "C3", "peak detector")):
        v = stored(25.0, delays, mode, n_ev=4000)
        one = []
        for j in range(len(delays)):
            s = v[:, j]; sel = s[(s > 0.5) & (s < 1.5)] if delays[j] >= 10 else s[(s > 0.2) & (s < 1.5)]
            one.append(np.median(sel) if len(sel) else np.nan)
        ax[1].plot(delays, one, "o-" if mode == "th" else "s-", c=c, mfc="none" if mode == "pd" else c, label=lab)
    ax[1].axvline(d25, c="gray", ls=":")
    ax[1].set_xlabel("hold delay desde el trigger (ns)"); ax[1].set_ylabel("posición del pico de 1 pe (pe)")
    ax[1].set_title("Con ruido y DP (toy)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "modos.png", dpi=150); plt.close(fig)


def fig_spectra(d25):
    bins = np.linspace(-0.5, 4.0, 181)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    cases = (("th", [d25, d25 - 20, d25 + 40], "Track & hold"),
             ("pd", [d25, d25 - 20, 120], "Peak detector"))
    for a, (mode, ds, title) in zip(ax, cases):
        v = stored(25.0, ds, mode, n_ev=20000)
        for j, d in enumerate(ds):
            lab = f"hold delay {d:.0f} ns" + (" (máximo)" if d == d25 else "")
            a.hist(v[:, j], bins, histtype="step", lw=1.3, label=lab)
        a.set_title(title); a.set_xlabel("valor guardado (pe)"); a.legend(fontsize=8)
    ax[0].set_ylabel("eventos")
    fig.tight_layout(); fig.savefig(OUT / "espectros.png", dpi=150); plt.close(fig)


def fig_staircase():
    """Staircase of one channel at 5 V: trigger rate against the threshold (in pe). Dark
    pulses at DARK_MHZ with branching crosstalk (Borel), amplitude n pe plus the noise of the
    fast path (0.15 pe); noise crossings by Rice's formula with a 15 ns fast shaper."""
    lam = -np.log(1 - XT)
    n = np.arange(1, 30)
    from math import lgamma
    pn = np.exp(-lam * n + (n - 1) * np.log(lam * n) - np.array([lgamma(k + 1) for k in n]))
    sig = 0.15
    thr = np.linspace(0.0, 4.5, 901)
    from math import erf
    cdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / np.sqrt(2))))
    dark = DARK_MHZ * np.array([np.sum(pn * cdf((n - t) / sig)) for t in thr])
    noise = 1 / (2 * 15.0e-3) * np.exp(-thr**2 / (2 * sig**2))       # MHz
    rate = dark + noise
    spec = -np.gradient(dark, thr)
    fig, ax = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    ax[0].semilogy(thr, rate, "k")
    ax[0].set_ylim(1e-3, 1e2); ax[0].set_ylabel("tasa de trigger (MHz)")
    ax[0].axvline(0.5, c="C3", ls="--")
    ax[0].text(0.55, 40, "umbral elegido: 0,5 pe", c="C3")
    for k, lab in ((1, "meseta de 1 pe:\ntodos los DP"), (2, "meseta de 2 pe:\nDP con crosstalk"),
                   (3, "meseta de 3 pe")):
        r = np.interp(k - 0.35, thr, rate)
        ax[0].text(k - 0.45, r / 25, f"{lab}\n({r:.2g} MHz)", fontsize=8)
    ax[0].text(0.03, 8, "ruido", fontsize=8)
    ax[0].set_title("Staircase de un canal a 5 V (modelo)")
    ax[1].plot(thr, spec, "C0")
    ax[1].axvline(0.5, c="C3", ls="--")
    ax[1].set_ylabel("−d(tasa)/d(umbral)")
    ax[1].set_xlabel("umbral (pe)")
    ax[1].set_title("Su derivada: el espectro de amplitudes de los DP", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "staircase.png", dpi=150); plt.close(fig)


OFFSET = 0.0                   # possible arming-to-hold offset added to the window (ns): the
                               # window is HoldDelay (gtrg to hold, FERSlib); the latency from
                               # the crossing to gtrg shifts it along the pulse, not lengthens it


def shaped_noise(tp, n):
    """White noise through the slow shaper (CR-RC^2, theta = tp/2), normalised to unit rms."""
    from scipy.signal import lfilter
    a = np.exp(-DT / (tp / 2))
    y = lfilter([a, -a], [1, -a], RNG.normal(size=n))
    for _ in range(2):
        y = lfilter([1 - a], [1, -a], y)
    y = y[5000:]
    return y / y.std()


def fig_window():
    """Peak stretch: the stored value is the maximum between trigger and hold. Left, a 1 pe
    pulse at 25 ns against the hold delay (no noise). Centre and right, noise alone (SiPM
    below breakdown): distribution of the stored value, in units of the rms of one sample,
    for windows OFFSET + HoldDelay, at 25 and 87.5 ns (HoldDelay 0 is one sample)."""
    tc = crossing(FAST, THR)
    y = shaped(25.0)
    delays = np.arange(0, 121, 8)
    pd = np.interp(tc + delays, T, np.maximum.accumulate(np.where(T >= tc, y, 0)))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(T - tc, y, "k", lw=1, label="pulso del slow shaper, 25 ns")
    ax[0].plot(delays, pd, "s", mfc="none", c="C3", label="peak stretch: máximo hasta el hold")
    ax[0].set_xlim(-20, 130); ax[0].set_ylim(-0.4, 1.15); ax[0].axhline(0, c="k", lw=0.5)
    ax[0].set_xlabel("HoldDelay desde el trigger (ns)"); ax[0].set_ylabel("valor guardado (pe)")
    ax[0].set_title("Señal de 1 pe, sin ruido"); ax[0].legend(fontsize=8, loc="lower right")
    bins = np.linspace(-4, 6, 201)
    n_ev = 20000
    for a, tp in zip(ax[1:], (25.0, 87.5)):
        z = shaped_noise(tp, n_ev * 1200 + 6000)
        starts = np.arange(n_ev) * 1200
        for hd, c in ((0, "C0"), (96, "C1"), (296, "C3")):
            w = int(OFFSET + hd)
            m = np.array([z[s:s + w + 1].max() for s in starts])
            lab = "HoldDelay 0 ns (un instante)" if hd == 0 else f"HoldDelay {hd} ns"
            a.hist(m, bins, histtype="step", lw=1.3, color=c,
                   label=f"{lab}: media {m.mean():.2f}, anchura {m.std():.2f}")
        a.set_title(f"Solo ruido, slow shaper {tp:g} ns")
        a.set_xlabel("valor guardado (rms del ruido en un instante)"); a.legend(fontsize=7)
    ax[1].set_ylabel("eventos")
    fig.tight_layout(); fig.savefig(OUT / "ventana.png", dpi=150); plt.close(fig)


def fig_scan_ps():
    """Peak stretch, the hold delay scan expected for the channel that triggers: mean stored
    value against HoldDelay (no latency) at 25, 50 and 87.5 ns. Solid, the toy with noise,
    dark pulses and crosstalk; dashed, one avalanche without noise. Dotted lines at the model
    minimum HoldDelay."""
    tc = crossing(FAST, THR)
    fig, ax = plt.subplots(figsize=(7, 4))
    delays = np.arange(0, 201, 8)
    for tp, c in ((25.0, "C0"), (50.0, "C1"), (87.5, "C2")):
        v = stored(tp, delays, "pd", n_ev=2000)
        y = shaped(tp)
        clean = np.interp(tc + delays, T, np.maximum.accumulate(np.where(T >= tc, y, 0)))
        d = model_delay(tp)
        ax.plot(delays, v.mean(axis=0), "s-", c=c, mfc="none", label=f"slow shaper {tp:g} ns (mínimo {d:.0f} ns)")
        ax.plot(delays, clean, "--", c=c, lw=1)
        ax.axvline(d, c=c, ls=":")
    ax.set_xlabel("HoldDelay desde el trigger (ns), sin latencia")
    ax.set_ylabel("valor guardado medio (pe)")
    ax.set_title("Peak stretch: canal que dispara")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "scan_ps.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d25, d87 = fig_mechanism()
    print(f"model hold delay: {d25:.0f} ns at 25 ns, {model_delay(50.0):.0f} ns at 50 ns, {d87:.0f} ns at 87.5 ns")
    fig_modes(d25)
    fig_spectra(d25)
    fig_staircase()
    fig_window()
    fig_mechanism(peak_stretch=True)
    fig_scan_ps()
    print("figures in", OUT)
