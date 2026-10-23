"""Noise of the Citiroc front end, model against bench data (docs_fullsim/model_data/noise.tex).
Reads, for a day of pedestal runs (SiPMs below breakdown, periodic trigger),
fullsim_analysis/results/bench/<day>/runs.csv, channels.csv (inspect_run.jl) and
noise_components.csv (noise_components.jl).

The model is that of the simulation: white series noise through the slow shaper (one CR and
two RC with time constant tp/2, as src/frontend/shaping.jl), whose rms scales as tp^-1/2,
the same rms in ADC counts in the low gain as in the high gain at the same shaping, and the
peak stretch storing the maximum of the shaped noise between the trigger and the hold, with
HoldDelay 0 a single sample (bench/hold_scan_figs.py, figure ventana.png).

Writes fullsim_analysis/results/bench/<day>/:
    noise_vs_shaping.png    left, total, coherent and incoherent noise of the high gain
                            against the shaping time, with HoldDelay 0 (median over the
                            chips, bars from the lowest to the highest chip), the low gain at
                            87.5 ns, and the model scaled to the incoherent noise at 87.5 ns;
                            right, the same divided by its value at 87.5 ns, with the model
                            ratios;
    noise_vs_holddelay.png  high gain at 25 ns against HoldDelay: left, the rise of the
                            pedestal from HoldDelay 0 in units of the total rms at HoldDelay 0
                            (median over the channels of each board, one curve per board);
                            right, total, coherent and incoherent rms divided by their value
                            at HoldDelay 0 (median over the chips); the model in black.
and prints the numbers drawn.

    python3 fullsim_analysis/plots/model_data_noise.py <YYYY-MM-DD>
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from common import plt, results_dir, save

TP_REF = 87.5
N_EV = 20000
SPAN = 600          # ns of noise per model event, longer than the longest window


def shaped_noise(tp, n, rng):
    """White noise through CR-RC^2 (time constant tp/2) on a 1 ns grid, unit rms."""
    a = np.exp(-1.0 / (tp / 2))
    y = lfilter([a, -a], [1, -a], rng.normal(size=n + 5000))
    for _ in range(2):
        y = lfilter([1 - a], [1, -a], y)
    y = y[5000:]
    return y / y.std()


def window_model(tp, delays, rng=np.random.default_rng(1)):
    """Mean and rms of the maximum of the shaped noise over HoldDelay + 1 samples."""
    z = shaped_noise(tp, N_EV * SPAN, rng).reshape(N_EV, SPAN)
    m = [z[:, : int(d) + 1].max(axis=1) for d in delays]
    return np.array([x.mean() for x in m]), np.array([x.std() for x in m])


def ns(s):
    return s.astype(str).str.split().str[0].astype(float)


def main(day: str):
    d = results_dir("bench", day)
    runs = pd.read_csv(d / "runs.csv")[["run", "board", "HG_ShapingTime", "HoldDelay"]]
    runs["tp"], runs["hold"] = ns(runs.HG_ShapingTime), ns(runs.HoldDelay)
    nc = pd.read_csv(d / "noise_components.csv").merge(runs, on=["run", "board"])
    ch = pd.read_csv(d / "channels.csv").merge(runs, on=["run", "board"])
    ch = ch[ch.hg_zero < 1]
    first = nc.groupby(["tp", "hold"]).run.min().rename("first")
    nc = nc.merge(first, on=["tp", "hold"])
    nc = nc[nc.run == nc["first"]]
    ch = ch.merge(first, on=["tp", "hold"])
    ch = ch[ch.run == ch["first"]]

    # noise against shaping, HoldDelay 0
    s = nc[nc.hold == 0]
    comps = [("hg_total", "total", "C0"), ("hg_coherent", "coherent", "C1"),
             ("hg_incoherent", "incoherent", "C2")]
    g = s.groupby("tp")
    tps = np.array(sorted(s.tp.unique()))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t = np.linspace(20, 95, 100)
    print("HG noise against shaping (median over chips, ADC counts; ratio to 87.5 ns):")
    for col, lab, c in comps:
        med, lo, hi = g[col].median(), g[col].min(), g[col].max()
        ax[0].errorbar(tps, med, [med - lo, hi - med], fmt="o-", c=c, capsize=3, label=f"HG {lab}")
        ref = med.loc[TP_REF]
        ax[1].plot(tps, med / ref, "o-", c=c, label=f"HG {lab}")
        print(f"  {lab:10s} " + "  ".join(f"{tp:g} ns {m:5.2f} ({m / ref:.2f})"
                                           for tp, m in med.items()))
    inc_ref = g.hg_incoherent.median().loc[TP_REF]
    ax[0].plot(t, inc_ref * np.sqrt(TP_REF / t), "k--", lw=1,
               label="model: series noise, scaled to incoherent at 87.5 ns")
    ax[1].plot(t, np.sqrt(TP_REF / t), "k--", lw=1, label="model: (87.5 ns / tp)$^{1/2}$")
    lg = s[s.tp == TP_REF]
    for col, lab, mk in (("lg_total", "LG total", "s"), ("lg_coherent", "LG coherent", "v"),
                         ("lg_incoherent", "LG incoherent", "^")):
        ax[0].plot([TP_REF], [lg[col].median()], mk, mfc="none", c="k", ms=7, label=lab)
    print(f"  LG at 87.5 ns: total {lg.lg_total.median():.2f}, coherent "
          f"{lg.lg_coherent.median():.2f}, incoherent {lg.lg_incoherent.median():.2f}")
    ax[0].set_xlabel("HG shaping time (ns)"); ax[0].set_ylabel("rms (ADC counts)")
    ax[0].set_ylim(0, 16); ax[0].legend(fontsize=7, ncol=2, loc="lower right")
    ax[1].set_xlabel("HG shaping time (ns)"); ax[1].set_ylabel("rms / rms at 87.5 ns")
    ax[1].set_ylim(0.5, 2.0); ax[1].legend(fontsize=7)
    fig.suptitle(f"{day}: pedestal noise against shaping, HoldDelay 0, SiPMs below breakdown",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "bench", day, "noise_vs_shaping")

    # noise against HoldDelay, 25 ns
    tp = 25.0
    h = nc[nc.tp == tp]
    holds = np.array(sorted(h.hold.unique()))
    mm, ms = window_model(tp, holds)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c25 = ch[ch.tp == tp]
    base = c25[c25.hold == 0].set_index(["board", "channel"])
    print(f"HG at {tp:g} ns against HoldDelay {holds.astype(int).tolist()}:")
    for b in sorted(c25.board.unique()):
        x = c25[c25.board == b].set_index(["board", "channel"])
        rise = ((x.hg_mean - base.hg_mean.reindex(x.index)) / base.hg_rms.reindex(x.index))
        r = rise.groupby(x.hold).median()
        ax[0].plot(r.index, r.values, "o-", ms=4, label=f"board {b}")
        print(f"  board {b} rise / rms(0): {np.round(r.values, 2).tolist()}")
    ax[0].plot(holds, mm - mm[0], "ks--", label="model: maximum of series noise")
    print(f"  model rise: {np.round(mm - mm[0], 2).tolist()}")
    ax[0].set_xlabel("HoldDelay (ns)"); ax[0].set_ylabel("pedestal rise / rms at HoldDelay 0")
    ax[0].legend(fontsize=7)
    gh = h.groupby("hold")
    for col, lab, c in comps:
        med = gh[col].median()
        ax[1].plot(med.index, med / med.loc[0], "o-", c=c, label=f"HG {lab}")
        print(f"  {lab:10s} rms / rms(0): {np.round((med / med.loc[0]).values, 2).tolist()}")
    ax[1].plot(holds, ms / ms[0], "ks--", label="model: maximum of series noise")
    print(f"  model rms / rms(0): {np.round(ms / ms[0], 2).tolist()}")
    ax[1].set_xlabel("HoldDelay (ns)"); ax[1].set_ylabel("rms / rms at HoldDelay 0")
    ax[1].set_ylim(0, 1.1); ax[1].legend(fontsize=7)
    fig.suptitle(f"{day}: pedestal against HoldDelay, HG {tp:g} ns, SiPMs below breakdown",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "bench", day, "noise_vs_holddelay")


if __name__ == "__main__":
    main(sys.argv[1])
