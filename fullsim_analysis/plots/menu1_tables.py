"""Tables and the per-chip figure of the pedestal noise of one day of bench data (menu 1), for
the report of the day. Reads fullsim_analysis/results/bench/<day>/runs.csv, channels.csv
(inspect_run.jl) and noise_components.csv (noise_components.jl); pedestal runs only (SiPMs
below breakdown), the first run of each shaping and HoldDelay.

Writes fullsim_analysis/results/bench/<day>/:
    menu1_chips.png       total, coherent and incoherent noise of the high gain of each chip at
                          25 ns and HoldDelay 0, and the coherent fraction of the variance;
    menu1_chips.tex       the same as a LaTeX table, with the low gain;
    menu1_shaping.tex     LaTeX table: median over the chips and range of the three components
                          at HoldDelay 0 for each shaping, high gain, and the low gain total;
    menu1_holddelay.tex   LaTeX table, 25 ns: for each HoldDelay the total, coherent and
                          incoherent rms (median over the chips) and their ratio to HoldDelay
                          0, and the rise of the pedestal (median over the channels) in counts
                          and in units of the rms at HoldDelay 0;
    menu1_runs.tex        LaTeX table of the pedestal runs: run, shaping, HoldDelay, events per
                          board, detector temperature of each board.

    python3 fullsim_analysis/plots/menu1_tables.py <YYYY-MM-DD>
"""
import sys

import numpy as np
import pandas as pd

from common import board_name, plt, results_dir, save

PEDESTAL_HV_MAX = 42.0
FS = 18
COMPS = [("hg_total", "total", "o"), ("hg_coherent", "coherent", "^"),
         ("hg_incoherent", "incoherent", "s")]


def ns(s):
    return s.astype(str).str.split().str[0].astype(float)


def main(day):
    d = results_dir("bench", day)
    runs = pd.read_csv(d / "runs.csv")
    runs["tp"], runs["hold"] = ns(runs.HG_ShapingTime), ns(runs.HoldDelay)
    runs = runs[ns(runs.HV_Vbias) <= PEDESTAL_HV_MAX].copy()
    runs["name"] = [board_name(day, b) for b in runs.board]
    first = runs.groupby(["tp", "hold"]).run.min().rename("first").reset_index()
    sel = runs.merge(first, on=["tp", "hold"])
    sel = sel[sel.run == sel["first"]]
    nc = pd.read_csv(d / "noise_components.csv").merge(
        sel[["run", "board", "name", "tp", "hold"]], on=["run", "board"])
    ch = pd.read_csv(d / "channels.csv").merge(
        sel[["run", "board", "name", "tp", "hold"]], on=["run", "board"])
    ch = ch[(ch.hg_zero < 1) & (ch.hg_rms > 0)]

    # runs table
    lines = [r"\begin{tabular}{lllll}", r"\toprule",
             r"run & HG shaping & HoldDelay & events per board & detector temperature ($^\circ$C) \\",
             r"\midrule"]
    for r, g in runs.sort_values(["run", "name"]).groupby("run"):
        g = g.sort_values("name")
        temps = ", ".join(f"{t:.1f}" for t in g.T_detector)
        lines.append(f"{r} & {g.tp.iloc[0]:g} ns & {g.hold.iloc[0]:g} ns & "
                     f"{int(g.events.iloc[0] // len(g))} & {temps} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (d / "menu1_runs.tex").write_text("\n".join(lines) + "\n")

    # chips at 25 ns, HoldDelay 0
    x = nc[(nc.tp == 25) & (nc.hold == 0)].sort_values(["name", "chip"])
    x = x.assign(frac=x.hg_coherent**2 / x.hg_total**2)
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"board & channels & $\sigma_\mathrm{tot}$ & $\sigma_\mathrm{coh}$ & "
             r"$\sigma_\mathrm{inc}$ & $\sigma_\mathrm{coh}^2/\sigma_\mathrm{tot}^2$ & "
             r"$\sigma_\mathrm{tot}^\mathrm{LG}$ \\", r"\midrule"]
    for r in x.itertuples():
        lines.append(f"{r.name} & {32 * r.chip}--{32 * r.chip + 31} & {r.hg_total:.1f} & "
                     f"{r.hg_coherent:.1f} & {r.hg_incoherent:.1f} & {r.frac:.2f} & "
                     f"{r.lg_total:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (d / "menu1_chips.tex").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [2, 1]})
    labels = [f"{r.name}: {32 * r.chip}\u2013{32 * r.chip + 31}" for r in x.itertuples()]
    pos = np.arange(len(x))
    for k, (col, name, mk) in enumerate(COMPS):
        ax[0].bar(pos + (k - 1) * 0.27, x[col], 0.27, label=name)
    ax[0].set_xticks(pos, labels, fontsize=FS - 4, rotation=20)
    ax[0].set_ylabel("noise (ADC counts)", fontsize=FS)
    ax[0].set_ylim(0, 15)
    ax[0].legend(fontsize=FS - 4, loc="upper right", ncol=3)
    ax[0].tick_params(labelsize=FS - 2)
    ax[0].set_title("high gain, 25 ns, HoldDelay 0: board and channels of each chip",
                    fontsize=FS - 2)
    ax[1].bar(pos, x.frac, 0.6, color="tab:gray")
    ax[1].set_xticks(pos, labels, fontsize=FS - 4, rotation=20)
    ax[1].set_ylabel(r"$\sigma_\mathrm{coh}^2 / \sigma_\mathrm{tot}^2$", fontsize=FS)
    ax[1].set_ylim(0, 0.5)
    ax[1].tick_params(labelsize=FS - 2)
    ax[1].set_title("coherent fraction of the variance", fontsize=FS - 2)
    fig.tight_layout()
    save(fig, "bench", day, "menu1_chips")

    # shaping table, HoldDelay 0
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"HG shaping & $\sigma_\mathrm{tot}$ & $\sigma_\mathrm{coh}$ & "
             r"$\sigma_\mathrm{inc}$ & $\sigma_\mathrm{tot}^\mathrm{LG}$ \\", r"\midrule"]
    for tp, g in nc[nc.hold == 0].groupby("tp"):
        cells = [f"{g[c].median():.1f} ({g[c].min():.1f}--{g[c].max():.1f})"
                 for c in ("hg_total", "hg_coherent", "hg_incoherent", "lg_total")]
        lines.append(f"{tp:g} ns & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (d / "menu1_shaping.tex").write_text("\n".join(lines) + "\n")

    # HoldDelay table, 25 ns
    h = nc[nc.tp == 25].groupby("hold")[["hg_total", "hg_coherent", "hg_incoherent"]].median()
    c25 = ch[ch.tp == 25]
    base = c25[c25.hold == 0].set_index(["board", "channel"])
    xx = c25.set_index(["board", "channel"])
    shift = (xx.hg_mean - base.hg_mean.reindex(xx.index)).groupby(xx.hold).median()
    rise = ((xx.hg_mean - base.hg_mean.reindex(xx.index))
            / base.hg_rms.reindex(xx.index)).groupby(xx.hold).median()
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"HoldDelay & $\sigma_\mathrm{tot}$ & $\sigma_\mathrm{coh}$ & $\sigma_\mathrm{inc}$ & "
             r"$\sigma_\mathrm{tot}/\sigma_\mathrm{tot}(0)$ & $\sigma_\mathrm{coh}/\sigma_\mathrm{coh}(0)$ & "
             r"$\sigma_\mathrm{inc}/\sigma_\mathrm{inc}(0)$ & $\Delta p$ & $\Delta p/\sigma_\mathrm{tot}(0)$ \\",
             r"\midrule"]
    for hold in sorted(h.index):
        r = h.loc[hold]; r0 = h.loc[0]
        lines.append(f"{hold:g} ns & {r.hg_total:.1f} & {r.hg_coherent:.1f} & {r.hg_incoherent:.1f} & "
                     f"{r.hg_total / r0.hg_total:.2f} & {r.hg_coherent / r0.hg_coherent:.2f} & "
                     f"{r.hg_incoherent / r0.hg_incoherent:.2f} & {shift.loc[hold]:.1f} & "
                     f"{rise.loc[hold]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (d / "menu1_holddelay.tex").write_text("\n".join(lines) + "\n")
    for name in ("menu1_runs", "menu1_chips", "menu1_shaping", "menu1_holddelay"):
        print("written", d / f"{name}.tex")


if __name__ == "__main__":
    main(sys.argv[1])
