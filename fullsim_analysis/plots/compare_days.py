"""Pedestal noise of the three bench sessions (22, 23 and 27 Oct 2026) side by side: the five
boards with our ("green") adapter boards on 22 Oct, boards 0 and 1 with the CAEN ("red")
adapters on 23 Oct, boards 1, 2 and 4 with the red adapters and the new FPGA firmware on
27 Oct (read by the DAQ as boards 0, 1, 2; here named by their own number). Reads, for each
day, fullsim_analysis/results/bench/<day>/runs.csv, channels.csv (inspect_run.jl) and
noise_components.csv (noise_components.jl). The pedestal runs are those with the SiPMs below
breakdown (HV_Vbias 40.7 V); the first run of each shaping and HoldDelay is used.

Writes fullsim_analysis/results/bench/2026-10-27/:
    compare_days_shaping.png     total, coherent and incoherent noise of the high gain
                                 against the shaping time at HoldDelay 0: median over the
                                 chips of each day, bars from the lowest to the highest chip;
    compare_days_holddelay.png   high gain at 25 ns against HoldDelay: pedestal rise in units
                                 of the rms at HoldDelay 0 (top) and total, coherent and
                                 incoherent rms relative to HoldDelay 0 (bottom), each day;
    compare_days_components.tex  LaTeX table: for each day, board and chip, the total,
                                 coherent and incoherent noise and the coherent fraction at
                                 25 ns and HoldDelay 0;
    compare_days_components_last.tex  the same for the chips of the last day alone;
    compare_days_summary.tex     LaTeX table: for each day and shaping, the median and the
                                 range over the chips of the three components at HoldDelay 0,
                                 and the number of chips;
and prints both tables.

    python3 fullsim_analysis/plots/compare_days.py
"""
import numpy as np
import pandas as pd

from common import plt, results_dir, save

DAYS = [("2026-10-22", "22 Oct, green adapters, boards 0--4", "tab:green", "s", "--", -1.2),
        ("2026-10-23", "23 Oct, red adapters, boards 0, 1", "tab:red", "o", "-.", 0.0),
        ("2026-10-27", "27 Oct, red adapters, boards 1, 2, 4", "tab:blue", "^", "-", 1.2)]
# DAQ board number -> board number of the bench, when they differ
BOARD_NAMES = {"2026-10-27": {0: 1, 1: 2, 2: 4}}
OUT = "2026-10-27"
PEDESTAL_HV_MAX = 42.0
FS = 18
COMPS = [("hg_total", "total", "o"), ("hg_coherent", "coherent", "^"),
         ("hg_incoherent", "incoherent", "s")]


def ns(s):
    return s.astype(str).str.split().str[0].astype(float)


def load(day):
    d = results_dir("bench", day)
    runs = pd.read_csv(d / "runs.csv")[["run", "board", "HG_ShapingTime", "HoldDelay",
                                        "HV_Vbias"]]
    runs["tp"], runs["hold"] = ns(runs.HG_ShapingTime), ns(runs.HoldDelay)
    runs = runs[ns(runs.HV_Vbias) <= PEDESTAL_HV_MAX]
    first = runs.groupby(["tp", "hold"]).run.min().rename("first").reset_index()
    runs = runs.merge(first, on=["tp", "hold"])
    runs = runs[runs.run == runs["first"]]
    nc = pd.read_csv(d / "noise_components.csv").merge(runs, on=["run", "board"])
    ch = pd.read_csv(d / "channels.csv").merge(runs, on=["run", "board"])
    ch = ch[(ch.hg_zero < 1) & (ch.hg_rms > 0)]          # channels that are read
    names = BOARD_NAMES.get(day, {})
    for t in (nc, ch):
        t["board"] = t.board.map(lambda b: names.get(b, b))
    return nc, ch


def main():
    data = {day: load(day) for day, *_ in DAYS}

    # noise against shaping, HoldDelay 0
    fig, ax = plt.subplots(figsize=(12, 8))
    for day, lab, c, mk, ls, dx in DAYS:
        nc = data[day][0]
        g = nc[nc.hold == 0].groupby("tp")
        for col, name, cmk in COMPS:
            med, lo, hi = g[col].median(), g[col].min(), g[col].max()
            ax.errorbar(med.index + dx, med, [med - lo, hi - med], fmt=cmk, ls=ls, c=c,
                        ms=10, lw=2, capsize=4, label=f"{name}, {lab}")
    ax.set_xlabel("HG shaping time (ns)", fontsize=FS)
    ax.set_ylabel("noise (ADC counts)", fontsize=FS)
    ax.set_xlim(15, 95); ax.set_ylim(0, 15)
    ax.tick_params(labelsize=FS - 2)
    ax.legend(fontsize=FS - 7, ncol=1, loc="lower left")
    ax.set_title("HoldDelay 0: median over the chips of each day, bars to the extreme chips",
                 fontsize=FS - 2)
    fig.tight_layout()
    save(fig, "bench", OUT, "compare_days_shaping")

    # noise against HoldDelay, 25 ns
    fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    for day, lab, c, mk, ls, dx in DAYS:
        nc, ch = data[day]
        c25 = ch[ch.tp == 25]
        base = c25[c25.hold == 0].set_index(["board", "channel"])
        x = c25.set_index(["board", "channel"])
        rise = (x.hg_mean - base.hg_mean.reindex(x.index)) / base.hg_rms.reindex(x.index)
        r = rise.groupby(x.hold).median()
        ax[0].plot(r.index, r.values, mk, ls=ls, c=c, ms=10, lw=2, label=lab)
        h = nc[nc.tp == 25].groupby("hold")
        for col, name, cmk in COMPS:
            med = h[col].median()
            ax[1].plot(med.index, med / med.loc[0], cmk, ls=ls, c=c, ms=10, lw=2,
                       label=f"{name}, {lab}")
    ax[0].set_ylabel("pedestal rise / rms at 0", fontsize=FS)
    ax[0].legend(fontsize=FS - 6, loc="upper left")
    ax[0].set_title("HG 25 ns: pedestal against the window, median over the channels",
                    fontsize=FS - 2)
    ax[1].set_ylim(0, 1.1)
    ax[1].set_xlabel("HoldDelay (ns)", fontsize=FS)
    ax[1].set_ylabel("rms / rms at HoldDelay 0", fontsize=FS)
    ax[1].legend(fontsize=FS - 7, ncol=1, loc="lower left")
    for a in ax:
        a.tick_params(labelsize=FS - 2)
    fig.tight_layout()
    save(fig, "bench", OUT, "compare_days_holddelay")

    # table per chip, 25 ns, HoldDelay 0
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"day & adapter & board & channels & $\sigma_\mathrm{tot}$ & $\sigma_\mathrm{coh}$ & "
             r"$\sigma_\mathrm{inc}$ & $\sigma_\mathrm{coh}^2/\sigma_\mathrm{tot}^2$ \\", r"\midrule"]
    for day, lab, *_ in DAYS:
        nc = data[day][0]
        x = nc[(nc.hold == 0) & (nc.tp == 25)].sort_values(["board", "chip"])
        adapter = "green" if day == "2026-10-22" else "red"
        for r in x.itertuples():
            lines.append(f"{day[8:]} Oct & {adapter} & {r.board} & {32 * r.chip}--{32 * r.chip + 31} & "
                         f"{r.hg_total:.1f} & {r.hg_coherent:.1f} & {r.hg_incoherent:.1f} & "
                         f"{r.hg_coherent**2 / r.hg_total**2:.2f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path = results_dir("bench", OUT) / "compare_days_components.tex"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("written", path)

    # the chips of the last day alone, for the talk
    day = DAYS[-1][0]
    nc = data[day][0]
    x = nc[(nc.hold == 0) & (nc.tp == 25)].sort_values(["board", "chip"])
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"board & channels & $\sigma_\mathrm{tot}$ & $\sigma_\mathrm{coh}$ & "
             r"$\sigma_\mathrm{inc}$ & $\sigma_\mathrm{coh}^2/\sigma_\mathrm{tot}^2$ \\", r"\midrule"]
    for r in x.itertuples():
        lines.append(f"{r.board} & {32 * r.chip}--{32 * r.chip + 31} & {r.hg_total:.1f} & "
                     f"{r.hg_coherent:.1f} & {r.hg_incoherent:.1f} & "
                     f"{r.hg_coherent**2 / r.hg_total**2:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = results_dir("bench", OUT) / "compare_days_components_last.tex"
    path.write_text("\n".join(lines) + "\n")
    print("written", path)

    # summary per day and shaping, HoldDelay 0: median (min--max) over the chips
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"day & shaping & chips & $\sigma_\mathrm{tot}$ & $\sigma_\mathrm{coh}$ & "
             r"$\sigma_\mathrm{inc}$ \\", r"\midrule"]
    for day, lab, *_ in DAYS:
        nc = data[day][0]
        for tp, g in nc[nc.hold == 0].groupby("tp"):
            cells = [f"{g[col].median():.1f} ({g[col].min():.1f}--{g[col].max():.1f})"
                     for col, *_ in COMPS]
            lines.append(f"{day[8:]} Oct & {tp:g} ns & {len(g)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path = results_dir("bench", OUT) / "compare_days_summary.tex"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("written", path)


if __name__ == "__main__":
    main()
