"""Noise of boards 0 and 1 with our ("green") adapter boards (22 Oct 2026, runs 26-34) and with
the CAEN ("red") adapter boards and the new FPGA firmware (23 Oct 2026, runs 42-50), for the
report of 23 Oct. Reads, for both days, fullsim_analysis/results/bench/<day>/runs.csv,
channels.csv (inspect_run.jl), noise_components.csv, channel_noise.csv and common_mode.csv
(noise_components.jl).

Writes fullsim_analysis/results/bench/2026-10-23/:
    compare_shaping.png      total, coherent and incoherent noise of the high gain against the
                             shaping time, HoldDelay 0, median over the three chips read on both
                             days (board 0, both chips; board 1, channels 32-63), bars from the
                             lowest to the highest chip;
    compare_holddelay.png    pedestal rise per board (top) and total, coherent and incoherent
                             rms relative to HoldDelay 0 (bottom) against HoldDelay, 25 ns, both
                             days;
    compare_slopes.png       slope of each channel against the common mode of its chip, median
                             over the runs with HoldDelay 0, boards 0 and 1, both days, with the
                             cut BLIND_SLOPE;
    compare_components.tex   LaTeX table: total, coherent and incoherent noise and the coherent
                             fraction of each chip of boards 0 and 1 in the first run with
                             HoldDelay 0 of each day (25 ns);
and prints the jumps per run and chip of board 0 and the correlations of the common modes.

    python3 fullsim_analysis/plots/compare_adapters.py
"""
import numpy as np
import pandas as pd

from common import plt, results_dir, save

DAYS = [("2026-10-22", "green adapters (22 Oct)", "tab:green", "--"),
        ("2026-10-23", "red adapters (23 Oct)", "tab:red", "-")]
OUT = "2026-10-23"
BOARDS = [0, 1]
BLIND_SLOPE = 0.75
FS = 18
COMPS = [("hg_total", "total", "o"), ("hg_coherent", "coherent", "^"),
         ("hg_incoherent", "incoherent", "s")]


def ns(s):
    return s.astype(str).str.split().str[0].astype(float)


def load(day, common_only=True):
    d = results_dir("bench", day)
    runs = pd.read_csv(d / "runs.csv")[["run", "board", "HG_ShapingTime", "HoldDelay"]]
    runs["tp"], runs["hold"] = ns(runs.HG_ShapingTime), ns(runs.HoldDelay)
    nc = pd.read_csv(d / "noise_components.csv").merge(runs, on=["run", "board"])
    nc = nc[nc.board.isin(BOARDS)]
    if common_only:
        nc = nc[~((nc.board == 1) & (nc.chip == 0))]
    first = nc.groupby(["tp", "hold"]).run.min().rename("first")
    nc = nc.merge(first, on=["tp", "hold"])
    ch = pd.read_csv(d / "channels.csv").merge(runs, on=["run", "board"])
    ch = ch[ch.board.isin(BOARDS) & (ch.hg_zero < 1)].merge(first, on=["tp", "hold"])
    if common_only:
        ch = ch[~((ch.board == 1) & (ch.channel < 32))]
    sl = pd.read_csv(d / "channel_noise.csv").merge(runs, on=["run", "board"])
    sl = sl[sl.board.isin(BOARDS) & (sl.hold == 0)]
    cm = pd.read_csv(d / "common_mode.csv")
    cm = cm[cm.board_a.isin(BOARDS) & cm.board_b.isin(BOARDS)].merge(
        runs.drop_duplicates("run")[["run", "tp", "hold"]], on="run")
    return nc, ch, sl, cm


def main():
    # the comparison uses the three chips read on both days (chip 0 of board 1 was not read on
    # 22 Oct, F7); the table lists all four
    data = {day: load(day) for day, *_ in DAYS}
    full = {day: load(day, common_only=False) for day, *_ in DAYS}

    # noise against shaping
    fig, ax = plt.subplots(figsize=(12, 8))
    for day, lab, c, ls in DAYS:
        nc = data[day][0]
        s = nc[(nc.hold == 0) & (nc.run == nc["first"])]
        g = s.groupby("tp")
        for col, name, mk in COMPS:
            med, lo, hi = g[col].median(), g[col].min(), g[col].max()
            ax.errorbar(med.index + (0.8 if c == "tab:red" else -0.8), med,
                        [med - lo, hi - med], fmt=mk, ls=ls, c=c, ms=11, lw=2, capsize=5,
                        label=f"{name}, {lab}")
            print(f"{day} {name:10s} " + "  ".join(f"{tp:g} ns {m:5.2f}" for tp, m in med.items()))
    ax.set_xlabel("HG shaping time (ns)", fontsize=FS)
    ax.set_ylabel("noise (ADC counts)", fontsize=FS)
    ax.set_xlim(15, 95); ax.set_ylim(0, 15)
    ax.tick_params(labelsize=FS - 2)
    ax.legend(fontsize=FS - 6, ncol=2, loc="lower left")
    ax.set_title("HoldDelay 0: median over the three chips read on both days", fontsize=FS - 2)
    fig.tight_layout()
    save(fig, "bench", OUT, "compare_shaping")

    # noise against HoldDelay, 25 ns
    fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    for day, lab, c, ls in DAYS:
        nc, ch = data[day][0], data[day][1]
        c25 = ch[(ch.tp == 25) & (ch.run == ch["first"])]
        base = c25[c25.hold == 0].set_index(["board", "channel"])
        for b, mk in zip(BOARDS, ("o", "s")):
            x = c25[c25.board == b].set_index(["board", "channel"])
            rise = (x.hg_mean - base.hg_mean.reindex(x.index)) / base.hg_rms.reindex(x.index)
            r = rise.groupby(x.hold).median()
            ax[0].plot(r.index, r.values, mk, ls=ls, c=c, ms=10, lw=2, label=f"board {b}, {lab}")
        h = nc[(nc.tp == 25) & (nc.run == nc["first"])].groupby("hold")
        for col, name, mk in COMPS:
            med = h[col].median()
            ax[1].plot(med.index, med / med.loc[0], mk, ls=ls, c=c, ms=10, lw=2,
                       label=f"{name}, {lab}")
    ax[0].set_ylabel("pedestal rise / rms at 0", fontsize=FS)
    ax[0].legend(fontsize=FS - 6, ncol=2, loc="upper left")
    ax[0].set_title("HG 25 ns: pedestal against the window", fontsize=FS)
    ax[1].set_ylim(0, 1.1)
    ax[1].set_xlabel("HoldDelay (ns)", fontsize=FS)
    ax[1].set_ylabel("rms / rms at HoldDelay 0", fontsize=FS)
    ax[1].legend(fontsize=FS - 6, ncol=2, loc="lower left")
    for a in ax:
        a.tick_params(labelsize=FS - 2)
    fig.tight_layout()
    save(fig, "bench", OUT, "compare_holddelay")

    # slopes per channel
    fig, axes = plt.subplots(len(BOARDS), 1, figsize=(14, 5 * len(BOARDS)), sharex=True)
    for a, b in zip(axes, BOARDS):
        for day, lab, c, ls in DAYS:
            sl = data[day][2]
            m = sl[sl.board == b].groupby("channel").slope.median()
            a.plot(m.index, m.values, "o" if c == "tab:red" else "s", c=c, ms=8,
                   mfc="none" if c == "tab:green" else c, mew=1.8, label=lab)
            print(f"{day} board {b}: blind {m[m < BLIND_SLOPE].index.tolist()}")
        a.axhline(BLIND_SLOPE, c="k", ls="--", lw=1.5)
        a.set_ylim(-0.1, 1.3); a.set_xlim(-1, 64)
        a.set_ylabel("slope b$_i$", fontsize=FS)
        a.set_title(f"board {b}", fontsize=FS)
        a.tick_params(labelsize=FS - 2)
        a.legend(fontsize=FS - 4, loc="lower left")
    axes[-1].set_xlabel("channel", fontsize=FS)
    fig.tight_layout()
    save(fig, "bench", OUT, "compare_slopes")

    # table of the components, first run with HoldDelay 0 of each day
    rows = []
    for day, lab, c, ls in DAYS:
        nc = full[day][0]
        x = nc[(nc.hold == 0) & (nc.tp == 25) & (nc.run == nc["first"])]
        for r in x.sort_values(["board", "chip"]).itertuples():
            rows.append((r.board, r.chip, day, r.run, r.hg_total, r.hg_coherent, r.hg_incoherent,
                         r.hg_coherent**2 / r.hg_total**2))
    t = pd.DataFrame(rows, columns=["board", "chip", "day", "run", "tot", "coh", "inc", "frac"])
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"board & channels & adapter & run & $\sigma_\mathrm{tot}$ & $\sigma_\mathrm{coh}$ & "
             r"$\sigma_\mathrm{inc}$ & $\sigma_\mathrm{coh}^2/\sigma_\mathrm{tot}^2$ \\", r"\midrule"]
    for (b, chip), g in t.groupby(["board", "chip"]):
        if not (g.day == "2026-10-22").any():
            lines.append(f"{b} & {32 * chip}--{32 * chip + 31} & green & -- & \\multicolumn{{4}}{{c}}"
                         "{not read (F7)} \\\\")
        for r in g.sort_values("day").itertuples():
            adapter = "green" if r.day == "2026-10-22" else "red"
            lines.append(f"{b} & {32 * chip}--{32 * chip + 31} & {adapter} & {r.run} & {r.tot:.1f} & "
                         f"{r.coh:.1f} & {r.inc:.1f} & {r.frac:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = results_dir("bench", OUT) / "compare_components.tex"
    path.write_text("\n".join(lines) + "\n")
    print("written", path)

    # jumps of board 0 and correlations of the common modes
    for day, *_ in DAYS:
        nc, cm = full[day][0], full[day][3]
        j = nc[nc.board == 0].pivot_table(index="run", columns="chip", values="jumps")
        print(day, "jumps of board 0 per run and chip:", j.astype(int).to_dict("index"))
        c0 = cm[cm.hold == 0].copy()
        c0["pair"] = [f"{a}{ca}-{b}{cb}" for a, ca, b, cb in
                      zip(c0.board_a, c0.chip_a, c0.board_b, c0.chip_b)]
        print(day, "CM correlations, HoldDelay 0:",
              c0.pivot_table(index="pair", columns="tp", values="correlation").round(2).to_dict())


if __name__ == "__main__":
    main()
