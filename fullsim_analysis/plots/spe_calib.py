"""Single-photoelectron calibration with the periodic trigger (menu 3) for a day of bench data.
Reads fullsim_analysis/results/bench/<day>/pedestal_spectra.csv (pedestal_spectra.jl: the
high-gain spectrum of every channel of the runs listed in RUNS and of the pedestal run
PED_RUN), runs.csv and channels.csv (inspect_run.jl), noise_components.csv and
channel_noise.csv (noise_components.jl), hold_scan_knees.csv (hold_scan.py) and
staircase_<N>.csv, staircase_<N>_channels.csv (staircase.py).

Method, per channel and run. The spectrum between 50 counts below the pedestal (the mode of the
spectrum of the pedestal run) and NPEAKS gains above it is fitted, in bins of REBIN counts, with
a comb of NPEAKS Gaussians at p + n g (n = 0, 1, ...), of width sqrt(s0^2 + n s1^2) and free
amplitudes: p is the 0-pe peak, g the gain in counts per photoelectron, s0 the width of the
0-pe peak and s1 the extra width per photoelectron. The seed of g is the 1-2 pe spacing of the
peaks found at the lowest bias (local maxima of the smoothed spectrum), then the gain of the
previous bias scaled by the overvoltage, then the 25 ns gain for the other shapings. The ladder
is resolved when the 1-pe peak is narrower than the gain over SEP_MIN. The breakdown voltage of
a channel is the intercept of the line through its gain at the three biases at 25 ns, when the
three are resolved. Blind channels (BLIND) are left out.

The photoelectron spacing of a staircase, in DAC units, is the median distance between the
first consecutive minima of the derivative of the log rate against the threshold (the drops of
the staircase), above the edge of the noise and while the rate is above 300 Hz.

Writes fullsim_analysis/results/bench/<day>/:
    spe_gains.csv          one row per run, board (bench number) and channel: bias, shaping,
                           pedestal, the 0, 1, 2 pe peaks found, the fitted 0-pe position, gain,
                           widths, the separation g / width of the 1-pe peak, chi2 per degree
                           of freedom, and resolved (1 when the ladder is resolved);
    spe_breakdown.csv      one row per board and channel with the three biases resolved: gains,
                           slope (counts per volt) and breakdown voltage;
    spe_gains_table.tex    per board, bias and shaping: channels resolved, median and range of
                           the gain, median width of the 1-pe peak;
    spe_breakdown_table.tex per board: median and range of the breakdown voltage and slope;
    spe_noise_pe.tex       per board and shaping: the pedestal noise of menu 1 (total,
                           coherent, incoherent) in counts and in photoelectrons at 43 V;
    spe_staircases.tex     scans 113-115: bias, edge of the noise and photoelectron spacing in
                           DAC units (median over the channels);
    spe_zeros.tex          runs 108-112: fraction of high-gain entries at 0 per board;
    spe_spectra.png        one channel: spectra and fits at the three biases (25 ns) and at the
                           three shapings (43 V);
    spe_gains.png          gain of every channel of each board at the three biases, 25 ns;
    spe_breakdown.png      gain against bias for a few channels, and the distribution of the
                           breakdown voltage per board;
    spe_shaping.png        gain against the shaping at 43 V, per board, with the menu-2 points;
    spe_noise_pe.png       pedestal noise of every channel in photoelectrons at 43 V, 25 ns.

    python3 fullsim_analysis/plots/spe_calib.py <YYYY-MM-DD>
"""
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from common import plt, results_dir, save, board_name

RUNS = {112: (41.0, 25.0), 111: (42.0, 25.0), 108: (43.0, 25.0), 109: (43.0, 50.0),
        110: (43.0, 87.5)}                      # run -> (bias at the SiPM in V, HG shaping in ns)
ORDER = (112, 111, 108, 109, 110)
PED_RUN = 95                                    # 25 ns, HoldDelay 200 ns, SiPMs below breakdown
NOISE_RUNS = {25.0: 91, 50.0: 92, 87.5: 93}     # menu 1, HoldDelay 0
BLIND = {0: [61, 63], 2: [3, 59, 61, 63]}       # DAQ board -> channels without SiPM signal
STAIRCASES = {113: 43.0, 114: 41.0, 115: 41.0}  # scan -> bias
MENU2_BOARD = 0                                 # DAQ board of the hold-delay scans
SHOW = (1, 20)                                  # DAQ board, channel of the spectra figure
NADC = 4096
NPEAKS = 6
REBIN = 4
SEP_MIN = 2.0      # resolved: gain / width of the 1-pe peak at least this
SEED = {112: 90.0, 111: 130.0, 108: 170.0}      # gains when no better seed exists
FS = 16


def histogram(d, run, board, channel):
    x = d[(d.run == run) & (d.board == board) & (d.channel == channel)]
    h = np.zeros(NADC)
    h[x.adc.values] = x["count"].values
    return h


def low_peaks(h, ped, sigma=4.0, min_sep=35):
    """0, 1, 2 pe peaks: local maxima of the smoothed spectrum from the pedestal up."""
    s = gaussian_filter1d(h, sigma)
    pk, _ = find_peaks(s, prominence=0.02 * s.max(), distance=min_sep)
    pk = pk[pk >= ped - 25]
    out = list(pk[:3].astype(float))
    return out + [np.nan] * (3 - len(out))


def comb(x, p, g, s0, s1, *amp):
    y = np.zeros_like(x)
    for n, a in enumerate(amp):
        s = np.sqrt(s0 ** 2 + n * s1 ** 2)
        y = y + a * np.exp(-0.5 * ((x - p - n * g) / s) ** 2)
    return y


def fit_comb(h, ped, g0):
    """Fit of the photoelectron ladder; returns (p, g, s0, s1, amplitudes, chi2/ndf) or None."""
    a, b = int(max(ped - 50, 0)), int(min(ped + (NPEAKS - 0.5) * g0, NADC - 1))
    x = np.arange(a, b, 1.0); y = h[a:b]
    m = len(x) // REBIN
    x = x[:REBIN * m].reshape(m, REBIN).mean(1); y = y[:REBIN * m].reshape(m, REBIN).sum(1)
    sm = gaussian_filter1d(y, 2.0)
    amp0 = [max(sm[min(int((ped + n * g0 - a) / REBIN), m - 1)], 1.0) for n in range(NPEAKS)]
    p0 = [ped, g0, 12.0, 20.0] + amp0
    lo = [ped - 40, 0.7 * g0, 3.0, 0.0] + [0.0] * NPEAKS
    hi = [ped + 40, 1.4 * g0, 60.0, 120.0] + [np.inf] * NPEAKS
    try:
        popt, _ = curve_fit(comb, x, y, p0=p0, bounds=(lo, hi), sigma=np.sqrt(y + 1),
                            maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    chi2 = np.sum((y - comb(x, *popt)) ** 2 / (y + 1)) / (len(x) - len(popt))
    return popt[0], popt[1], popt[2], popt[3], popt[4:], chi2


def staircase_spacing(thr, rate, min_rate=300.0, nsteps=5):
    """Median distance between the first nsteps consecutive drops of a staircase (DAC units),
    over the thresholds where the rate is above min_rate (0.5 s per point)."""
    keep = rate > min_rate
    thr, rate = thr[keep], rate[keep]
    if len(thr) < 20:
        return np.nan
    y = gaussian_filter1d(np.gradient(np.log10(rate)), 1.0)
    mins, _ = find_peaks(-y, distance=6, prominence=0.6 * np.std(y))
    if len(mins) < 3:
        return np.nan
    return float(np.median(np.diff(thr[mins])[:nsteps]))


def main(day):
    d = results_dir("bench", day)
    sp = pd.read_csv(d / "pedestal_spectra.csv")
    runs = pd.read_csv(d / "runs.csv")
    boards = sorted(sp.board.unique())
    tp_of = {25.0: "25 ns", 50.0: "50 ns", 87.5: "87.5 ns"}

    # fits
    rows, fits = [], {}
    for board in boards:
        for c in range(64):
            blind = c in BLIND.get(board, [])
            ped = float(np.argmax(gaussian_filter1d(histogram(sp, PED_RUN, board, c), 2.0)))
            gain = {}
            for run in ORDER:
                bias, tp = RUNS[run]
                h = histogram(sp, run, board, c)
                p0, p1, p2 = low_peaks(h, ped)
                g_pk = (p2 - p1) if np.isfinite(p2) else (p1 - p0)
                if run == 112:
                    g0 = g_pk if np.isfinite(g_pk) and 50 < g_pk < 200 else SEED[run]
                elif run == 111:
                    g0 = gain.get(112, SEED[111] * 3 / 4) * 4 / 3
                elif run == 108:
                    g0 = gain.get(111, SEED[108] * 4 / 5) * 5 / 4
                else:
                    g0 = gain.get(108, SEED[108])
                res = None if blind else fit_comb(h, ped, g0)
                if res is None:
                    p, g, s0, s1, chi2, sep, resolved = (np.nan,) * 6 + (0,)
                else:
                    p, g, s0, s1, amp, chi2 = res
                    sep = g / np.sqrt(s0 ** 2 + s1 ** 2)
                    resolved = int(sep >= SEP_MIN)
                    fits[(run, board, c)] = (p, g, s0, s1, amp)
                if resolved:
                    gain[run] = g
                rows.append((run, board_name(day, board), c, bias, tp, ped, p0, p1, p2, g_pk, p, g,
                             s0, s1, sep, chi2, resolved, int(blind)))
    gt = pd.DataFrame(rows, columns=["run", "board", "channel", "bias", "shaping", "pedestal",
                                     "p0", "p1", "p2", "gain_peaks", "p_fit", "gain", "s0", "s1",
                                     "sep", "chi2", "resolved", "blind"])
    gt.to_csv(d / "spe_gains.csv", index=False)
    ok = gt[(gt.blind == 0) & (gt.resolved == 1)]
    for run in ORDER:
        x = gt[(gt.run == run) & (gt.blind == 0)]
        print(f"run {run}: ladder resolved in {x.resolved.sum()} of {len(x)} channels")

    # breakdown voltage from the three biases at 25 ns
    rows = []
    for (board, c), g in ok[ok.shaping == 25.0].groupby(["board", "channel"]):
        if len(g) < 3:
            continue
        g = g.sort_values("bias")
        a, b = np.polyfit(g.bias, g.gain, 1)
        rows.append((board, c, *g.gain.values, a, -b / a))
    bd = pd.DataFrame(rows, columns=["board", "channel", "gain41", "gain42", "gain43", "slope",
                                     "vbd"])
    bd.to_csv(d / "spe_breakdown.csv", index=False)

    # table of gains
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"board & bias & shaping & channels & gain (counts per pe) & 1-pe width \\",
             r"\midrule"]
    print("gain per board: median (min--max), rms / median, median width of the 1-pe peak")
    for board, g in ok.groupby("board"):
        for run in ORDER:
            x = g[g.run == run]
            bias, tp = RUNS[run]
            w = np.sqrt(x.s0 ** 2 + x.s1 ** 2).median()
            lines.append(f"{board} & {bias:g} V & {tp_of[tp]} & {len(x)} & "
                         f"{x.gain.median():.0f} ({x.gain.min():.0f}--{x.gain.max():.0f}) & {w:.0f} \\\\")
            print(f"  board {board} {bias:g} V {tp_of[tp]}: {x.gain.median():.1f} ({x.gain.min():.1f}--"
                  f"{x.gain.max():.1f}), rms/median {x.gain.std() / x.gain.median():.3f}, "
                  f"width {w:.1f}, n {len(x)}")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (d / "spe_gains_table.tex").write_text("\n".join(lines) + "\n")

    # table of breakdown voltages
    temp = runs[runs.run == 108].set_index("board").T_detector
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"board & channels & $V_\mathrm{bd}$ (V) & slope (counts per pe per V) & "
             r"$T_\mathrm{det}$ ($^\circ$C) \\", r"\midrule"]
    print("breakdown per board: median (min--max)")
    for board, g in bd.groupby("board"):
        daq = [b for b in boards if board_name(day, b) == board][0]
        lines.append(f"{board} & {len(g)} & {g.vbd.median():.2f} ({g.vbd.min():.2f}--{g.vbd.max():.2f}) & "
                     f"{g.slope.median():.1f} ({g.slope.min():.1f}--{g.slope.max():.1f}) & "
                     f"{temp[daq]:.1f} \\\\")
        print(f"  board {board}: Vbd {g.vbd.median():.2f} ({g.vbd.min():.2f}--{g.vbd.max():.2f}), "
              f"rms {g.vbd.std():.2f}, slope {g.slope.median():.1f} ({g.slope.min():.1f}--"
              f"{g.slope.max():.1f}), n {len(g)}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (d / "spe_breakdown_table.tex").write_text("\n".join(lines) + "\n")

    # noise of menu 1 in photoelectrons
    nc = pd.read_csv(d / "noise_components.csv")
    cn = pd.read_csv(d / "channel_noise.csv")
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"board & shaping & gain & $\sigma_\mathrm{tot}$ & $\sigma_\mathrm{coh}$ & "
             r"$\sigma_\mathrm{inc}$ & $\sigma_\mathrm{tot}$ (pe) & $\sigma_\mathrm{coh}$ (pe) & "
             r"$\sigma_\mathrm{inc}$ (pe) \\", r"\midrule"]
    print("noise of menu 1 in pe at 43 V")
    noise_pe = []
    for daq in boards:
        board = board_name(day, daq)
        for tp, run_noise in NOISE_RUNS.items():
            run = [r for r, (b, t) in RUNS.items() if b == 43.0 and t == tp][0]
            x = ok[(ok.board == board) & (ok.run == run)]
            gmed, n = x.gain.median(), len(x)
            z = nc[(nc.run == run_noise) & (nc.board == daq)]
            tot, coh, inc = z.hg_total.mean(), z.hg_coherent.mean(), z.hg_incoherent.mean()
            cells = (f"{tot / gmed:.3f} & {coh / gmed:.3f} & {inc / gmed:.3f}" if n >= 10
                     else r"\multicolumn{3}{c}{ladder not resolved}")
            gcell = f"{gmed:.0f}" if n >= 10 else "--"
            lines.append(f"{board} & {tp_of[tp]} & {gcell} & {tot:.1f} & {coh:.1f} & {inc:.1f} & "
                         f"{cells} \\\\")
            print(f"  board {board} {tp_of[tp]}: gain {gmed:.0f} (n {n}), total {tot / gmed:.3f} "
                  f"coherent {coh / gmed:.3f} incoherent {inc / gmed:.3f} pe")
            if tp == 25.0:
                y = cn[(cn.run == run_noise) & (cn.board == daq)][["channel", "hg_rms"]]
                y = y.merge(x[["channel", "gain"]], on="channel")
                noise_pe.append((board, y.channel.values, (y.hg_rms / y.gain).values))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (d / "spe_noise_pe.tex").write_text("\n".join(lines) + "\n")

    # staircases at 5 and 3 V: edge and photoelectron spacing in DAC units
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"scan & bias & edge (DAC) & pe spacing (DAC) & rate 30 DAC above the edge (Hz) \\",
             r"\midrule"]
    print("staircases")
    for scan, bias in STAIRCASES.items():
        st = pd.read_csv(d / f"staircase_{scan}.csv")
        chs = pd.read_csv(d / f"staircase_{scan}_channels.csv")
        blind = BLIND.get(MENU2_BOARD, [])
        spacing = []
        for c, g in st[(st.valid == 1) & (~st.channel.isin(blind))].groupby("channel"):
            edge = chs[chs.channel == c].edge.iloc[0]
            g = g[(g.threshold > edge + 2) & (g.rate_cps > 0)].sort_values("threshold")
            if len(g) >= 30:
                s = staircase_spacing(g.threshold.values.astype(float), g.rate_cps.values)
                if np.isfinite(s):
                    spacing.append(s)
        spacing = np.array(spacing)
        good = chs[~chs.channel.isin(blind)]
        lines.append(f"{scan} & {bias:g} V & {good.edge.median():.0f} ({good.edge.min():.0f}--"
                     f"{good.edge.max():.0f}) & {np.median(spacing):.0f} ({spacing.min():.0f}--"
                     f"{spacing.max():.0f}) & {good.rate_edge_plus_30.median():.1e} \\\\")
        print(f"  scan {scan} {bias:g} V: edge {good.edge.median():.0f} ({good.edge.min():.0f}--"
              f"{good.edge.max():.0f}), spacing {np.median(spacing):.1f} ({spacing.min():.0f}--"
              f"{spacing.max():.0f}) over {len(spacing)} channels, rate +30 "
              f"{good.rate_edge_plus_30.median():.2e}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (d / "spe_staircases.tex").write_text("\n".join(lines) + "\n")

    # entries at zero (F12)
    ch = pd.read_csv(d / "channels.csv")
    names = [board_name(day, b) for b in boards]
    lines = [r"\begin{tabular}{lcc" + "c" * len(boards) + "}", r"\toprule",
             r"run & bias & shaping & " + " & ".join(f"board {n}" for n in names) + r" \\",
             r"\midrule"]
    print("entries at 0 (median over the channels, %)")
    for run in (108, 109, 110, 111, 112):
        bias, tp = RUNS[run]
        z = [100 * ch[(ch.run == run) & (ch.board == b)].hg_zero.median() for b in boards]
        lines.append(f"{run} & {bias:g} V & {tp_of[tp]} & " + " & ".join(f"{v:.1f}" for v in z) + r" \\")
        print(f"  run {run} {bias:g} V {tp_of[tp]}: " + ", ".join(f"{v:.1f}" for v in z))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (d / "spe_zeros.tex").write_text("\n".join(lines) + "\n")

    # cross-check with menu 2
    kn = pd.read_csv(d / "hold_scan_knees.csv")
    b2 = board_name(day, MENU2_BOARD)
    print(f"menu 2 against menu 3, board {b2}, 43 V:")
    for r in kn.itertuples():
        run = [k for k, (b, t) in RUNS.items() if b == 43.0 and t == r.shaping][0]
        g = gt[(gt.board == b2) & (gt.run == run) & (gt.channel == r.channel)].iloc[0]
        print(f"  channel {r.channel} {r.shaping:g} ns: hold scan {r.gain_plateau:.0f}, periodic "
              f"trigger {g.gain:.0f} ({'resolved' if g.resolved else 'not resolved'})")

    # figure: spectra and fits of one channel
    daq, c = SHOW
    board = board_name(day, daq)
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for k, run in enumerate((112, 111, 108, 108, 109, 110)):
        a = ax[k // 3, k % 3]
        bias, tp = RUNS[run]
        h = histogram(sp, run, daq, c)
        row = gt[(gt.run == run) & (gt.board == board) & (gt.channel == c)].iloc[0]
        hb = h[:NADC // REBIN * REBIN].reshape(-1, REBIN).sum(1)
        xb = np.arange(len(hb)) * REBIN + REBIN / 2
        a.step(xb, hb, where="mid", lw=1.0, c="k")
        if (run, daq, c) in fits:
            p, g, s0, s1, amp = fits[(run, daq, c)]
            xx = np.arange(max(p - 60, 0), p + (NPEAKS - 0.5) * g)
            a.plot(xx, comb(xx, p, g, s0, s1, *amp), c="tab:red", lw=2)
            for n in range(NPEAKS):
                a.axvline(p + n * g, c="tab:red", ls=":", lw=1)
            gtxt = f"gain {g:.0f}, 1-pe width {np.sqrt(s0**2 + s1**2):.0f}" + \
                   ("" if row.resolved else " (not resolved)")
            a.set_xlim(max(p - 60, 0), p + (NPEAKS - 0.5) * g)
        else:
            gtxt = "no fit"
            a.set_xlim(max(row.pedestal - 60, 0), row.pedestal + 1000)
        a.set_yscale("log"); a.set_ylim(1, hb.max() * 2)
        a.set_title(f"board {board} channel {c}: {bias:g} V, {tp_of[tp]}\n{gtxt}", fontsize=FS - 1)
        a.set_xlabel("high gain (ADC counts)", fontsize=FS - 2)
        a.tick_params(labelsize=FS - 3)
    ax[0, 0].set_ylabel(f"events per {REBIN} counts", fontsize=FS - 2)
    ax[1, 0].set_ylabel(f"events per {REBIN} counts", fontsize=FS - 2)
    fig.suptitle("red: comb of Gaussians fitted to the ladder; dotted: n photoelectrons",
                 fontsize=FS)
    fig.tight_layout()
    save(fig, "bench", day, "spe_spectra")

    # figure: gain of every channel at the three biases
    bl = sorted(ok.board.unique())
    fig, ax = plt.subplots(len(bl), 1, figsize=(14, 3.6 * len(bl)), sharex=True)
    for a, board in zip(ax, bl):
        for run, mk in ((108, "o"), (111, "s"), (112, "^")):
            g = ok[(ok.board == board) & (ok.run == run)]
            a.plot(g.channel, g.gain, mk, ms=6, label=f"{RUNS[run][0]:g} V")
        daq = [b for b in boards if board_name(day, b) == board][0]
        for cc in BLIND.get(daq, []):
            a.axvline(cc, c="tab:red", ls=":", lw=1)
        a.set_ylabel("gain (counts per pe)", fontsize=FS - 2)
        a.set_title(f"board {board}, 25 ns (channels with the ladder resolved; dotted: no SiPM "
                    "signal)", fontsize=FS - 1)
        a.set_ylim(0, 260); a.set_xlim(-1, 64)
        a.tick_params(labelsize=FS - 3)
        a.legend(fontsize=FS - 4, ncol=3, loc="lower right")
    ax[-1].set_xlabel("channel", fontsize=FS - 2)
    fig.tight_layout()
    save(fig, "bench", day, "spe_gains")

    # figure: gain against bias and breakdown voltage
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for board in bl:
        x = bd[bd.board == board]
        if len(x) == 0:
            continue
        r = x.iloc[len(x) // 2]
        g = ok[(ok.board == board) & (ok.channel == r.channel) & (ok.shaping == 25.0)].sort_values("bias")
        v = np.array([r.vbd, 43.5])
        line, = ax[0].plot(g.bias, g.gain, "o", ms=9, label=f"board {board} channel {int(r.channel)}: "
                           f"$V_\\mathrm{{bd}}$ = {r.vbd:.2f} V")
        ax[0].plot(v, r.slope * (v - r.vbd), "-", c=line.get_color(), lw=1.5)
    ax[0].set_xlabel("bias at the SiPM (V)", fontsize=FS)
    ax[0].set_ylabel("gain (counts per pe)", fontsize=FS)
    ax[0].set_ylim(0, 220)
    ax[0].legend(fontsize=FS - 4); ax[0].tick_params(labelsize=FS - 2)
    ax[0].set_title("25 ns: the line through the three biases", fontsize=FS)
    for board in bl:
        ax[1].hist(bd[bd.board == board].vbd, bins=np.arange(37.5, 40.5, 0.1), histtype="step",
                   lw=2, label=f"board {board}")
    ax[1].set_xlabel("breakdown voltage (V)", fontsize=FS)
    ax[1].set_ylabel("channels per 0.1 V", fontsize=FS)
    ax[1].legend(fontsize=FS - 2); ax[1].tick_params(labelsize=FS - 2)
    fig.tight_layout()
    save(fig, "bench", day, "spe_breakdown")

    # figure: gain against shaping at 43 V
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, board in enumerate(bl):
        for tp, run in ((25.0, 108), (50.0, 109), (87.5, 110)):
            x = ok[(ok.board == board) & (ok.run == run)].gain
            if len(x) < 5:
                continue
            ax.errorbar(tp + 1.5 * (i - 1), x.median(), [[x.median() - x.quantile(0.16)],
                        [x.quantile(0.84) - x.median()]], fmt="o", ms=9, capsize=4,
                        c=f"C{i}", label=f"board {board}, {tp_of[tp]}: {len(x)} channels")
    ax.plot(kn.shaping, kn.gain_plateau, "k*", ms=13, label="hold-delay scans, board 1")
    ax.set_xlabel("HG shaping time (ns)", fontsize=FS)
    ax.set_ylabel("gain at 43 V (counts per pe)", fontsize=FS)
    ax.set_ylim(0, 260); ax.set_xlim(15, 95)
    ax.legend(fontsize=FS - 5, ncol=2); ax.tick_params(labelsize=FS - 2)
    ax.set_title("channels with the ladder resolved: median, bars at the 16th and 84th percentiles",
                 fontsize=FS - 1)
    fig.tight_layout()
    save(fig, "bench", day, "spe_shaping")

    # figure: noise in photoelectrons per channel
    fig, ax = plt.subplots(figsize=(14, 6))
    for board, chs_, npe in noise_pe:
        ax.plot(chs_, npe, "o", ms=6, label=f"board {board}")
    ax.set_xlabel("channel", fontsize=FS)
    ax.set_ylabel("pedestal rms / gain at 43 V (pe)", fontsize=FS)
    ax.set_ylim(0, 0.15); ax.set_xlim(-1, 64)
    ax.legend(fontsize=FS - 2); ax.tick_params(labelsize=FS - 2)
    ax.set_title("25 ns, HoldDelay 0", fontsize=FS)
    fig.tight_layout()
    save(fig, "bench", day, "spe_noise_pe")


if __name__ == "__main__":
    main(sys.argv[1])
