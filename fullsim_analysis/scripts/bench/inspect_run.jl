# First look at a day of bench data from the PET CAEN DAQ: what each run and the hold-delay
# scan contain, board by board, before any study. The day folder is benchdata/<YYYY-MM-DD>/
# with one subdirectory per DAQ run (run-N/run_N.0000.h5 and manifest.json) or scan
# (hold-scan-N/hold-delay.h5); the staircases are read by fullsim_analysis/plots/staircase.py.
# Values are raw ADC counts; 8191 is the full scale of the 13-bit converter and 0 its floor.
#
# Boards: the manifest lists each board with its chain, node and firmware revision. The
# service records carry no board field; they come in groups of one record per board, in
# board order, which is how they are assigned (checked against the detector temperatures
# written in the logbook of 22 Oct 2026).
#
# Writes fullsim_analysis/results/bench/<day>/:
#   runs.csv       one row per run and board: events, triggers and lost triggers of the run
#                  (manifest), board, chain, node, firmware, the settings of that board in
#                  the stored configuration, the medians of its service records
#                  (temperatures in C, HV in V and A), and the fractions of its measurements
#                  at full scale and at zero;
#   channels.csv   one row per run, board and channel: number of measurements, mean, rms,
#                  minimum, maximum, fraction at full scale and fraction at zero of the high
#                  and low gain, fraction with the discriminator set;
#   hold_scan.csv  one row per point and channel of each hold-delay scan (board 0): requested
#                  and applied delay, events, events missing the channel, mean and rms of the
#                  high gain from the binned spectrum.
# and prints a summary per run and board and the warnings: channels at full scale, channels
# at 0 in every event (not read), channels often at 0 (a pedestal clipped by the ADC floor),
# channels that do not see their SiPM (NO_SIPM_RATIO), settings that differ between runs or boards,
# missing channels.
#   julia --project=. fullsim_analysis/scripts/bench/inspect_run.jl benchdata/<YYYY-MM-DD>
include(joinpath(@__DIR__, "..", "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using Statistics, Printf

const FULL_SCALE = 8191
const FLAG_FRACTION = 0.01   # a channel with more than this fraction at 0 or 8191 is flagged
# A SiPM at the input adds capacitance and raises the noise of the preamplifier, most visibly
# with the longest shaping: a channel whose high-gain rms is below NO_SIPM_RATIO times the
# median of its board, in the runs with the longest HG shaping and HoldDelay 0, is flagged as
# not seeing its SiPM (bad contact). The cut is deliberately loose: on board 0 (22 Oct 2026,
# 87.5 ns) the six channels without avalanches in the staircases of 21 Oct are at 0.82-0.87
# of the median and channels with avalanches go down to 0.89, too close to cut between them;
# the distribution of the ratio is drawn by fullsim_analysis/plots/bench_noise.py.
const NO_SIPM_RATIO = 0.6
const SETTINGS = ["HG_ShapingTime", "LG_ShapingTime", "HoldDelay", "PeakDetectorMode",
                  "HV_Vbias", "HV_IndivAdj", "HV_Adjust_Range", "PtrgPeriod", "BunchTrgSource",
                  "HG_Gain", "LG_Gain", "TD_CoarseThreshold", "QD_CoarseThreshold", "Pedestal"]

day_dir = rstrip(ARGS[1], '/')
day = basename(day_dir)

# the integer value of a "key": "value" pair of the manifest (first occurrence)
manifest_int(text, key) = (m = match(Regex("\"$(key)\":\\s*\"?(-?[0-9]+)"), text);
                           m === nothing ? -1 : parse(Int, m.captures[1]))

# boards of the run: (board, chain, node, firmware) from the topology of the manifest
function manifest_boards(text)
    re = r"\"board\":\s*(\d+),\s*\"chain\":\s*(\d+),\s*\"node\":\s*(\d+),\s*\"product_id\":\s*[^,]+,\s*\"firmware_revision\":\s*(\d+)"
    b = unique([(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]), parse(Int, m[4]))
                for m in eachmatch(re, text)])
    isempty(b) ? [(0, 0, 0, -1)] : sort(b)
end

subdirs(prefix) = sort(filter(d -> startswith(d, prefix) && isdir(joinpath(day_dir, d)),
                              readdir(day_dir)); by = d -> parse(Int, split(d, '-')[end]))

frac(v, x) = isempty(v) ? NaN : count(==(x), v) / length(v)

# channel list as ranges: [0, 1, 2, 5] -> "0-2, 5"
function ranges(c)
    isempty(c) && return ""
    out = String[]; a = b = c[1]
    for x in c[2:end]
        x == b + 1 ? (b = x) : (push!(out, a == b ? "$a" : "$a-$b"); a = b = x)
    end
    push!(out, a == b ? "$a" : "$a-$b")
    join(out, ", ")
end
stat(f, v) = isempty(v) ? NaN : f(v)

warnings = String[]
RCOLS = ["run", "events", "triggers", "lost_triggers", "board", "chain", "node", "firmware",
         SETTINGS..., "T_detector", "T_board", "T_fpga", "T_hv", "hv_v", "hv_i",
         "hg_full", "lg_full", "hg_zero", "lg_zero"]
CCOLS = ["run", "board", "chain", "node", "channel", "n",
         "hg_mean", "hg_rms", "hg_min", "hg_max", "hg_full", "hg_zero",
         "lg_mean", "lg_rms", "lg_min", "lg_max", "lg_full", "lg_zero", "discriminator"]
runs = Dict{String,Vector}(k => [] for k in RCOLS)
chans = Dict{String,Vector}(k => [] for k in CCOLS)
summary = String[]

for d in subdirs("run-")
    n = parse(Int, split(d, '-')[end])
    r = read_daq_run(joinpath(day_dir, d, "run_$(n).0000.h5"))
    man = read(joinpath(day_dir, d, "manifest.json"), String)
    p, o, s = r.config.params, r.obs, r.service
    boards = manifest_boards(man)
    nb = length(boards)
    for (ib, (b, ch, nd, fw)) in enumerate(boards)
        rows = [i for i in eachindex(s.timestamp) if (i - 1) % nb == ib - 1 && s.validity[i] != 0]
        med(v) = isempty(rows) ? NaN : median(v[rows])
        push!(runs["run"], n); push!(runs["events"], r.nevents)
        push!(runs["triggers"], manifest_int(man, "trigger_count"))
        push!(runs["lost_triggers"], manifest_int(man, "lost_trigger_count"))
        push!(runs["board"], b); push!(runs["chain"], ch); push!(runs["node"], nd)
        push!(runs["firmware"], fw)
        for k in SETTINGS
            v = daq_param(p, k; board = b)
            push!(runs[k], v === nothing ? "" : v)
        end
        push!(runs["T_detector"], med(s.detector_t)); push!(runs["T_board"], med(s.board_t))
        push!(runs["T_fpga"], med(s.fpga_t)); push!(runs["T_hv"], med(s.hv_t))
        push!(runs["hv_v"], med(s.hv_v)); push!(runs["hv_i"], med(s.hv_i))
        hg = daq_channel_values(r; gain = :hg, chain = ch, node = nd)
        lg = daq_channel_values(r; gain = :lg, chain = ch, node = nd)
        disc = [Bool[] for _ in 1:64]
        for i in eachindex(o.channel)
            (o.chain[i] == ch && o.node[i] == nd) && push!(disc[o.channel[i] + 1], o.discriminator[i])
        end
        for c in 1:64
            push!(chans["run"], n); push!(chans["board"], b); push!(chans["chain"], ch)
            push!(chans["node"], nd); push!(chans["channel"], c - 1); push!(chans["n"], length(hg[c]))
            for (g, v) in (("hg", hg[c]), ("lg", lg[c]))
                push!(chans["$(g)_mean"], stat(mean, v)); push!(chans["$(g)_rms"], stat(std, v))
                push!(chans["$(g)_min"], stat(minimum, v)); push!(chans["$(g)_max"], stat(maximum, v))
                push!(chans["$(g)_full"], frac(v, FULL_SCALE)); push!(chans["$(g)_zero"], frac(v, 0))
            end
            push!(chans["discriminator"], isempty(disc[c]) ? NaN : mean(disc[c]))
        end
        allhg, alllg = reduce(vcat, hg), reduce(vcat, lg)
        push!(runs["hg_full"], frac(allhg, FULL_SCALE)); push!(runs["lg_full"], frac(alllg, FULL_SCALE))
        push!(runs["hg_zero"], frac(allhg, 0)); push!(runs["lg_zero"], frac(alllg, 0))
        tag = "run $n board $b ($ch,$nd)"
        missing_ch = findall(isempty, hg) .- 1
        isempty(missing_ch) || push!(warnings, "$tag: channels without high gain $(missing_ch)")
        for (g, v, all) in (("high", hg, allhg), ("low", lg, alllg))
            nfull = findall(c -> frac(v[c], FULL_SCALE) > FLAG_FRACTION, 1:64) .- 1
            dead = findall(c -> frac(v[c], 0) == 1, 1:64) .- 1
            nzero = setdiff(findall(c -> frac(v[c], 0) > FLAG_FRACTION, 1:64) .- 1, dead)
            isempty(nfull) || push!(warnings, @sprintf("%s: %s gain at full scale in %d channels (%.1f%% of all values): %s",
                                                       tag, g, length(nfull), 100 * frac(all, FULL_SCALE), ranges(nfull)))
            isempty(dead) || push!(warnings, "$tag: $g gain is 0 in every event in $(length(dead)) channels: $(ranges(dead))")
            isempty(nzero) || push!(warnings, @sprintf("%s: %s gain at 0 in more than %.0f%% of the events (pedestal clipped by the ADC floor?) in %d channels: %s",
                                                       tag, g, 100 * FLAG_FRACTION, length(nzero), ranges(nzero)))
        end
        live = [c for c in 1:64 if !isempty(hg[c]) && frac(hg[c], 0) < 1]
        m = isempty(live) ? NaN : median(median.(hg[live]))
        sd = isempty(live) ? NaN : median(std.(hg[live]))
        push!(summary, @sprintf("  run %2d  board %d (%d,%d)  %6d events  HG %-7s Hold %-6s Vbias %-5s Ped %-3s  T det %.1f board %.1f C  I %.1e A  HG ped %5.0f rms %5.2f  full %.0f%% zero %.1f%%",
            n, b, ch, nd, length(hg[1]), runs["HG_ShapingTime"][end], runs["HoldDelay"][end],
            runs["HV_Vbias"][end], runs["Pedestal"][end], runs["T_detector"][end], runs["T_board"][end],
            runs["hv_i"][end], m, sd, 100 * runs["hg_full"][end], 100 * runs["hg_zero"][end]))
    end
end

if !isempty(runs["run"])
    for k in ("HV_Vbias", "HV_IndivAdj", "PtrgPeriod", "PeakDetectorMode", "HG_Gain", "LG_Gain", "Pedestal")
        u = unique(runs[k])
        length(u) > 1 && push!(warnings, "setting $k differs between runs or boards: $(u)")
    end
    fws = unique(zip(runs["board"], runs["firmware"]))
    length(unique(last.(fws))) > 1 && push!(warnings, "firmware revisions differ between boards: " *
        join(["board $b 0x$(string(f, base = 16))" for (b, f) in sort(fws)], ", "))
    shaping(v) = (x = tryparse(Float64, first(split(string(v)))); x === nothing ? 0.0 : x)
    longest = maximum(shaping.(runs["HG_ShapingTime"]))
    sel = unique(runs["run"][i] for i in eachindex(runs["run"])
                 if shaping(runs["HG_ShapingTime"][i]) == longest && startswith(string(runs["HoldDelay"][i]), "0"))
    for n in sel, b in unique(runs["board"])
        idx = [i for i in eachindex(chans["run"]) if chans["run"][i] == n && chans["board"][i] == b &&
               chans["hg_zero"][i] < 1 && !isnan(chans["hg_rms"][i])]
        isempty(idx) && continue
        m = median(chans["hg_rms"][idx])
        low = [chans["channel"][i] for i in idx if chans["hg_rms"][i] < NO_SIPM_RATIO * m]
        isempty(low) || push!(warnings, @sprintf("run %d board %d (HG %g ns): high-gain rms below %.0f%% of the board median (%.2f) in channels %s: SiPM not seen",
                                                 n, b, longest, 100 * NO_SIPM_RATIO, m, ranges(low)))
    end
    println(write_table("bench", day, "runs", [k => runs[k] for k in RCOLS]))
    println(write_table("bench", day, "channels", [k => chans[k] for k in CCOLS]))
end

hold = Dict{String,Vector}(k => [] for k in ["scan", "point", "delay_ns", "effective_delay_ns",
        "events", "channel", "missing", "hg_mean", "hg_rms"])
for d in subdirs("hold-scan-")
    n = parse(Int, split(d, '-')[end])
    h = read_daq_hold_delay(joinpath(day_dir, d, "hold-delay.h5"))
    x = h.bin_low .+ (h.bin_low[2] - h.bin_low[1]) / 2
    for pt in eachindex(h.delay_ns), c in 1:64
        w = h.counts[:, c, pt]; nw = sum(w)
        m = nw > 0 ? sum(w .* x) / nw : NaN
        sd = nw > 1 ? sqrt(sum(w .* (x .- m) .^ 2) / (nw - 1)) : NaN
        push!(hold["scan"], n); push!(hold["point"], pt - 1); push!(hold["delay_ns"], h.delay_ns[pt])
        push!(hold["effective_delay_ns"], h.effective_delay_ns[pt]); push!(hold["events"], h.events[pt])
        push!(hold["channel"], c - 1); push!(hold["missing"], h.missing[c, pt])
        push!(hold["hg_mean"], m); push!(hold["hg_rms"], sd)
    end
    top = h.bin_low[end]
    frac_top = sum(h.counts[end, :, :]) / max(sum(h.counts), 1)
    frac_top > 0.5 && push!(warnings, @sprintf("hold scan %d: %.1f%% of the entries in the last bin (from %d); the high gain is at full scale",
                                               n, 100 * frac_top, top))
    sum(h.missing) > 0 && push!(warnings, "hold scan $n: $(sum(h.missing)) missing channel readings")
end
isempty(hold["scan"]) || println(write_table("bench", day, "hold_scan",
    [k => hold[k] for k in ["scan", "point", "delay_ns", "effective_delay_ns", "events",
                            "channel", "missing", "hg_mean", "hg_rms"]]))

println("\nRuns of $(day) (HG ped and rms: medians over the channels that are not 0 in every event, ADC counts):")
foreach(println, summary)
println(isempty(warnings) ? "\nNo warnings." : "\nWarnings:")
foreach(w -> println("  ", w), warnings)
