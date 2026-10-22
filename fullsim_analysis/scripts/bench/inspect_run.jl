# First look at a day of bench data from the PET CAEN DAQ: what each run and the hold-delay
# scan contain, before any study. The day folder is benchdata/<YYYY-MM-DD>/ with one
# subdirectory per DAQ run (run-N/run_N.0000.h5 and manifest.json) or scan
# (hold-scan-N/hold-delay.h5); the staircases are read by fullsim_analysis/plots/staircase.py.
# Values are raw ADC counts; 8191 is the full scale of the 13-bit converter.
#
# Writes fullsim_analysis/results/bench/<day>/:
#   runs.csv       one row per run: events, triggers and lost triggers from the manifest,
#                  the settings of the stored configuration, the medians of the service
#                  records (temperatures in C, HV in V and A) and the fraction of
#                  measurements at full scale;
#   channels.csv   one row per run and channel: mean, rms, minimum, maximum and fraction at
#                  full scale of the high and low gain, fraction with the discriminator set;
#   hold_scan.csv  one row per point and channel of each hold-delay scan: requested and
#                  applied delay, events, events missing the channel, mean and rms of the
#                  high gain from the binned spectrum.
# and prints the warnings (channels at full scale, settings that differ between runs,
# missing channels).
#   julia --project=. fullsim_analysis/scripts/bench/inspect_run.jl benchdata/<YYYY-MM-DD>
include(joinpath(@__DIR__, "..", "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using Statistics, Printf

const FULL_SCALE = 8191
const SETTINGS = ["HG_ShapingTime", "LG_ShapingTime", "HoldDelay", "PeakDetectorMode",
                  "HV_Vbias", "HV_IndivAdj", "HV_Adjust_Range", "PtrgPeriod", "BunchTrgSource",
                  "HG_Gain", "LG_Gain", "TD_CoarseThreshold", "QD_CoarseThreshold", "Pedestal"]

day_dir = rstrip(ARGS[1], '/')
day = basename(day_dir)

# the integer value of a "key": "value" pair of the manifest (first occurrence)
manifest_int(text, key) = (m = match(Regex("\"$(key)\":\\s*\"?(-?[0-9]+)"), text);
                           m === nothing ? -1 : parse(Int, m.captures[1]))

subdirs(prefix) = sort(filter(d -> startswith(d, prefix) && isdir(joinpath(day_dir, d)),
                              readdir(day_dir)); by = d -> parse(Int, split(d, '-')[end]))

warnings = String[]
runs = Dict{String,Vector}(k => [] for k in ["run", "events", "triggers", "lost_triggers",
        SETTINGS..., "T_detector", "T_board", "T_fpga", "T_hv", "hv_v", "hv_i", "hg_full", "lg_full"])
chans = Dict{String,Vector}(k => [] for k in ["run", "channel", "n", "hg_mean", "hg_rms", "hg_min",
        "hg_max", "hg_full", "lg_mean", "lg_rms", "lg_min", "lg_max", "lg_full", "discriminator"])

for d in subdirs("run-")
    n = parse(Int, split(d, '-')[end])
    r = read_daq_run(joinpath(day_dir, d, "run_$(n).0000.h5"))
    man = read(joinpath(day_dir, d, "manifest.json"), String)
    p, o, s = r.config.params, r.obs, r.service
    ok = s.validity .!= 0
    med(v) = isempty(v[ok]) ? NaN : median(v[ok])
    push!(runs["run"], n); push!(runs["events"], r.nevents)
    push!(runs["triggers"], manifest_int(man, "trigger_count"))
    push!(runs["lost_triggers"], manifest_int(man, "lost_trigger_count"))
    for k in SETTINGS
        v = daq_param(p, k)
        push!(runs[k], v === nothing ? "" : v)
    end
    push!(runs["T_detector"], med(s.detector_t)); push!(runs["T_board"], med(s.board_t))
    push!(runs["T_fpga"], med(s.fpga_t)); push!(runs["T_hv"], med(s.hv_t))
    push!(runs["hv_v"], med(s.hv_v)); push!(runs["hv_i"], med(s.hv_i))
    hg, lg = daq_channel_values(r; gain = :hg), daq_channel_values(r; gain = :lg)
    disc = [Bool[] for _ in 1:64]
    for i in eachindex(o.channel)
        (o.chain[i] == 0 && o.node[i] == 0) && push!(disc[o.channel[i] + 1], o.discriminator[i])
    end
    full(v) = isempty(v) ? NaN : count(==(FULL_SCALE), v) / length(v)
    stat(f, v) = isempty(v) ? NaN : f(v)
    for c in 1:64
        push!(chans["run"], n); push!(chans["channel"], c - 1); push!(chans["n"], length(hg[c]))
        for (g, v) in (("hg", hg[c]), ("lg", lg[c]))
            push!(chans["$(g)_mean"], stat(mean, v)); push!(chans["$(g)_rms"], stat(std, v))
            push!(chans["$(g)_min"], stat(minimum, v)); push!(chans["$(g)_max"], stat(maximum, v))
            push!(chans["$(g)_full"], full(v))
        end
        push!(chans["discriminator"], isempty(disc[c]) ? NaN : mean(disc[c]))
    end
    allhg, alllg = reduce(vcat, hg), reduce(vcat, lg)
    push!(runs["hg_full"], full(allhg)); push!(runs["lg_full"], full(alllg))
    missing_ch = findall(isempty, hg) .- 1
    isempty(missing_ch) || push!(warnings, "run $n: channels without high gain $(missing_ch)")
    nfull = count(c -> full(hg[c]) > 0.5, 1:64)
    nfull > 0 && push!(warnings, @sprintf("run %d: %d of 64 channels at full scale (%d) in more than half of the events; high gain %.1f%%, low gain %.1f%% of all measurements at full scale",
                                          n, nfull, FULL_SCALE, 100 * full(allhg), 100 * full(alllg)))
end

if !isempty(runs["run"])
    for k in ("HV_Vbias", "HV_IndivAdj", "PtrgPeriod", "PeakDetectorMode", "HG_Gain", "LG_Gain")
        u = unique(runs[k])
        length(u) > 1 && push!(warnings, "setting $k differs between runs: $(u)")
    end
    cols = ["run", "events", "triggers", "lost_triggers", SETTINGS..., "T_detector", "T_board",
            "T_fpga", "T_hv", "hv_v", "hv_i", "hg_full", "lg_full"]
    println(write_table("bench", day, "runs", [k => runs[k] for k in cols]))
    ccols = ["run", "channel", "n", "hg_mean", "hg_rms", "hg_min", "hg_max", "hg_full",
             "lg_mean", "lg_rms", "lg_min", "lg_max", "lg_full", "discriminator"]
    println(write_table("bench", day, "channels", [k => chans[k] for k in ccols]))
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

println("\nRuns of $(day):")
for i in eachindex(runs["run"])
    @printf("  run %2d  %6d events  lost %d  HG %-7s Hold %-6s Vbias %s  T det %.1f board %.1f C  I %.1e A  HG full %.0f%%\n",
            runs["run"][i], runs["events"][i], runs["lost_triggers"][i], runs["HG_ShapingTime"][i],
            runs["HoldDelay"][i], runs["HV_Vbias"][i], runs["T_detector"][i], runs["T_board"][i],
            runs["hv_i"][i], 100 * runs["hg_full"][i])
end
println(isempty(warnings) ? "\nNo warnings." : "\nWarnings:")
foreach(w -> println("  ", w), warnings)
