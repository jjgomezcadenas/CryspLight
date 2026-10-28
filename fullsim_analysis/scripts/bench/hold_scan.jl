# Hold-delay scans of a day of bench data (menu 2): with the SiPMs above breakdown and the
# trigger limited to one channel (the shooting channel), the DAQ scans HoldDelay and stores, for
# every channel and delay, the spectrum of the high gain (bins of 16 ADC codes). The shooting
# channel measures the response of the peak detector to its own dark pulse: the stored value
# rises with the delay until the window holds the maximum of the shaped pulse. The other 63
# channels see a window at a random time with respect to their own dark pulses: their mean rises
# with the length of the window.
#
# The DAQ applies a new delay with a lag of a few points: at the start of each scan but the
# first, the channels stay at the values of the last delay of the previous scan. Those points are
# flagged (stale = 1): a point is stale while the median over the non-shooting channels of the
# mean has not yet reached its minimum over the scan.
#
# Writes fullsim_analysis/results/bench/<day>/:
#   hold_scan_points.csv   one row per scan and point: shooting channel, HG shaping, delay,
#                          events, mean and rms of the shooting channel, the median over the
#                          other channels of their mean, and the stale flag;
#   hold_scan_spectra.csv  one row per scan, point and bin with at least one entry, for the
#                          shooting channel and the reference channel REFERENCE: delay,
#                          channel, bin_low (ADC), count.
#   julia --project=. fullsim_analysis/scripts/bench/hold_scan.jl benchdata/<YYYY-MM-DD>
include(joinpath(@__DIR__, "..", "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using Statistics, Printf

# shooting channel and HG shaping (ns) of each scan, from the logbook of the day
const SCANS = Dict(101 => (0, 25.0), 102 => (0, 50.0), 103 => (0, 50.0),
                   104 => (20, 25.0), 105 => (40, 25.0), 106 => (60, 25.0))
const REFERENCE = 32   # a channel of the other chip, never the shooting one

day_dir = rstrip(ARGS[1], '/')
day = basename(day_dir)
scans = sort([parse(Int, split(d, '-')[end]) for d in readdir(day_dir)
              if startswith(d, "hold-scan-") && isdir(joinpath(day_dir, d))])

PCOLS = ["scan", "point", "channel", "shaping", "delay_ns", "events", "hg_mean", "hg_rms",
         "others_mean", "stale"]
SCOLS = ["scan", "point", "delay_ns", "channel", "bin_low", "count"]
pt = Dict{String,Vector}(k => [] for k in PCOLS)
st = Dict{String,Vector}(k => [] for k in SCOLS)

function moments(counts, centres)
    n = sum(counts)
    n == 0 && return (NaN, NaN)
    m = sum(counts .* centres) / n
    (m, sqrt(max(sum(counts .* (centres .- m) .^ 2) / n, 0.0)))
end

for s in scans
    haskey(SCANS, s) || (println("scan $(s): not in the logbook table, skipped"); continue)
    shooting, shaping = SCANS[s]
    r = read_daq_hold_delay(joinpath(day_dir, "hold-scan-$(s)", "hold-delay.h5"))
    centres = r.bin_low .+ 8.0
    np = length(r.delay_ns)
    means = [moments(r.counts[:, c, p], centres)[1] for c in 1:64, p in 1:np]
    others = [median(means[[c for c in 1:64 if c != shooting + 1], p]) for p in 1:np]
    first_ok = argmin(others)
    @printf("scan %d: shooting channel %d, %g ns, %d points, stale points %d (others' mean %.0f -> %.0f)\n",
            s, shooting, shaping, np, first_ok - 1, others[1], others[first_ok])
    for p in 1:np
        m, sd = moments(r.counts[:, shooting + 1, p], centres)
        push!(pt["scan"], s); push!(pt["point"], p - 1); push!(pt["channel"], shooting)
        push!(pt["shaping"], shaping); push!(pt["delay_ns"], r.effective_delay_ns[p])
        push!(pt["events"], r.events[p]); push!(pt["hg_mean"], m); push!(pt["hg_rms"], sd)
        push!(pt["others_mean"], others[p]); push!(pt["stale"], p < first_ok ? 1 : 0)
        for c in (shooting, REFERENCE), b in 1:512
            r.counts[b, c + 1, p] == 0 && continue
            push!(st["scan"], s); push!(st["point"], p - 1)
            push!(st["delay_ns"], r.effective_delay_ns[p]); push!(st["channel"], c)
            push!(st["bin_low"], r.bin_low[b]); push!(st["count"], r.counts[b, c + 1, p])
        end
    end
end

println(write_table("bench", day, "hold_scan_points", [k => pt[k] for k in PCOLS]))
println(write_table("bench", day, "hold_scan_spectra", [k => st[k] for k in SCOLS]))
