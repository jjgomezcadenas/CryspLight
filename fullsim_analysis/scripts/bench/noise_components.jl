# Coherent and incoherent noise of the pedestal runs of a day of bench data. In each event the
# 32 channels of a Citiroc are read at the same trigger; a fluctuation shared by all of them
# (common mode: the reference, the supply or a pick-up that reaches the whole chip) shows as
# an event-by-event shift of their mean, while the noise of each preamplifier is independent
# from channel to channel. For each run, board and chip:
#   1. events in which 10 or more channels of the chip lie more than OUTLIER counts from their
#      median are counted as outlier events (the whole chip jumps) and set aside;
#   2. in the remaining events each channel is referred to its median, and the common mode of
#      the event is the mean of the 32 deviations;
#   3. coherent = rms of the common mode over the events, with the share of the incoherent
#      noise that survives the mean (incoherent^2 / 32) subtracted in quadrature;
#      incoherent = median over the channels of the rms of each channel after subtracting the
#      common mode, scaled by sqrt(32/31) for the share of itself it removes;
#      total = median over the channels of the rms of each channel.
# Chips whose channels are 0 in every event (not read, findings F7) are skipped. Values are
# raw ADC counts, high and low gain.
#
# Writes fullsim_analysis/results/bench/<day>/:
#   noise_components.csv  one row per run, board and chip: events, outlier events, and total,
#                         coherent and incoherent noise of the high and low gain;
#   common_mode.csv       one row per run and pair of chips (same or different board): the
#                         correlation coefficient of their common modes over the events both
#                         read (events matched by trigger id).
#   julia --project=. fullsim_analysis/scripts/bench/noise_components.jl benchdata/<YYYY-MM-DD>
include(joinpath(@__DIR__, "..", "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using Statistics, Printf

const OUTLIER = 60        # counts, five times the pedestal rms of the high gain
const OUTLIER_CHANNELS = 10

day_dir = rstrip(ARGS[1], '/')
day = basename(day_dir)
subdirs(prefix) = sort(filter(d -> startswith(d, prefix) && isdir(joinpath(day_dir, d)),
                              readdir(day_dir)); by = d -> parse(Int, split(d, '-')[end]))
# board number from chain and node, as in the manifest topology
function board_numbers(text)
    re = r"\"board\":\s*(\d+),\s*\"chain\":\s*(\d+),\s*\"node\":\s*(\d+)"
    Dict((parse(Int, m[2]), parse(Int, m[3])) => parse(Int, m[1]) for m in eachmatch(re, text))
end

# events x 32 matrix of one gain of one chip, rows ordered by trigger id
function chip_matrix(o, sel, gain)
    tids = sort(unique(o.trigger_id[sel]))
    row = Dict(t => k for (k, t) in enumerate(tids))
    v = gain === :hg ? o.hg : o.lg
    M = fill(NaN, length(tids), 32)
    for i in sel
        M[row[o.trigger_id[i]], o.channel[i] % 32 + 1] = v[i]
    end
    tids, M
end

function components(M)
    D = M .- median(M; dims = 1)
    out = vec(sum(abs.(D) .> OUTLIER; dims = 2)) .>= OUTLIER_CHANNELS
    D = D[.!out, :]
    cm = vec(mean(D; dims = 2))
    total = median(vec(std(D; dims = 1)))
    inc = median(vec(std(D .- cm; dims = 1))) * sqrt(32 / 31)
    coh = sqrt(max(var(cm) - inc^2 / 32, 0.0))
    (; out, cm, total, coh, inc)
end

NCOLS = ["run", "board", "chip", "events", "outlier_events",
         "hg_total", "hg_coherent", "hg_incoherent", "lg_total", "lg_coherent", "lg_incoherent"]
MCOLS = ["run", "board_a", "chip_a", "board_b", "chip_b", "events", "correlation"]
nt = Dict{String,Vector}(k => [] for k in NCOLS)
mt = Dict{String,Vector}(k => [] for k in MCOLS)

for d in subdirs("run-")
    n = parse(Int, split(d, '-')[end])
    r = read_daq_run(joinpath(day_dir, d, "run_$(n).0000.h5"))
    bn = board_numbers(read(joinpath(day_dir, d, "manifest.json"), String))
    o = r.obs
    cms = Dict{Tuple{Int,Int},Dict{UInt64,Float64}}()
    for ((c, nd), b) in sort(collect(bn); by = last), chip in 0:1
        sel = findall(i -> o.chain[i] == c && o.node[i] == nd && o.channel[i] ÷ 32 == chip && o.has_hg[i],
                      eachindex(o.hg))
        isempty(sel) && continue
        tids, H = chip_matrix(o, sel, :hg)
        all(H .== 0) && continue
        _, L = chip_matrix(o, sel, :lg)
        h, l = components(H), components(L)
        cms[(b, chip)] = Dict(t => x for (t, x) in zip(tids[.!h.out], h.cm))
        for (k, v) in zip(NCOLS, (n, b, chip, size(H, 1), count(h.out),
                                  h.total, h.coh, h.inc, l.total, l.coh, l.inc))
            push!(nt[k], v)
        end
        @printf("run %d board %d chip %d: %d events, %d outlier; HG total %.2f coherent %.2f incoherent %.2f; LG total %.2f coherent %.2f incoherent %.2f\n",
                n, b, chip, size(H, 1), count(h.out), h.total, h.coh, h.inc, l.total, l.coh, l.inc)
    end
    keys_ = sort(collect(keys(cms)))
    for i in eachindex(keys_), j in i+1:length(keys_)
        a, b = cms[keys_[i]], cms[keys_[j]]
        common = [t for t in keys(a) if haskey(b, t)]
        length(common) < 100 && continue
        rho = cor([a[t] for t in common], [b[t] for t in common])
        for (k, v) in zip(MCOLS, (n, keys_[i]..., keys_[j]..., length(common), rho))
            push!(mt[k], v)
        end
    end
end
println(write_table("bench", day, "noise_components", [k => nt[k] for k in NCOLS]))
println(write_table("bench", day, "common_mode", [k => mt[k] for k in MCOLS]))
