# Coherent and incoherent noise of the pedestal runs of a day of bench data (definitions and
# formulas in fullsim_analysis/lib/chip_noise.jl). In each event the 32 channels of a Citiroc
# are read at the same trigger, so a fluctuation shared by all of them (common mode) can be
# separated from the noise of each channel. For each run, board and chip: outliers of single
# channels are left out, jumps of the whole chip (seen in the common mode) are set aside, and
# the total, coherent and incoherent noise and the slope of each channel against the common
# mode are computed on the rest.
# Chips whose channels are 0 in every event (not read, findings F7) are skipped. Values are
# raw ADC counts, high and low gain.
#
# Writes fullsim_analysis/results/bench/<day>/:
#   noise_components.csv  one row per run, board and chip: events, jumps of the chip and
#                         outliers of single channels (high gain), and total, coherent and
#                         incoherent noise of the high and low gain;
#   common_mode.csv       one row per run and pair of chips (same or different board): the
#                         correlation coefficient of their common modes over the events both
#                         read (events matched by trigger id);
#   channel_slopes.csv    one row per run, board, chip and channel: the slope b_i of the
#                         deviation of the high gain against the common mode (1 if the
#                         channel takes the full common mode) and the number of events in
#                         which the channel is an outlier.
# and prints the channels whose median slope over the runs is below NO_SIPM_SLOPE: they do
# not receive the common mode, which enters through the SiPM, so their SiPM is not seen (on
# 22 Oct 2026 the six channels of board 0 silent in the staircases are at 0.02-0.6, all the
# others above 0.85).
#   julia --project=. fullsim_analysis/scripts/bench/noise_components.jl benchdata/<YYYY-MM-DD>
include(joinpath(@__DIR__, "..", "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using Statistics, Printf

day_dir = rstrip(ARGS[1], '/')
day = basename(day_dir)
subdirs(prefix) = sort(filter(d -> startswith(d, prefix) && isdir(joinpath(day_dir, d)),
                              readdir(day_dir)); by = d -> parse(Int, split(d, '-')[end]))

NCOLS = ["run", "board", "chip", "events", "jumps", "channel_outliers",
         "hg_total", "hg_coherent", "hg_incoherent", "lg_total", "lg_coherent", "lg_incoherent"]
MCOLS = ["run", "board_a", "chip_a", "board_b", "chip_b", "events", "correlation"]
nt = Dict{String,Vector}(k => [] for k in NCOLS)
mt = Dict{String,Vector}(k => [] for k in MCOLS)
SCOLS = ["run", "board", "chip", "channel", "slope", "outliers"]
st = Dict{String,Vector}(k => [] for k in SCOLS)

for d in subdirs("run-")
    n = parse(Int, split(d, '-')[end])
    r = read_daq_run(joinpath(day_dir, d, "run_$(n).0000.h5"))
    bn = daq_board_numbers(read(joinpath(day_dir, d, "manifest.json"), String))
    o = r.obs
    cms = Dict{Tuple{Int,Int},Dict{UInt64,Float64}}()
    for ((c, nd), b) in sort(collect(bn); by = last), chip in 0:1
        sel = chip_rows(o, c, nd, chip)
        isempty(sel) && continue
        tids, H = chip_matrix(o, sel, :hg)
        all(H .== 0) && continue
        _, L = chip_matrix(o, sel, :lg)
        h, l = chip_noise(H), chip_noise(L)
        for i in 1:32
            for (k, v) in zip(SCOLS, (n, b, chip, 32chip + i - 1, h.slope[i], h.outliers[i]))
                push!(st[k], v)
            end
        end
        cms[(b, chip)] = Dict(t => x for (t, x) in zip(tids[.!h.jump], h.cm))
        for (k, v) in zip(NCOLS, (n, b, chip, size(H, 1), count(h.jump), sum(h.outliers),
                                  h.total, h.coh, h.inc, l.total, l.coh, l.inc))
            push!(nt[k], v)
        end
        @printf("run %d board %d chip %d: %d events, %d jumps, %d channel outliers; HG total %.2f coherent %.2f incoherent %.2f; LG total %.2f coherent %.2f incoherent %.2f\n",
                n, b, chip, size(H, 1), count(h.jump), sum(h.outliers), h.total, h.coh, h.inc, l.total, l.coh, l.inc)
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
println(write_table("bench", day, "channel_slopes", [k => st[k] for k in SCOLS]))

const NO_SIPM_SLOPE = 0.75
for b in sort(unique(st["board"]))
    chs = sort(unique(st["channel"][i] for i in eachindex(st["board"]) if st["board"][i] == b))
    low = [c for c in chs if median([st["slope"][i] for i in eachindex(st["board"])
                                     if st["board"][i] == b && st["channel"][i] == c]) < NO_SIPM_SLOPE]
    isempty(low) || println("board $b: common-mode slope below $(NO_SIPM_SLOPE) in channels $(low): SiPM not seen")
end
