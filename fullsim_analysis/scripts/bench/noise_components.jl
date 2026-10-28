# Coherent and incoherent noise of the pedestal runs of a day of bench data (definitions and
# formulas in fullsim_analysis/lib/chip_noise.jl). In each event the 32 channels of a Citiroc
# are read at the same trigger, so a fluctuation shared by all of them (common mode) can be
# separated from the noise of each channel. For each run, board and chip: outliers of single
# channels are left out (in the runs with HoldDelay 0 only, where the noise is symmetric), jumps of the whole chip (seen in the common mode) are set aside, and
# the total, coherent and incoherent noise and the slope of each channel against the common
# mode are computed on the rest.
# Chips whose channels are 0 in every event (not read, findings F7) are skipped, and so are the
# runs with the SiPMs above breakdown (HV_Vbias above PEDESTAL_HV_MAX). Values are raw ADC
# counts, high and low gain.
#
# Writes fullsim_analysis/results/bench/<day>/:
#   noise_components.csv  one row per run, board and chip: events, jumps of the chip,
#                         outliers of single channels and channels in the common mode (the
#                         blind ones left out) of the high gain, and total, coherent and
#                         incoherent noise of the high and low gain;
#   common_mode.csv       one row per run and pair of chips (same or different board): the
#                         correlation coefficient of their common modes over the events both
#                         read (events matched by trigger id);
#   channel_noise.csv     one row per run, board, chip and channel: the slope b_i of the
#                         deviation of the high gain against the common mode (1 if the
#                         channel takes the full common mode), the number of events in which
#                         the channel is an outlier (high gain), and the rms of the channel in
#                         the high and low gain, without jumps and outliers;
#   outlier_pulls.csv     one row per run and bin of 0.1: the number of entries of the pull
#                         z = (d_i - r) / s_i of the high gain, over all channels of all chips
#                         and the events that are not jumps, from -10 to 10 (the first and
#                         last bins include everything beyond).
# and prints the channels whose median slope over the runs is below NO_SIPM_SLOPE: they do
# not receive the common mode, which enters through the SiPM, so their SiPM is not seen (on
# 22 Oct 2026 the six channels of board 0 silent in the staircases are below 0.45, all the
# others above 0.82; computed against the common mode without the blind channels).
#   julia --project=. fullsim_analysis/scripts/bench/noise_components.jl benchdata/<YYYY-MM-DD>
include(joinpath(@__DIR__, "..", "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using Statistics, Printf

day_dir = rstrip(ARGS[1], '/')
day = basename(day_dir)
subdirs(prefix) = sort(filter(d -> startswith(d, prefix) && isdir(joinpath(day_dir, d)),
                              readdir(day_dir)); by = d -> parse(Int, split(d, '-')[end]))

NCOLS = ["run", "board", "chip", "events", "jumps", "channel_outliers", "channels_in_cm",
         "hg_total", "hg_coherent", "hg_incoherent", "lg_total", "lg_coherent", "lg_incoherent"]
MCOLS = ["run", "board_a", "chip_a", "board_b", "chip_b", "events", "correlation"]
nt = Dict{String,Vector}(k => [] for k in NCOLS)
mt = Dict{String,Vector}(k => [] for k in MCOLS)
SCOLS = ["run", "board", "chip", "channel", "slope", "outliers", "hg_rms", "lg_rms"]
const ZEDGES = -10.0:0.1:10.0
# Pedestal runs have the SiPMs below breakdown (HV_Vbias 40.7 V, 36.5 V at the SiPM); runs with
# a higher bias (the calibration with the periodic trigger, menu 3) are not noise runs and are skipped.
const PEDESTAL_HV_MAX = 42.0
PCOLS = ["run", "z_low", "count"]
pt = Dict{String,Vector}(k => [] for k in PCOLS)
st = Dict{String,Vector}(k => [] for k in SCOLS)

for d in subdirs("run-")
    n = parse(Int, split(d, '-')[end])
    r = read_daq_run(joinpath(day_dir, d, "run_$(n).0000.h5"))
    bn = daq_board_numbers(read(joinpath(day_dir, d, "manifest.json"), String))
    o = r.obs
    hold = daq_value(r.config.params, "HoldDelay")
    cut_outliers = hold !== nothing && hold == 0   # symmetric noise only with HoldDelay 0
    hv = daq_value(r.config.params, "HV_Vbias")
    if hv !== nothing && hv > PEDESTAL_HV_MAX
        println("run $(n): HV_Vbias $(hv) V, SiPMs above breakdown, not a pedestal run, skipped")
        continue
    end
    cms = Dict{Tuple{Int,Int},Dict{UInt64,Float64}}()
    zall = Float64[]
    for ((c, nd), b) in sort(collect(bn); by = last), chip in 0:1
        sel = chip_rows(o, c, nd, chip)
        isempty(sel) && continue
        tids, H = chip_matrix(o, sel, :hg)
        all(H .== 0) && continue
        _, L = chip_matrix(o, sel, :lg)
        h = chip_noise(H; outliers = cut_outliers)
        l = chip_noise(L; outliers = cut_outliers, blind = h.blind)   # blind from the high gain
        for i in 1:32
            for (k, v) in zip(SCOLS, (n, b, chip, 32chip + i - 1, h.slope[i], h.outliers[i], h.rms[i], l.rms[i]))
                push!(st[k], v)
            end
        end
        append!(zall, vec(h.Z[.!h.jump, :]))
        cms[(b, chip)] = Dict(t => x for (t, x) in zip(tids[.!h.jump], h.cm))
        for (k, v) in zip(NCOLS, (n, b, chip, size(H, 1), count(h.jump), sum(h.outliers), count(.!h.blind),
                                  h.total, h.coh, h.inc, l.total, l.coh, l.inc))
            push!(nt[k], v)
        end
        @printf("run %d board %d chip %d: %d events, %d jumps, %d channel outliers; HG total %.2f coherent %.2f incoherent %.2f; LG total %.2f coherent %.2f incoherent %.2f\n",
                n, b, chip, size(H, 1), count(h.jump), sum(h.outliers), h.total, h.coh, h.inc, l.total, l.coh, l.inc)
    end
    zc = zeros(Int, length(ZEDGES) - 1)
    for z in zall
        k = clamp(floor(Int, (z - first(ZEDGES)) / step(ZEDGES)) + 1, 1, length(zc))
        zc[k] += 1
    end
    for (k, c) in enumerate(zc)
        push!(pt["run"], n); push!(pt["z_low"], ZEDGES[k]); push!(pt["count"], c)
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
println(write_table("bench", day, "channel_noise", [k => st[k] for k in SCOLS]))
println(write_table("bench", day, "outlier_pulls", [k => pt[k] for k in PCOLS]))

const NO_SIPM_SLOPE = BLIND_SLOPE
for b in sort(unique(st["board"]))
    chs = sort(unique(st["channel"][i] for i in eachindex(st["board"]) if st["board"][i] == b))
    low = [c for c in chs if median([st["slope"][i] for i in eachindex(st["board"])
                                     if st["board"][i] == b && st["channel"][i] == c]) < NO_SIPM_SLOPE]
    isempty(low) || println("board $b: common-mode slope below $(NO_SIPM_SLOPE) in channels $(low): SiPM not seen")
end
