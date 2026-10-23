# Spectra of the high gain, channel by channel and board by board, for some runs of a day of
# bench data. The board is identified by the pair (chain, node) of each measurement and
# numbered with the topology stored in the manifest of the run; grouping by node alone would
# mix boards of different chains (board 0 with board 2, board 1 with board 3, on 22 Oct 2026).
#
# Writes fullsim_analysis/results/bench/<day>/pedestal_spectra.csv: one row per run, board,
# channel and ADC value with at least one entry (value, count), raw ADC counts; read by
# fullsim_analysis/plots/pedestal_spectra.py.
#   julia --project=. fullsim_analysis/scripts/bench/pedestal_spectra.jl benchdata/<YYYY-MM-DD> 26 27 28
include(joinpath(@__DIR__, "..", "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis

day_dir = rstrip(ARGS[1], '/')
day = basename(day_dir)
runs = parse.(Int, ARGS[2:end])

cols = Dict(k => Int[] for k in ("run", "board", "channel", "adc", "count"))
for n in runs
    d = joinpath(day_dir, "run-$(n)")
    r = read_daq_run(joinpath(d, "run_$(n).0000.h5"))
    bn = daq_board_numbers(read(joinpath(d, "manifest.json"), String))
    o = r.obs
    h = Dict{NTuple{3,Int},Int}()
    for i in eachindex(o.hg)
        o.has_hg[i] || continue
        k = (bn[(o.chain[i], o.node[i])], o.channel[i], o.hg[i])
        h[k] = get(h, k, 0) + 1
    end
    for (b, c, v) in sort(collect(keys(h)))
        push!(cols["run"], n); push!(cols["board"], b); push!(cols["channel"], c)
        push!(cols["adc"], v); push!(cols["count"], h[(b, c, v)])
    end
    println("run $n: boards ", sort(collect(values(bn))), " from (chain, node) ", sort(collect(keys(bn))))
end
println(write_table("bench", day, "pedestal_spectra",
                    [k => cols[k] for k in ("run", "board", "channel", "adc", "count")]))
