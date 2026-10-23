# The steps of the common-mode analysis (fullsim_analysis/lib/chip_noise.jl) on one run, to
# illustrate them in the report and in the noise note: an example chip (EXAMPLE), the chip with jumps (JUMPS), and the
# common mode of the chips used to show its correlation (CORRELATED). High gain, ADC counts.
#
# Writes fullsim_analysis/results/bench/<day>/:
#   noise_steps_matrix.csv  deviations d_i(e) of the 32 channels for NSHOW consecutive events:
#                           panel "example" from the first event of the example chip, panel
#                           "jump" centred on the first jump of the chip with jumps; columns
#                           panel, event (row in the run), channel, d;
#   noise_steps_cuts.csv    for the example chip and the chip with jumps: centre and half-width
#                           of the band of the common mode outside which an event is a jump;
#   noise_steps_hist.csv    histograms (bins of 1 count, -100 to 100) of d_i and of d_i - CM
#                           over all channels and kept events of the example chip;
#   noise_steps_events.csv  one row per trigger id: the event (row in the run), CM(e) of every
#                           event and the jump flag of the example chip and of the chip with
#                           jumps, CM(e) of the CORRELATED chips (empty for jumps), and d_i(e)
#                           of the channel SHOWN of the example chip.
#   julia --project=. fullsim_analysis/scripts/bench/noise_steps.jl benchdata/<YYYY-MM-DD> <run>
include(joinpath(@__DIR__, "..", "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using Statistics

const EXAMPLE = (board = 2, chip = 1)
const JUMPS = (board = 0, chip = 1)
const CORRELATED = [(2, 0), (2, 1), (3, 1)]   # (board, chip)
const SHOWN = 40                               # channel of the example chip in noise_steps_events
const NSHOW = 300

day_dir = rstrip(ARGS[1], '/')
day = basename(day_dir)
n = parse(Int, ARGS[2])
d = joinpath(day_dir, "run-$(n)")
r = read_daq_run(joinpath(d, "run_$(n).0000.h5"))
bn = daq_board_numbers(read(joinpath(d, "manifest.json"), String))
cn = Dict(v => k for (k, v) in bn)
o = r.obs

analyse(b, chip) = (sel = chip_rows(o, cn[b]..., chip); (tids, M) = chip_matrix(o, sel, :hg);
                    (tids = tids, c = chip_noise(M)))
chips = Dict((b, c) => analyse(b, c) for (b, c) in unique([(EXAMPLE...,), (JUMPS...,), CORRELATED...]))

# matrices
mcols = Dict(k => [] for k in ("panel", "event", "channel", "d"))
ex, ju = chips[(EXAMPLE...,)], chips[(JUMPS...,)]
j0 = findfirst(ju.c.jump)
for (panel, a, first) in (("example", ex, 1), ("jump", ju, max(1, j0 - NSHOW ÷ 2)))
    for e in first:min(first + NSHOW - 1, size(a.c.D0, 1)), i in 1:32
        push!(mcols["panel"], panel); push!(mcols["event"], e - 1)
        push!(mcols["channel"], 32 * (panel == "example" ? EXAMPLE.chip : JUMPS.chip) + i - 1)
        push!(mcols["d"], a.c.D0[e, i])
    end
end
println(write_table("bench", day, "noise_steps_matrix",
                    [k => mcols[k] for k in ("panel", "event", "channel", "d")]))

# histograms of the example chip
edges = -100.5:1:100.5
hist(v) = [count(x -> lo <= x < lo + 1, v) for lo in edges[1:end-1]]
println(write_table("bench", day, "noise_steps_hist",
                    ["d" => collect(-100:100), "count_d" => hist(vec(ex.c.D)),
                     "count_d_minus_cm" => hist(vec(ex.c.D .- ex.c.cm))]))

# per-event quantities, matched by trigger id
alltids = sort(unique(vcat([a.tids for a in values(chips)]...)))
ecols = Pair{String,Vector}["trigger_id" => Int.(alltids)]
lookup(a, v) = (m = Dict(zip(a.tids, v)); [get(m, t, NaN) for t in alltids])
kept(a) = a.tids[.!a.c.jump]
cmof(a) = (m = Dict(zip(kept(a), a.c.cm)); [get(m, t, NaN) for t in alltids])
push!(ecols, "event" => lookup(ex, collect(0:length(ex.tids) - 1)))
push!(ecols, "cm_all_example" => lookup(ex, ex.c.cm_all))
push!(ecols, "jump_example" => lookup(ex, Float64.(ex.c.jump)))
push!(ecols, "cm_all_jumps" => lookup(ju, ju.c.cm_all))
push!(ecols, "jump_jumps" => lookup(ju, Float64.(ju.c.jump)))
for (b, c) in CORRELATED
    push!(ecols, "cm_b$(b)c$(c)" => cmof(chips[(b, c)]))
end
ishow = SHOWN - 32 * EXAMPLE.chip + 1
push!(ecols, "d_ch$(SHOWN)" => (m = Dict(zip(kept(ex), ex.c.D[:, ishow])); [get(m, t, NaN) for t in alltids]))
println(write_table("bench", day, "noise_steps_events", ecols))
println(write_table("bench", day, "noise_steps_cuts",
                    ["chip" => ["example", "jumps"], "centre" => [ex.c.cut[1], ju.c.cut[1]],
                     "half_width" => [ex.c.cut[2], ju.c.cut[2]]]))
println("example board $(EXAMPLE.board) chip $(EXAMPLE.chip): total $(round(ex.c.total, digits=2)) coherent $(round(ex.c.coh, digits=2)) incoherent $(round(ex.c.inc, digits=2)), slopes $(round.(extrema(ex.c.slope), digits=2)), channel outliers $(sum(ex.c.outliers)), jump cut ±$(round(ex.c.cut[2], digits=1)); jumps in board $(JUMPS.board) chip $(JUMPS.chip): $(count(ju.c.jump)), first at row $(j0 - 1)")
