# Trigger threshold scan on an optical run: for each front end and threshold, the
# fraction of 511 keV photopeak decays that trigger and the fraction of decays without
# any energy deposit that trigger on dark pulses (the dark-trigger probability per event
# window, converted to a rate). Writes fullsim_analysis/results/<campaign>/trigger_scan_<tag>.toml
# and a figure.
#   julia --project=. fullsim_analysis/trigger_scan.jl <campaign> output/<tag>/pes.h5 [n_events]
using CryspLight, Statistics, Printf, TOML
using Random: Xoshiro, seed!
using CairoMakie

campaign, pes_path = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1500
repo = dirname(@__DIR__)
pes = read_pes(pes_path)
tag = pes.attributes["tag"]
ev = pes.events
pk = findall((ev.edep_511_kev .> 510) .& (ev.edep_1275_kev .== 0))[1:nmax]
compton = findall((ev.edep_kev .> 200) .& (ev.edep_kev .< 300))[1:nmax]
empty = findall(ev.edep_kev .== 0)[1:nmax]
dev = SipmDevice()
results = Dict{String,Any}("tag" => tag, "n_photopeak" => length(pk), "n_compton" => length(compton),
                           "n_empty" => length(empty))

"Fraction of the events of `list` that trigger, with one chain per thread (reseeded per event)."
function trigger_fraction(kind, op, settings, list)
    chains = [kind == "citiroc" ? CitirocChain(settings, op, Xoshiro(t)) :
                                   DigitizerChain(settings, op, Xoshiro(t)) for t in 1:Threads.maxthreadid()]
    apply = kind == "citiroc" ? citiroc_event : digitizer_event
    n = Threads.Atomic{Int}(0)
    Threads.@threads :static for k in list
        ch = chains[Threads.threadid()]
        seed!(ch.rng, hash((5, k, :noise)))
        trains = event_trains(op, event_pes(pes, k), 64, Xoshiro(hash((5, k))))
        apply(ch, trains).triggered && Threads.atomic_add!(n, 1)
    end
    return n[] / length(list)
end

function scan(kind, op, thresholds, window_len_ns, make_settings)
    eff = Float64[]; effc = Float64[]; dark = Float64[]
    for thr in thresholds
        st = make_settings(thr)
        push!(eff, trigger_fraction(kind, op, st, pk))
        push!(effc, trigger_fraction(kind, op, st, compton))
        push!(dark, trigger_fraction(kind, op, st, empty))
        @printf("%-9s threshold %6.1f: efficiency 511 keV %.3f, 200-300 keV %.3f; dark trigger per window %.3f (%.1f kHz over %.1f us)\n",
                kind, thr, eff[end], effc[end], dark[end], dark[end] / window_len_ns * 1e6, window_len_ns / 1000)
        flush(stdout)
    end
    return eff, effc, dark
end

# Citiroc at 3 V: single-channel threshold in photoelectron-equivalents, and majority 2
op3 = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 64, window_ns = (-1500.0, 3000.0))
thr_c = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
thr_c2 = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]
e1, c1, d1 = scan("citiroc", op3, thr_c, 2200.0, t -> CitirocSettings(threshold_pe = t, majority = 1))
e2, c2, d2 = scan("citiroc", op3, thr_c2, 2200.0, t -> CitirocSettings(threshold_pe = t, majority = 2))
# digitizer at 5 V (43 V), gain 1: threshold in counts, majority 2 within 100 ns
op5 = SipmOperation(dev; t_k = 293.15, vov = 5.0, n_channels = 64, window_ns = (-2500.0, 3000.0))
thr_d = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0, 250.0, 300.0, 400.0]
ed2, cd2, dd2 = scan("digitizer", op5, thr_d, 5000.0, t -> DigitizerSettings(threshold_counts = t, majority = 2))
ed1, cd1, dd1 = scan("digitizer", op5, thr_d, 5000.0, t -> DigitizerSettings(threshold_counts = t, majority = 1))
tbl(th, e, c, d, w; key = "threshold") = Dict(key => th, "efficiency_511" => e, "efficiency_200_300" => c,
                                              "dark_per_window" => d, "window_ns" => w)
results["citiroc_3v_majority1"] = tbl(thr_c, e1, c1, d1, 2200.0; key = "threshold_pe")
results["citiroc_3v_majority2"] = tbl(thr_c2, e2, c2, d2, 2200.0; key = "threshold_pe")
results["digitizer_5v_majority2"] = tbl(thr_d, ed2, cd2, dd2, 5000.0; key = "threshold_counts")
results["digitizer_5v_majority1"] = tbl(thr_d, ed1, cd1, dd1, 5000.0; key = "threshold_counts")
outdir = joinpath(repo, "fullsim_analysis", "results", campaign)
mkpath(outdir)
open(joinpath(outdir, "trigger_scan_$(tag).toml"), "w") do io; TOML.print(io, results); end
fig = Figure(size = (1200, 500))
ax1 = Axis(fig[1, 1]; xlabel = "threshold (photoelectron-equivalents, one channel)", ylabel = "fraction",
           title = "Citiroc 1A, 3 V, $(tag)")
lines!(ax1, thr_c, e1; label = "511 keV efficiency, OR"); lines!(ax1, thr_c, c1; label = "200-300 keV, OR")
lines!(ax1, thr_c, d1; label = "dark trigger per 2.2 us window, OR")
lines!(ax1, thr_c2, e2; linestyle = :dash, label = "511 keV, majority 2"); lines!(ax1, thr_c2, c2; linestyle = :dash, label = "200-300 keV, majority 2")
lines!(ax1, thr_c2, d2; linestyle = :dash, label = "dark, majority 2")
axislegend(ax1; position = :rc)
ax2 = Axis(fig[1, 2]; xlabel = "threshold (counts, 16 per photoelectron)", ylabel = "fraction",
           title = "DT2745, 43 V gain 1, $(tag)")
lines!(ax2, thr_d, ed2; label = "511 keV efficiency, majority 2"); lines!(ax2, thr_d, cd2; label = "200-300 keV, majority 2")
lines!(ax2, thr_d, dd2; label = "dark trigger per 5 us window, majority 2")
lines!(ax2, thr_d, ed1; linestyle = :dash, label = "511 keV, OR"); lines!(ax2, thr_d, cd1; linestyle = :dash, label = "200-300 keV, OR")
lines!(ax2, thr_d, dd1; linestyle = :dash, label = "dark, OR")
axislegend(ax2; position = :rc)
save(joinpath(outdir, "trigger_scan_$(tag).png"), fig)
println("written ", outdir)
