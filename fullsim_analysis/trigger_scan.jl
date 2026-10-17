# Trigger threshold scan on an optical run: for each front end and threshold, the
# fraction of 511 keV photopeak decays that trigger and the fraction of decays without
# any energy deposit that trigger on dark pulses (the dark-trigger probability per event
# window, converted to a rate). Writes fullsim_analysis/results/<campaign>/trigger_scan_<tag>.toml
# and a figure.
#   julia --project=. fullsim_analysis/trigger_scan.jl <campaign> output/<tag>/pes.h5 [n_events]
using CryspLight, Statistics, Printf, TOML
using Random: Xoshiro
using CairoMakie

campaign, pes_path = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1500
repo = dirname(@__DIR__)
pes = read_pes(pes_path)
tag = pes.attributes["tag"]
ev = pes.events
pk = findall((ev.edep_511_kev .> 510) .& (ev.edep_1275_kev .== 0))[1:nmax]
empty = findall(ev.edep_kev .== 0)[1:nmax]
dev = SipmDevice()
results = Dict{String,Any}("tag" => tag, "n_photopeak" => length(pk), "n_empty" => length(empty))

function scan(kind, op, thresholds, window_len_ns, make_chain, make_settings)
    eff = Float64[]; dark = Float64[]
    for thr in thresholds
        chain = make_chain(make_settings(thr), op, Xoshiro(1))
        apply = kind == "citiroc" ? citiroc_event : digitizer_event
        ntrig_pk = 0; ntrig_empty = 0
        for (list, counter) in ((pk, :pk), (empty, :empty))
            n = 0
            for k in list
                rng = Xoshiro(hash((5, k)))
                trains = event_trains(op, event_pes(pes, k), 64, rng)
                apply(chain, trains).triggered && (n += 1)
            end
            counter == :pk ? (ntrig_pk = n) : (ntrig_empty = n)
        end
        push!(eff, ntrig_pk / length(pk)); push!(dark, ntrig_empty / length(empty))
        @printf("%-9s threshold %6.1f: photopeak efficiency %.3f, dark trigger per window %.3f (%.1f kHz over %.1f us)\n",
                kind, thr, eff[end], dark[end], dark[end] / window_len_ns * 1e6, window_len_ns / 1000)
    end
    return eff, dark
end

# Citiroc at 3 V: single-channel threshold in photoelectron-equivalents, and majority 2
op3 = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 64, window_ns = (-1500.0, 3000.0))
thr_c = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
effc1, darkc1 = scan("citiroc", op3, thr_c, 2200.0, CitirocChain,
                     t -> CitirocSettings(threshold_pe = t, majority = 1))
effc2, darkc2 = scan("citiroc", op3, [2.0, 3.0, 4.0, 5.0, 6.0], 2200.0, CitirocChain,
                     t -> CitirocSettings(threshold_pe = t, majority = 2))
# digitizer at 5 V (43 V), gain 1: threshold in counts, majority 2 within 100 ns
op5 = SipmOperation(dev; t_k = 293.15, vov = 5.0, n_channels = 64, window_ns = (-2500.0, 3000.0))
thr_d = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0]
effd2, darkd2 = scan("digitizer", op5, thr_d, 5000.0, DigitizerChain,
                     t -> DigitizerSettings(threshold_counts = t, majority = 2))
effd1, darkd1 = scan("digitizer", op5, thr_d, 5000.0, DigitizerChain,
                     t -> DigitizerSettings(threshold_counts = t, majority = 1))
results["citiroc_3v_majority1"] = Dict("threshold_pe" => thr_c, "efficiency_511" => effc1, "dark_per_window" => darkc1, "window_ns" => 2200.0)
results["citiroc_3v_majority2"] = Dict("threshold_pe" => [2.0, 3.0, 4.0, 5.0, 6.0], "efficiency_511" => effc2, "dark_per_window" => darkc2, "window_ns" => 2200.0)
results["digitizer_5v_majority2"] = Dict("threshold_counts" => thr_d, "efficiency_511" => effd2, "dark_per_window" => darkd2, "window_ns" => 5000.0)
results["digitizer_5v_majority1"] = Dict("threshold_counts" => thr_d, "efficiency_511" => effd1, "dark_per_window" => darkd1, "window_ns" => 5000.0)
outdir = joinpath(repo, "fullsim_analysis", "results", campaign)
mkpath(outdir)
open(joinpath(outdir, "trigger_scan_$(tag).toml"), "w") do io; TOML.print(io, results); end
fig = Figure(size = (1100, 450))
ax1 = Axis(fig[1, 1]; xlabel = "threshold (photoelectron-equivalents, one channel)", ylabel = "fraction", title = "Citiroc, 3 V, $(tag)")
lines!(ax1, thr_c, effc1; label = "511 keV efficiency, OR"); lines!(ax1, thr_c, darkc1; label = "dark trigger per 2.2 us window, OR")
lines!(ax1, [2.0, 3.0, 4.0, 5.0, 6.0], effc2; linestyle = :dash, label = "511 keV, majority 2"); lines!(ax1, [2.0, 3.0, 4.0, 5.0, 6.0], darkc2; linestyle = :dash, label = "dark, majority 2")
axislegend(ax1; position = :lb)
ax2 = Axis(fig[1, 2]; xlabel = "threshold (counts, 16 per photoelectron)", ylabel = "fraction", title = "DT2745, 43 V gain 1, $(tag)")
lines!(ax2, thr_d, effd2; label = "511 keV efficiency, majority 2"); lines!(ax2, thr_d, darkd2; label = "dark trigger per 5 us window, majority 2")
lines!(ax2, thr_d, effd1; linestyle = :dash, label = "511 keV, OR"); lines!(ax2, thr_d, darkd1; linestyle = :dash, label = "dark, OR")
axislegend(ax2; position = :lb)
save(joinpath(outdir, "trigger_scan_$(tag).png"), fig)
println("written ", outdir)
