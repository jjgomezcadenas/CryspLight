# Gate scan of the digitizer on the photon file of a front-end configuration: for a set
# of gate starts (relative to the trigger) and gate lengths, the 511 keV peak, its width
# (fitted and from the tagged decays), the 1275 to 511 ratio from the tagged decays, the
# fraction of the photons stored, the noise and dark charge of the gate measured on empty
# decays with an external trigger, and the triggered fraction. Writes
# results/<campaign>/gate_scan/gate_scan.csv (rows of this tag replaced) and spectra into
# gate_scan.h5. Plot with fullsim_analysis/plots/gate_scan.py.
#   julia --project=. --threads=8 fullsim_analysis/scripts/gate_scan.jl <campaign> <frontend config> [n_decays]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, HDF5, Statistics, Printf

campaign, cfg = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 40_000
r = open_pes_run(cfg)
st, pes, op = r.setup, r.pes, r.op
st.frontend == "digitizer" || error("the gate scan is for the digitizer configurations")
tag = st.tag
ev = pes.events
n = min(nmax, length(ev.edep_kev))
all_list = collect(1:n)
inrange(v) = v .& (1:length(ev.edep_kev) .<= n)
sel511 = inrange((ev.edep_511_kev .> 510) .& (ev.edep_1275_kev .== 0))[1:n]
sel1275 = inrange((ev.edep_1275_kev .> 1273) .& (ev.edep_511_kev .== 0))[1:n]
empty = findall(ev.edep_kev .== 0)[1:min(3000, count(ev.edep_kev .== 0))]
# the scan: gate starts at the configured length, then gate lengths at the best start
starts = [-50.0, -100.0, -200.0, -300.0, -500.0]
lengths = sort(unique([0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 10.0] .* 1000.0))
points = vcat([(s0, st.settings.gate_ns) for s0 in starts],
              [(-300.0, g) for g in lengths if g != st.settings.gate_ns || -300.0 ∉ starts])
cols = Dict{String,Vector}(k => Float64[] for k in
    ("gate_start_ns", "gate_ns", "peak511", "fwhm511", "fwhm511_mc", "ratio_1275_511_mc", "stored_fraction",
     "empty_charge_mean", "empty_charge_rms", "triggered_photopeak", "triggered_empty"))
open_results(campaign, "gate_scan"; mode = isfile(joinpath(results_dir(campaign, "gate_scan"), "gate_scan.h5")) ? "r+" : "w") do h5
    haskey(h5, tag) && delete_object(h5, tag)
    gt = create_group(h5, tag)
    stamp!(gt, "config" => st.cfg_text, "n_decays" => n)
    for (g0, gl) in points
        s = with(st.settings; gate_start_ns = g0, gate_ns = gl,
                 t_start_ns = min(st.settings.t_start_ns, g0 - 1000.0),
                 t_end_ns = max(st.settings.t_end_ns, gl + 2000.0))
        # dark avalanches must cover the baseline window and the whole gate
        opg = with_window(op, (min(op.window_ns[1], g0 - s.baseline_ns - 1000.0),
                               max(op.window_ns[2], g0 + gl + 1000.0)))
        res = emulate(pes, opg, "digitizer", s, all_list)
        tot = vec(sum(Float64.(res.stored); dims = 1))
        trig = res.triggered
        tg = tot[trig]
        p511 = find_peak511(tg)
        t511 = tot[sel511 .& trig]; t1275 = tot[sel1275 .& trig]
        # empty decays with an external trigger at the decay time: the charge of the
        # gate without light, noise plus dark pulses after the baseline subtraction
        emp = vec(sum(Float64.(emulate(pes, opg, "digitizer", with(s; threshold_counts = 0.0), empty).stored); dims = 1))
        push!(cols["gate_start_ns"], g0); push!(cols["gate_ns"], gl)
        push!(cols["peak511"], p511.mean); push!(cols["fwhm511"], 2.355p511.sigma / p511.mean)
        push!(cols["fwhm511_mc"], 2.355std(t511) / mean(t511))
        push!(cols["ratio_1275_511_mc"], median(t1275) / median(t511))
        push!(cols["stored_fraction"], median(t511 ./ ev.n_detected[1:n][sel511 .& trig]))
        push!(cols["empty_charge_mean"], mean(emp)); push!(cols["empty_charge_rms"], std(emp))
        push!(cols["triggered_photopeak"], mean(trig[sel511])); push!(cols["triggered_empty"], mean(trig[ev.edep_kev[1:n] .== 0]))
        g = create_group(gt, "start$(Int(g0))_gate$(Int(gl))")
        hi = quantile(tg, 0.999)
        write_hist!(g, "sum", histogram(tg, 0.0, hi, 300)...)
        write_hist!(g, "empty", histogram(emp, quantile(emp, 0.001) - 1, quantile(emp, 0.999) + 1, 80)...)
        @printf("%-26s start %5.0f ns gate %6.0f ns: 511 keV at %6.0f, FWHM %.1f%% (tagged %.1f%%), 1275/511 %.2f, stored %.3f, empty gate %5.1f ± %4.1f, triggered photopeak %.3f empty %.3f\n",
                tag, g0, gl, p511.mean, 100cols["fwhm511"][end], 100cols["fwhm511_mc"][end],
                cols["ratio_1275_511_mc"][end], cols["stored_fraction"][end], mean(emp), std(emp),
                cols["triggered_photopeak"][end], cols["triggered_empty"][end])
        flush(stdout)
    end
end
order = ["gate_start_ns", "gate_ns", "peak511", "fwhm511", "fwhm511_mc", "ratio_1275_511_mc", "stored_fraction",
         "empty_charge_mean", "empty_charge_rms", "triggered_photopeak", "triggered_empty"]
path = joinpath(results_dir(campaign, "gate_scan"), "gate_scan.csv")
update_table(campaign, "gate_scan", "gate_scan",
            vcat(["tag" => fill(tag, length(points))], [k => cols[k] for k in order]))
println("written ", path)
