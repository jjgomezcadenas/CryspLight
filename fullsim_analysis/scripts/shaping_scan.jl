# Shaping-time scan of the Radioroc2 on the photon file of a front-end configuration: for
# each peaking time the 511 keV peak, its width, the 1275 to 511 ratio (fitted and from the
# tagged decays), the pedestal stored by the 64 peak detectors without light (empty decays
# with an external trigger, as a pulser run), the fraction of 511 and 1275 keV events with
# a channel above the high-gain ceiling, and the triggered fractions. Writes
# results/<campaign>/shaping_scan/shaping_scan.csv (rows of this tag replaced) and the
# spectra into shaping_scan.h5. Plot with fullsim_analysis/plots/shaping_scan.py.
#   julia --project=. --threads=8 fullsim_analysis/scripts/shaping_scan.jl <campaign> <frontend config> [n_decays]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, HDF5, Statistics, Printf

campaign, cfg = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 60_000
r = open_pes_run(cfg)
st, pes, op = r.setup, r.pes, r.op
st.frontend == "radioroc" || error("the shaping scan is for the Radioroc2 configurations")
tag = st.tag
ev = pes.events
n = min(nmax, length(ev.edep_kev))
all_list = collect(1:n)
pk = findall((ev.edep_511_kev .> 510) .& (ev.edep_1275_kev .== 0) .& (1:length(ev.edep_kev) .<= n))
pk2 = findall((ev.edep_1275_kev .> 1273) .& (ev.edep_511_kev .== 0) .& (1:length(ev.edep_kev) .<= n))
empty = findall(ev.edep_kev .== 0)[1:min(3000, count(ev.edep_kev .== 0))]
tps = [300.0, 600.0, 1200.0, 1800.0]
hg_ceiling = st.settings.lg_ceiling_pe * st.settings.lg_ratio * 1e6 / op.gain

cols = Dict{String,Vector}(k => Float64[] for k in
    ("tp_ns", "tq_threshold_pe", "peak511", "fwhm511", "fwhm511_mc", "ratio_1275_511_fit", "ratio_1275_511_mc",
     "pedestal", "pedestal_rms", "stored_fraction", "hg_saturated_511", "hg_saturated_1275",
     "triggered_photopeak", "triggered_empty"))
open_results(campaign, "shaping_scan"; mode = isfile(joinpath(results_dir(campaign, "shaping_scan"), "shaping_scan.h5")) ? "r+" : "w") do h5
    haskey(h5, tag) && delete_object(h5, tag)
    gt = create_group(h5, tag)
    stamp!(gt, "config" => st.cfg_text, "n_decays" => n, "hg_ceiling" => hg_ceiling)
    for tp in tps
        tq = tp <= 600 ? 20.0 : 40.0     # above the dark pedestal of one channel
        s = with(st.settings; tp_ns = fill(tp, 64), hold_ns = 1.7tp + 500, tq_threshold_pe = tq,
                 t_end_ns = max(st.settings.t_end_ns, 6tp + 4000))
        res = emulate(pes, op, "radioroc", s, all_list)
        tot = vec(sum(Float64.(res.stored); dims = 1))
        trig = res.triggered
        tg = tot[trig]
        p511 = find_peak511(tg)
        p1275 = find_peak1275(tg, p511)
        sel511 = falses(n); sel511[pk] .= true
        sel1275 = falses(n); sel1275[pk2] .= true
        t511 = tot[sel511 .& trig]; t1275 = tot[sel1275 .& trig]
        br511 = brightest(res.stored[:, sel511 .& trig]); br1275 = brightest(res.stored[:, sel1275 .& trig])
        # pedestal: empty decays with an external trigger at the decay time
        sp = with(s; trigger = "external")
        ped = vec(sum(Float64.(emulate(pes, op, "radioroc", sp, empty).stored); dims = 1))
        push!(cols["tp_ns"], tp); push!(cols["tq_threshold_pe"], tq)
        push!(cols["peak511"], p511.mean); push!(cols["fwhm511"], 2.355p511.sigma / p511.mean)
        push!(cols["fwhm511_mc"], 2.355std(t511) / mean(t511))
        push!(cols["ratio_1275_511_fit"], p1275.mean / p511.mean)
        push!(cols["ratio_1275_511_mc"], median(t1275) / median(t511))
        push!(cols["pedestal"], median(ped)); push!(cols["pedestal_rms"], std(ped))
        push!(cols["stored_fraction"], median(t511 ./ ev.n_detected[1:n][sel511 .& trig]))
        push!(cols["hg_saturated_511"], mean(br511 .> hg_ceiling))
        push!(cols["hg_saturated_1275"], mean(br1275 .> hg_ceiling))
        push!(cols["triggered_photopeak"], mean(trig[sel511])); push!(cols["triggered_empty"], mean(trig[(ev.edep_kev[1:n] .== 0)]))
        g = create_group(gt, "tp_$(Int(tp))")
        hi = isfinite(p1275.mean) ? 1.6p1275.mean : quantile(tg, 0.999)
        write_hist!(g, "sum", histogram(tg, 0.0, hi, 300)...)
        write_hist!(g, "brightest_511", histogram(br511, 0.0, max(quantile(br511, 0.999), hg_ceiling * 1.2), 120)...)
        write_hist!(g, "brightest_1275", histogram(br1275, 0.0, max(quantile(br1275, 0.999), hg_ceiling * 1.2), 120)...)
        write_hist!(g, "pedestal", histogram(ped, minimum(ped) - 1, maximum(ped) + 1, 60)...)
        @printf("%-28s tp %5.0f ns: 511 keV at %6.0f, FWHM %.1f%% (tagged %.1f%%), 1275/511 %.2f (fit) %.2f (mc), pedestal %5.0f ± %4.0f, HG saturated %.1f%% / %.1f%%, triggered photopeak %.3f empty %.3f\n",
                tag, tp, p511.mean, 100cols["fwhm511"][end], 100cols["fwhm511_mc"][end],
                cols["ratio_1275_511_fit"][end], cols["ratio_1275_511_mc"][end], median(ped), std(ped),
                100cols["hg_saturated_511"][end], 100cols["hg_saturated_1275"][end],
                cols["triggered_photopeak"][end], cols["triggered_empty"][end])
        flush(stdout)
    end
end
order = ["tp_ns", "tq_threshold_pe", "peak511", "fwhm511", "fwhm511_mc", "ratio_1275_511_fit", "ratio_1275_511_mc",
         "pedestal", "pedestal_rms", "stored_fraction", "hg_saturated_511", "hg_saturated_1275",
         "triggered_photopeak", "triggered_empty"]
path = joinpath(results_dir(campaign, "shaping_scan"), "shaping_scan.csv")
update_table(campaign, "shaping_scan", "shaping_scan",
            vcat(["tag" => fill(tag, length(tps))], [k => cols[k] for k in order]))
println("written ", path)
