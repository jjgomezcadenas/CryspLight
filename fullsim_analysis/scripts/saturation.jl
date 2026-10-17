# Dynamic range of a front end on the photon file of a front-end configuration: for the
# 511 and 1275 keV photopeak decays, the distribution of the brightest channel and of the
# dimmest one, the fraction of events with a channel above the ceiling (the high-gain one
# for the Citiroc and the Radioroc2, the input range for the digitizer), and the pedestal
# per channel from empty decays. For the Radioroc2 the scan repeats over the peaking
# times. Writes results/<campaign>/saturation/saturation.csv and saturation.h5. Plot with
# fullsim_analysis/plots/saturation.py.
#   julia --project=. --threads=8 fullsim_analysis/scripts/saturation.jl <campaign> <frontend config> [n_per_class]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, HDF5, Statistics, Printf

campaign, cfg = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3000
r = open_pes_run(cfg)
st, pes, op = r.setup, r.pes, r.op
tag = st.tag
lists = event_lists(pes; nmax = nmax)
kind = st.frontend

# the settings to run and the ceiling of each, in avalanches
function cases()
    if kind == "radioroc"
        hgc = st.settings.lg_ceiling_pe * st.settings.lg_ratio * 1e6 / op.gain
        lgc = st.settings.lg_ceiling_pe * 1e6 / op.gain
        return [(string(Int(tp)), with(st.settings; tp_ns = fill(tp, 64), hold_ns = 1.7tp + 500,
                                       tq_threshold_pe = tp <= 600 ? 20.0 : 40.0,
                                       t_end_ns = max(st.settings.t_end_ns, 6tp + 4000)), hgc, lgc)
                for tp in (300.0, 600.0, 1200.0, 1800.0)]
    elseif kind == "citiroc"
        c = st.settings.ceiling_pc / (1.602176634e-7 * op.gain)
        return [("tp$(Int(st.settings.tp_ns))", st.settings, c, c)]
    else
        return [("gain$(st.settings.gain)", st.settings, Inf, Inf)]
    end
end

cols = Dict{String,Vector}(k => [] for k in
    ("case", "ceiling_hg", "ceiling_lg", "brightest511_median", "brightest511_p99", "brightest1275_median",
     "brightest1275_p99", "above_hg_511", "above_hg_1275", "above_lg_511", "above_lg_1275",
     "dimmest511_median", "dimmest511_p10", "pedestal_channel", "pedestal_channel_rms", "noise_hg_pe", "noise_lg_pe"))
open_results(campaign, "saturation"; mode = isfile(joinpath(results_dir(campaign, "saturation"), "saturation.h5")) ? "r+" : "w") do h5
    haskey(h5, tag) && delete_object(h5, tag)
    gt = create_group(h5, tag)
    stamp!(gt, "config" => st.cfg_text, "frontend" => kind)
    for (name, s, hgc, lgc) in cases()
        r511 = emulate(pes, op, kind, s, lists.photopeak511)
        r1275 = emulate(pes, op, kind, s, lists.photopeak1275)
        b511 = brightest(r511.stored[:, r511.triggered]); b1275 = brightest(r1275.stored[:, r1275.triggered])
        d511 = dimmest(r511.stored[:, r511.triggered])
        # pedestal per channel: empty decays with an external trigger where the chain has
        # one (Radioroc2); for the others the dim channels of 511 keV events stand in
        ped = if kind == "radioroc"
            vec(Float64.(emulate(pes, op, kind, with(s; trigger = "external"), lists.empty).stored))
        else
            Float64.(vec(r511.stored[:, r511.triggered]))[vec(r511.stored[:, r511.triggered]) .< quantile(vec(r511.stored[:, r511.triggered]), 0.1)]
        end
        nhg = kind == "radioroc" ? noise_at(s, s.tp_ns[1]) : kind == "citiroc" ? s.noise_pe : NaN
        nlg = kind == "radioroc" ? nhg / s.lg_ratio : kind == "citiroc" ? nhg / s.lg_ratio : NaN
        push!(cols["case"], name); push!(cols["ceiling_hg"], hgc); push!(cols["ceiling_lg"], lgc)
        push!(cols["brightest511_median"], median(b511)); push!(cols["brightest511_p99"], quantile(b511, 0.99))
        push!(cols["brightest1275_median"], median(b1275)); push!(cols["brightest1275_p99"], quantile(b1275, 0.99))
        push!(cols["above_hg_511"], mean(b511 .>= hgc * (1 - 1e-6))); push!(cols["above_hg_1275"], mean(b1275 .>= hgc * (1 - 1e-6)))
        push!(cols["above_lg_511"], mean(b511 .>= lgc * (1 - 1e-6))); push!(cols["above_lg_1275"], mean(b1275 .>= lgc * (1 - 1e-6)))
        push!(cols["dimmest511_median"], median(d511)); push!(cols["dimmest511_p10"], quantile(d511, 0.1))
        push!(cols["pedestal_channel"], mean(ped)); push!(cols["pedestal_channel_rms"], std(ped))
        push!(cols["noise_hg_pe"], nhg); push!(cols["noise_lg_pe"], nlg)
        g = create_group(gt, name)
        top = max(quantile(b1275, 0.999), isfinite(hgc) ? 1.2hgc : 0.0)
        write_hist!(g, "brightest_511", histogram(b511, 0.0, top, 150)...)
        write_hist!(g, "brightest_1275", histogram(b1275, 0.0, top, 150)...)
        write_hist!(g, "dimmest_511", histogram(d511, min(0.0, minimum(d511)), quantile(d511, 0.999), 100)...)
        write_hist!(g, "pedestal_channel", histogram(ped, minimum(ped) - 0.5, quantile(ped, 0.999) + 0.5, 100)...)
        attrs(g)["ceiling_hg"] = hgc; attrs(g)["ceiling_lg"] = lgc
        @printf("%-28s %-8s brightest 511 keV median %6.1f p99 %6.1f, 1275 keV median %6.1f p99 %6.1f; above HG ceiling %.0f: %.1f%% / %.1f%%; dimmest 511 keV median %.1f; pedestal per channel %.1f ± %.1f\n",
                tag, name, median(b511), quantile(b511, 0.99), median(b1275), quantile(b1275, 0.99), hgc,
                100mean(b511 .>= hgc * (1 - 1e-6)), 100mean(b1275 .>= hgc * (1 - 1e-6)), median(d511), mean(ped), std(ped))
        flush(stdout)
    end
end
order = ["case", "ceiling_hg", "ceiling_lg", "brightest511_median", "brightest511_p99", "brightest1275_median",
         "brightest1275_p99", "above_hg_511", "above_hg_1275", "above_lg_511", "above_lg_1275",
         "dimmest511_median", "dimmest511_p10", "pedestal_channel", "pedestal_channel_rms", "noise_hg_pe", "noise_lg_pe"]
path = joinpath(results_dir(campaign, "saturation"), "saturation.csv")
update_table(campaign, "saturation", "saturation",
            vcat(["tag" => fill(tag, length(cols["case"]))], [k => cols[k] for k in order]))
println("written ", path)
