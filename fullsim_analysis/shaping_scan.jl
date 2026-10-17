# Shaping-time scan of the Radioroc2 on an optical run: for each peaking time of the slow
# mode (and the longest of the fast mode) the front-end stage runs on a subset of decays
# and the summary numbers are collected: 511 keV peak position in avalanches, its FWHM,
# the 1275 to 511 ratio, the pedestal of the empty decays (dark pulses and noise stored by
# the 64 peak detectors), the triggered fractions of photopeak and empty decays. Writes
# fullsim_analysis/results/<campaign>/shaping_scan_<tag>.toml and a figure.
#   julia --project=. --threads=8 fullsim_analysis/shaping_scan.jl <campaign> <frontend config> [n_decays]
# The front-end configuration gives the photon file, the operating point and the trigger;
# only the peaking time and the hold (set to 1.7 tp + 500 ns) are varied.
using CryspLight, HDF5, Statistics, Printf, TOML
using CairoMakie
include(joinpath(@__DIR__, "summary.jl"))   # peak_fit (the script part is guarded below)

function shaping_scan(campaign::AbstractString, cfg_path::AbstractString, nmax::Int)
    repo = dirname(@__DIR__)
    cfg_text = read(cfg_path, String)
    base = TOML.parse(cfg_text)
    tag = splitext(basename(cfg_path))[1]
    tps = [300.0, 600.0, 1200.0, 1800.0]
    tmp = mktempdir()
    rows = Dict{String,Vector{Float64}}(k => Float64[] for k in
        ("tp_ns", "peak511", "fwhm511", "ratio_1275_511", "ratio_1275_511_mc", "pedestal_empty",
         "pedestal_rms_empty", "triggered_photopeak", "triggered_empty", "stored_fraction"))
    for tp in tps
        c = deepcopy(base)
        c["run"]["tag"] = "$(tag)_tp$(Int(tp))"
        c["run"]["n_events"] = nmax
        c["run"]["output_dir"] = tmp
        c["frontend"]["tp_ns"] = tp
        c["frontend"]["hold_ns"] = 1.7 * tp + 500.0
        c["frontend"]["t_end_ns"] = max(c["frontend"]["t_end_ns"], 6 * tp + 4000.0)
        p = joinpath(tmp, "cfg_$(Int(tp)).toml")
        open(p, "w") do io; TOML.print(io, c); end
        out = run_fullsim_frontend(p; verbose = false)
        h5open(out, "r") do f
            q = Float64.(read(f["events"]["charge_pe"]))
            trig = Bool.(read(f["events"]["triggered"]))
            e511 = read(f["mc"]["edep_511_kev"]); e1275 = read(f["mc"]["edep_1275_kev"])
            edep = read(f["mc"]["edep_kev"])
            tot = vec(sum(q; dims = 1))
            pk = (e511 .> 510) .& (e1275 .== 0)
            empty = edep .== 0
            tg = tot[trig]
            hi = quantile(tg, 0.995)
            p511 = peak_fit(tg, 0.25hi, hi)
            p1275 = isfinite(p511.mean) ? peak_fit(tg, 2.2p511.mean, 2.8p511.mean) : (mean = NaN,)
            # 1275 to 511 ratio from the decays tagged by the simulation, no fit
            pk1275 = (e1275 .> 1273) .& (e511 .== 0) .& trig
            ratio_mc = median(tot[pk1275]) / median(tot[pk .& trig])
            # pedestal: what the 64 peak detectors store without light, from a second pass
            # over empty decays with an external trigger at the decay time (a pulser run)
            ped = Float64[]
            push!(rows["tp_ns"], tp)
            push!(rows["peak511"], p511.mean); push!(rows["fwhm511"], 2.355p511.sigma / p511.mean)
            push!(rows["ratio_1275_511"], p1275.mean / p511.mean)
            push!(rows["ratio_1275_511_mc"], ratio_mc)
            push!(rows["triggered_photopeak"], mean(trig[pk])); push!(rows["triggered_empty"], mean(trig[empty]))
            nph = read(f["mc"]["n_photons_at_sipm"])
            push!(rows["stored_fraction"], median(tot[pk .& trig] ./ nph[pk .& trig]))
        end
        # pedestal pass: empty decays, external trigger, 3000 of them
        c2 = deepcopy(c)
        c2["run"]["tag"] = "$(tag)_tp$(Int(tp))_ped"
        c2["frontend"]["trigger"] = "external"
        c2["run"]["n_events"] = min(nmax, 20_000)
        p2 = joinpath(tmp, "cfg_$(Int(tp))_ped.toml")
        open(p2, "w") do io; TOML.print(io, c2); end
        out2 = run_fullsim_frontend(p2; verbose = false)
        ped = h5open(out2, "r") do f
            q = Float64.(read(f["events"]["charge_pe"])); edep = read(f["mc"]["edep_kev"])
            vec(sum(q; dims = 1))[edep .== 0]
        end
        push!(rows["pedestal_empty"], median(ped)); push!(rows["pedestal_rms_empty"], std(ped))
        @printf("%-28s tp %5.0f ns: 511 keV at %6.0f, FWHM %.1f%%, 1275/511 %.2f (fit) %.2f (mc), pedestal %5.0f ± %4.0f, triggered photopeak %.3f empty %.3f\n",
                tag, tp, rows["peak511"][end], 100rows["fwhm511"][end], rows["ratio_1275_511"][end],
                rows["ratio_1275_511_mc"][end], rows["pedestal_empty"][end], rows["pedestal_rms_empty"][end],
                rows["triggered_photopeak"][end], rows["triggered_empty"][end])
        flush(stdout)
    end
    outdir = joinpath(repo, "fullsim_analysis", "results", campaign)
    mkpath(outdir)
    open(joinpath(outdir, "shaping_scan_$(tag).toml"), "w") do io
        TOML.print(io, merge(Dict{String,Any}("tag" => tag, "n_decays" => nmax), rows))
    end
    fig = Figure(size = (1200, 420))
    ax1 = Axis(fig[1, 1]; xlabel = "peaking time (ns)", ylabel = "FWHM at 511 keV", title = tag)
    scatterlines!(ax1, rows["tp_ns"], rows["fwhm511"])
    ax2 = Axis(fig[1, 2]; xlabel = "peaking time (ns)", ylabel = "stored fraction of the photons")
    scatterlines!(ax2, rows["tp_ns"], rows["stored_fraction"])
    ax3 = Axis(fig[1, 3]; xlabel = "peaking time (ns)", ylabel = "1275 / 511 peak ratio")
    scatterlines!(ax3, rows["tp_ns"], rows["ratio_1275_511"])
    save(joinpath(outdir, "shaping_scan_$(tag).png"), fig)
    println("written ", outdir)
    return rows
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    length(ARGS) >= 2 || error("usage: shaping_scan.jl <campaign> <frontend config> [n_decays]")
    shaping_scan(ARGS[1], ARGS[2], length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 60_000)
end
