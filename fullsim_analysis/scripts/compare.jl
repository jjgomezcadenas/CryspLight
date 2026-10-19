# Comparison of the front ends on one crystal: the five questions of the first study
# (working plan, "The first study") answered from emulated event files of the same photon
# file. For each event file, on the 511 keV photopeak decays tagged by the simulation:
#   calibration     the noise of the calibration path in avalanches and the resulting
#                   single-photoelectron signal-to-noise ratio (for the digitizer the
#                   amplitude of one avalanche over the noise per sample, the quantity
#                   that decides whether the dark-pulse spectrum shows its peaks)
#   resolution      the 511 and 1275 keV peaks (Gaussian plus linear fits on the
#                   triggered sum), the FWHM at 511 keV relative to the peak position and
#                   from the energy scale of the two peaks, and the width of the tagged
#                   photopeak sum
#   light           photons emitted, photons at the active areas, the detection
#                   efficiency at the peak wavelength, the stored signal per emitted
#                   photon and per photon at the SiPMs
#   charges         mean charge per channel and charge of the seed channel (median and
#                   1% tail), the dimmest channel, and the fraction of photopeak events
#                   flagged as saturated by the front end
# The energy is the sum of the channels of the energy path: the charge in avalanches for
# the digitizer and the Citiroc (high gain), the low-gain charge for the Radioroc2. For the
# ASICs the pedestal of the peak detectors (fullsim_analysis/scripts/pedestal.jl, which
# must have been run on the same configuration) is subtracted channel by channel before
# anything else; the two-peak energy scale then measures what offset remains.
# Writes results/<campaign>/compare/compare.csv (one row per event file, rows of the same
# tag replaced) and compare.h5 (sum spectra of the tagged photopeak events, seed
# distributions). Plot with fullsim_analysis/plots/compare.py.
#   julia --project=. fullsim_analysis/scripts/compare.jl <campaign> <events.h5> [more...]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using HDF5, Statistics, Printf

const E_PC = 1.602176634e-7

"Energy path: charge per channel in avalanches, and its ceiling in avalanches (Inf if none)."
function energy_path(e)
    if e.frontend == "radioroc"
        e.qlg === nothing && error("$(e.tag): no low-gain calibration in the event file")
        gain = median(Float64.(e.calib["gain_electrons"]))
        ceil = e.run["fe_lg_ceiling_pe"] * 1e6 / gain
        return e.qlg, ceil
    elseif e.frontend == "citiroc"
        gain = median(Float64.(e.calib["gain_electrons"]))
        return e.q, e.run["fe_ceiling_pc"] / (E_PC * gain)
    else
        return e.q, Inf
    end
end

"Noise of the calibration path in avalanches (rms of the stored value without light)."
calibration_noise(e) = median(Float64.(e.calib["pedestal_rms"]) ./ Float64.(e.calib["gain_per_pe"]))

"Single-avalanche signal over noise on the path used to calibrate the gain."
spe_snr(e) = e.frontend == "digitizer" ?
    median(Float64.(e.calib["peak_counts_per_pe"])) / e.run["fe_noise_counts"] :
    1 / calibration_noise(e)

"Peak-detector pedestal per channel of the energy path (zeros for the digitizer)."
function pedestal_of(campaign, e)
    e.frontend == "digitizer" && return zeros(e.nch)
    path = joinpath(results_dir(campaign, "pedestal"), "pedestal.h5")
    isfile(path) || error("no pedestals: run fullsim_analysis/scripts/pedestal.jl first")
    h5open(path, "r") do f
        haskey(f, e.tag) || error("$(e.tag): no pedestal in $(path)")
        read(f[e.tag]["energy_mean"])
    end
end

function compare_one!(h5, campaign, e)
    qraw, ceil = energy_path(e)
    ped = pedestal_of(campaign, e)
    q = qraw .- ped
    pk = select_photopeak511(e) .& e.trig
    tot_all = vec(sum(q[:, e.trig]; dims = 1))
    p511 = find_peak511(tot_all)
    p1275 = find_peak1275(tot_all, p511)
    slope = (p1275.mean - p511.mean) / (1275 - 511)      # sum per keV
    offset_kev = 511 - p511.mean / slope                 # residual offset of the scale
    tot = vec(sum(q[:, pk]; dims = 1))
    seed = brightest(q[:, pk]); dim = dimmest(q[:, pk])
    nemit = Float64.(e.mc.nemit[pk]); nph = Float64.(e.mc.nph[pk])
    noise = calibration_noise(e)
    row = Dict{String,Any}(
        "tag" => e.tag, "frontend" => e.frontend, "crystal" => e.run["crystal"],
        "overvoltage_v" => e.run["overvoltage_v"], "events" => e.n,
        "photopeak_events" => count(pk), "photopeak_triggered" => mean(e.trig[select_photopeak511(e)]),
        "calib_noise_pe" => noise, "spe_snr" => spe_snr(e),
        "pedestal_sum" => sum(ped), "peak511" => p511.mean, "peak1275" => p1275.mean,
        "ratio_1275_511" => p1275.mean / p511.mean, "offset_kev" => offset_kev,
        "fwhm511_fit" => 2.355 * p511.sigma / p511.mean,
        "fwhm511_scale" => 2.355 * p511.sigma / slope / 511,
        "fwhm511_tagged" => 2.355 * std(tot) / mean(tot),
        "photons_emitted" => mean(nemit), "photons_at_sipm" => mean(nph),
        "collection" => mean(nph ./ nemit), "pde_peak" => median(Float64.(e.calib["pde_peak"])),
        "stored_per_emitted" => mean(tot ./ nemit), "stored_per_photon_at_sipm" => mean(tot ./ nph),
        "charge_per_channel_mean" => mean(tot) / e.nch,
        "seed_median" => median(seed), "seed_p99" => quantile(seed, 0.99),
        "seed_fraction_median" => median(seed ./ tot),
        "dimmest_median" => median(dim), "dimmest_p10" => quantile(dim, 0.1),
        "ceiling" => ceil, "saturated" => mean((e.flags[pk] .& 0x01) .!= 0))
    haskey(h5, e.tag) && delete_object(h5, e.tag)
    g = create_group(h5, e.tag)
    stamp!(g, "frontend" => e.frontend, "path" => e.path)
    lo, hi = quantile(tot, 0.002), quantile(tot, 0.998)
    write_hist!(g, "photopeak_sum", histogram(tot ./ mean(tot), lo / mean(tot) - 0.05, hi / mean(tot) + 0.05, 120)...)
    write_hist!(g, "seed", histogram(seed, 0.0, quantile(seed, 0.999), 120)...)
    write_hist!(g, "dimmest", histogram(dim, min(0.0, minimum(dim)), quantile(dim, 0.999), 80)...)
    @printf("%-32s %-9s %4.1f V: pedestal %.0f; 511 keV FWHM %.1f%% (two-peak scale %.1f%%, offset %.0f keV, ratio %.3f; tagged %.1f%%); stored/emitted %.3f; per channel %.1f, seed %.0f (p99 %.0f), dimmest %.1f; saturated %.2f%%; SPE S/N %.1f\n",
            e.tag, e.frontend, e.run["overvoltage_v"], sum(ped), 100row["fwhm511_fit"], 100row["fwhm511_scale"],
            offset_kev, row["ratio_1275_511"], 100row["fwhm511_tagged"],
            row["stored_per_emitted"], row["charge_per_channel_mean"], median(seed), quantile(seed, 0.99),
            median(dim), 100row["saturated"], row["spe_snr"])
    return row
end

length(ARGS) >= 2 || error("usage: compare.jl <campaign> <events.h5> [more...]")
campaign = ARGS[1]
rows = Dict{String,Any}[]
mode = isfile(joinpath(results_dir(campaign, "compare"), "compare.h5")) ? "r+" : "w"
open_results(campaign, "compare"; mode = mode) do h5
    for p in ARGS[2:end]
        push!(rows, compare_one!(h5, campaign, read_events(p)))
    end
end
cols = ["tag", "frontend", "crystal", "overvoltage_v", "events", "photopeak_events", "photopeak_triggered",
        "calib_noise_pe", "spe_snr", "pedestal_sum", "peak511", "peak1275", "ratio_1275_511", "offset_kev",
        "fwhm511_fit", "fwhm511_scale", "fwhm511_tagged", "photons_emitted",
        "photons_at_sipm", "collection", "pde_peak", "stored_per_emitted", "stored_per_photon_at_sipm",
        "charge_per_channel_mean", "seed_median", "seed_p99", "seed_fraction_median", "dimmest_median",
        "dimmest_p10", "ceiling", "saturated"]
println("written ", update_table(campaign, "compare", "compare", [c => [r[c] for r in rows] for c in cols]))
