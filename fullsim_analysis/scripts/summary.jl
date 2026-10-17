# First look at event files (emulated or measured, same layout): spectrum of the
# equalised sum, 511 and 1275 keV peaks, resolution, brightest channel, channel maps.
# Writes results/<campaign>/summary/summary.h5 (one group per tag) and summary.csv (one
# row per tag). Plot with fullsim_analysis/plots/summary.py.
#   julia --project=. fullsim_analysis/scripts/summary.jl <campaign> <events.h5> [more...]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using HDF5, Statistics, Printf

function summarise!(h5, e)
    tot = equalised_sum(e)
    good = e.trig
    tg = tot[good]
    p511 = find_peak511(tg)
    p1275 = find_peak1275(tg, p511)
    win = peak_window(tg, p511)
    qg = e.q[:, good]
    br = brightest(qg)[win]
    res = 2.355 * p511.sigma / p511.mean
    hi = isfinite(p1275.mean) ? 1.6 * p1275.mean : quantile(tg, 0.999)
    g = create_group(h5, e.tag)
    stamp!(g, "frontend" => e.frontend, "kind" => e.kind, "events" => e.n, "path" => e.path)
    write_hist!(g, "sum", histogram(tg, 0.0, hi, 300)...)
    write_hist!(g, "brightest", histogram(br, 0.0, quantile(br, 0.999), 150)...)
    write_matrix!(g, "mean_map", mean_map(e, findall(good)[win] |> idx -> (m = falses(e.n); m[idx] .= true; m)))
    write_matrix!(g, "occupancy_map", occupancy_map(e, findall(good)[win] |> idx -> (m = falses(e.n); m[idx] .= true; m)))
    write_matrix!(g, "seed_map", seed_peak_map(e, tot, good, 0.25 * quantile(tg, 0.995), quantile(tg, 0.995)))
    write_vector!(g, "channel_x_mm", e.x); write_vector!(g, "channel_y_mm", e.y)
    row = Dict{String,Any}(
        "tag" => e.tag, "frontend" => e.frontend, "kind" => e.kind, "events" => e.n,
        "triggered_fraction" => mean(e.trig),
        "saturated_fraction" => mean((e.flags .& 0x01) .!= 0),
        "peak511" => p511.mean, "sigma511" => p511.sigma, "fwhm511" => res,
        "peak1275" => p1275.mean, "ratio_1275_511" => p1275.mean / p511.mean,
        "brightest_median" => median(br), "brightest_p99" => quantile(br, 0.99),
        "brightest_fraction" => median(br ./ tg[win]),
        "hits_median" => median(vec(sum(e.hits[:, good]; dims = 1))))
    if e.mc !== nothing
        pk = select_photopeak511(e)
        row["photopeak_triggered"] = mean(e.trig[pk])
        row["empty_triggered"] = mean(e.trig[select_empty(e)])
        row["ratio_1275_511_mc"] = median(tot[select_photopeak1275(e) .& e.trig]) / median(tot[pk .& e.trig])
        row["fwhm511_mc"] = 2.355 * std(tot[pk .& e.trig]) / mean(tot[pk .& e.trig])
    end
    for (k, v) in row
        attrs(g)[k] = v
    end
    @printf("%-26s %-9s %7d events, %5.1f%% triggered; 511 keV at %6.0f, FWHM %.1f%%; 1275/511 %.2f; brightest %.0f (p99 %.0f); saturated %.2f%%\n",
            e.tag, e.frontend, e.n, 100row["triggered_fraction"], p511.mean, 100res,
            row["ratio_1275_511"], median(br), quantile(br, 0.99), 100row["saturated_fraction"])
    return row
end

length(ARGS) >= 2 || error("usage: summary.jl <campaign> <events.h5> [more...]")
campaign = ARGS[1]
rows = Dict{String,Any}[]
open_results(campaign, "summary"; mode = isfile(joinpath(results_dir(campaign, "summary"), "summary.h5")) ? "r+" : "w") do h5
    for p in ARGS[2:end]
        e = read_events(p)
        haskey(h5, e.tag) && delete_object(h5, e.tag)
        push!(rows, summarise!(h5, e))
    end
end
# the table carries every tag of the file, so that repeated runs accumulate
all_rows = Dict{String,Any}[]
h5open(joinpath(results_dir(campaign, "summary"), "summary.h5"), "r") do h5
    for tag in keys(h5)
        a = attrs(h5[tag])
        push!(all_rows, Dict{String,Any}(k => a[k] for k in keys(a)))
    end
end
cols = ["tag", "frontend", "kind", "events", "triggered_fraction", "photopeak_triggered",
        "empty_triggered", "peak511", "sigma511", "fwhm511", "fwhm511_mc", "peak1275",
        "ratio_1275_511", "ratio_1275_511_mc", "brightest_median", "brightest_p99",
        "brightest_fraction", "saturated_fraction", "hits_median"]
write_table(campaign, "summary", "summary",
            [c => [get(r, c, NaN) for r in all_rows] for c in cols])
println("written ", results_dir(campaign, "summary"))
