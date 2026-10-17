# First-look summary of an event file (emulated or measured, same layout): the energy
# spectrum of the equalised sum, the 511 keV peak position and resolution, the 1275 to
# 511 ratio, the trigger fraction, the brightest channel and the channel maps. Writes the
# figures and a TOML of numbers under fullsim_analysis/results/<campaign>/<tag>/.
#   julia --project=. fullsim_analysis/summary.jl <campaign> output/<tag>/events.h5 [more...]
using HDF5, Statistics, Printf, TOML
using CairoMakie

"Mean and sigma of a peak by truncated moments: iterate the window ±2 sigma around the mean."
function peak_fit(x::AbstractVector, lo::Real, hi::Real; iterations::Int = 8)
    sel = x[(x .>= lo) .& (x .<= hi)]
    isempty(sel) && return (mean = NaN, sigma = NaN, n = 0)
    # start from the mode of a coarse histogram
    edges = range(lo, hi; length = 61)
    h = [count(v -> edges[i] <= v < edges[i + 1], sel) for i in 1:60]
    m = 0.5 * (edges[argmax(h)] + edges[argmax(h) + 1])
    s = (hi - lo) / 8
    for _ in 1:iterations
        w = sel[abs.(sel .- m) .<= 2s]
        length(w) < 10 && break
        m, s = mean(w), std(w)
    end
    return (mean = m, sigma = s, n = count(abs.(sel .- m) .<= 2s))
end

function summarise(path::AbstractString, campaign::AbstractString)
    repo = dirname(@__DIR__)
    f = h5open(path, "r")
    run = attrs(f["run"])
    tag = run["tag"]; fe = run["frontend"]; kind = run["kind"]
    q = Float64.(read(f["events"]["charge_pe"]))        # 64 x n, in avalanches
    trig = Bool.(read(f["events"]["triggered"]))
    hits = read(f["events"]["hits"])
    flags = read(f["events"]["flags"])
    x = Float64.(read(f["channels"]["x_mm"])); y = Float64.(read(f["channels"]["y_mm"]))
    nch, n = size(q)
    hasmc = haskey(f, "mc")
    edep = hasmc ? read(f["mc"]["edep_kev"]) : nothing
    close(f)

    outdir = joinpath(repo, "fullsim_analysis", "results", campaign, tag)
    mkpath(outdir)
    good = trig
    tot = vec(sum(q; dims = 1))
    tg = tot[good]
    # 511 keV peak: the highest populated region of the upper spectrum
    hi = quantile(tg, 0.999)
    p511 = peak_fit(tg, 0.3hi, hi)
    # look for the 1275 keV peak above the 511 one
    p1275 = peak_fit(tg, 2.0 * p511.mean, 3.2 * p511.mean)
    res = 2.355 * p511.sigma / p511.mean
    win = abs.(tg .- p511.mean) .<= 2 * p511.sigma
    qg = q[:, good]
    brightest = vec(maximum(qg; dims = 1))
    bwin = brightest[win]
    occ = vec(mean(qg[:, win] .> 0.5; dims = 2))
    mq = vec(mean(qg[:, win]; dims = 2))
    # seed-channel photopeak map: position of the peak for the events whose brightest
    # channel is c, in units of the median over channels
    seed = vec(map(i -> i[1], argmax(qg; dims = 1)))
    seedpk = fill(NaN, nch)
    for c in 1:nch
        sel = tg[seed .== c]
        length(sel) > 200 || continue
        seedpk[c] = peak_fit(sel, 0.3hi, hi).mean
    end
    seedpk ./= median(filter(isfinite, seedpk))

    numbers = Dict(
        "tag" => tag, "frontend" => fe, "kind" => kind, "events" => n,
        "triggered_fraction" => mean(trig),
        "saturated_fraction" => mean((flags .& 0x01) .!= 0),
        "peak511_pe" => p511.mean, "peak511_sigma_pe" => p511.sigma,
        "resolution511_fwhm" => res, "events_in_511_window" => count(win),
        "peak1275_pe" => p1275.mean, "ratio_1275_511" => p1275.mean / p511.mean,
        "brightest_median_pe" => median(bwin), "brightest_p99_pe" => quantile(bwin, 0.99),
        "brightest_fraction_median" => median(bwin ./ tg[win]),
        "hits_per_event_median" => median(vec(sum(hits[:, good]; dims = 1))),
        "occupancy_min" => minimum(occ), "occupancy_max" => maximum(occ),
        "seed_peak_rms" => std(filter(isfinite, seedpk)))
    if hasmc
        numbers["photopeak_mc_fraction_triggered"] = mean(trig[edep .> 510])
        numbers["no_deposit_fraction_triggered"] = mean(trig[edep .== 0])
        # photostatistics of the stored value against the measured width
        numbers["mc_511_events"] = count(edep .> 510)
    end
    open(joinpath(outdir, "summary.toml"), "w") do io
        TOML.print(io, numbers)
    end

    # figures
    fig = Figure(size = (1200, 900))
    ax = Axis(fig[1, 1]; xlabel = "sum of the channels (avalanches)", ylabel = "events",
              title = "$(tag): equalised sum, $(fe)")
    hist!(ax, tg; bins = 300)
    vlines!(ax, [p511.mean]; color = :red, linestyle = :dash)
    text!(ax, p511.mean, 0.9 * maximum(hist(tg; bins = 300).plots[1][1][]...); text = "")
    ax2 = Axis(fig[1, 2]; xlabel = "brightest channel (avalanches)", ylabel = "events",
               title = @sprintf("511 keV window: median %.0f, p99 %.0f", median(bwin), quantile(bwin, 0.99)))
    hist!(ax2, bwin; bins = 150)
    ax3 = Axis(fig[2, 1]; title = "mean charge per channel, 511 keV window", aspect = 1)
    hm = heatmap!(ax3, x, y, mq)
    Colorbar(fig[2, 1][1, 2], hm)
    ax4 = Axis(fig[2, 2]; title = "seed-channel photopeak / median", aspect = 1)
    hm2 = heatmap!(ax4, x, y, seedpk; colorrange = (0.9, 1.1))
    Colorbar(fig[2, 2][1, 2], hm2)
    save(joinpath(outdir, "summary.png"), fig)

    @printf("%-24s %s: %d events, %.1f%% triggered; 511 keV at %.0f avalanches, FWHM %.1f%%; 1275/511 = %.2f; brightest median %.0f (p99 %.0f); saturated %.2f%%\n",
            tag, fe, n, 100mean(trig), p511.mean, 100res, p1275.mean / p511.mean,
            median(bwin), quantile(bwin, 0.99), 100numbers["saturated_fraction"])
    println("  written ", outdir)
    return numbers
end

length(ARGS) >= 2 || error("usage: summary.jl <campaign> <events.h5> [more...]")
for p in ARGS[2:end]
    summarise(p, ARGS[1])
end
