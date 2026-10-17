# First-look summary of an event file (emulated or measured, same layout): the energy
# spectrum of the equalised sum, the 511 keV peak position and resolution, the 1275 to
# 511 ratio, the trigger fraction, the brightest channel and the channel maps. Writes the
# figures and a TOML of numbers under fullsim_analysis/results/<campaign>/<tag>/.
#   julia --project=. fullsim_analysis/summary.jl <campaign> output/<tag>/events.h5 [more...]
using HDF5, Statistics, Printf, TOML, LinearAlgebra
using CairoMakie

"""
Gaussian plus linear background fitted to a histogram (bin centres xc, counts c) by
Gauss-Newton with Poisson weights: returns (A, mu, sigma, b0, b1) or nothing.
"""
function fit_gauss_linear(xc::AbstractVector, c::AbstractVector, mu0::Real, sigma0::Real)
    x0 = mu0
    model(p) = p[1] .* exp.(-0.5 .* ((xc .- p[2]) ./ p[3]) .^ 2) .+ p[4] .+ p[5] .* (xc .- x0)
    bkg = 0.5 * (mean(c[1:max(1, length(c) ÷ 10)]) + mean(c[end - max(0, length(c) ÷ 10 - 1):end]))
    p = [max(maximum(c) - bkg, 1.0), mu0, sigma0, bkg, 0.0]
    w = 1 ./ max.(c, 1.0)
    lambda = 1e-3
    chi2(p) = sum(w .* (c .- model(p)) .^ 2)
    best = chi2(p)
    for _ in 1:60
        J = zeros(length(xc), 5)
        for j in 1:5
            dp = zeros(5); dp[j] = max(1e-6, 1e-4 * abs(p[j]))
            J[:, j] = (model(p .+ dp) .- model(p .- dp)) ./ (2dp[j])
        end
        r = c .- model(p)
        H = J' * (w .* J)
        g = J' * (w .* r)
        step = (H + lambda * Diagonal(diag(H) .+ 1e-12)) \ g
        pn = p .+ step
        pn[3] = abs(pn[3])
        cn = chi2(pn)
        if cn < best
            p, best, lambda = pn, cn, lambda / 3
        else
            lambda *= 10
        end
        lambda > 1e8 && break
    end
    (isfinite(p[2]) && p[3] > 0 && p[1] > 0) || return nothing
    return (A = p[1], mu = p[2], sigma = p[3], b0 = p[4], b1 = p[5])
end

"""
Mean and sigma of the most populated peak of x within [lo, hi]: the mode of a histogram
starts a Gaussian plus linear fit in a window of ±4 initial sigmas (5% of the mode),
repeated once with the fitted width.
"""
function peak_fit(x::AbstractVector, lo::Real, hi::Real; nbins::Int = 100)
    sel = x[(x .>= lo) .& (x .<= hi)]
    length(sel) < 100 && return (mean = NaN, sigma = NaN, n = 0)
    edges = range(lo, hi; length = nbins + 1)
    c = [count(v -> edges[i] <= v < edges[i + 1], sel) for i in 1:nbins]
    xc = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    mu, sg = xc[argmax(c)], 0.05 * xc[argmax(c)]
    for _ in 1:2
        w0, w1 = mu - 4sg, mu + 4sg
        e2 = range(w0, w1; length = 41)
        c2 = [count(v -> e2[i] <= v < e2[i + 1], sel) for i in 1:40]
        x2 = 0.5 .* (e2[1:end-1] .+ e2[2:end])
        f = fit_gauss_linear(x2, c2, mu, sg)
        f === nothing && return (mean = NaN, sigma = NaN, n = 0)
        mu, sg = f.mu, f.sigma
    end
    return (mean = mu, sigma = sg, n = count(abs.(sel .- mu) .<= 2sg))
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
    hi = quantile(tg, 0.995)
    p511 = peak_fit(tg, 0.25hi, hi)
    # the 1275 keV peak: the most populated region between 2.2 and 2.8 times the 511 one
    p1275 = isfinite(p511.mean) ? peak_fit(tg, 2.2 * p511.mean, 2.8 * p511.mean) :
                                  (mean = NaN, sigma = NaN, n = 0)
    res = 2.355 * p511.sigma / p511.mean
    win = isfinite(p511.mean) ? abs.(tg .- p511.mean) .<= 2 * p511.sigma : trues(length(tg))
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
        length(sel) > 100 || continue
        seedpk[c] = peak_fit(sel, 0.3hi, hi).mean
    end
    any(isfinite, seedpk) && (seedpk ./= median(filter(isfinite, seedpk)))

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
        "seed_peak_rms" => count(isfinite, seedpk) > 1 ? std(filter(isfinite, seedpk)) : NaN)
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
