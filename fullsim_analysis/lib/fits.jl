# Peak fits shared by every script: a Gaussian plus a linear background on a histogram.

"""
    fit_gauss_linear(xc, c, mu0, sigma0) -> (A, mu, sigma, b0, b1) or nothing

Gauss-Newton with Levenberg damping and Poisson weights on the bin counts c at centres xc.
The background is b0 + b1 (x - mu0).
"""
function fit_gauss_linear(xc::AbstractVector, c::AbstractVector, mu0::Real, sigma0::Real)
    x0 = mu0
    model(p) = p[1] .* exp.(-0.5 .* ((xc .- p[2]) ./ p[3]) .^ 2) .+ p[4] .+ p[5] .* (xc .- x0)
    ntail = max(1, length(c) ÷ 10)
    bkg = 0.5 * (mean(c[1:ntail]) + mean(c[end - ntail + 1:end]))
    p = [max(maximum(c) - bkg, 1.0), mu0, sigma0, bkg, 0.0]
    w = 1 ./ max.(c, 1.0)
    chi2(p) = sum(w .* (c .- model(p)) .^ 2)
    best = chi2(p)
    lambda = 1e-3
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
    peak_fit(x, lo, hi; nbins = 100) -> (mean, sigma, n)

Mean and sigma of the most populated peak of x within [lo, hi]: the mode of a histogram
starts a Gaussian plus linear fit in ±4 initial sigmas (5% of the mode), repeated with the
fitted width. The range only locates the mode: the fit window takes all of x, so a peak
near an edge of the range is not cut. n is the number of entries within ±2 sigma. NaN
when the fit fails.
"""
function peak_fit(x::AbstractVector, lo::Real, hi::Real; nbins::Int = 100)
    sel = x[(x .>= lo) .& (x .<= hi)]
    length(sel) < 100 && return (mean = NaN, sigma = NaN, n = 0)
    edges = range(lo, hi; length = nbins + 1)
    c = [count(v -> edges[i] <= v < edges[i + 1], sel) for i in 1:nbins]
    xc = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    mu, sg = xc[argmax(c)], 0.05 * xc[argmax(c)]
    for _ in 1:2
        e2 = range(mu - 4sg, mu + 4sg; length = 41)
        c2 = [count(v -> e2[i] <= v < e2[i + 1], x) for i in 1:40]
        x2 = 0.5 .* (e2[1:end-1] .+ e2[2:end])
        f = fit_gauss_linear(x2, c2, mu, sg)
        f === nothing && return (mean = NaN, sigma = NaN, n = 0)
        mu, sg = f.mu, f.sigma
    end
    return (mean = mu, sigma = sg, n = count(abs.(x .- mu) .<= 2sg))
end

"""
    peak_fit_window(x, lo, hi, w; nbins = 40) -> (mean, sigma, n)

The most populated peak of x within [lo, hi], fitted with a Gaussian plus a linear
background in a window of ±w around the mode of the histogram (nbins bins over [lo, hi]),
then again in ±2.5 fitted sigmas (at least ±w/2). n is the number of entries within ±2
sigma. NaN when there are fewer than 100 entries in [lo, hi] or the fit fails.
"""
function peak_fit_window(x::AbstractVector, lo::Real, hi::Real, w::Real; nbins::Int = 40)
    sel = x[(x .>= lo) .& (x .<= hi)]
    length(sel) < 100 && return (mean = NaN, sigma = NaN, n = 0)
    edges = range(lo, hi; length = nbins + 1)
    c = [count(v -> edges[i] <= v < edges[i + 1], sel) for i in 1:nbins]
    mu, sg, half = 0.5 * (edges[argmax(c)] + edges[argmax(c) + 1]), w / 4, w
    for _ in 1:2
        e2 = range(mu - half, mu + half; length = 31)
        c2 = [count(v -> e2[i] <= v < e2[i + 1], x) for i in 1:30]
        f = fit_gauss_linear(0.5 .* (e2[1:end-1] .+ e2[2:end]), c2, mu, sg)
        (f === nothing || abs(f.mu - mu) > half) && return (mean = NaN, sigma = NaN, n = 0)
        mu, sg = f.mu, f.sigma
        half = max(2.5sg, w / 2)
    end
    return (mean = mu, sigma = sg, n = count(abs.(x .- mu) .<= 2sg))
end

"""
    photoelectron_ladder(x, nominal; pedestal) -> (gain, zero, peaks)

Gain and zero level of a self-triggered dark-count spectrum x (ADC counts). The peaks
are located as the minima of the second derivative of the histogram (bins of a
fiftieth of the nominal gain), smoothed with a Gaussian of 0.12 nominal gains, which
resolves a peak that is only a shoulder on the tail of the previous one; each minimum is
refined by a parabola. Starting from the most prominent peak, the chain of peaks spaced
by 0.6 to 1.4 times the nominal gain (then the running spacing) is kept, up and down.
The one-photoelectron peak is the first of the chain more than 0.3 spacings above the
pedestal (the random-trigger value), which excludes a peak at the pedestal itself; it
and the next two, the 2 and 3 photoelectron peaks, give the gain and the zero as the
slope and the intercept of the straight line through their positions against their
numbers (the higher structures are compressed by the pile-up and left out). NaN with
fewer than two peaks.
"""
function photoelectron_ladder(x::AbstractVector, nominal::Real; pedestal::Real = 0.0)
    nan = (gain = NaN, zero = NaN, peaks = Float64[])
    lo, hi = quantile(x, 0.001), quantile(x, 0.999)
    w = nominal / 50
    nb = ceil(Int, (hi - lo) / w) + 1
    nb < 20 && return nan
    c = zeros(nb)
    for v in x
        k = floor(Int, (v - lo) / w) + 1
        1 <= k <= nb && (c[k] += 1)
    end
    sg = 0.12 * nominal / w                       # kernel sigma in bins
    hw = ceil(Int, 5sg)
    ker = [((t / sg)^2 - 1) / sg^2 * exp(-0.5 * (t / sg)^2) for t in -hw:hw]
    d2 = [sum(ker[j + hw + 1] * c[k + j] for j in -hw:hw if 1 <= k + j <= nb) for k in 1:nb]
    dmin = minimum(d2)
    dmin < 0 || return nan
    mins = [k for k in 2:nb-1 if d2[k] < d2[k - 1] && d2[k] <= d2[k + 1] && d2[k] < 0.003dmin]
    isempty(mins) && return nan
    pos(k) = lo + (k - 0.5 + 0.5 * (d2[k - 1] - d2[k + 1]) / (d2[k - 1] - 2d2[k] + d2[k + 1])) * w
    ref = mins[argmin(d2[mins])]
    chain = [pos(ref)]
    for dir in (1, -1)
        last, sp = pos(ref), Float64(nominal)
        cand = dir > 0 ? [k for k in mins if k > ref] : reverse([k for k in mins if k < ref])
        for k in cand
            d = dir * (pos(k) - last)
            d < 0.6sp && continue
            d > 1.4sp && break
            sp = length(chain) > 1 ? d : sp
            last = pos(k)
            dir > 0 ? push!(chain, last) : pushfirst!(chain, last)
        end
    end
    length(chain) < 2 && return nan
    g0 = median(diff(chain))
    i1 = findfirst(p -> p - pedestal > 0.3g0, chain)
    i1 === nothing && return nan
    chain = chain[i1:min(i1 + 2, end)]
    length(chain) < 2 && return nan
    n = collect(1.0:length(chain))
    nm, pm = sum(n) / length(n), sum(chain) / length(chain)
    gain = sum((n .- nm) .* (chain .- pm)) / sum(abs2, n .- nm)
    return (gain = gain, zero = pm - gain * nm, peaks = chain)
end
