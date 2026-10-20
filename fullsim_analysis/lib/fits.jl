# Peak fits shared by every script: a Gaussian plus a linear background on a histogram.

"""
    fit_gauss_linear(xc, c, mu0, sigma0) -> (A, mu, sigma, b0, b1) or nothing

Fit of A exp(-(x - mu)^2 / 2 sigma^2) + b0 + b1 (x - mu0) to the bin counts c at the
centres xc, by Gauss-Newton with Levenberg damping and Poisson weights, starting from mu0
and sigma0. mu and sigma are in the unit of xc, A, b0 and b1 in counts per bin (b1 per
unit of xc). nothing when the fit does not converge to a positive peak.
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
near an edge of the range is not cut. mean and sigma are in the unit of x, n is the
number of entries within ±2 sigma. NaN (and n = 0) when there are fewer than 100 entries
in [lo, hi] or the fit fails.
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
then again in ±2.5 fitted sigmas (at least ±w/2). mean, sigma and w are in the unit of
x, n is the number of entries within ±2 sigma. NaN when there are fewer than 100 entries
in [lo, hi], the fit fails or its mean moves out of the fit window.
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
    photoelectron_ladder(x, nominal; pedestal, baseline) -> (gain, zero, peaks, cells)

Gain (ADC counts per cell) and zero level (ADC counts) of a self-triggered dark-count
spectrum x in ADC counts, given the nominal gain and the pedestal in ADC counts and the
baseline in gains. The peaks are located as the minima of the second derivative of the
histogram (bins of a fiftieth of the nominal gain), smoothed with a Gaussian of 0.12 nominal gains, which
resolves a peak that is only a shoulder on the tail of the previous one; each minimum is
refined by a parabola. Starting from the most prominent peak, the chain of peaks spaced
by 0.6 to 1.4 times the nominal gain (then the running spacing) is kept, up and down.

The number of cells of each peak is not counted from the pedestal: the chain removes the
mean current of the dark counts before shaping, so a held signal with no avalanche sits
`baseline` gains below the pedestal of the random trigger (dark_baseline of the chain;
zero behind a voltage-sensitive preamplifier, about 1.5 cells on the Radioroc2 at 100 ns
and room temperature), and the one-cell peak of such a chain lies below the pedestal.
Each peak of the chain is given the number of cells nearest to its distance from that
expected zero in units of the spacing of the chain; the peaks of one cell or more, the
first three of them (the higher structures are compressed by the pile-up and left out),
give the gain and the zero as the slope and the intercept of the straight line through
their positions against their numbers. The ladder is rejected (NaN) when it has fewer
than two such peaks, when the gain is outside 0.7 to 1.3 times the nominal, when the
intercept differs from the expected zero by more than 0.35 gains, when a peak is more
than 0.15 gains from the line, or when two peaks receive the same number: a spectrum in
which the peaks are not equally spaced at the expected positions gives no gain rather
than a wrong one. `cells` holds the numbers assigned to `peaks`.
"""
function photoelectron_ladder(x::AbstractVector, nominal::Real; pedestal::Real = 0.0,
                              baseline::Real = 0.0)
    nan = (gain = NaN, zero = NaN, peaks = Float64[], cells = Int[])
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
    # cells of each peak from the expected zero, pedestal - baseline x spacing; the first
    # three peaks of one cell or more
    cells = [round(Int, (p - pedestal) / g0 + baseline) for p in chain]
    keep = findall(>=(1), cells)
    length(keep) < 2 && return nan
    keep = keep[1:min(3, end)]
    chain, cells = chain[keep], cells[keep]
    allunique(cells) || return nan
    n = Float64.(cells)
    nm, pm = sum(n) / length(n), sum(chain) / length(chain)
    gain = sum((n .- nm) .* (chain .- pm)) / sum(abs2, n .- nm)
    zero = pm - gain * nm
    0.7nominal <= gain <= 1.3nominal || return nan
    abs((zero - pedestal) / gain + baseline) <= 0.35 || return nan
    maximum(abs.(chain .- (zero .+ gain .* n))) <= 0.15gain || return nan
    return (gain = gain, zero = zero, peaks = chain, cells = cells)
end
