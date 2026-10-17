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
fitted width. n is the number of entries within ±2 sigma. NaN when the fit fails.
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
        c2 = [count(v -> e2[i] <= v < e2[i + 1], sel) for i in 1:40]
        x2 = 0.5 .* (e2[1:end-1] .+ e2[2:end])
        f = fit_gauss_linear(x2, c2, mu, sg)
        f === nothing && return (mean = NaN, sigma = NaN, n = 0)
        mu, sg = f.mu, f.sigma
    end
    return (mean = mu, sigma = sg, n = count(abs.(sel .- mu) .<= 2sg))
end
