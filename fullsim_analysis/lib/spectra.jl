# Spectra of the equalised sum and the quantities taken from them.

"""
    equalised_sum(e) -> Vector{Float64}

Sum of the channel charges q of an event file (read_events) in avalanches, one value per
event (the equalised sum).
"""
equalised_sum(e) = vec(sum(e.q; dims = 1))

"""
    histogram(x, lo, hi, nbins) -> (edges, counts)

Histogram of x in [lo, hi) with nbins equal bins: the nbins + 1 edges (unit of x) and the
Int counts; values outside are dropped.
"""
function histogram(x::AbstractVector, lo::Real, hi::Real, nbins::Int)
    edges = collect(range(lo, hi; length = nbins + 1))
    counts = zeros(Int, nbins)
    w = nbins / (hi - lo)
    for v in x
        (lo <= v < hi) || continue
        counts[1 + floor(Int, (v - lo) * w)] += 1
    end
    return edges, counts
end

"""
    find_peak511(tot) -> (mean, sigma, n)

The 511 keV peak (peak_fit, in the unit of tot) of the sums tot of the triggered events:
the most populated peak of the upper spectrum, between a fifth of the 99.5% quantile and that quantile (the coincidence of
the 1275 keV gamma with a 511 keV one sets the quantile near 3.5 times the 511 keV peak).
"""
function find_peak511(tot::AbstractVector)
    hi = quantile(tot, 0.995)
    return peak_fit(tot, 0.2hi, hi)
end

"""
    find_peak1275(tot, p511) -> (mean, sigma, n)

The 1275 keV peak of the sums tot, searched between 2.2 and 2.8 times the mean of the
511 keV peak p511 (find_peak511); NaN when p511 failed.
"""
find_peak1275(tot::AbstractVector, p511) = isfinite(p511.mean) ?
    peak_fit(tot, 2.2 * p511.mean, 2.8 * p511.mean) : (mean = NaN, sigma = NaN, n = 0)

"""
    peak_window(tot, p, nsig = 2) -> BitVector

The events whose sum tot lies within ±nsig sigma of the fitted peak p (mean, sigma); none
when the fit failed.
"""
peak_window(tot::AbstractVector, p, nsig::Real = 2) =
    isfinite(p.mean) ? abs.(tot .- p.mean) .<= nsig * p.sigma : falses(length(tot))

"""
    brightest(q::AbstractMatrix) -> Vector

Largest channel value of each event (columns of the nch x n matrix q, in its unit).
"""
brightest(q::AbstractMatrix) = vec(maximum(q; dims = 1))
"""
    dimmest(q::AbstractMatrix) -> Vector

Smallest channel value of each event (columns of q).
"""
dimmest(q::AbstractMatrix) = vec(minimum(q; dims = 1))
