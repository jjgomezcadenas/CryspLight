# Spectra of the equalised sum and the quantities taken from them.

"Sum of the channels in avalanches, one value per event (the equalised sum)."
equalised_sum(e) = vec(sum(e.q; dims = 1))

"Histogram of x in [lo, hi] with nbins bins: (edges, counts)."
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
The 511 keV peak of the sums of the triggered events: the most populated peak of the upper
spectrum, between a quarter of the 99.5% quantile and that quantile.
"""
function find_peak511(tot::AbstractVector)
    hi = quantile(tot, 0.995)
    return peak_fit(tot, 0.25hi, hi)
end

"The 1275 keV peak, searched between 2.2 and 2.8 times the 511 keV one."
find_peak1275(tot::AbstractVector, p511) = isfinite(p511.mean) ?
    peak_fit(tot, 2.2 * p511.mean, 2.8 * p511.mean) : (mean = NaN, sigma = NaN, n = 0)

"Events within ±nsig sigma of a fitted peak."
peak_window(tot::AbstractVector, p, nsig::Real = 2) =
    isfinite(p.mean) ? abs.(tot .- p.mean) .<= nsig * p.sigma : falses(length(tot))

"Largest channel value of each event (columns of q)."
brightest(q::AbstractMatrix) = vec(maximum(q; dims = 1))
"Smallest channel value of each event."
dimmest(q::AbstractMatrix) = vec(minimum(q; dims = 1))
