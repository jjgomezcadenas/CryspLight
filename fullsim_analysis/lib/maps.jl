# Channel maps as 8 x 8 arrays (i along x, j along y, channel c = (j - 1) nx + i).

"""
    channel_grid(e) -> (nx, ny)

Grid shape of an event file (read_events), from the number of distinct channel centres
along x and y.
"""
channel_grid(e) = (length(unique(e.x)), length(unique(e.y)))

"Vector in channel order to an nx x ny matrix."
function as_map(v::AbstractVector, nx::Int, ny::Int)
    m = fill(NaN, nx, ny)
    for c in eachindex(v)
        m[(c - 1) % nx + 1, (c - 1) ÷ nx + 1] = v[c]
    end
    return m
end

"""
    occupancy_map(e, sel; thr = 0.5) -> Matrix{Float64} (nx x ny)

Fraction of the events selected by sel in which each channel charge q is above thr
(avalanches), as a channel map.
"""
function occupancy_map(e, sel::AbstractVector{Bool}; thr::Real = 0.5)
    nx, ny = channel_grid(e)
    return as_map(vec(mean(e.q[:, sel] .> thr; dims = 2)), nx, ny)
end

"""
    mean_map(e, sel) -> Matrix{Float64} (nx x ny)

Mean charge q of each channel over the events selected by sel, in avalanches, as a
channel map.
"""
function mean_map(e, sel::AbstractVector{Bool})
    nx, ny = channel_grid(e)
    return as_map(vec(mean(e.q[:, sel]; dims = 2)), nx, ny)
end

"""
    seed_peak_map(e, tot, sel, lo, hi; nmin = 100) -> Matrix{Float64} (nx x ny)

Position of the 511 keV peak of the sums tot (peak_fit) for the events grouped by their
brightest channel, divided by the median over the channels (NaN where fewer than nmin
events): a dimensionless channel map. sel selects the events to group (normally the
triggered ones); the search window is [lo, hi], in the unit of tot.
"""
function seed_peak_map(e, tot::AbstractVector, sel::AbstractVector{Bool}, lo::Real, hi::Real;
                       nmin::Int = 100)
    nx, ny = channel_grid(e)
    idx = findall(sel)
    seed = [argmax(view(e.q, :, k)) for k in idx]
    v = fill(NaN, e.nch)
    for c in 1:e.nch
        t = tot[idx[seed .== c]]
        length(t) >= nmin || continue
        v[c] = peak_fit(t, lo, hi).mean
    end
    med = median(filter(isfinite, v))
    isfinite(med) && (v ./= med)
    return as_map(v, nx, ny)
end
