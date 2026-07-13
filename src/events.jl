# Per-event pipeline: 511 keV gammas -> vendored gamma transport (deposits) ->
# optical photons (Poisson yield per deposit, deposit times as emission offsets) ->
# batched KA/Metal transport -> per-event SiPM outputs. The optical stage runs through
# run_photons_ka! with per-photon records; photons of one event occupy a contiguous id
# range, so the per-event reduction needs no event array in the kernel. Batches use
# pid_offset for disjoint Philox streams.

using Random: Xoshiro, randn, rand

"Poisson sampler: Knuth below lambda = 50, Gaussian approximation above."
function rand_poisson(rng::AbstractRNG, lam::Float64)
    lam <= 0.0 && return 0
    lam > 50.0 && return max(0, round(Int, lam + sqrt(lam) * randn(rng)))
    L = exp(-lam)
    k = 0
    p = 1.0
    while true
        p *= rand(rng)
        p < L && return k
        k += 1
    end
end

"""
    run_events!(box, op, ro, tb, pv, yield_per_mev; n_events, seed, tau_ns, ...)
        -> (edep_kev, npe, maps)

Shoot n_events gammas of e0_mev entering uniformly over the front (z = 0) face along
+z; transport deposits with the vendored gamma core, expand into optical photons and
transport them in batches. Returns per event: the deposited energy [keV], the number
of detected photoelectrons, and the time-integrated nx x ny photoelectron map.
Events with no deposit (gamma crosses without interacting) have edep = 0, npe = 0.
"""
function run_events!(box::Box, op::OpticalParams, ro::Readout, tb::TimeBinning,
                     pv, yield_per_mev::Real;
                     n_events::Int, seed::Integer, tau_ns::Float32,
                     e0_mev::Float64 = 0.511, max_bounces::Int = 100_000,
                     batch_photons::Int = 2_000_000, ArrayT = Array)
    L = box.L
    nx, ny = Int(ro.grid.nx), Int(ro.grid.ny)
    edep = zeros(Float32, n_events)
    npe = zeros(Int32, n_events)
    maps = zeros(UInt16, nx, ny, n_events)

    # batch buffers: one entry per photon (position = its deposit, t0 = deposit time)
    positions = NTuple{3,Float32}[]
    t0s = Float32[]
    ev_range = Tuple{Int,Int,Int}[]     # (event, first photon, last photon) in batch
    pid0 = 0

    function flush!()
        isempty(positions) && return
        n = length(positions)
        _, recs = run_photons_ka!(box, op, ro, tb; n_photons = n, seed = seed,
                                  positions = positions, t0s = t0s,
                                  pid_offset = pid0, tau_ns = tau_ns,
                                  max_bounces = max_bounces, ArrayT = ArrayT,
                                  return_records = true)
        for (ev, lo, hi) in ev_range
            c = Int32(0)
            for k in lo:hi
                if Int32(recs.status[k]) == STATUS_DETECTED
                    c += Int32(1)
                    lin = abs(Int(recs.idx[k]))
                    ii = (lin - 1) % nx + 1
                    jj = (lin - 1) ÷ nx + 1
                    maps[ii, jj, ev] += UInt16(1)
                end
            end
            npe[ev] += c
        end
        pid0 += n
        empty!(positions); empty!(t0s); empty!(ev_range)
        return
    end

    for ev in 1:n_events
        rng = Xoshiro(hash((seed, ev)))
        entry = (Float32(rand(rng) * L[1]), Float32(rand(rng) * L[2]), 0f0)
        deps = gamma_deposits(pv, L, entry, (0.0, 0.0, 1.0), rng; e0_mev = e0_mev)
        isempty(deps) && continue
        first_photon = length(positions) + 1
        for d in deps
            edep[ev] += d[4]
            nph = rand_poisson(rng, Float64(yield_per_mev) * d[4] / 1000)
            for _ in 1:nph
                push!(positions, (d[1], d[2], d[3]))
                push!(t0s, d[5])
            end
        end
        last_photon = length(positions)
        last_photon >= first_photon && push!(ev_range, (ev, first_photon, last_photon))
        length(positions) >= batch_photons && flush!()
    end
    flush!()
    return edep, npe, maps
end
