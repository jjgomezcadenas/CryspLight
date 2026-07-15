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
        -> NamedTuple (edep, npe, maps, tmin, xyz1, e1, xyz2, e2, er, int_type, n_int)

Shoot n_events gammas of e0_mev entering uniformly over the front (z = 0) face along
+z; transport deposits with the vendored gamma core, expand into optical photons and
transport them in batches. Per event: the deposited energy [keV], the detected
photoelectron count, the time-integrated nx x ny photoelectron map, and the nx x ny
matrix of first-photoelectron times [ns] (Inf32 where a SiPM saw nothing). Truth:
position/energy of the first and second interaction (xyz1 3 x n [mm], e1 [keV], xyz2,
e2; zeros when absent), the rest energy er = e0 - e1 - e2 [keV] (escaped energy plus
any deposits beyond the second), int_type (Int8: -1 = crossed without interacting,
0 = direct photoelectric, X >= 1 = X Compton scatters), and n_int (total deposits).
Events with no deposit have edep = 0, npe = 0, er = e0.
"""
function run_events!(box::Box, op::OpticalParams, ro::Readout, tb::TimeBinning,
                     pv, yield_per_mev::Real;
                     n_events::Int, seed::Integer, tau_ns::Float32,
                     e0_mev::Float64 = 0.511, max_bounces::Int = 100_000,
                     batch_photons::Int = 2_000_000, ArrayT = Array)
    L = box.L
    nx, ny = Int(ro.grid.nx), Int(ro.grid.ny)
    e0_kev = Float32(e0_mev * 1000)
    edep = zeros(Float32, n_events)
    npe = zeros(Int32, n_events)
    maps = zeros(UInt16, nx, ny, n_events)
    tmin = fill(Inf32, nx, ny, n_events)
    xyz1 = zeros(Float32, 3, n_events)
    e1 = zeros(Float32, n_events)
    xyz2 = zeros(Float32, 3, n_events)
    e2 = zeros(Float32, n_events)
    er = fill(e0_kev, n_events)
    int_type = fill(Int8(-1), n_events)
    n_int = zeros(Int16, n_events)

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
                    t = recs.t[k]
                    t < tmin[ii, jj, ev] && (tmin[ii, jj, ev] = t)
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
        ncompt = count(d -> d[6] == DEP_COMPTON, deps)
        int_type[ev] = Int8(min(ncompt, 127))          # 0 = direct photoelectric
        n_int[ev] = Int16(length(deps))
        xyz1[1, ev], xyz1[2, ev], xyz1[3, ev] = deps[1][1], deps[1][2], deps[1][3]
        e1[ev] = deps[1][4]
        if length(deps) >= 2
            xyz2[1, ev], xyz2[2, ev], xyz2[3, ev] = deps[2][1], deps[2][2], deps[2][3]
            e2[ev] = deps[2][4]
        end
        er[ev] = e0_kev - e1[ev] - e2[ev]
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
    return (edep = edep, npe = npe, maps = maps, tmin = tmin,
            xyz1 = xyz1, e1 = e1, xyz2 = xyz2, e2 = e2, er = er,
            int_type = int_type, n_int = n_int)
end
