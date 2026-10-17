# Per-decay pipeline of the full simulation: Na-22 decays at a point source -> the gammas
# that meet the crystal -> vendored gamma transport (deposits, with the flight time from
# the source) -> optical photons (Poisson yield per deposit, two-component emission) ->
# batched KA/Metal transport -> per-event interaction list, photoelectron records
# (channel, time) and integrated map. One event is one decay, whether or not any of its
# gammas interacted, so that the fraction of decays giving a signal is a result.

using Random: Xoshiro, randn

"""
    run_decays!(box, op, ro, tb, pv, yield_per_mev; source_mm, n_decays, seed, tau_ns,
                tau2_ns, frac1, first_decay, batch_photons, max_bounces, ArrayT)
        -> NamedTuple with
           events: edep_kev, edep_511_kev, edep_1275_kev, n_gammas_in, n_emitted,
                   n_detected, maps (nx x ny x n), first_decay
           interactions: offset (n + 1), x_mm, y_mm, z_mm, e_kev, t_ns, gamma_kind, proc
           pes: offset (n + 1), channel (UInt8, (j - 1) nx + i), t_ns

The intrinsic resolution of the crystal is a Gaussian smearing of the light of each
decay: the Poisson means of all its deposits are scaled by one factor of relative width
intrinsic_fwhm / 2.355 x (511 keV / E_dep)^intrinsic_exponent, E_dep the energy
deposited by the decay (0 = no smearing).

Decay k (first_decay <= k < first_decay + n_decays) uses the stream Xoshiro(hash((seed,
k))) for the decay and the gamma transport, and photon ids offset so that chunks with
disjoint decay ranges never share a Philox stream: chunks are reproducible and can be
written in sequence. Times are in ns from the decay: flight from the source, gamma path
in the crystal, scintillation delay, optical path.
"""
function run_decays!(box::Box, op::OpticalParams, ro::Readout, tb::TimeBinning,
                     pv, yield_per_mev::Real;
                     source_mm, n_decays::Int, seed::Integer, tau_ns::Float32,
                     tau2_ns::Float32 = 0f0, frac1::Float32 = 1f0,
                     intrinsic_fwhm::Real = 0.0, intrinsic_exponent::Real = 0.5,
                     first_decay::Int = 1, batch_photons::Int = 2_000_000,
                     max_bounces::Int = 100_000, ArrayT = Array)
    L = box.L
    nx, ny = Int(ro.grid.nx), Int(ro.grid.ny)
    n = n_decays
    edep = zeros(Float32, n); edep511 = zeros(Float32, n); edep1275 = zeros(Float32, n)
    ngin = zeros(Int8, n); nemit = zeros(Int32, n); ndet = zeros(Int32, n)
    maps = zeros(UInt16, nx, ny, n)
    # interaction lists
    ioff = zeros(Int64, n + 1)
    ix = Float32[]; iy = Float32[]; iz = Float32[]; ie = Float32[]; it = Float32[]
    ikind = Int8[]; iproc = Int8[]
    # photoelectron records
    poff = zeros(Int64, n + 1)
    pch = UInt8[]; pt = Float32[]

    positions = NTuple{3,Float32}[]
    t0s = Float32[]
    ev_range = Tuple{Int,Int,Int}[]
    # photon ids: a decay k owns ids starting at a block unique to k; the block size is
    # the largest photon count a decay can produce, bounded by the yield at 1786 keV plus
    # a wide margin, so that chunks are stream-disjoint without coordination
    block = Int64(ceil(yield_per_mev * 1.7865 * 1.5)) + 1000
    pid0 = (first_decay - 1) * block
    pending = Int64[]              # the pid offset of each batch flush

    function flush!()
        isempty(positions) && return
        np = length(positions)
        _, recs = run_photons_ka!(box, op, ro, tb; n_photons = np, seed = seed,
                                  positions = positions, t0s = t0s,
                                  pid_offset = pid0, tau_ns = tau_ns,
                                  tau2_ns = tau2_ns, frac1 = frac1,
                                  max_bounces = max_bounces, ArrayT = ArrayT,
                                  return_records = true)
        for (ev, lo, hi) in ev_range
            c = Int32(0)
            for k in lo:hi
                Int32(recs.status[k]) == STATUS_DETECTED || continue
                c += Int32(1)
                lin = abs(Int(recs.idx[k]))
                ii = (lin - 1) % nx + 1
                jj = (lin - 1) ÷ nx + 1
                maps[ii, jj, ev] += UInt16(1)
                push!(pch, UInt8(lin))
                push!(pt, recs.t[k])
            end
            ndet[ev] += c
            poff[ev + 1] = length(pch)
        end
        pid0 += np
        empty!(positions); empty!(t0s); empty!(ev_range)
        return
    end

    src = (Float64(source_mm[1]), Float64(source_mm[2]), Float64(source_mm[3]))
    for ev in 1:n
        k = first_decay + ev - 1
        rng = Xoshiro(hash((seed, k)))
        gammas = na22_decay(rng)
        first_photon = length(positions) + 1
        # gamma transport of the whole decay first, so that the intrinsic smearing can
        # scale the light of all its deposits by one factor
        evdeps = NTuple{6,Float32}[]
        evkind = Int8[]
        evtfl = Float32[]
        for g in gammas
            hit = entry_point(src, g.dir, L)
            hit === nothing && continue
            entry, path = hit
            deps = gamma_deposits(pv, L, entry, g.dir, rng; e0_mev = g.e_mev)
            isempty(deps) && continue
            ngin[ev] += Int8(1)
            tflight = Float32(path / C_MM_NS)
            for d in deps
                push!(evdeps, d); push!(evkind, g.kind); push!(evtfl, tflight)
            end
        end
        etot = sum(d -> d[4], evdeps; init = 0f0)
        scale = 1.0
        if intrinsic_fwhm > 0 && etot > 0
            srel = intrinsic_fwhm / 2.355 * (511.0 / etot)^intrinsic_exponent
            scale = max(0.0, 1.0 + srel * randn(rng))
        end
        for (d, kind, tflight) in zip(evdeps, evkind, evtfl)
            edep[ev] += d[4]
            kind == GAMMA_511 ? (edep511[ev] += d[4]) : (edep1275[ev] += d[4])
            push!(ix, d[1]); push!(iy, d[2]); push!(iz, d[3]); push!(ie, d[4])
            push!(it, d[5] + tflight); push!(ikind, kind)
            push!(iproc, Int8(round(d[6])))
            nph = rand_poisson(rng, scale * Float64(yield_per_mev) * d[4] / 1000)
            nemit[ev] += Int32(nph)
            for _ in 1:nph
                push!(positions, (d[1], d[2], d[3]))
                push!(t0s, d[5] + tflight)
            end
        end
        ioff[ev + 1] = length(ix)
        last_photon = length(positions)
        if last_photon >= first_photon
            push!(ev_range, (ev, first_photon, last_photon))
        else
            poff[ev + 1] = length(pch)       # empty event: offsets stay contiguous
        end
        length(positions) >= batch_photons && flush!()
    end
    flush!()
    # events flushed earlier than later empty ones already set their offsets; make the
    # offset array monotone for events that had no photons after the last flush
    for ev in 1:n
        poff[ev + 1] = max(poff[ev + 1], poff[ev])
    end
    return (events = (edep_kev = edep, edep_511_kev = edep511, edep_1275_kev = edep1275,
                      n_gammas_in = ngin, n_emitted = nemit, n_detected = ndet,
                      maps = maps, first_decay = first_decay),
            interactions = (offset = ioff, x_mm = ix, y_mm = iy, z_mm = iz, e_kev = ie,
                            t_ns = it, gamma_kind = ikind, proc = iproc),
            pes = (offset = poff, channel = pch, t_ns = pt))
end
