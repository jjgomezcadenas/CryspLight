# The bounce loop — the code that becomes the Metal kernel. One call = one photon,
# Float32 throughout, no allocation. Physics: design note Sec. 3.

const C_MM_NS = 299.792458f0

const STATUS_DETECTED = Int32(1)
const STATUS_ABS_BULK = Int32(2)   # bulk absorption in the crystal
const STATUS_ABS_WALL = Int32(3)   # killed by the wrap (1 - R)
const STATUS_ABS_SIPM = Int32(4)   # transmitted to the sensor but failed the PDE draw
const STATUS_CAP      = Int32(5)   # bounce cap (diagnostic; ~0 with physical R < 1)
const STATUS_ABS_SUR  = Int32(6)   # transmitted outside the active region, killed by
                                   # the surround (1 - R_surround)

struct OpticalParams
    n_crystal::Float32
    n_coupling::Float32     # grease, 1.45
    abs_len_mm::Float32     # bulk absorption (Inf32 disables)
    rayleigh_mm::Float32    # bulk Rayleigh scattering length (Inf32 disables)
    wrap_R::Float32         # wrap reflectivity
    wrap_specular::Bool     # true = ESR (specular), false = PTFE (Lambertian)
    air_gap::Bool           # true = back-painted (air gap), false = front-painted (contact)
    sigma_alpha::Float32    # UNIFIED surface roughness (rad) at the crystal-air interface
    pde::Float32            # applied on the fly at the SiPM plane
end

struct SipmGrid
    pitch_mm::Float32
    nx::Int32
    ny::Int32
end

"""
Readout description of the back face. Default (disc = false): the whole face is active,
binned by the SiPM grid. Disc mode (a PMT photocathode): only a disc of radius r about
(cx, cy) is active; photons transmitted outside it hit the surround, which reflects them
back into the crystal with probability sur_R (specular or Lambertian re-injection, like a
back-painted wrap) or kills them. sur_R = 0 is an absorbing surround (bare setup);
sur_R ~ 0.95, Lambertian is a Tyvek-covered rear face. Detected photons are binned on the
grid either way (the light map).
"""
struct Readout
    grid::SipmGrid
    disc::Bool
    cx::Float32
    cy::Float32
    r2::Float32                  # disc radius squared
    sur_R::Float32
    sur_specular::Bool
end

Readout(grid::SipmGrid; center = nothing, radius = nothing, sur_R = 0,
        sur_specular::Bool = false) =
    radius === nothing ?
        Readout(grid, false, 0f0, 0f0, 0f0, 0f0, false) :
        Readout(grid, true, Float32(center[1]), Float32(center[2]), Float32(radius)^2,
                Float32(sur_R), sur_specular)

struct TimeBinning
    bin_ns::Float32
    nbins::Int32            # counts array has nbins + 1, the last is the overflow bin
end

mutable struct Accumulator
    counts::Array{UInt32,3}         # nx × ny × (nbins+1) photoelectron counts
    first_ns::Matrix{Float32}       # earliest detected time per SiPM (Inf if none)
    ndet::Int64
    nabs_bulk::Int64
    nabs_wall::Int64
    nabs_sipm::Int64
    ncap::Int64
    nabs_sur::Int64                 # killed by the readout-face surround
    nscat::Int64                    # total Rayleigh scatters
    sum_bounces_det::Int64
end

Accumulator(grid::SipmGrid, tb::TimeBinning) =
    Accumulator(zeros(UInt32, grid.nx, grid.ny, tb.nbins + 1),
                fill(Inf32, grid.nx, grid.ny), 0, 0, 0, 0, 0, 0, 0, 0)

function Base.merge!(a::Accumulator, b::Accumulator)
    a.counts .+= b.counts
    a.first_ns .= min.(a.first_ns, b.first_ns)
    a.ndet += b.ndet; a.nabs_bulk += b.nabs_bulk; a.nabs_wall += b.nabs_wall
    a.nabs_sipm += b.nabs_sipm; a.ncap += b.ncap; a.nabs_sur += b.nabs_sur
    a.nscat += b.nscat
    a.sum_bounces_det += b.sum_bounces_det
    return a
end

total_terminated(a::Accumulator) =
    a.ndet + a.nabs_bulk + a.nabs_wall + a.nabs_sipm + a.ncap + a.nabs_sur

@inline function record_detection!(acc::Accumulator, grid::SipmGrid, tb::TimeBinning,
                                   x::Float32, y::Float32, t::Float32, bounces::Int)
    i = min(grid.nx, Int32(1) + unsafe_trunc(Int32, x / grid.pitch_mm))
    j = min(grid.ny, Int32(1) + unsafe_trunc(Int32, y / grid.pitch_mm))
    bin = Int32(1) + unsafe_trunc(Int32, t / tb.bin_ns)
    bin = bin > tb.nbins ? tb.nbins + Int32(1) : bin
    acc.counts[i, j, bin] += UInt32(1)
    if t < acc.first_ns[i, j]
        acc.first_ns[i, j] = t
    end
    acc.ndet += 1
    acc.sum_bounces_det += bounces
end

propagate_photon!(acc::Accumulator, box::Box, op::OpticalParams, grid::SipmGrid,
                  tb::TimeBinning, s::PhiloxStream, x, y, z, ux, uy, uz, t0; kw...) =
    propagate_photon!(acc, box, op, Readout(grid), tb, s, x, y, z, ux, uy, uz, t0; kw...)

"""
Transport one photon from (x,y,z) with direction (ux,uy,uz) and emission time t0 (ns)
until it is detected or absorbed. Accumulates into acc and returns the status.
"""
function propagate_photon!(acc::Accumulator, box::Box, op::OpticalParams,
                           ro::Readout, tb::TimeBinning, s::PhiloxStream,
                           x::Float32, y::Float32, z::Float32,
                           ux::Float32, uy::Float32, uz::Float32,
                           t0::Float32; max_bounces::Int = 100_000)
    t = t0
    inv_v = op.n_crystal / C_MM_NS          # ns per mm in the crystal
    bounces = 0
    while true
        d, face = wall_hit(x, y, z, ux, uy, uz, box)
        # bulk processes compete with the wall: absorption and Rayleigh scattering
        d_abs = isfinite(op.abs_len_mm) ? -op.abs_len_mm * log(randu(s)) : Inf32
        d_ray = isfinite(op.rayleigh_mm) ? -op.rayleigh_mm * log(randu(s)) : Inf32
        if d_abs < d && d_abs <= d_ray
            acc.nabs_bulk += 1
            return STATUS_ABS_BULK
        elseif d_ray < d
            x += d_ray * ux; y += d_ray * uy; z += d_ray * uz
            t += d_ray * inv_v
            ux, uy, uz = rayleigh_dir(s, ux, uy, uz)
            acc.nscat += 1
            bounces += 1                     # scatters count toward the cap too
            if bounces >= max_bounces
                acc.ncap += 1
                return STATUS_CAP
            end
            continue
        end
        # advance to the wall; snap the hit coordinate onto it, clamp the others
        x = clamp(x + d * ux, 0f0, box.L[1])
        y = clamp(y + d * uy, 0f0, box.L[2])
        z = clamp(z + d * uz, 0f0, box.L[3])
        a = face_axis(face)
        if a == 1
            x = face == FACE_XP ? box.L[1] : 0f0
        elseif a == 2
            y = face == FACE_YP ? box.L[2] : 0f0
        else
            z = face == FACE_BACK ? box.L[3] : 0f0
        end
        t += d * inv_v
        cosi = a == 1 ? abs(ux) : (a == 2 ? abs(uy) : abs(uz))

        if face == FACE_BACK
            # single Fresnel surface crystal -> coupling medium (grease, or air gap)
            tir, R = fresnel(op.n_crystal, op.n_coupling, cosi)
            if tir || randu(s) < R
                ux, uy, uz = specular(ux, uy, uz, face)
            elseif !ro.disc || (x - ro.cx)^2 + (y - ro.cy)^2 <= ro.r2
                if randu(s) <= op.pde
                    record_detection!(acc, ro.grid, tb, x, y, t, bounces)
                    return STATUS_DETECTED
                else
                    acc.nabs_sipm += 1
                    return STATUS_ABS_SIPM
                end
            elseif randu(s) <= ro.sur_R
                # surround reflector: re-inject, like a back-painted wrap
                if ro.sur_specular
                    ux, uy, uz = specular(ux, uy, uz, face)
                else
                    ux, uy, uz = lambertian_dir(s, face)
                end
            else
                acc.nabs_sur += 1
                return STATUS_ABS_SUR
            end
        elseif !op.air_gap
            # front-painted (wrap in optical contact): no Fresnel stage, the reflector
            # acts on every hit — specular (polished) or Lambertian (ground) with prob R
            if randu(s) <= op.wrap_R
                if op.wrap_specular
                    ux, uy, uz = specular(ux, uy, uz, face)
                else
                    ux, uy, uz = lambertian_dir(s, face)
                end
            else
                acc.nabs_wall += 1
                return STATUS_ABS_WALL
            end
        else
            # back-painted (air gap): crystal -> air Fresnel/TIR at the (possibly rough)
            # crystal surface, then the reflector for the transmitted fraction
            transmitted = false
            if op.sigma_alpha > 0f0
                ux, uy, uz, transmitted =
                    surface_interact(s, ux, uy, uz, face, op.n_crystal, op.sigma_alpha)
            else
                tir, R = fresnel(op.n_crystal, 1f0, cosi)
                if tir || randu(s) < R
                    ux, uy, uz = specular(ux, uy, uz, face)
                else
                    transmitted = true
                end
            end
            if transmitted
                if randu(s) <= op.wrap_R    # <=: randu hits 1.0 with p = 2^-24, and the
                                            # R = 1 idealized limit must never absorb
                    if op.wrap_specular
                        ux, uy, uz = specular(ux, uy, uz, face)   # ESR: net specular flip
                    else
                        ux, uy, uz = lambertian_dir(s, face)      # PTFE: diffuse
                    end
                else
                    acc.nabs_wall += 1
                    return STATUS_ABS_WALL
                end
            end
        end
        bounces += 1
        if bounces >= max_bounces
            acc.ncap += 1
            return STATUS_CAP
        end
    end
end

run_photons!(box::Box, op::OpticalParams, grid::SipmGrid, tb::TimeBinning; kw...) =
    run_photons!(box, op, Readout(grid), tb; kw...)

"""
Run n_photons from an isotropic point source with exponential (tau_ns > 0) or delta
(tau_ns = 0) emission time. Pass either pos (a single point) or positions (a list —
photons are assigned round-robin, giving an equal-weight scan). Deterministic in
(seed, photon id) regardless of threading. Returns the merged Accumulator.
"""
function run_photons!(box::Box, op::OpticalParams, ro::Readout, tb::TimeBinning;
                      n_photons::Int, seed::Integer,
                      pos::Union{Nothing,NTuple{3,Float32}} = nothing,
                      positions::Union{Nothing,Vector{NTuple{3,Float32}}} = nothing,
                      tau_ns::Float32, max_bounces::Int = 100_000)
    plist = positions === nothing ? [something(pos)] : positions
    npos = length(plist)
    nt = Threads.nthreads()
    accs = [Accumulator(ro.grid, tb) for _ in 1:nt]
    # NB: the threaded body must not assign any variable name also assigned in the outer
    # scope — Julia would box it into a single shared binding and the threads would race.
    Threads.@threads :static for tid in 1:nt
        acc = accs[tid]
        for pid in tid:nt:n_photons
            s = PhiloxStream(seed, pid)
            p = plist[1 + (pid - 1) % npos]
            ux, uy, uz = isotropic_dir(s)
            t0 = emission_time(s, tau_ns)
            propagate_photon!(acc, box, op, ro, tb, s,
                              p[1], p[2], p[3], ux, uy, uz, t0;
                              max_bounces = max_bounces)
        end
    end
    out = accs[1]
    for k in 2:nt
        merge!(out, accs[k])
    end
    return out
end
