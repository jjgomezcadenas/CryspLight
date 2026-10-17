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
    n_window::Float32       # SiPM window (resin) behind the coupling layer; 0 = none
end

"Nine-argument form: no window, the readout face is the single crystal-coupling interface."
OpticalParams(n_crystal, n_coupling, abs_len_mm, rayleigh_mm, wrap_R, wrap_specular::Bool,
              air_gap::Bool, sigma_alpha, pde) =
    OpticalParams(Float32(n_crystal), Float32(n_coupling), Float32(abs_len_mm),
                  Float32(rayleigh_mm), Float32(wrap_R), wrap_specular, air_gap,
                  Float32(sigma_alpha), Float32(pde), 0f0)

struct SipmGrid
    pitch_mm::Float32
    nx::Int32
    ny::Int32
end

"""
Readout description of the back face. Default (disc = false, tiles = false): the whole
face is active, binned by the SiPM grid. Disc mode (a PMT photocathode): only a disc of
radius r about (cx, cy) is active; photons transmitted outside it hit the surround, which
reflects them back into the crystal with probability sur_R (specular or Lambertian
re-injection, like a back-painted wrap) or kills them. sur_R = 0 is an absorbing
surround (bare setup); sur_R ~ 0.95, Lambertian is a Tyvek-covered rear face. Tiles mode
(the bench array): the active area is a set of squares of width active_w, tile_n per
tile and axis on the pitch tile_pitch, with a dead margin tile_edge beyond the outer
pitch cells of each tile; tiles are butted, grid.nx ÷ tile_n of them along x, and the
array corner sits at (x0, y0) in the crystal frame (negative when the array overhangs
the crystal). The package surface between the active squares, and the board beyond the
array, act as the surround (sur_R, sur_specular). Detected photons are binned on the
grid either way (the light map); in tiles mode the bin is the channel.
"""
struct Readout
    grid::SipmGrid
    disc::Bool
    cx::Float32
    cy::Float32
    r2::Float32                  # disc radius squared
    sur_R::Float32
    sur_specular::Bool
    two_sided::Bool              # sensors on BOTH z faces (Kim/Ding geometries)
    tiles::Bool                  # active squares of a tiled array instead of the full face
    tile_n::Int32                # channels per tile along one axis
    tile_pitch::Float32          # channel pitch inside a tile
    tile_edge::Float32           # dead margin between the outer pitch cell and the package edge
    active_w::Float32            # active width of a channel
    tile_w::Float32              # tile width = 2 edge + (n - 1) pitch + active
    x0::Float32                  # array corner in the crystal frame
    y0::Float32
end

"""
    Readout(grid; center, radius, sur_R, sur_specular, two_sided,
            tiles, tile_n, tile_pitch, tile_edge, active_w, face_mm, origin)

Keyword constructor. radius selects the disc mode; tiles = true the tiled array, whose
corner is placed either at origin = (x0, y0) or, given face_mm = (Lx, Ly), centred on the
face. Defaults of the tile parameters are the Hamamatsu S14161-6050HS-04 (4 x 4 channels
of 6.0 mm on a 6.2 mm pitch, 0.2 mm margin: 25.0 mm tiles).
"""
function Readout(grid::SipmGrid; center = nothing, radius = nothing, sur_R = 0,
                 sur_specular::Bool = false, two_sided::Bool = false,
                 tiles::Bool = false, tile_n::Integer = 4, tile_pitch::Real = 6.2,
                 tile_edge::Real = 0.2, active_w::Real = 6.0,
                 face_mm = nothing, origin = nothing)
    disc = radius !== nothing
    (disc && tiles) && error("a readout is either a disc or a tiled array, not both")
    cx = disc ? Float32(center[1]) : 0f0
    cy = disc ? Float32(center[2]) : 0f0
    r2 = disc ? Float32(radius)^2 : 0f0
    sR = (disc || tiles) ? Float32(sur_R) : 0f0
    if tiles
        (grid.nx % tile_n == 0 && grid.ny % tile_n == 0) ||
            error("grid $(grid.nx) x $(grid.ny) is not a whole number of $tile_n x $tile_n tiles")
        tw = Float32(2 * tile_edge + (tile_n - 1) * tile_pitch + active_w)
        if origin === nothing
            face_mm === nothing && error("tiles need origin = (x0, y0) or face_mm = (Lx, Ly)")
            x0 = Float32((face_mm[1] - (grid.nx ÷ tile_n) * tw) / 2)
            y0 = Float32((face_mm[2] - (grid.ny ÷ tile_n) * tw) / 2)
        else
            x0, y0 = Float32(origin[1]), Float32(origin[2])
        end
        return Readout(grid, false, 0f0, 0f0, 0f0, sR, sur_specular, two_sided, true,
                       Int32(tile_n), Float32(tile_pitch), Float32(tile_edge),
                       Float32(active_w), tw, x0, y0)
    end
    return Readout(grid, disc, cx, cy, r2, sR, sur_specular, two_sided, false,
                   Int32(0), 0f0, 0f0, 0f0, 0f0, 0f0, 0f0)
end

"""
Channel index (1-based) along one axis of a tiled array for the coordinate s measured
from the array corner, or 0 on a dead margin, between tiles, or outside the array.
"""
@inline function tile_channel(s::Float32, ro::Readout, nchan::Int32)
    s < 0f0 && return Int32(0)
    tile = unsafe_trunc(Int32, s / ro.tile_w)
    r = s - Float32(tile) * ro.tile_w - ro.tile_edge
    r < 0f0 && return Int32(0)
    k = unsafe_trunc(Int32, r / ro.tile_pitch)
    k >= ro.tile_n && return Int32(0)
    r - Float32(k) * ro.tile_pitch > ro.active_w && return Int32(0)
    i = tile * ro.tile_n + k + Int32(1)
    return i > nchan ? Int32(0) : i
end

"""
Whether the point (x, y) of the readout face is active, and the channel (i, j) it falls
in. Full face: active everywhere, binned by the pitch. Disc: binned by the pitch, active
inside the disc. Tiles: active on the squares, the channel being the square; on a dead
area the channel indices returned are placeholders (1) and must not be used.
"""
@inline function active_channel(ro::Readout, x::Float32, y::Float32)
    g = ro.grid
    if ro.tiles
        i = tile_channel(x - ro.x0, ro, g.nx)
        j = tile_channel(y - ro.y0, ro, g.ny)
        return (i > Int32(0)) & (j > Int32(0)), max(i, Int32(1)), max(j, Int32(1))
    end
    i = min(g.nx, Int32(1) + unsafe_trunc(Int32, x / g.pitch_mm))
    j = min(g.ny, Int32(1) + unsafe_trunc(Int32, y / g.pitch_mm))
    return !ro.disc || (x - ro.cx)^2 + (y - ro.cy)^2 <= ro.r2, i, j
end

"Centres (mm, crystal frame) of the channels of a tiled readout along x and y."
function channel_centres(ro::Readout)
    ro.tiles || error("channel_centres needs a tiled readout")
    axis(n, o) = [o + (k ÷ ro.tile_n) * ro.tile_w + ro.tile_edge +
                  (k % ro.tile_n) * ro.tile_pitch + ro.active_w / 2 for k in 0:n-1]
    return axis(ro.grid.nx, ro.x0), axis(ro.grid.ny, ro.y0)
end

struct TimeBinning
    bin_ns::Float32
    nbins::Int32            # counts array has nbins + 1, the last is the overflow bin
end

mutable struct Accumulator
    counts::Array{UInt32,3}         # nx × ny × (nbins+1) photoelectron counts
    first_ns::Matrix{Float32}       # earliest detected time per SiPM (Inf if none)
    ndet::Int64
    ndet_front::Int64               # subset of ndet on the z = 0 face (two-sided only)
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
                fill(Inf32, grid.nx, grid.ny), 0, 0, 0, 0, 0, 0, 0, 0, 0)

function Base.merge!(a::Accumulator, b::Accumulator)
    a.counts .+= b.counts
    a.first_ns .= min.(a.first_ns, b.first_ns)
    a.ndet += b.ndet; a.ndet_front += b.ndet_front
    a.nabs_bulk += b.nabs_bulk; a.nabs_wall += b.nabs_wall
    a.nabs_sipm += b.nabs_sipm; a.ncap += b.ncap; a.nabs_sur += b.nabs_sur
    a.nscat += b.nscat
    a.sum_bounces_det += b.sum_bounces_det
    return a
end

total_terminated(a::Accumulator) =
    a.ndet + a.nabs_bulk + a.nabs_wall + a.nabs_sipm + a.ncap + a.nabs_sur

@inline function record_detection!(acc::Accumulator, tb::TimeBinning,
                                   i::Int32, j::Int32, t::Float32, bounces::Int)
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

        if face == FACE_BACK || (ro.two_sided && face == FACE_FRONT)
            # single Fresnel surface crystal -> coupling medium (grease, or air gap)
            tir, R = fresnel(op.n_crystal, op.n_coupling, cosi)
            if tir || randu(s) < R
                ux, uy, uz = specular(ux, uy, uz, face)
            elseif op.n_window > 0f0 &&
                   !window_pass(s, op.n_crystal, op.n_coupling, op.n_window, cosi)
                # reflected by the coupling-to-window interface, back into the crystal
                ux, uy, uz = specular(ux, uy, uz, face)
            else
                active, ii, jj = active_channel(ro, x, y)
                if active
                    if randu(s) <= op.pde
                        record_detection!(acc, tb, ii, jj, t, bounces)
                        face == FACE_FRONT && (acc.ndet_front += 1)
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
