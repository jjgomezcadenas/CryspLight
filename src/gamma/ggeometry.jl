# VENDORED from PTCryspMC.jl @ d4982db (src/geometry.jl), trimmed to what
# propagate_photon needs for a single box crystal: Box (origin-centred, half-extents,
# cm) with the slab method, and the LogicalVolume/PhysicalVolume placement (translation
# only). The Cylinder/Sphere/CylShell solids and the Scanner machinery are not vendored.

const SURFACE_EPS  = 1e-10
const PARALLEL_EPS = 1e-20

"An axis-aligned box centred at the origin, half-extents in cm."
struct Box
    half_x_cm::Float64
    half_y_cm::Float64
    half_z_cm::Float64
end

is_inside(b::Box, p)::Bool =
    (abs(p[1]) <= b.half_x_cm) && (abs(p[2]) <= b.half_y_cm) && (abs(p[3]) <= b.half_z_cm)

@inline function _slab_crossings(pos, dir, b::Box)::Tuple{Float64,Float64}
    t_near = -Inf; t_far = Inf
    @inbounds for i in 1:3
        h = i == 1 ? b.half_x_cm : i == 2 ? b.half_y_cm : b.half_z_cm
        p = pos[i]; d = dir[i]
        if abs(d) > PARALLEL_EPS
            t1 = (-h - p) / d; t2 = (h - p) / d
            lo = min(t1, t2); hi = max(t1, t2)
            t_near = max(t_near, lo); t_far = min(t_far, hi)
        elseif p < -h || p > h
            return (Inf, -Inf)
        end
    end
    t_far < t_near ? (Inf, -Inf) : (t_near, t_far)
end

function distance_to_exit(pos, dir, b::Box)::Float64
    _, t_far = _slab_crossings(pos, dir, b)
    t_far > SURFACE_EPS ? t_far : Inf
end

function distance_to_entry(pos, dir, b::Box)::Float64
    t_near, t_far = _slab_crossings(pos, dir, b)
    (t_near > SURFACE_EPS && t_near < t_far) ? t_near : Inf
end

"A solid together with its material."
struct LogicalVolume
    name::String
    solid::Box
    material::Material
end

material(lv::LogicalVolume) = lv.material

"A logical volume placed at `position` in the world frame [cm] (translation only)."
struct PhysicalVolume
    logical::LogicalVolume
    position::NTuple{3,Float64}
end

solid(pv::PhysicalVolume)    = pv.logical.solid
material(pv::PhysicalVolume) = pv.logical.material

@inline _to_local(pv::PhysicalVolume, p) =
    (p[1] - pv.position[1], p[2] - pv.position[2], p[3] - pv.position[3])

is_inside(pv::PhysicalVolume, p)::Bool = is_inside(solid(pv), _to_local(pv, p))
distance_to_exit(pos, dir, pv::PhysicalVolume)::Float64 =
    distance_to_exit(_to_local(pv, pos), dir, solid(pv))
distance_to_entry(pos, dir, pv::PhysicalVolume)::Float64 =
    distance_to_entry(_to_local(pv, pos), dir, solid(pv))
