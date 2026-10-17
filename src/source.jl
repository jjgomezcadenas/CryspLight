# A radioactive point source outside the crystal (full simulation, design document
# Section "Source and gamma transport"). Na-22: one gamma of 1274.5 keV per decay and, in
# 90.3% of the decays, a positron whose annihilation gives two back-to-back gammas of
# 511 keV. Directions are isotropic; the gammas whose line meets the crystal are handed
# to the vendored gamma transport at their entry point.

using Random: AbstractRNG, rand

const E_511_MEV = 0.511
const E_1275_MEV = 1.2745
const NA22_POSITRON_FRACTION = 0.903

const GAMMA_511 = Int8(1)
const GAMMA_1275 = Int8(2)

struct GammaRay
    e_mev::Float64
    dir::NTuple{3,Float64}
    kind::Int8                  # GAMMA_511 or GAMMA_1275
end

"Isotropic unit vector in Float64 from a Random RNG."
@inline function isotropic_dir64(rng::AbstractRNG)
    uz = 1.0 - 2.0 * rand(rng)
    r = sqrt(max(0.0, 1.0 - uz * uz))
    phi = 2pi * rand(rng)
    return (r * cos(phi), r * sin(phi), uz)
end

"""
    na22_decay(rng) -> Vector{GammaRay}

The gammas of one Na-22 decay: the 1275 keV gamma, then the two 511 keV gammas when the
decay goes through the positron (opposite directions).
"""
function na22_decay(rng::AbstractRNG)
    out = GammaRay[GammaRay(E_1275_MEV, isotropic_dir64(rng), GAMMA_1275)]
    if rand(rng) < NA22_POSITRON_FRACTION
        d = isotropic_dir64(rng)
        push!(out, GammaRay(E_511_MEV, d, GAMMA_511))
        push!(out, GammaRay(E_511_MEV, (-d[1], -d[2], -d[3]), GAMMA_511))
    end
    return out
end

"""
    entry_point(source_mm, dir, L) -> (entry_mm, path_mm) or nothing

Where the ray from source_mm (crystal frame, outside the box [0, L]) along dir first meets
the box, and the distance to that point. nothing when the ray misses the crystal.
"""
function entry_point(source_mm, dir, L)
    t_near = -Inf
    t_far = Inf
    for a in 1:3
        if abs(dir[a]) < 1e-12
            (0.0 <= source_mm[a] <= L[a]) || return nothing
        else
            t1 = (0.0 - source_mm[a]) / dir[a]
            t2 = (L[a] - source_mm[a]) / dir[a]
            t1 > t2 && ((t1, t2) = (t2, t1))
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
        end
    end
    (t_near < t_far && t_near > 0.0) || return nothing
    e = (source_mm[1] + t_near * dir[1], source_mm[2] + t_near * dir[2],
         source_mm[3] + t_near * dir[3])
    entry = (clamp(Float32(e[1]), 0f0, L[1]), clamp(Float32(e[2]), 0f0, L[2]),
             clamp(Float32(e[3]), 0f0, L[3]))
    return entry, t_near
end

"""
Fraction of the sphere, seen from a point on the axis at distance d from the centre of a
rectangular face of sides a x b, covered by that face (the geometric acceptance of one
gamma).
"""
face_acceptance(a::Real, b::Real, d::Real) =
    4 * atan(a * b / (4 * d * sqrt((a / 2)^2 + (b / 2)^2 + d^2))) / (4pi)
