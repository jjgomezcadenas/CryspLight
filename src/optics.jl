# Surface optics: unpolarized Fresnel with TIR, specular reflection off a box face,
# and Lambertian re-injection (diffuse wrap), per the UNIFIED back-painted spec of the
# design note.

"""
Unpolarized Fresnel reflectance for a photon in medium n1 hitting an interface to n2 at
|cos(theta_i)| = cosi. Returns (tir, R): tir = total internal reflection (R meaningless),
otherwise R = (Rs + Rp)/2.
"""
@inline function fresnel(n1::Float32, n2::Float32, cosi::Float32)
    sin2t = (n1 / n2)^2 * (1f0 - cosi^2)
    if sin2t >= 1f0
        return true, 1f0
    end
    cost = sqrt(1f0 - sin2t)
    rs = ((n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost))^2
    rp = ((n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi))^2
    return false, 0.5f0 * (rs + rp)
end

"Axis index (1,2,3) of a face code."
@inline face_axis(face::Int32) = face <= 2 ? 1 : (face <= 4 ? 2 : 3)

"Specular reflection at a face: flip the direction component normal to it."
@inline function specular(ux::Float32, uy::Float32, uz::Float32, face::Int32)
    a = face_axis(face)
    return a == 1 ? (-ux, uy, uz) : (a == 2 ? (ux, -uy, uz) : (ux, uy, -uz))
end

"Outward-normal axis (1,2,3) and sign of a face."
@inline function face_frame(face::Int32)
    a = face_axis(face)
    sgn = (face == FACE_XP || face == FACE_YP || face == FACE_BACK) ? 1f0 : -1f0
    return a, sgn
end

"Standard normal via Box-Muller."
@inline gaussian(s::PhiloxStream) =
    sqrt(-2f0 * log(randu(s))) * cos(2f0 * Float32(pi) * randu(s))

"""
UNIFIED specular-lobe interaction at a rough crystal--air interface (air-gap finishes):
sample a micro-facet normal tilted by alpha ~ |N(0, sigma_alpha)| from the average normal,
apply Fresnel/TIR against the facet. Returns (ux, uy, uz, transmitted): the lobe-reflected
direction (transmitted = false), or the unchanged direction with transmitted = true (the
photon enters the gap; its direction there is not needed). Rejection loops guard facets not
seen by the photon and reflections that would leave the crystal; fallback = ideal specular.
"""
@inline function surface_interact(s::PhiloxStream, ux::Float32, uy::Float32, uz::Float32,
                                  face::Int32, n_crystal::Float32, sigma_alpha::Float32)
    a, sgn = face_frame(face)
    # outward normal N and in-plane tangents T1, T2 (the other two axes)
    Nx = a == 1 ? sgn : 0f0; Ny = a == 2 ? sgn : 0f0; Nz = a == 3 ? sgn : 0f0
    for _ in 1:20
        alpha = sigma_alpha * abs(gaussian(s))
        alpha >= 1.2f0 && continue                       # ~69 deg safety cut
        ca = cos(alpha); sa = sin(alpha)
        phi = 2f0 * Float32(pi) * randu(s)
        b = sa * cos(phi); c = sa * sin(phi)
        # facet normal: rotate N by alpha toward a random in-plane azimuth
        # (components b, c go on the two tangent axes, ca*sgn on the face axis)
        Nfx = a == 1 ? ca * sgn : b
        Nfy = a == 2 ? ca * sgn : (a == 1 ? b : c)
        Nfz = a == 3 ? ca * sgn : c
        cosi = ux * Nfx + uy * Nfy + uz * Nfz
        cosi <= 0f0 && continue                          # facet not illuminated
        tir, R = fresnel(n_crystal, 1f0, cosi)
        if tir || randu(s) < R
            rx = ux - 2f0 * cosi * Nfx
            ry = uy - 2f0 * cosi * Nfy
            rz = uz - 2f0 * cosi * Nfz
            if rx * Nx + ry * Ny + rz * Nz < 0f0         # must point back inside
                return rx, ry, rz, false
            end
        else
            return ux, uy, uz, true
        end
    end
    r = specular(ux, uy, uz, face)
    return r[1], r[2], r[3], false
end

"""
Rayleigh scattering: new direction with the (1 + cos^2 theta) phase function about the
current one (polarization not tracked).
"""
@inline function rayleigh_dir(s::PhiloxStream, ux::Float32, uy::Float32, uz::Float32)
    mu = 0f0
    while true
        mu = 1f0 - 2f0 * randu(s)
        2f0 * randu(s) <= 1f0 + mu * mu && break
    end
    st = sqrt(max(0f0, 1f0 - mu * mu))
    phi = 2f0 * Float32(pi) * randu(s)
    cb = st * cos(phi); sb = st * sin(phi)
    # orthonormal basis (t1, t2) transverse to u
    if abs(uz) < 0.9f0
        inv = 1f0 / sqrt(ux * ux + uy * uy)
        t1x = -uy * inv; t1y = ux * inv; t1z = 0f0
    else
        inv = 1f0 / sqrt(uy * uy + uz * uz)
        t1x = 0f0; t1y = -uz * inv; t1z = uy * inv
    end
    t2x = uy * t1z - uz * t1y
    t2y = uz * t1x - ux * t1z
    t2z = ux * t1y - uy * t1x
    vx = mu * ux + cb * t1x + sb * t2x
    vy = mu * uy + cb * t1y + sb * t2y
    vz = mu * uz + cb * t1z + sb * t2z
    inv = 1f0 / sqrt(vx * vx + vy * vy + vz * vz)
    return vx * inv, vy * inv, vz * inv
end

"""
Lambertian direction about the inward normal of a face: cos(theta) = sqrt(xi) about the
normal (pdf ∝ cos), uniform azimuth in the face plane.
"""
@inline function lambertian_dir(s::PhiloxStream, face::Int32)
    ct = sqrt(randu(s))                 # cos(theta) w.r.t. inward normal
    st = sqrt(max(0f0, 1f0 - ct * ct))
    phi = 2f0 * Float32(pi) * randu(s)
    a = st * cos(phi)
    b = st * sin(phi)
    # inward normal: -axis for the "+" faces (odd codes 1,3,5), +axis for the "-" faces
    if face == FACE_XP
        return (-ct, a, b)
    elseif face == FACE_XM
        return (ct, a, b)
    elseif face == FACE_YP
        return (a, -ct, b)
    elseif face == FACE_YM
        return (a, ct, b)
    elseif face == FACE_BACK
        return (a, b, -ct)
    else
        return (a, b, ct)
    end
end
