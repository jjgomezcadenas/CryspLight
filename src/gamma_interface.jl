# Bridge between the vendored gamma core (cm/MeV/Float64, origin-centred volumes) and
# the CryspLight crystal frame (mm/keV/Float32, corner origin). Times are added here:
# the vendored Interaction records carry no time, so t_k = cumulative path length / c.

using Random: AbstractRNG

const C_CM_NS = 29.9792458    # speed of light [cm/ns]

"Vendored gamma volume for a crystal of size L (mm) and a gamma Material."
gamma_crystal(L::NTuple{3,Float32}, mat::Gamma.Material) =
    Gamma.PhysicalVolume(
        Gamma.LogicalVolume("crystal",
                            Gamma.Box(L[1] / 20, L[2] / 20, L[3] / 20), mat),
        (0.0, 0.0, 0.0))

"""
    gamma_deposits(pv, L, entry_mm, dir, rng; e0_mev = 0.511)
        -> Vector{NTuple{5,Float32}}  # (x, y, z [mm], E [keV], t [ns])

Transport one gamma entering the crystal at entry_mm (crystal frame) along dir and
return its energy deposits in the crystal frame, time-stamped from the path length.
"""
function gamma_deposits(pv, L::NTuple{3,Float32}, entry_mm, dir, rng::AbstractRNG;
                        e0_mev::Float64 = 0.511)
    p0 = ((entry_mm[1] - L[1] / 2) / 10, (entry_mm[2] - L[2] / 2) / 10,
          (entry_mm[3] - L[3] / 2) / 10)
    tr = Gamma.propagate_photon(e0_mev, p0, dir, pv, rng)
    out = NTuple{5,Float32}[]
    xp, yp, zp = p0
    path = 0.0
    for r in tr.recs
        path += sqrt((r.x - xp)^2 + (r.y - yp)^2 + (r.z - zp)^2)
        xp, yp, zp = r.x, r.y, r.z
        r.e_dep > 0.0 || continue
        push!(out, (Float32(r.x * 10 + L[1] / 2), Float32(r.y * 10 + L[2] / 2),
                    Float32(r.z * 10 + L[3] / 2), Float32(r.e_dep * 1000),
                    Float32(path / C_CM_NS)))
    end
    return out
end
