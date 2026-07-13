# VENDORED from PTCryspMC.jl @ d4982db (src/materials.jl), trimmed for CryspLight:
# the scintillation/readout fields and the materials.json/JSON layer are dropped —
# CryspLight keeps optical properties in its own data/*.toml and constructs gamma
# materials directly from an XCOM CSV + density. Physics unchanged: Sigma = mu/rho * rho
# (compound-level XCOM tables), log-log interpolation on pre-logged grids.

"A material's density and XCOM photon cross sections, pre-logged for interpolation."
struct Material
    name::String
    density::Float64                  # g/cm^3
    E::Vector{Float64}                # nudged energy grid [MeV]
    log_E::Vector{Float64}
    incoherent::Vector{Float64}       # mu/rho [cm^2/g]
    photoelectric::Vector{Float64}
    pair::Vector{Float64}             # nuclear + electron
    log_incoherent::Vector{Float64}
    log_photoelectric::Vector{Float64}
    log_pair::Vector{Float64}
end

"Build a Material from an XCOM CSV file and a density [g/cm^3]."
function make_material(name::AbstractString, density::Real,
                       xcom_path::AbstractString)::Material
    xc = load_xcom(xcom_path)
    E = _prepare_xcom_energy(xc)
    pair = xc.pair_nuclear .+ xc.pair_electron
    Material(String(name), Float64(density),
             E, log.(E), xc.incoherent, xc.photoelectric, pair,
             prelog_data(xc.incoherent), prelog_data(xc.photoelectric),
             prelog_data(pair))
end

"A material with no interactions (empty grids): photons fly straight through."
vacuum_material() = Material("Vacuum", 0.0, Float64[], Float64[], Float64[],
                             Float64[], Float64[], Float64[], Float64[], Float64[])

"""
    sigma_macro(mat, E_MeV) -> (Sigma_C, Sigma_Ph, Sigma_P)

Macroscopic cross sections [cm^-1] for Compton (incoherent), photoelectric and pair.
Energies outside the XCOM grid are extrapolated on the end interval (log-log), not
clamped — fine within 10 keV--10 MeV; the 10 keV transport cut keeps histories away
from the untrustworthy low-energy extrapolation.
"""
function sigma_macro(mat::Material, E_MeV::Float64)::Tuple{Float64,Float64,Float64}
    isempty(mat.E) && return (0.0, 0.0, 0.0)
    lx = log(E_MeV)
    n = length(mat.E)
    lo = clamp(searchsortedlast(mat.E, E_MeV), 1, n - 1)
    rho = mat.density
    SC  = interp_loglog_prelogged(lx, mat.log_E, mat.log_incoherent, mat.incoherent, lo) * rho
    SPh = interp_loglog_prelogged(lx, mat.log_E, mat.log_photoelectric, mat.photoelectric, lo) * rho
    SP  = interp_loglog_prelogged(lx, mat.log_E, mat.log_pair, mat.pair, lo) * rho
    (SC, SPh, SP)
end

"Photon mean free path [cm] in `mat` at energy `E_MeV`."
function mfp(mat::Material, E_MeV::Float64)::Float64
    S = sum(sigma_macro(mat, E_MeV))
    S > 0.0 ? 1.0 / S : Inf
end
