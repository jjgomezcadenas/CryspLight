"""
Gamma — the vendored 511 keV photon-transport core from PTCryspMC.jl @ d4982db
(see VENDORED.md). Internal conventions preserved: cm, MeV, Float64, origin-centred
volumes. CryspLight converts to its own mm/keV/Float32 crystal frame at the interface
(src/gamma_interface.jl). Physics: free path from XCOM cross sections, Compton
(Klein-Nishina) with local recoil deposit, photoelectric absorption, 10 keV cut;
no coherent scattering, no electron transport, pair closed below 1.022 MeV.
"""
module Gamma

using Random

include("nist_data.jl")
include("gmaterials.jl")
include("ggeometry.jl")
include("sampling.jl")
include("gtransport.jl")

export Material, make_material, vacuum_material, sigma_macro, mfp
export Box, LogicalVolume, PhysicalVolume, distance_to_exit, distance_to_entry, is_inside
export propagate_photon, Interaction, Transported

end # module
