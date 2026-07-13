"""
CryspLight — optical-photon transport in a wrapped scintillating crystal read out by a
SiPM matrix. v0: CPU reference implementation (Float32 throughout, counter-based Philox
RNG) of the physics specified in latex/crysp_light_metal.tex. The Metal port reuses the
same kernel logic.
"""
module CryspLight

using TOML
using Printf
using HDF5

include("philox.jl")
include("geometry.jl")
include("optics.jl")
include("materials.jl")
include("generation.jl")
include("transport.jl")
include("kernel.jl")
include("config.jl")
include("output.jl")

export PhiloxStream, randu
export Box, wall_hit, FACE_XP, FACE_XM, FACE_YP, FACE_YM, FACE_BACK, FACE_FRONT
export fresnel, specular!, lambertian_dir, isotropic_dir, rayleigh_dir, surface_interact
export OpticalParams, SipmGrid, Readout, TimeBinning, Accumulator, propagate_photon!
export total_terminated
export STATUS_DETECTED, STATUS_ABS_BULK, STATUS_ABS_WALL, STATUS_ABS_SIPM, STATUS_CAP,
       STATUS_ABS_SUR
export load_material, run_from_config, run_photons!, run_photons_ka!

end # module
