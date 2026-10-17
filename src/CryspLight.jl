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
include("gamma/Gamma.jl")
include("gamma_interface.jl")
include("events.jl")
include("source.jl")
include("decays.jl")
include("pes_file.jl")
include("fullsim_config.jl")
include("frontend/sipm.jl")
include("frontend/shaping.jl")
include("frontend/citiroc.jl")
include("frontend/digitizer.jl")
include("frontend/frontend_run.jl")
include("config.jl")
include("output.jl")

export PhiloxStream, randu
export Box, wall_hit, FACE_XP, FACE_XM, FACE_YP, FACE_YM, FACE_BACK, FACE_FRONT
export fresnel, specular!, lambertian_dir, isotropic_dir, rayleigh_dir, surface_interact,
       window_pass
export OpticalParams, SipmGrid, Readout, TimeBinning, Accumulator, propagate_photon!
export total_terminated, active_channel, tile_channel, channel_centres
export STATUS_DETECTED, STATUS_ABS_BULK, STATUS_ABS_WALL, STATUS_ABS_SIPM, STATUS_CAP,
       STATUS_ABS_SUR
export load_material, scintillation_times, surface_has_gap, run_from_config, run_photons!, run_photons_ka!
export Gamma, gamma_crystal, gamma_deposits, run_events!, rand_poisson
export GammaRay, na22_decay, isotropic_dir64, entry_point, face_acceptance, run_decays!
export open_pes, append_pes!, close_pes, read_pes, event_pes, code_version
export FullsimSetup, load_fullsim, run_fullsim_optical
export SipmDevice, SipmOperation, Avalanche, breakdown_voltage, gain, pde, crosstalk, dark_rate,
       afterpulse_prob, pulse_shape, respond, total_charge, interp_clamped
export TimeGrid, Shaper, bin_train!, shape!, device_pulse!, noise_generator, sample_index, sample_time
export CitirocSettings, CitirocChain, CitirocEvent, citiroc_event, stored_fraction, trigger_time
export DigitizerSettings, DigitizerChain, DigitizerEvent, digitizer_event, pulse_cdf, window_fraction
export FrontendSetup, load_frontend, run_fullsim_frontend, event_trains, settings_from

end # module
