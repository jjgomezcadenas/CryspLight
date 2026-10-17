# Optical stage of the full simulation: Na-22 decays at the source -> gammas in the
# crystal -> scintillation photons -> photons reaching the active areas of the tiled
# SiPM array, streamed to output/<tag>/pes.h5 chunk by chunk (Metal when available).
#   julia --project=. --threads=auto scripts/fullsim_optical.jl fullsim_runs/<campaign>/<tag>.toml [more...]
#   julia --project=. scripts/fullsim_optical.jl --cpu <config>      # force the CPU backend
using CryspLight
import Metal

args = copy(ARGS)
force_cpu = "--cpu" in args
filter!(a -> a != "--cpu", args)
isempty(args) && error("usage: fullsim_optical.jl [--cpu] <config.toml> [more...]")
ArrayT = (!force_cpu && Metal.functional()) ? Metal.MtlArray : Array
for cfg in args
    run_fullsim_optical(cfg; ArrayT = ArrayT)
end
