# SiPM and electronics stage of the full simulation: reads output/<tag>/pes.h5 of an
# optical run, applies the SiPM response and the configured front end (Citiroc 1A or
# DT2745 digitizer) to every decay, and writes output/<tag>/events.h5 in the layout common
# to measurements and simulation.
#   julia --project=. scripts/fullsim_frontend.jl fullsim_runs/<campaign>/frontend_<tag>.toml [more...]
using CryspLight
isempty(ARGS) && error("usage: fullsim_frontend.jl <config.toml> [more...]")
for cfg in ARGS
    run_fullsim_frontend(cfg)
end
