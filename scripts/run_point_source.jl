# Run one or more point-source configs:
#   julia --project=. --threads=auto scripts/run_point_source.jl runs/point_csi_teflon.toml [...]
using CryspLight

isempty(ARGS) && error("usage: run_point_source.jl <config.toml> [more configs...]")
for path in ARGS
    run_from_config(path)
end
