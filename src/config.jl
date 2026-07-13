# Run configuration: a TOML file in runs/ is the parameter source of truth and the run's
# provenance (PTCryspMC convention). The tag (= config filename) names the output dir
# output/<tag>/, which receives the results and a copy of the config.

struct RunSetup
    tag::String
    cfg::Dict{String,Any}
    box::Box
    op::OpticalParams
    grid::SipmGrid
    tb::TimeBinning
    pos::NTuple{3,Float32}
    tau_ns::Float32
    n_photons::Int
    seed::Int
    max_bounces::Int
    outdir::String
    cfg_path::String
end

function load_setup(cfg_path::AbstractString)
    cfg = TOML.parsefile(cfg_path)
    run = cfg["run"]
    tag = get(run, "tag", splitext(basename(cfg_path))[1])

    crystal = load_material(cfg["crystal"]["material"])
    wrap = load_material(cfg["wrap"]["material"])
    sipm = load_material(cfg["sipm"]["material"])

    sz = Float32.(cfg["crystal"]["size_mm"])
    box = Box((sz[1], sz[2], sz[3]))

    pde = Float32(get(cfg["sipm"], "pde_override", sipm["pde"]))
    surf = get(cfg, "surface", Dict{String,Any}())
    finish = get(surf, "finish", "backpainted")
    finish in ("backpainted", "frontpainted") || error("unknown finish: $finish")
    op = OpticalParams(Float32(crystal["n"]),
                       Float32(cfg["sipm"]["coupling_n"]),
                       Float32(crystal["abs_length_mm"]),
                       Float32(get(crystal, "rayleigh_mm", Inf)),
                       Float32(wrap["reflectivity"]),
                       wrap["model"] == "specular",
                       finish == "backpainted",
                       Float32(deg2rad(get(surf, "sigma_alpha_deg", 0.0))),
                       pde)

    grid = SipmGrid(Float32(sipm["pitch_mm"]), Int32(sipm["nx"]), Int32(sipm["ny"]))

    binning = cfg["binning"]
    bin_ns = Float32(binning["bin_ns"])
    tb = TimeBinning(bin_ns, Int32(round(binning["window_ns"] / bin_ns)))

    src = cfg["source"]
    src["type"] == "point" || error("v0 supports only the point source")
    p = Float32.(src["position_mm"])
    tau = src["time_profile"] == "delta" ? 0f0 : Float32(crystal["tau_ns"])

    outdir = joinpath(dirname(dirname(abspath(cfg_path))),
                      get(get(cfg, "output", Dict()), "dir", "output"), tag)

    RunSetup(tag, cfg, box, op, grid, tb, (p[1], p[2], p[3]), tau,
             Int(run["n_photons"]), Int(run["seed"]),
             Int(get(run, "max_bounces", 100_000)), outdir, abspath(cfg_path))
end

"Run a config end to end: transport, write output/<tag>/, print and return the summary."
function run_from_config(cfg_path::AbstractString)
    st = load_setup(cfg_path)
    elapsed = @elapsed acc = run_photons!(st.box, st.op, st.grid, st.tb;
                                          n_photons = st.n_photons, seed = st.seed,
                                          pos = st.pos, tau_ns = st.tau_ns,
                                          max_bounces = st.max_bounces)
    return write_output(st, acc, elapsed)
end
