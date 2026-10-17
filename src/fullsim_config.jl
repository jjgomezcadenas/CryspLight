# Configuration of a full-simulation optical run (fullsim_runs/<campaign>/<tag>.toml):
# crystal, wrapping, surface, coupling and window, tiled readout, source and run
# parameters, resolved into the objects of the transport. Separate from config.jl, which
# serves the pedagogical point-source runs and is left as it was.
#
# [run]      tag (default: file name), n_decays, seed, chunk_decays, batch_photons,
#            max_bounces, output_dir
# [crystal]  material (data/<name>.toml), size_mm, density, xcom
# [wrap]     material
# [surface]  finish, sigma_alpha_deg
# [coupling] n, window_n (0 = none)
# [readout]  pitch_mm, nx, ny, tiles, tile_n, tile_edge_mm, active_mm, dead_reflectivity,
#            dead_specular, pde (1 = record every photon reaching an active area)
# [source]   type = "na22", position_mm (crystal frame, z = 0 is the face opposite the
#            SiPMs)
# [binning]  bin_ns, window_ns (for the integrated maps of the optical stage)

struct FullsimSetup
    tag::String
    cfg::Dict{String,Any}
    cfg_text::String
    cfg_path::String
    box::Box
    op::OpticalParams
    ro::Readout
    tb::TimeBinning
    pv::Gamma.PhysicalVolume
    yield_per_mev::Float64
    times::NamedTuple
    lambda_nm::Float64
    source_mm::NTuple{3,Float64}
    n_decays::Int
    seed::Int
    chunk_decays::Int
    batch_photons::Int
    max_bounces::Int
    outdir::String
end

function load_fullsim(cfg_path::AbstractString)
    cfg_text = read(cfg_path, String)
    cfg = TOML.parse(cfg_text)
    run = cfg["run"]
    tag = get(run, "tag", splitext(basename(cfg_path))[1])
    repo = dirname(@__DIR__)

    cr = cfg["crystal"]
    mat = load_material(cr["material"])
    sz = Float32.(cr["size_mm"])
    box = Box((sz[1], sz[2], sz[3]))
    wrap = load_material(cfg["wrap"]["material"])
    surf = get(cfg, "surface", Dict{String,Any}())
    cp = cfg["coupling"]
    rd = cfg["readout"]
    op = OpticalParams(Float32(mat["n"]), Float32(cp["n"]), Float32(mat["abs_length_mm"]),
                       Float32(get(mat, "rayleigh_mm", Inf)), Float32(wrap["reflectivity"]),
                       wrap["model"] == "specular",
                       surface_has_gap(get(surf, "finish", "backpainted")),
                       Float32(deg2rad(get(surf, "sigma_alpha_deg", 0.0))),
                       Float32(get(rd, "pde", 1.0)), Float32(get(cp, "window_n", 0.0)))
    grid = SipmGrid(Float32(rd["pitch_mm"]), Int32(rd["nx"]), Int32(rd["ny"]))
    ro = get(rd, "tiles", false) ?
         Readout(grid; tiles = true, tile_n = Int(rd["tile_n"]),
                 tile_pitch = rd["pitch_mm"], tile_edge = rd["tile_edge_mm"],
                 active_w = rd["active_mm"], face_mm = (sz[1], sz[2]),
                 sur_R = get(rd, "dead_reflectivity", 0.5),
                 sur_specular = get(rd, "dead_specular", false)) :
         Readout(grid)
    bn = cfg["binning"]
    tb = TimeBinning(Float32(bn["bin_ns"]), Int32(round(bn["window_ns"] / bn["bin_ns"])))
    gmat = Gamma.make_material(mat["name"], cr["density"], joinpath(repo, "data", cr["xcom"]))
    pv = gamma_crystal(box.L, gmat)

    src = cfg["source"]
    src["type"] == "na22" || error("source type $(src["type"]) not supported (na22)")
    p = Float64.(src["position_mm"])
    outdir = joinpath(repo, get(run, "output_dir", "output"), tag)
    return FullsimSetup(tag, cfg, cfg_text, abspath(cfg_path), box, op, ro, tb, pv,
                        Float64(mat["yield_per_mev"]), scintillation_times(mat),
                        Float64(mat["lambda_nm"]), (p[1], p[2], p[3]),
                        Int(run["n_decays"]), Int(run["seed"]),
                        Int(get(run, "chunk_decays", 10_000)),
                        Int(get(run, "batch_photons", 2_000_000)),
                        Int(get(run, "max_bounces", 100_000)), outdir)
end

"""
    run_fullsim_optical(cfg_path; ArrayT, verbose) -> path of the photoelectron file

Run the optical stage of a full-simulation configuration chunk by chunk and stream the
result to output/<tag>/pes.h5, with a copy of the configuration beside it.
"""
function run_fullsim_optical(cfg_path::AbstractString; ArrayT = Array, verbose::Bool = true)
    st = load_fullsim(cfg_path)
    mkpath(st.outdir)
    cp(st.cfg_path, joinpath(st.outdir, basename(st.cfg_path)); force = true)
    path = joinpath(st.outdir, "pes.h5")
    w = open_pes(path, st.ro.grid.nx, st.ro.grid.ny; cfg_text = st.cfg_text,
                 pde_applied = st.op.pde, lambda_nm = st.lambda_nm,
                 attributes = Dict("tag" => st.tag, "seed" => st.seed,
                                   "backend" => ArrayT === Array ? "cpu" : "metal",
                                   "source_mm" => collect(st.source_mm)))
    t_all = @elapsed begin
        first = 1
        while first <= st.n_decays
            n = min(st.chunk_decays, st.n_decays - first + 1)
            r = run_decays!(st.box, st.op, st.ro, st.tb, st.pv, st.yield_per_mev;
                            source_mm = st.source_mm, n_decays = n, seed = st.seed,
                            first_decay = first, batch_photons = st.batch_photons,
                            max_bounces = st.max_bounces, ArrayT = ArrayT, st.times...)
            w = append_pes!(w, r)
            verbose && @printf("%-20s decays %7d / %d, %9d photoelectron records so far\n",
                               st.tag, w.n_decays, st.n_decays, length(w.file["pes"]["channel"]))
            first += n
        end
    end
    attrs(w.file)["elapsed_s"] = t_all
    close_pes(w)
    verbose && @printf("%-20s written %s (%.1f s)\n", st.tag, path, t_all)
    return path
end
