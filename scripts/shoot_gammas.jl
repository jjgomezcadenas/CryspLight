# End-to-end run: 511 keV gammas uniform over the front face -> vendored gamma
# transport -> optical transport (KA kernel; Metal if available) -> per-event SiPM
# outputs and interaction truth in output/<tag>/events.h5. Datasets are chunked along
# the event axis so a PyTorch Dataset can slice batches; Julia's column-major layout
# means maps/tmin (nx,ny,n) read as (n,ny,nx) in h5py, xyz as (n,3) -- CNN-ready.
#   julia --project=. --threads=auto scripts/shoot_gammas.jl runs/gammas_csitl.toml [more...]
using CryspLight
using TOML, Printf, HDF5, Statistics
import Metal

# Chunked dataset: full spatial dims, up to 4096 events per chunk.
function write_chunked(f, name, A)
    nd = ndims(A)
    ch = ntuple(i -> i < nd ? size(A, i) : min(4096, size(A, nd)), nd)
    d = create_dataset(f, name, eltype(A), size(A); chunk = ch)
    write(d, A)
end

for cfgpath in (isempty(ARGS) ? ["runs/gammas_csitl.toml", "runs/gammas_bgo.toml"] : ARGS)
    cfg = TOML.parsefile(cfgpath)
    camp = cfg["campaign"]
    mat = load_material(cfg["crystal"]["material"])
    sz = Float32.(cfg["crystal"]["size_mm"])
    box = Box((sz[1], sz[2], sz[3]))

    wrap = load_material(cfg["wrap"]["material"])
    sipm = load_material(cfg["sipm"]["material"])
    surf = cfg["surface"]
    finish = get(surf, "finish", "backpainted")
    op = OpticalParams(Float32(mat["n"]), Float32(cfg["sipm"]["coupling_n"]),
                       Float32(mat["abs_length_mm"]),
                       Float32(get(mat, "rayleigh_mm", Inf)),
                       Float32(wrap["reflectivity"]), wrap["model"] == "specular",
                       surface_has_gap(finish),
                       Float32(deg2rad(surf["sigma_alpha_deg"])),
                       Float32(get(cfg["sipm"], "pde_override", sipm["pde"])))
    grid = SipmGrid(Float32(sipm["pitch_mm"]), Int32(sipm["nx"]), Int32(sipm["ny"]))
    ro = Readout(grid)
    tb = TimeBinning(Float32(cfg["binning"]["bin_ns"]),
                     Int32(round(cfg["binning"]["window_ns"] / cfg["binning"]["bin_ns"])))

    gmat = Gamma.make_material(mat["name"], cfg["gamma"]["density"],
                               joinpath(dirname(dirname(abspath(cfgpath))), "data",
                                        cfg["gamma"]["xcom"]))
    pv = gamma_crystal(box.L, gmat)

    ArrayT = Metal.functional() ? Metal.MtlArray : Array
    nev = Int(camp["n_events"])
    t = @elapsed r = run_events!(box, op, ro, tb, pv,
                                 mat["yield_per_mev"];
                                 n_events = nev, seed = Int(camp["seed"]),
                                 tau_ns = Float32(mat["tau_ns"]),
                                 batch_photons = Int(camp["batch_photons"]),
                                 ArrayT = ArrayT)
    edep, npe = r.edep, r.npe

    outdir = joinpath(dirname(dirname(abspath(cfgpath))), "output", camp["tag"])
    mkpath(outdir)
    cp(cfgpath, joinpath(outdir, basename(cfgpath)); force = true)
    h5open(joinpath(outdir, "events.h5"), "w") do f
        write_chunked(f, "edep_kev", r.edep)
        write_chunked(f, "npe", r.npe)
        write_chunked(f, "maps", r.maps)          # pe counts after PDE (light matrix)
        write_chunked(f, "tmin_ns", r.tmin)       # first-pe time; Inf32 = no hit
        write_chunked(f, "xyz1_mm", r.xyz1)
        write_chunked(f, "e1_kev", r.e1)
        write_chunked(f, "xyz2_mm", r.xyz2)
        write_chunked(f, "e2_kev", r.e2)
        write_chunked(f, "er_kev", r.er)
        write_chunked(f, "int_type", r.int_type)
        write_chunked(f, "n_int", r.n_int)
        a = attrs(f)
        a["tag"] = camp["tag"]; a["n_events"] = nev; a["seed"] = Int(camp["seed"])
        a["crystal"] = cfg["crystal"]["material"]; a["pde"] = op.pde
        a["backend"] = ArrayT === Array ? "cpu" : "metal"
        a["int_type_key"] = "-1 = no interaction, 0 = direct photoelectric, X>=1 = X Compton scatters"
        a["tmin_nohit"] = "Inf"
    end

    interacted = edep .> 0
    peak = edep .> 510.0f0                       # full-absorption events
    pe_peak = npe[peak]
    @printf("%-14s %d events (%.1fs, %s): interacted %.1f%%, photopeak %.1f%%\n",
            camp["tag"], nev, t, ArrayT === Array ? "cpu" : "metal",
            100count(interacted) / nev, 100count(peak) / nev)
    if !isempty(pe_peak)
        m = mean(pe_peak); s = std(pe_peak)
        @printf("%-14s photopeak: %.0f pe mean, FWHM/mean %.2f%% (photostatistics+transport only)\n",
                "", m, 235.5s / m)
    end
end
