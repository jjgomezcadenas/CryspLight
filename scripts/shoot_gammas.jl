# End-to-end run: 511 keV gammas uniform over the front face -> vendored gamma
# transport -> optical transport (KA kernel; Metal if available) -> per-event SiPM
# outputs (deposited energy, photoelectrons, 8x8 maps) in output/<tag>/events.h5.
#   julia --project=. --threads=auto scripts/shoot_gammas.jl runs/gammas_csitl.toml [more...]
using CryspLight
using TOML, Printf, HDF5, Statistics
import Metal

for cfgpath in (isempty(ARGS) ? ["runs/gammas_csitl.toml", "runs/gammas_bgo.toml"] : ARGS)
    cfg = TOML.parsefile(cfgpath)
    camp = cfg["campaign"]
    mat = load_material(cfg["crystal"]["material"])
    sz = Float32.(cfg["crystal"]["size_mm"])
    box = Box((sz[1], sz[2], sz[3]))

    wrap = load_material(cfg["wrap"]["material"])
    sipm = load_material(cfg["sipm"]["material"])
    surf = cfg["surface"]
    op = OpticalParams(Float32(mat["n"]), Float32(cfg["sipm"]["coupling_n"]),
                       Float32(mat["abs_length_mm"]),
                       Float32(get(mat, "rayleigh_mm", Inf)),
                       Float32(wrap["reflectivity"]), wrap["model"] == "specular",
                       get(surf, "finish", "backpainted") == "backpainted",
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
    t = @elapsed edep, npe, maps = run_events!(box, op, ro, tb, pv,
                                               mat["yield_per_mev"];
                                               n_events = nev, seed = Int(camp["seed"]),
                                               tau_ns = Float32(mat["tau_ns"]),
                                               batch_photons = Int(camp["batch_photons"]),
                                               ArrayT = ArrayT)

    outdir = joinpath(dirname(dirname(abspath(cfgpath))), "output", camp["tag"])
    mkpath(outdir)
    cp(cfgpath, joinpath(outdir, basename(cfgpath)); force = true)
    h5open(joinpath(outdir, "events.h5"), "w") do f
        f["edep_kev"] = edep
        f["npe"] = npe
        f["maps"] = maps
        a = attrs(f)
        a["tag"] = camp["tag"]; a["n_events"] = nev; a["seed"] = Int(camp["seed"])
        a["crystal"] = cfg["crystal"]["material"]; a["pde"] = op.pde
        a["backend"] = ArrayT === Array ? "cpu" : "metal"
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
