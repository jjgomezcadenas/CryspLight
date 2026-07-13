# BGO surface/wrap calibration against the Ding & Liu 6 mm cube at 195 K —
# see runs/ding_calib.toml for the setup, anchors and every number.
# Maps eps_det over (sigma_alpha, wrap_R, Lambda_ext) and compares with the measured
# 0.825 +- 0.08 and their Geant4 MC 0.80.
#
#   julia --project=. --threads=auto scripts/ding_calib.jl [runs/ding_calib.toml]

using CryspLight
using TOML
using Printf

cfgpath = isempty(ARGS) ? joinpath(@__DIR__, "..", "runs", "ding_calib.toml") : ARGS[1]
cfg = TOML.parsefile(cfgpath)

camp = cfg["campaign"]
mat = load_material(cfg["crystal"]["material"])
sz = Float32.(cfg["crystal"]["size_mm"])
box = Box((sz[1], sz[2], sz[3]))
n_crystal = Float32(mat["n"])

grid = SipmGrid(sz[1], Int32(1), Int32(1))
ro = Readout(grid; two_sided = cfg["readout"]["two_sided"])
tb = TimeBinning(100f0, Int32(80))

# deposits just inside the entrance side face (x = 0): stratified truncated exponential
src = cfg["source"]
att = src["att_mm"]; L = Float64(sz[1]); N = src["n_points"]
yz = Float32.(src["yz_mm"])
cdf_end = 1 - exp(-L / att)
positions = [(Float32(-att * log(1 - ((k - 0.5) / N) * cdf_end)), yz[1], yz[2])
             for k in 1:N]

nph = Int(camp["n_photons"]); seed = Int(camp["seed"]); maxb = Int(camp["max_bounces"])
coupling = Float32(cfg["readout"]["coupling_n"])
f = cfg["grid"]["f_split"]

function eps_det(sigma_deg, wrap_R, lam_ext)
    lam_abs = f < 1 ? lam_ext / (1 - f) : Inf
    lam_ray = f > 0 ? lam_ext / f : Inf
    op = OpticalParams(n_crystal, coupling, Float32(lam_abs), Float32(lam_ray),
                       Float32(wrap_R), false, true, Float32(deg2rad(sigma_deg)), 1f0)
    acc = run_photons!(box, op, ro, tb; n_photons = nph, seed = seed,
                       positions = positions, tau_ns = 0f0, max_bounces = maxb)
    @assert total_terminated(acc) == nph
    return acc.ndet / nph
end

@printf("target: eps = %.3f +- 0.08 (measured, PDE band); their Geant4 MC %.2f\n\n",
        cfg["targets"]["eps_measured"], cfg["targets"]["eps_their_mc"])
@printf("%-12s %-10s | %s\n", "sigma_alpha", "Lam_ext", join([@sprintf("R=%.2f", R)
        for R in cfg["grid"]["wrap_R"]], "   "))
results = []
for sig in cfg["grid"]["sigma_alpha_deg"], lam in cfg["grid"]["lambda_ext_mm"]
    eps = [eps_det(sig, R, lam) for R in cfg["grid"]["wrap_R"]]
    push!(results, (sig, lam, eps))
    @printf("%-12.0f %-10.0f | %s\n", sig, lam,
            join([@sprintf("%.3f", e) for e in eps], "    "))
end

outdir = joinpath(dirname(dirname(abspath(cfgpath))), "output", camp["tag"])
mkpath(outdir)
cp(cfgpath, joinpath(outdir, basename(cfgpath)); force = true)
open(joinpath(outdir, "summary.csv"), "w") do io
    println(io, "sigma_alpha_deg,lambda_ext_mm," *
            join(["eps_R" * string(R) for R in cfg["grid"]["wrap_R"]], ","))
    for (sig, lam, eps) in results
        println(io, join([sig, lam, eps...], ","))
    end
end
println("\nwrote $(joinpath(outdir, "summary.csv"))")
