# Cold-CsI calibration against the Soleti et al. (CRYSP) 3x3x20 mm measurement —
# see runs/soleti_calib.toml for the setup, anchors and every number.
# Maps the simulated collection efficiency eps_det over (Lambda_ext, f, R_wrap) and
# compares with the measured 0.529/0.505 (Y = 100k, PDE = 0.17) and their LUT MC 0.58.
#
#   julia --project=. --threads=auto scripts/soleti_calib.jl [runs/soleti_calib.toml]

using CryspLight
using TOML
using Printf

cfgpath = isempty(ARGS) ? joinpath(@__DIR__, "..", "runs", "soleti_calib.toml") : ARGS[1]
cfg = TOML.parsefile(cfgpath)

camp = cfg["campaign"]
mat = load_material(cfg["crystal"]["material"])
sz = Float32.(cfg["crystal"]["size_mm"])
box = Box((sz[1], sz[2], sz[3]))
n_crystal = Float32(mat["n"])

grid = SipmGrid(sz[1], Int32(1), Int32(1))       # one SiPM covering the full end face
ro = Readout(grid)
tb = TimeBinning(100f0, Int32(80))

# stratified equal-weight quantiles of the truncated exponential depth profile
src = cfg["source"]
att = src["att_mm"]; L = Float64(sz[3]); N = src["n_points"]
xy = Float32.(src["xy_mm"])
cdf_end = 1 - exp(-L / att)
positions = [(xy[1], xy[2], Float32(-att * log(1 - ((k - 0.5) / N) * cdf_end)))
             for k in 1:N]

nph = Int(camp["n_photons"]); seed = Int(camp["seed"]); maxb = Int(camp["max_bounces"])
coupling = Float32(cfg["readout"]["coupling_n"])
sigma = Float32(deg2rad(cfg["surface"]["sigma_alpha_deg"]))

function eps_det(lam_ext, f, wrap_R)
    lam_abs = f < 1 ? lam_ext / (1 - f) : Inf
    lam_ray = f > 0 ? lam_ext / f : Inf
    op = OpticalParams(n_crystal, coupling, Float32(lam_abs), Float32(lam_ray),
                       Float32(wrap_R), false, true, sigma, 1f0)
    acc = run_photons!(box, op, ro, tb; n_photons = nph, seed = seed,
                       positions = positions, tau_ns = 0f0, max_bounces = maxb)
    @assert total_terminated(acc) == nph
    return acc.ndet / nph
end

t1, t2 = cfg["targets"]["eps_measured"]
tmc = cfg["targets"]["eps_their_mc"]
@printf("targets: eps = %.3f (crystal 1) / %.3f (crystal 2), their LUT MC %.2f\n\n",
        t1, t2, tmc)
@printf("%-14s %-6s | %s\n", "Lambda_ext", "f", join([@sprintf("R=%.2f", R)
        for R in cfg["grid"]["wrap_R"]], "   "))
results = []
for lam in cfg["grid"]["lambda_ext_mm"], f in cfg["grid"]["f_split"]
    eps = [eps_det(lam, f, R) for R in cfg["grid"]["wrap_R"]]
    push!(results, (lam, f, eps))
    @printf("%-14.0f %-6.1f | %s\n", lam, f,
            join([@sprintf("%.3f", e) for e in eps], "    "))
end

outdir = joinpath(dirname(dirname(abspath(cfgpath))), "output", camp["tag"])
mkpath(outdir)
cp(cfgpath, joinpath(outdir, basename(cfgpath)); force = true)
open(joinpath(outdir, "summary.csv"), "w") do io
    println(io, "lambda_ext_mm,f_split," *
            join(["eps_R" * string(R) for R in cfg["grid"]["wrap_R"]], ","))
    for (lam, f, eps) in results
        println(io, join([lam, f, eps...], ","))
    end
end
println("\nwrote $(joinpath(outdir, "summary.csv"))")
