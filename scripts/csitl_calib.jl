# CsI(Tl) calibration against the Brose et al. Fig. 7 wrapping ladder
# (arXiv physics/9805010) with Knyazev bulk parameters — see runs/csitl_calib.toml for
# every number and the campaign description.
#
# For each (f_split, sigma_alpha) grid point: simulate the bare crystal, then each
# Teflon case over its wrap_R scan; interpolate the R_eff that reproduces the measured
# LY(case)/LY(bare) ratio. Physical, monotone-in-thickness R_eff values (and a sane bare
# fraction) identify the preferred grid point.
#
#   julia --project=. --threads=auto scripts/csitl_calib.jl [runs/csitl_calib.toml]

using CryspLight
using TOML
using Printf

cfgpath = isempty(ARGS) ? joinpath(@__DIR__, "..", "runs", "csitl_calib.toml") : ARGS[1]
cfg = TOML.parsefile(cfgpath)

camp = cfg["campaign"]
mat = load_material(cfg["crystal"]["material"])
sz = Float32.(cfg["crystal"]["size_mm"])
box = Box((sz[1], sz[2], sz[3]))
# the campaign scans the absorption/scattering split of the MEASURED EXTINCTION, so
# recombine it from the material file (which stores the currently fitted split)
lam_ext = Float32(1 / (1 / mat["abs_length_mm"] + 1 / get(mat, "rayleigh_mm", Inf)))
n_crystal = Float32(mat["n"])

rd = cfg["readout"]
grid = SipmGrid(Float32(sz[1] / 8), Int32(8), Int32(8))   # light-map binning only
readout(sur_R) = Readout(grid; center = Tuple(rd["disc_center_mm"]),
                         radius = rd["disc_radius_mm"], sur_R = sur_R)
tb = TimeBinning(100f0, Int32(40))

src = cfg["source"]
xy = Float32.(src["xy_mm"])
positions = [(xy[1], xy[2], Float32(z)) for z in src["positions_z_mm"]]

nph = Int(camp["n_photons"])
seed = Int(camp["seed"])
maxb = Int(camp["max_bounces"])

function collection(f, sigma_deg, wrap_R, sur_R)
    lam_abs = f < 1 ? lam_ext / (1 - f) : Inf32
    lam_ray = f > 0 ? lam_ext / f : Inf32
    op = OpticalParams(n_crystal, Float32(rd["coupling_n"]), Float32(lam_abs),
                       Float32(lam_ray), Float32(wrap_R), false, true,
                       Float32(deg2rad(sigma_deg)), Float32(rd["pde"]))
    acc = run_photons!(box, op, readout(sur_R), tb; n_photons = nph, seed = seed,
                       positions = positions, tau_ns = 0f0, max_bounces = maxb)
    @assert total_terminated(acc) == nph
    return acc.ndet / nph
end

"Linear interpolation of the R at which frac(R)/base crosses the target ratio."
function fit_R(Rs, fracs, base, target)
    ratios = fracs ./ base
    for i in 1:length(Rs)-1
        lo, hi = ratios[i], ratios[i+1]
        if (lo - target) * (hi - target) <= 0
            w = (target - lo) / (hi - lo)
            return Rs[i] + w * (Rs[i+1] - Rs[i])
        end
    end
    return NaN
end

cases = cfg["cases"]
bare = only(filter(c -> c["name"] == "bare", cases))
teflon = filter(c -> haskey(c, "wrap_R_scan"), cases)
ly_bare = bare["target_ly"]

results = []
for f in cfg["grid"]["f_split"], sig in cfg["grid"]["sigma_alpha_deg"]
    fb = collection(f, sig, bare["wrap_R"], bare["surround_R"])
    @printf("\nf = %.2f  sigma_alpha = %.1f deg   bare collection = %.4f\n", f, sig, fb)
    row = Dict{String,Any}("f" => f, "sigma" => sig, "bare_frac" => fb)
    for c in teflon
        Rs = Float64.(c["wrap_R_scan"])
        # whole crystal wrapped: the rear-face surround is the same Teflon
        fr = [collection(f, sig, R, get(c, "surround", "") == "wrap" ? R :
                         get(c, "surround_R", 0.0)) for R in Rs]
        target = c["target_ly"] / ly_bare
        Rfit = fit_R(Rs, fr, fb, target)
        row[c["name"]] = Rfit
        @printf("  %-18s target ratio %5.2f   fracs %s   R_eff = %s\n",
                c["name"], target, join([@sprintf("%.3f", x) for x in fr], " "),
                isnan(Rfit) ? "out of scan range" : @sprintf("%.3f", Rfit))
    end
    push!(results, row)
end

# summary table + CSV
outdir = joinpath(dirname(dirname(abspath(cfgpath))), "output", camp["tag"])
mkpath(outdir)
cp(cfgpath, joinpath(outdir, basename(cfgpath)); force = true)
names = [c["name"] for c in teflon]
open(joinpath(outdir, "summary.csv"), "w") do io
    println(io, "f_split,sigma_alpha_deg,bare_frac," * join("Reff_" .* names, ","))
    for r in results
        vals = [get(r, n, NaN) for n in names]
        println(io, join([r["f"], r["sigma"], r["bare_frac"], vals...], ","))
    end
end

println("\n==== summary (R_eff fitted to each measured LY ratio; physical = monotone")
println("     in thickness, below saturation ~0.99, mono < multilayer) ====")
@printf("%-6s %-10s %-10s %s\n", "f", "sigma", "bare_frac", join(lpad.(names, 20)))
for r in results
    @printf("%-6.2f %-10.1f %-10.4f %s\n", r["f"], r["sigma"], r["bare_frac"],
            join([lpad(isnan(get(r, n, NaN)) ? "—" : @sprintf("%.3f", r[n]), 20)
                  for n in names]))
end
println("\nwrote $(joinpath(outdir, "summary.csv"))")
