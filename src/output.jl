# Output: output/<tag>/light.h5 (counts + first-photon matrix + counters, provenance in
# the root attributes) plus a copy of the config, and a printed summary.

function write_output(st::RunSetup, acc::Accumulator, elapsed::Float64)
    mkpath(st.outdir)
    cp(st.cfg_path, joinpath(st.outdir, basename(st.cfg_path)); force = true)

    n = st.n_photons
    fdet = acc.ndet / n
    h5open(joinpath(st.outdir, "light.h5"), "w") do f
        f["counts"] = acc.counts
        f["first_time_ns"] = acc.first_ns
        a = attrs(f)
        a["tag"] = st.tag
        a["n_photons"] = n
        a["seed"] = st.seed
        a["bin_ns"] = st.tb.bin_ns
        a["nbins"] = st.tb.nbins
        a["crystal"] = st.cfg["crystal"]["material"]
        a["wrap"] = st.cfg["wrap"]["material"]
        a["pde"] = st.op.pde
        a["ndet"] = acc.ndet
        a["nabs_bulk"] = acc.nabs_bulk
        a["nabs_wall"] = acc.nabs_wall
        a["nabs_sipm"] = acc.nabs_sipm
        a["ncap"] = acc.ncap
        a["nabs_surround"] = acc.nabs_sur
        a["nscat"] = acc.nscat
        surf = get(st.cfg, "surface", Dict{String,Any}())
        a["finish"] = get(surf, "finish", "backpainted")
        a["sigma_alpha_deg"] = Float64(get(surf, "sigma_alpha_deg", 0.0))
    end

    # 8-fold symmetry residual of the count map (point source at the centre):
    # max relative deviation of a SiPM from the mean of its symmetry class.
    m = dropdims(sum(acc.counts; dims = 3); dims = 3)
    sym = symmetry_residual(m)

    mean_bounces = acc.ndet > 0 ? acc.sum_bounces_det / acc.ndet : NaN
    summary = Dict(
        "tag" => st.tag, "n_photons" => n, "detected_fraction" => fdet,
        "abs_bulk_fraction" => acc.nabs_bulk / n, "abs_wall_fraction" => acc.nabs_wall / n,
        "abs_sipm_fraction" => acc.nabs_sipm / n, "cap_fraction" => acc.ncap / n,
        "mean_bounces_detected" => mean_bounces, "symmetry_residual" => sym,
        "elapsed_s" => elapsed)

    @printf("%-22s  detected %6.2f%%   bulk %6.2f%%   wall %6.2f%%   cap %.2e\n",
            st.tag, 100fdet, 100acc.nabs_bulk / n, 100acc.nabs_wall / n, acc.ncap / n)
    @printf("%-22s  <bounces|det> %.1f   symmetry residual %.3f   %.2fs (%d threads)\n",
            "", mean_bounces, sym, elapsed, Threads.nthreads())
    return summary
end

"""
Max relative deviation of the count map from its 8-fold symmetrization (x-mirror,
y-mirror, transpose) — 0 for a perfectly symmetric map; statistical noise otherwise.
"""
function symmetry_residual(m::AbstractMatrix)
    ms = (m .+ reverse(m; dims = 1) .+ reverse(m; dims = 2) .+ reverse(m) .+
          m' .+ reverse(m'; dims = 1) .+ reverse(m'; dims = 2) .+ reverse(m')) ./ 8
    return maximum(abs.(m .- ms) ./ max.(ms, 1))
end
