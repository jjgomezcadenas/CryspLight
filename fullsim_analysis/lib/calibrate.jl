# The charge of each channel in photoelectrons by the route of the measurement, from the
# ADC values of the event file and the calibration of the same readout on its own dark
# counts (calibration.jl). Nothing of the simulation's own calibration enters.
#   Citiroc and Radioroc2  low gain (the energy path) minus its zero level, divided by the
#                          gain of the high-gain path (short shaping, where the dark
#                          counts give peaks) times the ratio of the two paths, assumed
#                          known from charge injection (the value of the configuration);
#   digitiser              integrated charge over the gain in count-samples: the fitted
#                          amplitude of one photoelectron times the area over peak height
#                          of the averaged pulse.

"Per-channel vector of a case in an HDF5 results file."
read_case(campaign, study, file, case, name) =
    h5open(f -> Float64.(read(f[case][name])), joinpath(results_dir(campaign, study), file))

"Fitted gains with the failed fits replaced by the median of the others (and their count)."
function fitted_gains(campaign, case)
    g = read_case(campaign, "calibration", "calibration.h5", case, "gain_fitted")
    bad = .!isfinite.(g)
    g[bad] .= median(g[.!bad])
    return g, count(bad)
end

"""
    measured_charge(e, campaign; calib_case = e.tag) -> (q, info)

q: nch x n charges in photoelectrons by the route of the measurement; info: a named tuple
with the gains, pedestals and ratios used.
"""
function measured_charge(e, campaign; calib_case = e.tag)
    gain, nfail = fitted_gains(campaign, calib_case)
    if e.frontend == "digitizer"
        aop = median(Float64.(e.calib["gain_per_pe"]) ./ Float64.(e.calib["peak_counts_per_pe"]))
        q = Float64.(e.charge) ./ (gain .* aop)
        return q, (gain = gain, failed_fits = nfail, area_over_peak = aop)
    end
    ped_lg = read_case(campaign, "calibration", "calibration.h5", calib_case, "lg_pedestal")
    r = Float64(e.run["fe_lg_ratio"])
    q = (Float64.(e.lg) .- ped_lg) ./ (gain .* r)
    return q, (gain = gain, failed_fits = nfail, lg_over_hg = fill(r, e.nch), lg_pedestal = ped_lg)
end
