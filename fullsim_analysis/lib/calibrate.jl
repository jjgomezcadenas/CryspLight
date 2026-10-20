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

"""
    matched_filter_template(ch::DigitizerChain; length = 10, baseline_samples = 4)
        -> (tpl, nrm, norm, responses)

The matched filter of the digitiser calibration on the sampled waveform: `tpl` is the
pulse of one cell sampled from the sample of the avalanche, `length` samples long and
normalised to a peak of one, `nrm` its sum of squares, so that the correlation of the
template with the waveform (after subtracting the mean of the `baseline_samples`
preceding samples) divided by `nrm` equals the pulse amplitude when the template is
aligned with the avalanche. The estimator takes the local maxima of that correlation over
all alignments, and for the long pulse of the device the maximum lies one or two samples
after the avalanche and above the amplitude, by 4 to 6% depending on where the avalanche
falls between two samples. `norm` is the mean over the sampling phases (0 to one sample
in steps of the synthesis grid) of the local maximum for a noiseless isolated unit pulse,
and `responses` the value at each phase: divided by `norm`, the estimator gives one on
average for an isolated single cell.
"""
function matched_filter_template(ch::DigitizerChain; length::Int = 10, baseline_samples::Int = 4)
    s, g = ch.s, ch.grid
    step = max(1, round(Int, s.sample_ns / g.dt))
    x = zeros(Float64, g.n)
    k0 = 1 + step * cld(g.n, 4step)              # a sampling time well inside the record
    x[k0] = 1.0; device_pulse!(x, ch.pulse)
    tpl = x[k0:step:k0 + (length - 1) * step]; tpl ./= maximum(tpl)
    nrm = sum(abs2, tpl)
    responses = Float64[]
    for phase in 0:step-1
        fill!(x, 0.0); x[k0 + phase] = 1.0; device_pulse!(x, ch.pulse)
        y = matched_filter(x[1:step:end], tpl, nrm, 1.0; baseline_samples = baseline_samples)
        push!(responses, maximum(y))
    end
    return (tpl = tpl, nrm = nrm, norm = mean(responses), responses = responses)
end

"""
    matched_filter(v, tpl, nrm, norm; baseline_samples = 4) -> y

The filtered waveform: at each sample j, the template correlated with the samples from j
on, after subtracting the mean of the `baseline_samples` samples before j, divided by
`nrm` and by `norm` (matched_filter_template). Zero where the window does not fit.
"""
function matched_filter(v::AbstractVector, tpl::AbstractVector, nrm::Real, norm::Real;
                        baseline_samples::Int = 4)
    L, nb, n = length(tpl), baseline_samples, length(v)
    y = zeros(Float64, n)
    @inbounds for j in (nb + 1):(n - L)
        b = sum(view(v, (j - nb):(j - 1))) / nb
        acc = 0.0
        for m in 1:L
            acc += tpl[m] * (v[j + m - 1] - b)
        end
        y[j] = acc / (nrm * norm)
    end
    return y
end

"Local maxima of a filtered waveform above `floor`: the pulse amplitudes of the calibration."
function filter_amplitudes(y::AbstractVector, floor::Real)
    out = Float64[]
    @inbounds for j in 2:length(y)-1
        y[j] > y[j - 1] && y[j] >= y[j + 1] && y[j] > floor && push!(out, y[j])
    end
    return out
end

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
