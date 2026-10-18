# Single-photoelectron calibration of a front end from the dark counts of the SiPMs, as
# the bench measures it, on decays with no energy deposit:
#   Citiroc and Radioroc2  self-trigger of one channel on its own discriminator at half a
#                          photoelectron (the fast shaper of the Citiroc, the time
#                          preamplifier T1 of the Radioroc2), high-gain value held at the
#                          peaking time after the trigger (track and hold); the channels
#                          whose discriminator fired within 2 ns of the trigger, and whose
#                          value is therefore held at its peak, enter the spectrum;
#   digitiser              the sampled waveform of every channel with its noise, filtered
#                          with the sampled pulse of one cell (matched filter) after
#                          subtracting the mean of the 4 samples before, and the local
#                          maxima of the filtered waveform.
# For each channel the one-photoelectron peak is fitted; its position is the gain (the
# baseline being subtracted), compared with the gain the simulation used. The ratio is
# not one: the hold of the ASICs does not fall exactly at the maximum, and the sampling of
# the digitiser misses part of the peak; it is common to all channels. For the Radioroc2 an optional peaking
# time replaces the one of the configuration (the calibration can use a shorter one).
# Writes results/<campaign>/calibration/calibration.h5 (one group per case: spectrum
# normalised channel by channel to the true gain, raw spectrum, fitted and true gains)
# and calibration.csv (one row per case).
#   julia --project=. --threads=8 fullsim_analysis/scripts/calibration.jl <campaign> <frontend config> [n_empty] [tp_ns]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, HDF5, Statistics, Printf
using Random: Xoshiro, seed!

campaign, cfg = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20_000
tp_override = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : NaN
r = open_pes_run(cfg)
st, pes, op = r.setup, r.pes, r.op
kind, s = st.frontend, st.settings
case = isfinite(tp_override) ? "$(st.tag)_tp$(Int(tp_override))" : st.tag
empty = event_lists(pes; nmax = nmax).empty
nch = length(op.gain_factor)
vals = [Float64[] for _ in 1:nch]          # per channel, in counts above the baseline

if kind == "citiroc" || kind == "radioroc"
    cs = if kind == "citiroc"
        with(s; trigger = "majority", majority = 1, threshold_pe = 0.5, coincidence_ns = 2.0,
             peak_detector = false, hold_ns = s.tp_ns)
    else
        tp = isfinite(tp_override) ? tp_override : s.tp_ns[1]
        with(s; trigger = "t1", t1_threshold_pe = 0.5, coincidence_ns = 2.0, peak_detector = false,
             tp_ns = fill(tp, nch), tp_lg_ns = Float64[], hold_ns = tp)
    end
    res = emulate(pes, op, kind, cs, empty)
    # the ADC values are integers: a uniform spread over each count lets the peak fits
    # treat them as a continuous variable without binning artefacts
    rng = Xoshiro(17)
    for i in eachindex(empty), c in 1:nch
        res.triggered[i] && res.hits[c, i] &&
            push!(vals[c], Float64(res.hg[c, i]) - s.pedestal + rand(rng) - 0.5)
    end
    true_gain = s.adc_per_pe .* op.gain_factor
    unit = "ADC counts, high gain"
else
    # digitiser: the sampled waveform of each channel over the dark window
    ch0 = DigitizerChain(s, op, Xoshiro(1))
    g, step = ch0.grid, max(1, round(Int, s.sample_ns / ch0.grid.dt))
    nt = Threads.maxthreadid()
    chains = [DigitizerChain(s, op, Xoshiro(t)) for t in 1:nt]
    local_vals = [[Float64[] for _ in 1:nch] for _ in 1:nt]
    # template: the sampled pulse of one cell, peak 1; matched filter normalised so that a
    # single cell gives its peak amplitude
    fill!(ch0.x, 0.0); ch0.x[100] = 1.0; device_pulse!(ch0.x, ch0.pulse)
    tpl = ch0.x[100:step:100 + 9step]; tpl ./= maximum(tpl)
    L, nb, nrm = length(tpl), 4, sum(abs2, tpl)
    Threads.@threads :static for i in eachindex(empty)
        k = empty[i]
        ch = chains[Threads.threadid()]; lv = local_vals[Threads.threadid()]
        seed!(ch.rng, hash((5, k, :noise)))
        trains = event_trains(op, event_pes(pes, k), nch, Xoshiro(hash((5, k))))
        for c in 1:nch
            bin_train!(ch.x, g, trains[c])
            device_pulse!(ch.x, ch.pulse)
            amp = ch.peak_counts_per_pe * op.gain_factor[c]
            v = [amp * ch.x[j] + s.noise_counts * randn(ch.rng) for j in 1:step:g.n]
            n = length(v)
            y = zeros(n)
            for j in (nb + 1):(n - L)
                b = sum(view(v, (j - nb):(j - 1))) / nb
                y[j] = sum(tpl[m] * (v[j + m - 1] - b) for m in 1:L) / nrm
            end
            for j in (nb + 2):(n - L - 1)
                y[j] > y[j - 1] && y[j] >= y[j + 1] && y[j] > 0.3 * ch.peak_counts_per_pe && push!(lv[c], y[j])
            end
        end
    end
    for t in 1:nt, c in 1:nch
        append!(vals[c], local_vals[t][c])
    end
    true_gain = chains[1].peak_counts_per_pe .* op.gain_factor
    unit = "ADC counts, matched-filter amplitude"
end

# per channel: the one- and two-photoelectron peaks, searched around the nominal gain
nominal = median(true_gain)
fitted = fill(NaN, nch); sig1 = fill(NaN, nch)
for c in 1:nch
    length(vals[c]) < 100 && continue
    p1 = peak_fit(vals[c], 0.6nominal, 1.25nominal; nbins = 60)
    isfinite(p1.mean) || continue
    fitted[c] = p1.mean        # the baseline is subtracted: the one-photoelectron position is the gain
    sig1[c] = p1.sigma
end
ok = isfinite.(fitted)
ratio = fitted[ok] ./ true_gain[ok]
# robust summary: median, scaled median absolute deviation, and the channels beyond five
# of those (fits that caught another peak)
rmed = isempty(ratio) ? NaN : median(ratio)
rmad = isempty(ratio) ? NaN : 1.4826 * median(abs.(ratio .- rmed))
nout = count(abs.(ratio .- rmed) .> 5rmad)
allnorm = reduce(vcat, [vals[c] ./ true_gain[c] for c in 1:nch])
allraw = reduce(vcat, vals)

open_results(campaign, "calibration"; mode = isfile(joinpath(results_dir(campaign, "calibration"), "calibration.h5")) ? "r+" : "w") do h5
    haskey(h5, case) && delete_object(h5, case)
    gr = create_group(h5, case)
    stamp!(gr, "config" => st.cfg_text, "frontend" => kind, "unit" => unit, "n_empty" => length(empty),
           "tp_ns" => kind == "radioroc" ? (isfinite(tp_override) ? tp_override : s.tp_ns[1]) :
                      kind == "citiroc" ? s.tp_ns : NaN)
    write_hist!(gr, "spectrum_norm", histogram(allnorm, 0.0, 4.0, 160)...)
    write_hist!(gr, "spectrum_raw", histogram(allraw, 0.0, 4nominal, 160)...)
    write_vector!(gr, "gain_fitted", fitted); write_vector!(gr, "gain_true", true_gain)
    write_vector!(gr, "sigma_1pe", sig1); write_vector!(gr, "entries", [length(v) for v in vals])
end
@printf("%-40s %-9s %d empty decays: %d entries per channel (median); gain fitted in %d of %d channels, fitted/true %.3f ± %.3f (median, MAD), %d outliers; one-photoelectron peak width %.2f of the gain\n",
        case, kind, length(empty), median(length.(vals)), count(ok), nch, rmed, rmad, nout,
        count(ok) > 0 ? median(sig1[ok] ./ fitted[ok]) : NaN)
println("written ", update_table(campaign, "calibration", "calibration",
    ["tag" => [case], "frontend" => [kind], "unit" => [unit], "n_empty" => [length(empty)],
     "entries_per_channel" => [median(length.(vals))], "channels_fitted" => [count(ok)],
     "gain_nominal" => [nominal], "fitted_over_true_median" => [rmed], "fitted_over_true_mad" => [rmad], "outlier_channels" => [nout],
     "sigma1_over_gain" => [count(ok) > 0 ? median(sig1[ok] ./ fitted[ok]) : NaN]]))
