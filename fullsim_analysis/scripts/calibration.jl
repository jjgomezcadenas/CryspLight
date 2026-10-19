# Single-photoelectron calibration of a front end from the dark counts of the SiPMs, as
# the bench measures it, on decays with no energy deposit; each readout is calibrated
# with its own data only.
#   Citiroc and Radioroc2  the high gain at a short peaking time (where the dark counts do
#                          not pile up) is held at its maximum after every crossing of
#                          the discriminator of the channel at half a photoelectron (the
#                          fast shaper of the Citiroc, track and hold; T1 of the
#                          Radioroc2, peak detector), the hold delay being the one a scan
#                          on the bench would choose (crossing to maximum of a single
#                          avalanche). Per channel, the photoelectron peaks of the
#                          spectrum (minima of its smoothed second derivative, numbered
#                          from the random-trigger pedestal) give the gain (slope of the
#                          line through them) and the zero level of the high gain (its
#                          intercept). The zero level
#                          of the low gain (the energy path, production settings) is the
#                          mean low-gain value of the channels whose discriminator did not
#                          fire, on self-triggered dark-count events. The low-gain gain is
#                          the high-gain one times the ratio of the two paths, assumed
#                          known from charge injection;
#   digitiser              the sampled waveform of every channel with its noise, filtered
#                          with the sampled pulse of one cell (matched filter) after
#                          subtracting the mean of the 4 samples before, and the local
#                          maxima of the filtered waveform; the one-photoelectron peak is
#                          the gain, the zero level is zero by construction.
# The fitted gains are compared with those of the simulation.
# Writes results/<campaign>/calibration/calibration.h5 (one group per case: spectrum
# calibrated with the fitted gain and zero, the same with the simulation's values, raw
# spectrum of channel 1, fitted and true gains, zero levels) and calibration.csv (one row
# per case). tp_ns replaces the high-gain peaking time of the configuration (the case is
# then named <tag>_tp<tp>).
#   julia --project=. --threads=8 fullsim_analysis/scripts/calibration.jl <campaign> <frontend config> [n_empty] [tp_ns]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, HDF5, Statistics, Printf
using Random: Xoshiro, seed!

campaign, cfg = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5000
tp_override = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : NaN
r = open_pes_run(cfg)
st, pes, op = r.setup, r.pes, r.op
kind, s = st.frontend, st.settings
case = isfinite(tp_override) ? "$(st.tag)_tp$(replace(string(tp_override), ".0" => ""))" : st.tag
empty = event_lists(pes; nmax = nmax).empty
nch = length(op.gain_factor)
vals = [Float64[] for _ in 1:nch]          # per channel, ADC counts (with the pedestal for the ASICs)
zero_true = zeros(nch); lg_zero = fill(NaN, nch); hold_delay = NaN; tp_cal = NaN

if kind == "citiroc" || kind == "radioroc"
    # calibration settings: high gain at the calibration peaking time, low gain as produced
    cs = if kind == "citiroc"
        tp_cal = isfinite(tp_override) ? tp_override : s.tp_ns
        with(s; tp_ns = tp_cal, tp_lg_ns = CryspLight.tp_lg(s), threshold_pe = 0.5)
    else
        tp_cal = isfinite(tp_override) ? tp_override : s.tp_ns[1]
        with(s; tp_ns = fill(tp_cal, nch), tp_lg_ns = isempty(s.tp_lg_ns) ? s.tp_ns : s.tp_lg_ns,
             t1_threshold_pe = 0.5, peak_detector = true)
    end
    make = chain_maker(kind, cs, op)
    nt = Threads.maxthreadid()
    chains = [make(t) for t in 1:nt]
    hold_delay = hold_at_peak(chains[1])
    local_vals = [[Float64[] for _ in 1:nch] for _ in 1:nt]
    Threads.@threads :static for i in eachindex(empty)
        k = empty[i]
        ch = chains[Threads.threadid()]; lv = local_vals[Threads.threadid()]
        seed!(ch.rng, hash((5, k, :noise)))
        trains = event_trains(op, event_pes(pes, k), nch, Xoshiro(hash((5, k))))
        v = hg_at_crossings(ch, trains, hold_delay)
        for c in 1:nch
            append!(lv[c], v[c])
        end
    end
    # the ADC values are integers: a uniform spread over each count lets the peak fits
    # treat them as a continuous variable without binning artefacts
    rng = Xoshiro(17)
    for t in 1:nt, c in 1:nch
        append!(vals[c], local_vals[t][c] .+ rand(rng, length(local_vals[t][c])) .- 0.5)
    end
    # zero level of the low gain: self-triggered dark-count events with the production
    # settings, the channels whose discriminator did not fire
    ps = kind == "citiroc" ? with(s; trigger = "majority", majority = 1, threshold_pe = 0.5) :
                             with(s; trigger = "t1", t1_threshold_pe = 0.5)
    res = emulate(pes, op, kind, ps, empty[1:min(2000, length(empty))])
    for c in 1:nch
        sel = res.triggered .& .!res.hits[c, :]
        lg_zero[c] = mean(Float64.(res.lg[c, sel]))
    end
    true_gain = s.adc_per_pe .* op.gain_factor
    zero_true .= s.pedestal
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

# per channel: the gain and zero level from the photoelectron peaks (the ASICs), or the
# one-photoelectron peak (the digitiser, whose zero is zero)
nominal = median(true_gain)
fitted = fill(NaN, nch); zero_fit = fill(NaN, nch); npeaks = zeros(Int, nch)
for c in 1:nch
    length(vals[c]) < 300 && continue
    if kind == "digitizer"
        p1 = peak_fit(vals[c], 0.6nominal, 1.25nominal; nbins = 60)
        fitted[c] = p1.mean; zero_fit[c] = 0.0; npeaks[c] = isfinite(p1.mean)
    else
        lad = photoelectron_ladder(vals[c], nominal; pedestal = s.pedestal)
        fitted[c] = lad.gain; zero_fit[c] = lad.zero; npeaks[c] = length(lad.peaks)
    end
end
ok = isfinite.(fitted)
ratio = fitted[ok] ./ true_gain[ok]
# robust summary: median, scaled median absolute deviation, and the channels beyond five
# of those (fits that caught another peak)
rmed = isempty(ratio) ? NaN : median(ratio)
rmad = isempty(ratio) ? NaN : 1.4826 * median(abs.(ratio .- rmed))
nout = count(abs.(ratio .- rmed) .> 5rmad)
dzero = count(ok) > 0 ? median((zero_fit[ok] .- zero_true[ok]) ./ true_gain[ok]) : NaN
gcal = [ok[c] ? fitted[c] : any(ok) ? median(fitted[ok]) : true_gain[c] for c in 1:nch]
zcal = [ok[c] ? zero_fit[c] : any(ok) ? median(zero_fit[ok]) : zero_true[c] for c in 1:nch]
allcal = reduce(vcat, [(vals[c] .- zcal[c]) ./ gcal[c] for c in 1:nch])
allnorm = reduce(vcat, [(vals[c] .- zero_true[c]) ./ true_gain[c] for c in 1:nch])

open_results(campaign, "calibration"; mode = isfile(joinpath(results_dir(campaign, "calibration"), "calibration.h5")) ? "r+" : "w") do h5
    haskey(h5, case) && delete_object(h5, case)
    gr = create_group(h5, case)
    stamp!(gr, "config" => st.cfg_text, "frontend" => kind, "unit" => unit, "n_empty" => length(empty),
           "tp_ns" => tp_cal, "hold_delay_ns" => hold_delay)
    write_hist!(gr, "spectrum_cal", histogram(allcal, -0.5, 4.5, 250)...)
    write_hist!(gr, "spectrum_norm", histogram(allnorm, -0.5, 4.5, 250)...)
    lo1 = kind == "digitizer" ? 0.0 : s.pedestal - nominal
    write_hist!(gr, "spectrum_ch1", histogram(vals[1], lo1, lo1 + 5nominal, 250)...)
    write_vector!(gr, "gain_fitted", fitted); write_vector!(gr, "gain_true", true_gain)
    write_vector!(gr, "zero_fitted", zero_fit); write_vector!(gr, "zero_true", zero_true)
    write_vector!(gr, "lg_pedestal", lg_zero); write_vector!(gr, "peaks_found", Float64.(npeaks))
    write_vector!(gr, "entries", [length(v) for v in vals])
end
@printf("%-40s %-9s %d empty decays, hold %.1f ns: %d entries per channel (median); gain fitted in %d of %d channels (%d with three peaks), fitted/true %.3f ± %.3f (median, MAD), %d outliers; zero level %+.3f pe from the true one\n",
        case, kind, length(empty), hold_delay, median(length.(vals)), count(ok), nch, count(npeaks .>= 3),
        rmed, rmad, nout, dzero)
println("written ", update_table(campaign, "calibration", "calibration",
    ["tag" => [case], "frontend" => [kind], "unit" => [unit], "n_empty" => [length(empty)],
     "tp_ns" => [tp_cal], "hold_delay_ns" => [hold_delay],
     "entries_per_channel" => [median(length.(vals))], "channels_fitted" => [count(ok)],
     "channels_three_peaks" => [count(npeaks .>= 3)],
     "gain_nominal" => [nominal], "fitted_over_true_median" => [rmed], "fitted_over_true_mad" => [rmad],
     "outlier_channels" => [nout], "zero_offset_pe" => [dzero]]))
