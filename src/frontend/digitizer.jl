# The DT2745 digitizer of the full simulation (design document, Section "Electronics",
# "The DT2745 digitizer"), emulated on the avalanche train without building waveforms:
#   - the charge that the analysis extracts from a record is a weighted sum of samples
#     (one in the gate, minus the length ratio in the baseline window), and applied to
#     the pulse of one avalanche that weight depends only on where the avalanche falls;
#     the charge of the event is the sum over the avalanches of their area times that
#     weight, plus one Gaussian for the integrated white noise;
#   - the trigger is the one non-linear step: the sampled amplitude of each channel on
#     the 8 ns grid, with its noise, against a threshold in counts, then a majority of N
#     channels within a coincidence window; the trigger time places the gate and the
#     baseline window.
# Charges are in count-samples (as in the analysis of the measured records) with the
# single-avalanche area of the run as calibration.

"""
    DigitizerSettings(; gain = 1.0, threshold_counts = 60.0, gate_ns = 1000.0, ...)

The settings of the DT2745 digitizer emulation, a keyword struct (settings_from builds it
from the [frontend] table of a configuration). Amplitudes are in ADC counts: the peak of
one avalanche is counts_per_pe_per_volt x overvoltage x gain, the white noise has
noise_counts rms per sample, and each channel's leading-edge discriminator fires at
threshold_counts above the baseline (0 or less selects the external trigger at
external_trigger_ns). Times are in ns: the sampling period sample_ns, the coincidence
window of the majority trigger (majority channels), the gate (from gate_start_ns
relative to the trigger, gate_ns long) and the baseline window (baseline_ns, ending at
the gate start), and the synthesis grid dt_ns from t_start_ns to t_end_ns relative to
the decay. A channel whose sampled peak reaches 2^(bits - 1) counts is flagged
saturated. pretrigger_ns and fullscale_v (V) describe the acquisition and are written to
the event file, but the emulation does not use them.
"""
Base.@kwdef struct DigitizerSettings
    gain::Float64 = 1.0                    # digitizer gain setting (x1 ... x100)
    counts_per_pe_per_volt::Float64 = 3.2  # peak counts of one avalanche per volt of overvoltage, gain 1
    noise_counts::Float64 = 4.0            # rms per sample, white
    sample_ns::Float64 = 8.0
    threshold_counts::Float64 = 60.0       # per-channel leading edge, above the baseline;
                                           # 0 or less: external trigger at external_trigger_ns
    external_trigger_ns::Float64 = 0.0
    majority::Int = 2
    coincidence_ns::Float64 = 100.0
    pretrigger_ns::Float64 = 1000.0        # samples kept before the trigger
    gate_start_ns::Float64 = -50.0         # gate start relative to the trigger (before it,
                                           # so that the rise of the first pulse is inside)
    gate_ns::Float64 = 1000.0
    baseline_ns::Float64 = 1000.0          # baseline window, ending at the gate start
    fullscale_v::Float64 = 4.0
    bits::Int = 16
    dt_ns::Float64 = 2.0                   # synthesis grid of the sampled amplitude
    t_start_ns::Float64 = -2000.0
    t_end_ns::Float64 = 12_000.0
end

"""
    DigitizerEvent

One event as the analysis of the digitizer records would produce it, one entry per
channel: charge, the gate integral minus the scaled baseline integral in count-samples
(Int32), charge_pe the same in avalanches with the calibration of the run, hits the
channels whose discriminator fired within the coincidence window around the trigger,
saturated the channels whose sampled peak reached half the ADC range, and peak_counts
the largest sampled amplitude in ADC counts above the baseline (computed even without
trigger). t_trigger is the trigger time in ns from the decay (NaN without trigger) and
triggered whether there was one; without trigger the charges are zero.
"""
struct DigitizerEvent
    charge::Vector{Int32}       # count-samples, baseline subtracted
    charge_pe::Vector{Float32}  # the same in avalanches, with the calibration of the run
    hits::Vector{Bool}          # channels that crossed the threshold within the trigger window
    t_trigger::Float32          # ns from the decay; NaN when no trigger
    triggered::Bool
    saturated::Vector{Bool}     # peak amplitude beyond the input range
    peak_counts::Vector{Float32}  # largest sampled amplitude of the channel (diagnostic)
end

"""
    DigitizerChain(s::DigitizerSettings, op::SipmOperation, rng::AbstractRNG)

The emulation of one digitizer set-up: the grid with its warm-up, the device pulse, the
peak amplitude (ADC counts) and area (count-samples) of one avalanche at the operating
point op, the rms of the integrated noise of the charge (count-samples), the gate and
baseline lengths in samples, the work buffers and the noise stream rng. Built once per
run (one per thread); apply with digitizer_event, reseeding rng per event for
reproducibility.
"""
struct DigitizerChain
    s::DigitizerSettings
    op::SipmOperation
    grid::TimeGrid
    pulse::Shaper               # device pulse on the grid (no shaper)
    x::Vector{Float64}
    y::Vector{Float64}
    peak_counts_per_pe::Float64 # amplitude of one avalanche at gain_factor 1
    area_per_pe::Float64        # count-samples of one avalanche at gain_factor 1
    noise_charge::Float64       # rms of the integrated noise in count-samples
    n_gate::Int
    n_base::Int
    rng::AbstractRNG
end

function DigitizerChain(s::DigitizerSettings, op::SipmOperation, rng::AbstractRNG)
    # warm-up: ten falls of the SiPM pulse, in whole samples so that the sampling times
    # keep their phase with respect to the start of the record
    warm = s.sample_ns * ceil(10 * op.dev.pulse_fall_ns / s.sample_ns)
    grid = chain_grid(op, s.t_start_ns, s.t_end_ns, s.dt_ns, warm)
    n = grid.n
    pulse = Shaper(op.dev, 1.0, s.dt_ns)       # only its device-pulse calibration is used
    pk = s.counts_per_pe_per_volt * op.vov * s.gain
    # area of the avalanche pulse in samples per count of peak, from the pulse itself
    # (peak one on the grid), so that the charge and the waveform describe the same pulse
    aop = pulse_area_over_peak(pulse, s.sample_ns)
    ng = round(Int, s.gate_ns / s.sample_ns)
    nb = round(Int, s.baseline_ns / s.sample_ns)
    nq = s.noise_counts * sqrt(ng + ng^2 / nb)
    return DigitizerChain(s, op, grid, pulse, zeros(n), zeros(n), pk, pk * aop,
                          nq, ng, nb, rng)
end

"Area of the device pulse of one avalanche, in samples of sample_ns, per unit of its peak."
function pulse_area_over_peak(pulse::Shaper, sample_ns::Real)
    n = ceil(Int, 20 * pulse.dev.pulse_fall_ns / pulse.dt)
    y = zeros(Float64, n); y[1] = 1.0
    device_pulse!(y, pulse)
    return sum(y) * pulse.dt / maximum(y) / sample_ns
end

"""
    pulse_cdf(dev::SipmDevice, t) -> Float64

Fraction of the area of the avalanche pulse emitted before the time t [ns] after the
avalanche (the pulse is exp(-t/tf) - exp(-t/t2), t2 the combined rise constant): 0 for
t <= 0, rising to 1.
"""
function pulse_cdf(dev::SipmDevice, t::Real)
    t <= 0 && return 0.0
    tf = dev.pulse_fall_ns
    t2 = 1 / (1 / tf + 1 / dev.pulse_tr_ns)
    return (tf * (1 - exp(-t / tf)) - t2 * (1 - exp(-t / t2))) / (tf - t2)
end

"""
    window_fraction(dev::SipmDevice, ta, w0, w1) -> Float64

Fraction of the area of the pulse of an avalanche at time ta that falls in the window
[w0, w1] (all times in ns).
"""
@inline window_fraction(dev::SipmDevice, ta::Real, w0::Real, w1::Real) =
    pulse_cdf(dev, w1 - ta) - pulse_cdf(dev, w0 - ta)

"""
    digitizer_event(ch, trains) -> DigitizerEvent

Apply the digitizer to the avalanche trains of the channels of one event (one train per
channel, in avalanche units, scaled by the gain factors of the operating point). The
sampled amplitude of each channel, with its noise, feeds the leading-edge discriminators
and the majority (or external) trigger; the charge of each channel is the sum over its
avalanches of their area times the weight of the gate minus the scaled baseline window,
plus one Gaussian for the integrated noise, in count-samples.
"""
function digitizer_event(ch::DigitizerChain, trains::Vector{Vector{Avalanche}})
    s, g, dev = ch.s, ch.grid, ch.op.dev
    nch = length(trains)
    crossings = [Float64[] for _ in 1:nch]
    peaks = zeros(Float32, nch)
    step = max(1, round(Int, s.sample_ns / g.dt))
    for c in 1:nch
        amp = ch.peak_counts_per_pe * ch.op.gain_factor[c]
        bin_train!(ch.x, g, trains[c])
        any(!=(0.0), ch.x) || continue
        # the discriminator works on the signal above the baseline, which the steady
        # current of the dark counts raises (amp carries the gain factor of the channel)
        ch.x .-= dark_charge_per_ns(ch.op) * g.dt
        device_pulse!(ch.x, ch.pulse)
        pk = 0.0
        above = false
        @inbounds for k in 1:step:g.n
            v = amp * ch.x[k] + s.noise_counts * randn(ch.rng)
            v > pk && (pk = v)
            # leading-edge discriminator, re-armed when the sample falls below threshold
            if v >= s.threshold_counts
                above || (t = sample_time(g, k); t >= s.t_start_ns && push!(crossings[c], t))
                above = true
            else
                above = false
            end
        end
        peaks[c] = Float32(pk)
    end
    ttrig = s.threshold_counts <= 0 ? s.external_trigger_ns :
            trigger_time(crossings, s.majority, s.coincidence_ns)
    charge = zeros(Int32, nch); qpe = zeros(Float32, nch)
    hits = falses(nch); sat = falses(nch)
    halfscale = 2.0^(s.bits - 1)
    if isfinite(ttrig)
        g0 = ttrig + s.gate_start_ns
        g1 = g0 + s.gate_ns
        b0 = g0 - s.baseline_ns
        ratio = ch.n_gate / ch.n_base
        for c in 1:nch
            area = ch.area_per_pe * ch.op.gain_factor[c]
            q = 0.0
            for a in trains[c]
                ta = Float64(a.t_ns)
                ta > g1 && continue
                w = window_fraction(dev, ta, g0, g1) - ratio * window_fraction(dev, ta, b0, g0)
                q += a.q * area * w
            end
            q += ch.noise_charge * randn(ch.rng)
            charge[c] = Int32(round(q))
            qpe[c] = Float32(q / area)
            sat[c] = peaks[c] >= halfscale
            hits[c] = in_trigger_window(crossings[c], ttrig, s.coincidence_ns)
        end
    end
    return DigitizerEvent(charge, qpe, hits, Float32(ttrig), isfinite(ttrig), sat, peaks)
end
