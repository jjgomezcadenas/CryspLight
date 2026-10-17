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

Base.@kwdef struct DigitizerSettings
    gain::Float64 = 1.0                    # digitizer gain setting (x1 ... x100)
    counts_per_pe_per_volt::Float64 = 3.2  # peak counts of one avalanche per volt of overvoltage, gain 1
    area_over_peak::Float64 = 4.4          # area of the avalanche pulse in samples per count of peak
    noise_counts::Float64 = 4.0            # rms per sample, white
    sample_ns::Float64 = 8.0
    threshold_counts::Float64 = 60.0       # per-channel leading edge, above the baseline
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

"One event as the analysis of the digitizer records would produce it."
struct DigitizerEvent
    charge::Vector{Int32}       # count-samples, baseline subtracted
    charge_pe::Vector{Float32}  # the same in avalanches, with the calibration of the run
    hits::Vector{Bool}          # channels that crossed the threshold within the trigger window
    t_trigger::Float32          # ns from the decay; NaN when no trigger
    triggered::Bool
    saturated::Vector{Bool}     # peak amplitude beyond the input range
    peak_counts::Vector{Float32}  # largest sampled amplitude of the channel (diagnostic)
end

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
    n = ceil(Int, (s.t_end_ns - s.t_start_ns) / s.dt_ns)
    grid = TimeGrid(s.t_start_ns, s.dt_ns, n)
    pulse = Shaper(op.dev, 1.0, s.dt_ns)       # only its device-pulse calibration is used
    pk = s.counts_per_pe_per_volt * op.vov * s.gain
    ng = round(Int, s.gate_ns / s.sample_ns)
    nb = round(Int, s.baseline_ns / s.sample_ns)
    nq = s.noise_counts * sqrt(ng + ng^2 / nb)
    return DigitizerChain(s, op, grid, pulse, zeros(n), zeros(n), pk, pk * s.area_over_peak,
                          nq, ng, nb, rng)
end

"""
Fraction of the area of the avalanche pulse emitted before time t after the avalanche
(the pulse is exp(-t/tf) - exp(-t/t2), t2 the combined rise constant).
"""
function pulse_cdf(dev::SipmDevice, t::Real)
    t <= 0 && return 0.0
    tf = dev.pulse_fall_ns
    t2 = 1 / (1 / tf + 1 / dev.pulse_tr_ns)
    return (tf * (1 - exp(-t / tf)) - t2 * (1 - exp(-t / t2))) / (tf - t2)
end

"Fraction of the pulse of an avalanche at time ta that falls in the window [w0, w1]."
@inline window_fraction(dev::SipmDevice, ta::Real, w0::Real, w1::Real) =
    pulse_cdf(dev, w1 - ta) - pulse_cdf(dev, w0 - ta)

"""
    digitizer_event(ch, trains) -> DigitizerEvent

Apply the digitizer to the avalanche trains of the channels of one event.
"""
function digitizer_event(ch::DigitizerChain, trains::Vector{Vector{Avalanche}})
    s, g, dev = ch.s, ch.grid, ch.op.dev
    nch = length(trains)
    crossings = fill(NaN, nch)
    peaks = zeros(Float32, nch)
    step = max(1, round(Int, s.sample_ns / g.dt))
    for c in 1:nch
        amp = ch.peak_counts_per_pe * ch.op.gain_factor[c]
        bin_train!(ch.x, g, trains[c])
        any(!=(0.0), ch.x) || continue
        device_pulse!(ch.x, ch.pulse)
        pk = 0.0
        @inbounds for k in 1:step:g.n
            v = amp * ch.x[k] + s.noise_counts * randn(ch.rng)
            v > pk && (pk = v)
            if isnan(crossings[c]) && v >= s.threshold_counts
                crossings[c] = sample_time(g, k)
            end
        end
        peaks[c] = Float32(pk)
    end
    ttrig = trigger_time(crossings, s.majority, s.coincidence_ns)
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
            hits[c] = isfinite(crossings[c]) &&
                      ttrig - s.coincidence_ns <= crossings[c] <= ttrig + s.coincidence_ns
        end
    end
    return DigitizerEvent(charge, qpe, hits, Float32(ttrig), isfinite(ttrig), sat, peaks)
end
