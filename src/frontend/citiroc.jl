# The Citiroc 1A chain of the full simulation (design document, Section "Electronics",
# "The Citiroc 1A"), applied to the avalanche trains of the 64 channels of one event:
#   0. two AC-coupled voltage preamplifiers (high and low gain) amplify the voltage of the
#      current pulse across the 50 ohm termination, so that the CR stage of the shapers
#      differentiates the pulse itself (voltage_preamp; false = charge-sensitive chain);
#   1. the two slow shapers (CR-RC^2) on the high- and low-gain paths, each with its own
#      peaking time (common to the chip) and shaped white noise, whose rms falls as the
#      inverse square root of the peaking time (series noise of the large SiPM
#      capacitance); the low-gain noise is the same in ADC counts, hence larger by
#      1/lg_ratio in avalanches;
#   2. the fast shaper (15 ns) and the discriminator of each channel, whose crossing
#      times (it re-arms below threshold) feed the trigger: a majority of N channels within a coincidence
#      window (N = 1 is the OR), as the FPGA builds it, or an external trigger at a fixed
#      time (a random-trigger run, for the pedestals);
#   3. the peak detector (the maximum of the slow signal from the trigger to the hold) or
#      the track-and-hold (the value at the hold), per channel;
#   4. the conversion to ADC counts with the pedestal and the counts per avalanche of
#      the channel (the low gain lg_ratio of the high gain), and the saturation of the
#      input charge and of the low-gain ADC (a high gain at full scale is not a saturation:
#      the low gain measures the channel).
# The Radioroc2 uses the same code with its own settings.

using Random: AbstractRNG

const E_CHARGE_PC = 1.602176634e-7      # pC per electron

Base.@kwdef struct CitirocSettings
    tp_ns::Float64 = 87.5               # slow shaper peaking time, high-gain path
    tp_lg_ns::Float64 = 0.0             # low-gain path; 0 = same as tp_ns
    order::Int = 2                      # CR-RC^order
    voltage_preamp::Bool = true         # AC-coupled voltage preamplifiers (datasheet): the
                                        # shapers act on the current pulse, not on its charge
    fast_tp_ns::Float64 = 15.0          # fast shaper peaking time
    hold_ns::Float64 = 300.0            # delay from the trigger to the hold
    peak_detector::Bool = true          # false: track and hold
    threshold_pe::Float64 = 0.5         # discriminator threshold on the fast path, in avalanches
    noise_pe::Float64 = 0.1             # rms of the slow high-gain path at the output, at noise_tp_ns
    noise_tp_ns::Float64 = 87.5         # peaking time at which noise_pe is given
    noise_exponent::Float64 = 0.5       # noise ~ (noise_tp_ns / tp)^exponent (series noise)
    lg_noise_pe::Float64 = 0.0          # rms of the low-gain path in avalanches at noise_tp_ns; 0 = noise_pe / lg_ratio
    fast_noise_pe::Float64 = 0.15       # rms of the fast path at the output
    adc_bits::Int = 12
    adc_per_pe::Float64 = 10.0          # high-gain counts per avalanche, nominal
    pedestal::Float64 = 200.0
    lg_ratio::Float64 = 0.1             # low-gain over high-gain
    ceiling_pc::Float64 = 400.0         # input charge at which the chain saturates
    majority::Int = 1                   # channels within the coincidence window to trigger
    coincidence_ns::Float64 = 50.0
    trigger::String = "majority"        # "majority" of the discriminators, or "external"
    external_trigger_ns::Float64 = 0.0  # trigger time of the "external" mode (a random-trigger run)
    dt_ns::Float64 = 1.0                # grid step of the emulation
    t_start_ns::Float64 = -200.0        # grid start relative to the decay
    t_end_ns::Float64 = 2000.0          # grid end
end

"One event as the Citiroc acquisition records it, plus the emulation flags."
struct CitirocEvent
    hg::Vector{UInt16}
    lg::Vector{UInt16}
    hits::Vector{Bool}         # discriminator fired within the trigger window
    t_trigger::Float32         # ns from the decay; NaN when no trigger
    triggered::Bool
    saturated::Vector{Bool}
    stored_pe::Vector{Float32} # the stored value before the ADC, in avalanches (diagnostic)
end

"""
The emulation of one chip set-up: the two shapers, their noise generators, the grid and
the work buffers. Built once per run; apply with citiroc_event.
"""
struct CitirocChain
    s::CitirocSettings
    op::SipmOperation
    grid::TimeGrid
    slow::Shaper
    slow_lg::Shaper
    fast::Shaper
    slow_noise!::Function
    lg_noise!::Function
    fast_noise!::Function
    x::Vector{Float64}
    y::Vector{Float64}
    yf::Vector{Float64}
    nz::Vector{Float64}
    pe_ceiling::Float64        # avalanches at the input-charge ceiling
    rng::AbstractRNG           # the noise stream; reseed per event for reproducibility
end

tp_lg(s::CitirocSettings) = s.tp_lg_ns > 0 ? s.tp_lg_ns : s.tp_ns
noise_at(s::CitirocSettings, tp::Real) = s.noise_pe * (s.noise_tp_ns / tp)^s.noise_exponent
lg_noise(s::CitirocSettings) = (s.lg_noise_pe > 0 ? s.lg_noise_pe : s.noise_pe / s.lg_ratio) *
                               (s.noise_tp_ns / tp_lg(s))^s.noise_exponent

function CitirocChain(s::CitirocSettings, op::SipmOperation, rng::AbstractRNG)
    # warm-up: four peaking times of the longer slow shaper and ten falls of the SiPM pulse
    grid = chain_grid(op, s.t_start_ns, s.t_end_ns, s.dt_ns,
                      4 * max(s.tp_ns, tp_lg(s)) + 10 * op.dev.pulse_fall_ns)
    n = grid.n
    vp = s.voltage_preamp
    slow = Shaper(op.dev, s.tp_ns, s.dt_ns; order = s.order, voltage = vp)
    slow_lg = Shaper(op.dev, tp_lg(s), s.dt_ns; order = s.order, voltage = vp)
    fast = Shaper(op.dev, s.fast_tp_ns, s.dt_ns; order = s.order, voltage = vp)
    sn = noise_generator(slow, noise_at(s, s.tp_ns), n, rng)
    ln = noise_generator(slow_lg, lg_noise(s), n, rng)
    fn = noise_generator(fast, s.fast_noise_pe, n, rng)
    ceiling = s.ceiling_pc / (E_CHARGE_PC * op.gain)
    return CitirocChain(s, op, grid, slow, slow_lg, fast, sn, ln, fn, zeros(n), zeros(n), zeros(n),
                        zeros(n), ceiling, rng)
end

"First sample at which y crosses thr upwards, or 0."
function first_crossing(y::Vector{Float64}, thr::Float64)
    @inbounds for k in eachindex(y)
        y[k] >= thr && return k
    end
    return 0
end

"""
Times of all the upward crossings of thr by y on the grid g: the discriminator fires when
y reaches thr and re-arms when y falls below it again. Crossings before tmin (the warm-up) are ignored.
"""
function crossing_times(y::Vector{Float64}, thr::Float64, g::TimeGrid; tmin::Real = -Inf)
    ts = Float64[]
    above = false
    @inbounds for k in eachindex(y)
        if y[k] >= thr
            above || (t = sample_time(g, k); t >= tmin && push!(ts, t))
            above = true
        else
            above = false
        end
    end
    return ts
end

"""
Trigger time from the crossing times of the channels (one vector of times per channel):
the earliest crossing at which `majority` distinct channels have crossed within the
preceding `coincidence`; NaN when none.
"""
function trigger_time(crossings::Vector{Vector{Float64}}, majority::Int, coincidence::Float64)
    ev = sort!([(t, c) for c in eachindex(crossings) for t in crossings[c] if isfinite(t)])
    isempty(ev) && return NaN
    majority <= 1 && return ev[1][1]
    i0 = 1
    for j in eachindex(ev)
        tj = ev[j][1]
        while ev[i0][1] < tj - coincidence
            i0 += 1
        end
        length(unique(last.(view(ev, i0:j)))) >= majority && return tj
    end
    return NaN
end

"The same with one crossing time per channel (NaN when the channel did not cross)."
trigger_time(crossings::Vector{Float64}, majority::Int, coincidence::Float64) =
    trigger_time([isfinite(t) ? [t] : Float64[] for t in crossings], majority, coincidence)

"Whether any of the crossing times ts lies within coincidence of the trigger time."
in_trigger_window(ts::Vector{Float64}, ttrig::Real, coincidence::Real) =
    any(t -> ttrig - coincidence <= t <= ttrig + coincidence, ts)

"""
    citiroc_waves(ch, trains) -> (crossings, hg_wave, lg_wave)

The discriminator crossing times of each channel and its two slow signals (high and low
gain, in avalanches, with their noise) on the grid of the chain.
"""
function citiroc_waves(ch::CitirocChain, trains::Vector{Vector{Avalanche}})
    s, g = ch.s, ch.grid
    nch = length(trains)
    crossings = [Float64[] for _ in 1:nch]
    slow_wave = [zeros(Float64, g.n) for _ in 1:nch]
    lg_wave = [zeros(Float64, g.n) for _ in 1:nch]
    for c in 1:nch
        bin_train!(ch.x, g, trains[c])
        ch.x .*= ch.op.gain_factor[c]
        remove_dark_level!(ch.x, ch.op, c, g.dt)
        # fast path and discriminator
        copyto!(ch.yf, ch.x)
        shape!(ch.yf, ch.fast)
        ch.fast_noise!(ch.nz)
        ch.yf .+= ch.nz
        crossings[c] = crossing_times(ch.yf, s.threshold_pe, g; tmin = s.t_start_ns)
        # slow path
        y = slow_wave[c]
        copyto!(y, ch.x)
        shape!(y, ch.slow)
        ch.slow_noise!(ch.nz)
        y .+= ch.nz
        yl = lg_wave[c]
        copyto!(yl, ch.x)
        shape!(yl, ch.slow_lg)
        ch.lg_noise!(ch.nz)
        yl .+= ch.nz
    end
    return crossings, slow_wave, lg_wave
end

"""
    citiroc_event(ch, trains) -> CitirocEvent

Apply the chain to the avalanche trains of the channels of one event (a vector of
trains, one per channel, in avalanche units; the per-channel gain factors of the
operating point scale the charge).
"""
function citiroc_event(ch::CitirocChain, trains::Vector{Vector{Avalanche}})
    s, g = ch.s, ch.grid
    nch = length(trains)
    crossings, slow_wave, lg_wave = citiroc_waves(ch, trains)
    ttrig = s.trigger == "external" ? s.external_trigger_ns :
            s.trigger == "majority" ? trigger_time(crossings, s.majority, s.coincidence_ns) :
            error("trigger $(s.trigger) not supported (majority, external)")
    hg = zeros(UInt16, nch); lg = zeros(UInt16, nch)
    hits = falses(nch); sat = falses(nch); stored = zeros(Float32, nch)
    fullscale = 2^s.adc_bits - 1
    if isfinite(ttrig)
        k0 = clamp(sample_index(g, ttrig), 1, g.n)
        k1 = clamp(sample_index(g, ttrig + s.hold_ns), 1, g.n)
        for c in 1:nch
            y, yl = slow_wave[c], lg_wave[c]
            v = s.peak_detector ? maximum(view(y, k0:k1)) : y[k1]
            vl = s.peak_detector ? maximum(view(yl, k0:k1)) : yl[k1]
            # input saturation: the stored amplitude, in units of the peak of one
            # avalanche, is that of a prompt charge of v avalanches; beyond the input
            # ceiling it is clipped
            v = min(v, ch.pe_ceiling)
            if vl > ch.pe_ceiling
                sat[c] = true
                vl = ch.pe_ceiling
            end
            hg_counts = s.pedestal + s.adc_per_pe * v
            lg_counts = s.pedestal + s.adc_per_pe * s.lg_ratio * vl
            # dual gain: the high gain below its full scale, the low gain above it
            stored[c] = hg_counts < fullscale ? v : vl
            lg_counts >= fullscale && (sat[c] = true)
            hg[c] = UInt16(clamp(round(Int, hg_counts), 0, fullscale))
            lg[c] = UInt16(clamp(round(Int, lg_counts), 0, fullscale))
            hits[c] = in_trigger_window(crossings[c], ttrig, s.coincidence_ns)
        end
    end
    return CitirocEvent(hg, lg, hits, Float32(ttrig), isfinite(ttrig), sat, stored)
end

"""
    held_at_crossings(y, ts, g, delay, peak_detector) -> values

The value of the signal y (on grid g) held after each crossing time in ts: at the
crossing plus delay (track and hold), or its maximum from the crossing to then (peak
detector). Crossings whose hold falls beyond the grid are skipped, and so are those
before the hold of the previous one (the chip is busy holding): a discriminator that
crosses several times on the noise of one rising edge gives one value.
"""
function held_at_crossings(y::Vector{Float64}, ts::Vector{Float64}, g::TimeGrid, delay::Real,
                           peak_detector::Bool)
    out = Float64[]
    busy = -Inf
    for t in ts
        t <= busy && continue
        k0 = sample_index(g, t); k1 = sample_index(g, t + delay)
        (k0 >= 1 && k1 <= g.n) || continue
        push!(out, peak_detector ? maximum(view(y, k0:k1)) : y[k1])
        busy = t + delay
    end
    return out
end

"""
    hg_at_crossings(ch, trains, delay) -> per-channel vectors of high-gain ADC counts

The self-triggered calibration of the dark counts: every crossing of the discriminator of
a channel (threshold of the settings) holds that channel's high gain delay later (track
and hold), converted to ADC counts with the pedestal.
"""
function hg_at_crossings(ch::CitirocChain, trains::Vector{Vector{Avalanche}}, delay::Real)
    s = ch.s
    crossings, hgw, _ = citiroc_waves(ch, trains)
    full = 2^s.adc_bits - 1
    return [clamp.(round.(s.pedestal .+ s.adc_per_pe .* held_at_crossings(hgw[c], crossings[c], ch.grid, delay, false)), 0, full)
            for c in eachindex(trains)]
end

"""
Delay from the discriminator crossing of a single avalanche to the maximum of its
high-gain slow signal, without noise: the hold delay that a scan of the delay on the
bench would choose.
"""
function hold_at_peak(ch::CitirocChain)
    g = ch.grid
    fill!(ch.x, 0.0); ch.x[sample_index(g, 0.0)] = 1.0
    copyto!(ch.yf, ch.x); shape!(ch.yf, ch.fast)
    copyto!(ch.y, ch.x); shape!(ch.y, ch.slow)
    return sample_time(g, argmax(ch.y)) - sample_time(g, first_crossing(ch.yf, ch.s.threshold_pe))
end

"""
Fraction of the avalanches of an exponential train (decay tau, optional second component)
that the stored value keeps: the peak of the shaped signal of the train divided by that
of one avalanche, computed on the chain's grid without noise.
"""
function stored_fraction(ch::CitirocChain, tau_ns::Real; tau2_ns::Real = 0, frac1::Real = 1,
                         n::Int = 200_000)
    g = ch.grid
    fill!(ch.x, 0.0)
    cdf(t) = t <= 0 ? 0.0 :
             frac1 * (1 - exp(-t / tau_ns)) +
             (tau2_ns > 0 ? (1 - frac1) * (1 - exp(-t / tau2_ns)) : 0.0)
    for k in 1:g.n
        t = sample_time(g, k)
        ch.x[k] = cdf(t + g.dt) - cdf(t)          # charge emitted in the bin
    end
    copyto!(ch.y, ch.x)
    shape!(ch.y, ch.slow)
    return maximum(ch.y)
end

"The level removed with the mean dark current on the high-gain slow path (dark_baseline), in avalanches."
dark_baseline(ch::CitirocChain) = dark_baseline(ch.slow, ch.op)
