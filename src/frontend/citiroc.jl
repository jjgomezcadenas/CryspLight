# The Citiroc 1A chain of the full simulation (design document, Section "Electronics",
# "The Citiroc 1A"), applied to the avalanche trains of the 64 channels of one event:
#   1. the slow shaper (CR-RC^2, peaking time common to the chip) on the high-gain path,
#      with shaped white noise of the rms measured on the pedestals;
#   2. the fast shaper (15 ns) and the discriminator of each channel, whose first
#      crossing time feeds the trigger: a majority of N channels within a coincidence
#      window (N = 1 is the OR), as the FPGA builds it, or an external trigger at a fixed
#      time (a random-trigger run, for the pedestals);
#   3. the peak detector (the maximum of the slow signal from the trigger to the hold) or
#      the track-and-hold (the value at the hold), per channel;
#   4. the conversion to ADC counts with the pedestal and the counts per avalanche of
#      the channel, the saturation of the input charge and of the ADC, and the low-gain
#      path as a fixed fraction of the high-gain one.
# The Radioroc2 uses the same code with its own settings.

using Random: AbstractRNG

const E_CHARGE_PC = 1.602176634e-7      # pC per electron

Base.@kwdef struct CitirocSettings
    tp_ns::Float64 = 87.5               # slow shaper peaking time
    order::Int = 2                      # CR-RC^order
    fast_tp_ns::Float64 = 15.0          # fast shaper peaking time
    hold_ns::Float64 = 300.0            # delay from the trigger to the hold
    peak_detector::Bool = true          # false: track and hold
    threshold_pe::Float64 = 0.5         # discriminator threshold on the fast path, in avalanches
    noise_pe::Float64 = 0.1             # rms of the slow high-gain path at the output
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
    fast::Shaper
    slow_noise!::Function
    fast_noise!::Function
    x::Vector{Float64}
    y::Vector{Float64}
    yf::Vector{Float64}
    nz::Vector{Float64}
    pe_ceiling::Float64        # avalanches at the input-charge ceiling
    rng::AbstractRNG           # the noise stream; reseed per event for reproducibility
end

function CitirocChain(s::CitirocSettings, op::SipmOperation, rng::AbstractRNG)
    n = ceil(Int, (s.t_end_ns - s.t_start_ns) / s.dt_ns)
    grid = TimeGrid(s.t_start_ns, s.dt_ns, n)
    slow = Shaper(op.dev, s.tp_ns, s.dt_ns; order = s.order)
    fast = Shaper(op.dev, s.fast_tp_ns, s.dt_ns; order = s.order)
    sn = noise_generator(slow, s.noise_pe, n, rng)
    fn = noise_generator(fast, s.fast_noise_pe, n, rng)
    ceiling = s.ceiling_pc / (E_CHARGE_PC * op.gain)
    return CitirocChain(s, op, grid, slow, fast, sn, fn, zeros(n), zeros(n), zeros(n),
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
Trigger time from the crossing times of the channels: the earliest time at which
`majority` crossings fall within `coincidence` of each other; NaN when none.
"""
function trigger_time(crossings::Vector{Float64}, majority::Int, coincidence::Float64)
    ts = sort(filter(isfinite, crossings))
    length(ts) < majority && return NaN
    majority == 1 && return ts[1]
    for i in 1:length(ts) - majority + 1
        ts[i + majority - 1] - ts[i] <= coincidence && return ts[i + majority - 1]
    end
    return NaN
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
    crossings = fill(NaN, nch)
    slow_wave = [zeros(Float64, g.n) for _ in 1:nch]
    for c in 1:nch
        bin_train!(ch.x, g, trains[c])
        ch.x .*= ch.op.gain_factor[c]
        # fast path and discriminator
        copyto!(ch.yf, ch.x)
        shape!(ch.yf, ch.fast)
        ch.fast_noise!(ch.nz)
        ch.yf .+= ch.nz
        k = first_crossing(ch.yf, s.threshold_pe)
        k > 0 && (crossings[c] = sample_time(g, k))
        # slow path
        y = slow_wave[c]
        copyto!(y, ch.x)
        shape!(y, ch.slow)
        ch.slow_noise!(ch.nz)
        y .+= ch.nz
    end
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
            y = slow_wave[c]
            v = s.peak_detector ? maximum(view(y, k0:k1)) : y[k1]
            qin = sum(Float64(a.q) for a in trains[c]; init = 0.0) * ch.op.gain_factor[c]
            if qin > ch.pe_ceiling
                sat[c] = true
                v = min(v, ch.pe_ceiling)
            end
            stored[c] = v
            hg_counts = s.pedestal + s.adc_per_pe * v
            lg_counts = s.pedestal + s.adc_per_pe * s.lg_ratio * v
            hg_counts >= fullscale && (sat[c] = true)
            hg[c] = UInt16(clamp(round(Int, hg_counts), 0, fullscale))
            lg[c] = UInt16(clamp(round(Int, lg_counts), 0, fullscale))
            hits[c] = isfinite(crossings[c]) &&
                      ttrig - s.coincidence_ns <= crossings[c] <= ttrig + s.coincidence_ns
        end
    end
    return CitirocEvent(hg, lg, hits, Float32(ttrig), isfinite(ttrig), sat, stored)
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
