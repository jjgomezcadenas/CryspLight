# The Radioroc2 chain of the full simulation (design document, Section "Electronics",
# "The Radioroc2"), on the avalanche trains of the 64 channels of one event. It reuses the
# shaping, noise, peak detector and conversion of the Citiroc chain, with the differences
# of the chip:
#   - a peaking time per channel (fast mode 20 to 300 ns, slow mode 120 ns to 1.8 us), the
#     same on the high- and low-gain paths unless set apart;
#   - noise falling as the inverse square root of the peaking time;
#   - triggers: T1 and T2 on the time preamplifier, which sees the avalanche almost
#     unshaped (a pulse of its own, 5 ns rise and 20 ns fall by default), and the charge
#     trigger TQ on the low-gain shaped signal; the acquisition trigger is the chip-wide
#     OR of one of them, or a majority of the T1 crossings built outside the chip;
#   - the hold from the delay cell after the acquisition trigger, peak detectors on both
#     paths, and the saturation of the low-gain path at 2000 avalanches of 1e6 gain, the
#     high-gain one lower by the ratio of the gains.
# The event record is the CitirocEvent (high gain, low gain, hits, trigger time, flags).

Base.@kwdef struct RadiorocSettings
    tp_ns::Vector{Float64} = fill(300.0, 64)   # peaking time per channel, high-gain path
    tp_lg_ns::Vector{Float64} = Float64[]      # low-gain path; empty = same as tp_ns
    order::Int = 2
    hold_ns::Float64 = 500.0                   # delay cell: trigger to hold
    peak_detector::Bool = true
    trigger::String = "tq"                     # "tq", "t1", "t2", "majority" (of T1)
    t1_threshold_pe::Float64 = 0.5             # on the time preamplifier, in avalanches
    t2_threshold_pe::Float64 = 3.0
    tq_threshold_pe::Float64 = 50.0            # on the low-gain shaped signal, in avalanches
    majority::Int = 2
    coincidence_ns::Float64 = 50.0
    time_pulse_rise_ns::Float64 = 5.0          # pulse seen by the time preamplifier
    time_pulse_fall_ns::Float64 = 20.0
    time_noise_pe::Float64 = 0.1
    noise_pe_100ns::Float64 = 0.1              # high-gain noise at 100 ns peaking
    noise_exponent::Float64 = 0.5              # noise ~ (100 ns / tp)^exponent
    adc_bits::Int = 12
    adc_per_pe::Float64 = 10.0                 # high-gain counts per avalanche
    pedestal::Float64 = 200.0
    lg_ratio::Float64 = 0.1                    # low-gain over high-gain
    lg_ceiling_pe::Float64 = 2000.0            # at a gain of 1e6; scaled by the gain of the run
    dt_ns::Float64 = 1.0
    t_start_ns::Float64 = -200.0
    t_end_ns::Float64 = 12_000.0
    readout_ns::Float64 = 12_800.0             # multiplexed readout of 64 channels (recorded)
end

"A copy of a device with another avalanche pulse (the pulse of the time preamplifier)."
function with_pulse(dev::SipmDevice, rise_ns::Real, fall_ns::Real)
    SipmDevice(dev.name, dev.vbr_25c, dev.dvbr_dt, dev.gain_per_volt, dev.pde_vov,
               dev.pde_val, dev.pde_lambda, dev.pde_spectral, dev.ct_vov, dev.ct_val,
               dev.dark_hz_ref, dev.dark_t_ref, dev.dark_vov_ref, dev.dark_doubling_k,
               dev.ap_prob_ref, dev.ap_vov_ref, dev.ap_tau_ns, dev.recovery_ns,
               dev.n_cells, Float64(rise_ns), Float64(fall_ns),
               rise_constant(Float64(rise_ns), Float64(fall_ns)))
end

struct RadiorocChain
    s::RadiorocSettings
    op::SipmOperation
    grid::TimeGrid
    shapers::Dict{Float64,Shaper}        # one calibrated shaper per distinct peaking time
    noise::Dict{Float64,Function}        # shaped noise generator per peaking time
    time_pulse::Shaper                   # the time preamplifier pulse (device pulse only)
    x::Vector{Float64}
    y::Vector{Float64}
    ylg::Vector{Float64}
    yt::Vector{Float64}
    nz::Vector{Float64}
    lg_ceiling::Float64                  # avalanches at the low-gain ceiling for this gain
    hg_ceiling::Float64
    rng::AbstractRNG
end

noise_at(s::RadiorocSettings, tp::Real) = s.noise_pe_100ns * (100.0 / tp)^s.noise_exponent

function RadiorocChain(s::RadiorocSettings, op::SipmOperation, rng::AbstractRNG)
    n = ceil(Int, (s.t_end_ns - s.t_start_ns) / s.dt_ns)
    grid = TimeGrid(s.t_start_ns, s.dt_ns, n)
    tps = unique(vcat(s.tp_ns, s.tp_lg_ns))
    shapers = Dict(tp => Shaper(op.dev, tp, s.dt_ns; order = s.order) for tp in tps)
    noise = Dict(tp => noise_generator(shapers[tp], noise_at(s, tp), n, rng) for tp in tps)
    tdev = with_pulse(op.dev, s.time_pulse_rise_ns, s.time_pulse_fall_ns)
    tpulse = Shaper(tdev, 1.0, s.dt_ns)
    lgc = s.lg_ceiling_pe * 1e6 / op.gain
    return RadiorocChain(s, op, grid, shapers, noise, tpulse, zeros(n), zeros(n), zeros(n),
                         zeros(n), zeros(n), lgc, lgc * s.lg_ratio, rng)
end

tp_hg(ch::RadiorocChain, c::Int) = ch.s.tp_ns[c]
tp_lg(ch::RadiorocChain, c::Int) = isempty(ch.s.tp_lg_ns) ? ch.s.tp_ns[c] : ch.s.tp_lg_ns[c]

"""
    radioroc_event(ch, trains) -> CitirocEvent

Apply the Radioroc2 to the avalanche trains of one event. hits marks the channels whose
T1 fired within the coincidence window of the acquisition trigger.
"""
function radioroc_event(ch::RadiorocChain, trains::Vector{Vector{Avalanche}})
    s, g = ch.s, ch.grid
    nch = length(trains)
    t1 = fill(NaN, nch); t2 = fill(NaN, nch); tq = fill(NaN, nch)
    hg_wave = [zeros(Float64, g.n) for _ in 1:nch]
    lg_wave = [zeros(Float64, g.n) for _ in 1:nch]
    for c in 1:nch
        bin_train!(ch.x, g, trains[c])
        ch.x .*= ch.op.gain_factor[c]
        # time preamplifier: raw pulse plus its noise, two thresholds
        copyto!(ch.yt, ch.x)
        device_pulse!(ch.yt, ch.time_pulse)
        s.time_noise_pe > 0 && (ch.yt .+= s.time_noise_pe .* randn(ch.rng, g.n))
        k1 = first_crossing(ch.yt, s.t1_threshold_pe)
        k1 > 0 && (t1[c] = sample_time(g, k1))
        k2 = first_crossing(ch.yt, s.t2_threshold_pe)
        k2 > 0 && (t2[c] = sample_time(g, k2))
        # high-gain path
        y = hg_wave[c]
        copyto!(y, ch.x)
        shape!(y, ch.shapers[tp_hg(ch, c)])
        ch.noise[tp_hg(ch, c)](ch.nz)
        y .+= ch.nz
        # low-gain path (same avalanches, its own shaping and noise) and the charge trigger
        yl = lg_wave[c]
        copyto!(yl, ch.x)
        shape!(yl, ch.shapers[tp_lg(ch, c)])
        ch.noise[tp_lg(ch, c)](ch.nz)
        yl .+= ch.nz ./ s.lg_ratio          # the same input noise, larger in avalanche units
        kq = first_crossing(yl, s.tq_threshold_pe)
        kq > 0 && (tq[c] = sample_time(g, kq))
    end
    ttrig = s.trigger == "tq" ? trigger_time(tq, 1, s.coincidence_ns) :
            s.trigger == "t1" ? trigger_time(t1, 1, s.coincidence_ns) :
            s.trigger == "t2" ? trigger_time(t2, 1, s.coincidence_ns) :
            s.trigger == "majority" ? trigger_time(t1, s.majority, s.coincidence_ns) :
            error("trigger $(s.trigger) not supported (tq, t1, t2, majority)")
    hg = zeros(UInt16, nch); lg = zeros(UInt16, nch)
    hits = falses(nch); sat = falses(nch); stored = zeros(Float32, nch)
    fullscale = 2^s.adc_bits - 1
    if isfinite(ttrig)
        k0 = clamp(sample_index(g, ttrig), 1, g.n)
        k1 = clamp(sample_index(g, ttrig + s.hold_ns), 1, g.n)
        for c in 1:nch
            vh = s.peak_detector ? maximum(view(hg_wave[c], k0:k1)) : hg_wave[c][k1]
            vl = s.peak_detector ? maximum(view(lg_wave[c], k0:k1)) : lg_wave[c][k1]
            if vh > ch.hg_ceiling
                sat[c] = true; vh = ch.hg_ceiling
            end
            vl > ch.lg_ceiling && (sat[c] = true; vl = ch.lg_ceiling)
            stored[c] = vh
            hc = s.pedestal + s.adc_per_pe * vh
            lc = s.pedestal + s.adc_per_pe * s.lg_ratio * vl
            hc >= fullscale && (sat[c] = true)
            hg[c] = UInt16(clamp(round(Int, hc), 0, fullscale))
            lg[c] = UInt16(clamp(round(Int, lc), 0, fullscale))
            hits[c] = isfinite(t1[c]) && ttrig - s.coincidence_ns <= t1[c] <= ttrig + s.coincidence_ns
        end
    end
    return CitirocEvent(hg, lg, hits, Float32(ttrig), isfinite(ttrig), sat, stored)
end

"Stored fraction of an exponential train on the high-gain path of channel c (no noise)."
function stored_fraction(ch::RadiorocChain, tau_ns::Real; tau2_ns::Real = 0, frac1::Real = 1,
                         channel::Int = 1)
    g = ch.grid
    cdf(t) = t <= 0 ? 0.0 :
             frac1 * (1 - exp(-t / tau_ns)) +
             (tau2_ns > 0 ? (1 - frac1) * (1 - exp(-t / tau2_ns)) : 0.0)
    for k in 1:g.n
        t = sample_time(g, k)
        ch.x[k] = cdf(t + g.dt) - cdf(t)
    end
    copyto!(ch.y, ch.x)
    shape!(ch.y, ch.shapers[tp_hg(ch, channel)])
    return maximum(ch.y)
end
