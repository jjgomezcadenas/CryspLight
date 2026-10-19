# Waveforms of the front-end emulation as cascades of first-order filters on a uniform
# time grid. The avalanche current pulse (1 - exp(-t/tr)) exp(-t/tf) is the difference of
# two exponentials, and a CR-RC^n shaper is one differentiating and n integrating stages
# with the same time constant, so the whole chain from the avalanche train to the shaped
# signal is a product of first-order sections, applied recursively in O(N) with no
# transform. The impulse train is binned on the grid; with a step of 0.5 ns against time
# constants of tens of nanoseconds the discretisation error is far below the noise, and
# the single-avalanche peak is measured on the same grid, so the calibration is exact by
# construction.

using Statistics: std
using Random: AbstractRNG, randn, randn!

"A uniform time grid: t_k = t0 + (k - 1) dt, k = 1..n."
struct TimeGrid
    t0::Float64
    dt::Float64
    n::Int
end

sample_index(g::TimeGrid, t::Real) = 1 + floor(Int, (t - g.t0) / g.dt)
sample_time(g::TimeGrid, k::Integer) = g.t0 + (k - 1) * g.dt

"Bin a train of avalanches (time, charge) onto the grid: charge per sample."
function bin_train!(x::Vector{Float64}, g::TimeGrid, train::Vector{Avalanche})
    fill!(x, 0.0)
    for a in train
        k = sample_index(g, a.t_ns)
        1 <= k <= g.n || continue
        x[k] += a.q
    end
    return x
end

"""
The grid of a front-end chain: the record from t_start to t_end, preceded by warm_ns in
which the dark avalanches and the noise already run through the filters, so that the
response is stationary from the start of the record. The dark avalanches of the operating
point must cover the whole grid.
"""
function chain_grid(op::SipmOperation, t_start::Real, t_end::Real, dt::Real, warm_ns::Real)
    t0 = t_start - warm_ns
    op.window_ns[1] <= t0 + 1e-9 ||
        error("dark avalanches start at $(op.window_ns[1]) ns, after the start of the grid " *
              "with its warm-up ($t0 ns): set [sipm] window_ns to start at or before it")
    return TimeGrid(t0, dt, ceil(Int, (t_end - t0) / dt))
end

"""
Remove from the binned charge x of channel c the mean charge of the dark avalanches. The
chips hold their output at a fixed level, so the steady current of the dark counts gives
no offset; only its fluctuations remain.
"""
function remove_dark_level!(x::Vector{Float64}, op::SipmOperation, c::Integer, dt::Real)
    x .-= dark_charge_per_ns(op) * op.gain_factor[c] * dt
    return x
end

"RC integrator (low pass) with time constant tau, in place: y <- RC(y)."
function rc_lowpass!(y::Vector{Float64}, tau::Float64, dt::Float64)
    a = exp(-dt / tau)
    b = 1 - a
    prev = 0.0
    @inbounds for k in eachindex(y)
        prev = a * prev + b * y[k]
        y[k] = prev
    end
    return y
end

"CR differentiator (high pass) with time constant tau, in place."
function cr_highpass!(y::Vector{Float64}, tau::Float64, dt::Float64)
    a = exp(-dt / tau)
    prev_x = 0.0
    prev_y = 0.0
    @inbounds for k in eachindex(y)
        x = y[k]
        prev_y = a * (prev_y + x - prev_x)
        prev_x = x
        y[k] = prev_y
    end
    return y
end

"""
Avalanche current pulse of the device applied to a binned charge series, in place:
the difference of two integrators, exp(-t/tf) - exp(-t/(1/tf + 1/tr)^-1), normalised so
that one unit of charge gives a peak of one.
"""
function sipm_pulse!(y::Vector{Float64}, dev::SipmDevice, dt::Float64)
    tf = dev.pulse_fall_ns
    tr = dev.pulse_tr_ns
    t2 = 1 / (1 / tf + 1 / tr)
    y1 = copy(y)
    rc_lowpass!(y, tf, dt)
    rc_lowpass!(y1, t2, dt)
    # the impulse response of rc_lowpass! to a unit sample is (1 - a) a^(k-1): divide each
    # term by its (1 - a) so that both exponentials start at one and their difference is
    # the pulse (zero at the avalanche, peak at pulse_rise_ns); the absolute scale is set
    # by the caller from the single-avalanche peak
    s1 = 1 / (1 - exp(-dt / tf))
    s2 = 1 / (1 - exp(-dt / t2))
    @inbounds for k in eachindex(y)
        y[k] = y[k] * s1 - y1[k] * s2
    end
    return y
end

"""
Peak of the response to one unit of charge in sample 1 through the function f!(y, dt),
measured on a grid of n samples: the calibration of f in units of one avalanche.
"""
function unit_peak(f!, dt::Float64, n::Int)
    y = zeros(Float64, n)
    y[1] = 1.0
    f!(y, dt)
    return maximum(y)
end

"""
Semi-Gaussian shaper of order n and peaking time tp = n theta applied to a current
series, in place. The shaper output for one avalanche is (t/theta)^n exp(-t/theta): the
step response of the CR-RC^n chain, which is its response to the current pulse once the
preamplifier has integrated it. As a filter on the current that is n + 1 integrators
with the same time constant (the CR cancels the integration of the preamplifier).
"""
function crrc!(y::Vector{Float64}, tp::Float64, dt::Float64; order::Int = 2)
    theta = tp / order
    for _ in 1:order + 1
        rc_lowpass!(y, theta, dt)
    end
    return y
end

"""
CR-RC^n shaper of peaking time tp = n theta behind a voltage-sensitive preamplifier, in
place: the preamplifier amplifies the voltage of the current pulse across the input
termination, and the shaper differentiates it once and integrates it n times, all with
the time constant theta. A slow input is therefore suppressed, unlike behind a
charge-sensitive preamplifier, whose integration the differentiator cancels (crrc!).
"""
function crrc_voltage!(y::Vector{Float64}, tp::Float64, dt::Float64; order::Int = 2)
    theta = tp / order
    cr_highpass!(y, theta, dt)
    for _ in 1:order
        rc_lowpass!(y, theta, dt)
    end
    return y
end

"""
A shaping chain: the device pulse followed by a CR-RC^order shaper of peaking time tp,
behind a charge-sensitive preamplifier (voltage = false: the shaped output follows the
integrated charge) or a voltage-sensitive one (voltage = true: the shaper acts on the
current pulse itself), calibrated so that one avalanche gives a peak of one at the
output. The same object shapes white input noise, so that the noise at the output has
the spectrum of the chain.
"""
struct Shaper
    dev::SipmDevice
    tp_ns::Float64
    order::Int
    dt::Float64
    unit::Float64          # output peak of one avalanche before calibration
    unit_sipm::Float64     # peak of the device pulse of one avalanche
    voltage::Bool          # voltage-sensitive preamplifier
end

"The shaper filter alone (no device pulse) of a chain, in place."
shaper_filter!(y, tp, dt, order, voltage) =
    voltage ? crrc_voltage!(y, tp, dt; order = order) : crrc!(y, tp, dt; order = order)

function Shaper(dev::SipmDevice, tp_ns::Real, dt::Real; order::Int = 2, n_cal::Int = 0,
                voltage::Bool = false)
    n = n_cal > 0 ? n_cal : ceil(Int, (20 * tp_ns + 10 * dev.pulse_fall_ns) / dt)
    us = unit_peak((y, h) -> sipm_pulse!(y, dev, h), Float64(dt), n)
    u = unit_peak((y, h) -> shaper_filter!(sipm_pulse!(y, dev, h), Float64(tp_ns), h, order, voltage),
                  Float64(dt), n)
    return Shaper(dev, Float64(tp_ns), order, Float64(dt), u, us, voltage)
end

"Shaped output, in units of one avalanche, of a binned charge series (in place)."
function shape!(y::Vector{Float64}, sh::Shaper)
    sipm_pulse!(y, sh.dev, sh.dt)
    shaper_filter!(y, sh.tp_ns, sh.dt, sh.order, sh.voltage)
    y ./= sh.unit
    return y
end

"Device pulse only (no shaper), in units of one avalanche, of a binned series (in place)."
function device_pulse!(y::Vector{Float64}, sh::Shaper)
    sipm_pulse!(y, sh.dev, sh.dt)
    y ./= sh.unit_sipm
    return y
end

"""
Shaped white noise: a Gaussian series at the input of the shaper, shaped, and scaled so
that its rms at the output is rms_out (in units of one avalanche). The scale follows
from the impulse response h of the shaper, whose output variance for unit white input
is the sum of h squared; it is therefore the same for every chain built with the same
settings, whatever the random stream. Each series is shaped from a preceding stretch of
ten peaking times, so that the noise is stationary from its first sample.
"""
function noise_generator(sh::Shaper, rms_out::Real, n::Int, rng::AbstractRNG)
    nw = ceil(Int, 10 * sh.tp_ns / sh.dt)
    h = zeros(Float64, max(n, 2nw))
    h[1] = 1.0
    shaper_filter!(h, sh.tp_ns, sh.dt, sh.order, sh.voltage)
    s = rms_out / sqrt(sum(abs2, h))
    buf = zeros(Float64, nw + n)
    return function (out::Vector{Float64})
        length(buf) == nw + length(out) || resize!(buf, nw + length(out))
        randn!(rng, buf)
        shaper_filter!(buf, sh.tp_ns, sh.dt, sh.order, sh.voltage)
        @inbounds for k in eachindex(out)
            out[k] = s * buf[nw + k]
        end
        return out
    end
end
