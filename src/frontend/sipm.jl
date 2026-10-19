# The SiPM device model of the full simulation (design document, Section "Response of
# the SiPM"): the S14160-6050HS element, one channel at a time. It turns the photons that
# reach the active area of a channel (the records of the photoelectron file, written with
# the detection efficiency set to one) into a train of avalanches with their times and
# charges in units of one avalanche, by
#   1. thinning the records with the detection efficiency at the wavelength and the
#      overvoltage of the run,
#   2. adding, to each avalanche, its simultaneous crosstalk avalanches, by branching: each
#      fired cell fires a Poisson number of neighbours, and each of those does the same,
#   3. adding afterpulses, delayed by an exponential time and reduced by the fraction of
#      the cell recharge completed at that delay,
#   4. adding the dark avalanches of the acquisition window, with their crosstalk and
#      afterpulses.
# The electronics (Citiroc, digitizer) consume the train. All parameters come from the
# device file data/sipm_s14160.toml, whose values and provenance are in the design
# document, Table "Parameters".

using Random: AbstractRNG, Xoshiro, rand, randn, randexp

"Device parameters of one SiPM type, read from data/<name>.toml."
struct SipmDevice
    name::String
    vbr_25c::Float64            # breakdown voltage at 25 C [V]
    dvbr_dt::Float64            # temperature coefficient of the breakdown [V/K]
    gain_per_volt::Float64      # gain / overvoltage [1/V]
    pde_vov::Vector{Float64}    # overvoltage points of the PDE curve at the peak [V]
    pde_val::Vector{Float64}    # PDE at those points (fraction)
    pde_lambda::Vector{Float64} # wavelength points of the spectral factor [nm]
    pde_spectral::Vector{Float64} # spectral factor, 1 at the peak wavelength
    ct_vov::Vector{Float64}     # overvoltage points of the crosstalk curve [V]
    ct_val::Vector{Float64}     # crosstalk probability at those points
    dark_hz_ref::Float64        # dark rate per device at the reference point [Hz]
    dark_t_ref::Float64         # its temperature [K]
    dark_vov_ref::Float64       # its overvoltage [V]
    dark_doubling_k::Float64    # temperature for a factor 2 in the dark rate [K]
    ap_prob_ref::Float64        # afterpulse probability at ap_vov_ref
    ap_vov_ref::Float64
    ap_tau_ns::Float64          # afterpulse delay constant [ns]
    recovery_ns::Float64        # cell recharge time constant [ns]
    n_cells::Int
    pulse_rise_ns::Float64      # avalanche current pulse into the electronics: time to peak
    pulse_fall_ns::Float64      # decay constant after the peak
    pulse_tr_ns::Float64        # rise constant of (1 - exp(-t/tr)) exp(-t/fall) giving that peak
    ct_pf::Float64              # terminal capacitance of the device [pF]
    input_ohm::Float64          # input resistance of the electronics the pulse is computed for
end

"""
    physical_pulse(recovery_ns, ct_pf, r_ohm) -> (t_peak, fall, tr)

The current pulse of one avalanche in the input resistance r_ohm of the electronics: the
recharge current of the fired cell, exponential with the recovery time, flows into the
input resistance in parallel with the capacitance of the whole device, which filters it
with r C_t. The pulse is the difference of the two exponentials, rising with the shorter
time constant and falling with the longer one; it is returned in the form
(1 - exp(-t/tr)) exp(-t/fall) used by the shapers, with its time to peak. The fast
component through the parasitic capacitance of the quenching resistor is neglected.
"""
function physical_pulse(recovery_ns::Real, ct_pf::Real, r_ohm::Real)
    rc = r_ohm * ct_pf * 1e-3                       # ohm x pF = 1e-3 ns
    ta, tb = max(recovery_ns, rc), min(recovery_ns, rc)
    tb >= 0.999ta && (tb = 0.999ta)                 # equal constants: the limit t exp(-t/ta)
    tr = 1 / (1 / tb - 1 / ta)
    return (t_peak = ta * tb / (ta - tb) * log(ta / tb), fall = ta, tr = tr)
end

"A copy of the device with the pulse computed for the input resistance r_ohm."
function with_input(dev::SipmDevice, r_ohm::Real)
    p = physical_pulse(dev.recovery_ns, dev.ct_pf, r_ohm)
    SipmDevice(dev.name, dev.vbr_25c, dev.dvbr_dt, dev.gain_per_volt, dev.pde_vov,
               dev.pde_val, dev.pde_lambda, dev.pde_spectral, dev.ct_vov, dev.ct_val,
               dev.dark_hz_ref, dev.dark_t_ref, dev.dark_vov_ref, dev.dark_doubling_k,
               dev.ap_prob_ref, dev.ap_vov_ref, dev.ap_tau_ns, dev.recovery_ns,
               dev.n_cells, p.t_peak, p.fall, p.tr, dev.ct_pf, Float64(r_ohm))
end

"Rise constant tr such that (1 - exp(-t/tr)) exp(-t/fall) peaks at t_peak (bisection)."
function rise_constant(t_peak::Float64, fall::Float64)
    tpeak(tr) = tr * log(1 + fall / tr)
    lo, hi = 1e-3, 100 * fall
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        tpeak(mid) < t_peak ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

"Linear interpolation with constant extrapolation."
function interp_clamped(xs::Vector{Float64}, ys::Vector{Float64}, x::Real)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    k = searchsortedlast(xs, x)
    f = (x - xs[k]) / (xs[k + 1] - xs[k])
    return ys[k] + f * (ys[k + 1] - ys[k])
end

function SipmDevice(mat::AbstractDict)
    d = mat["device"]
    v(k) = Float64.(d[k])
    return SipmDevice(String(mat["name"]), d["vbr_25c_v"], d["dvbr_dt_v_per_k"],
                      d["gain_per_volt"], v("pde_vov_v"), v("pde_at_peak"),
                      v("pde_lambda_nm"), v("pde_spectral"), v("crosstalk_vov_v"),
                      v("crosstalk"), d["dark_hz"], d["dark_t_k"], d["dark_vov_v"],
                      d["dark_doubling_k"], d["afterpulse_prob"], d["afterpulse_vov_v"],
                      d["afterpulse_tau_ns"], d["recovery_ns"], Int(d["n_cells"]),
                      physical_pulse(d["recovery_ns"], d["terminal_capacitance_pf"], d["input_ohm"])...,
                      Float64(d["terminal_capacitance_pf"]), Float64(d["input_ohm"]))
end

SipmDevice(name::AbstractString = "sipm_s14160") = SipmDevice(load_material(name))

"Breakdown voltage at temperature T [K]."
breakdown_voltage(dev::SipmDevice, t_k::Real) = dev.vbr_25c + dev.dvbr_dt * (t_k - 298.15)

"Gain (electrons per avalanche) at the overvoltage vov [V]."
gain(dev::SipmDevice, vov::Real) = dev.gain_per_volt * vov

"Detection efficiency at the overvoltage vov [V] and wavelength lambda [nm]."
pde(dev::SipmDevice, vov::Real, lambda_nm::Real) =
    interp_clamped(dev.pde_vov, dev.pde_val, vov) *
    interp_clamped(dev.pde_lambda, dev.pde_spectral, lambda_nm)

"Crosstalk probability at the overvoltage vov [V]."
crosstalk(dev::SipmDevice, vov::Real) = clamp(interp_clamped(dev.ct_vov, dev.ct_val, vov), 0.0, 0.95)

"""
Dark rate per device [Hz] at temperature T [K] and overvoltage vov [V]: the reference
value scaled by a factor 2 per dark_doubling_k of temperature and by the ratio of the
detection efficiencies at the peak wavelength (the trigger probability of a thermal
carrier follows that of a photon).
"""
dark_rate(dev::SipmDevice, t_k::Real, vov::Real) =
    dev.dark_hz_ref * 2.0^((t_k - dev.dark_t_ref) / dev.dark_doubling_k) *
    interp_clamped(dev.pde_vov, dev.pde_val, vov) /
    interp_clamped(dev.pde_vov, dev.pde_val, dev.dark_vov_ref)

"Afterpulse probability at the overvoltage vov [V], scaling with its square."
afterpulse_prob(dev::SipmDevice, vov::Real) =
    clamp(dev.ap_prob_ref * (vov / dev.ap_vov_ref)^2, 0.0, 0.5)

"""
Normalised avalanche current pulse at time t [ns] after the avalanche: peak 1 at
pulse_rise_ns, exponential fall with pulse_fall_ns.
"""
function pulse_shape(dev::SipmDevice, t::Real)
    t <= 0 && return 0.0
    tr = dev.pulse_tr_ns
    v = (1 - exp(-t / tr)) * exp(-t / dev.pulse_fall_ns)
    tp = dev.pulse_rise_ns
    vmax = (1 - exp(-tp / tr)) * exp(-tp / dev.pulse_fall_ns)
    return v / vmax
end

"""
Operating point of one run: temperature, overvoltage and the derived quantities, with
the per-channel gain factors (unit mean, relative spread gain_spread) and the length of
the acquisition window over which dark avalanches are generated.
"""
struct SipmOperation
    dev::SipmDevice
    t_k::Float64
    vov::Float64
    vbr::Float64
    gain::Float64
    pde_peak::Float64
    crosstalk::Float64
    dark_hz::Float64
    afterpulse::Float64
    gain_factor::Vector{Float64}   # one per channel
    window_ns::Tuple{Float64,Float64}   # dark avalanches from t0 to t1 relative to the decay
end

function SipmOperation(dev::SipmDevice; t_k::Real, vov::Real, n_channels::Integer,
                       gain_spread::Real = 0.0, seed::Integer = 1,
                       window_ns = (-10_000.0, 10_000.0))
    rng = Xoshiro(hash((:gain, seed)))
    gf = gain_spread > 0 ? [max(0.5, 1 + gain_spread * randn(rng)) for _ in 1:n_channels] :
                           ones(Float64, n_channels)
    return SipmOperation(dev, t_k, vov, breakdown_voltage(dev, t_k), gain(dev, vov),
                         pde(dev, vov, dev.pde_lambda[argmax(dev.pde_spectral)]),
                         crosstalk(dev, vov), dark_rate(dev, t_k, vov),
                         afterpulse_prob(dev, vov), gf, (Float64(window_ns[1]), Float64(window_ns[2])))
end

"An avalanche of one channel: time [ns] and charge in units of one full avalanche."
struct Avalanche
    t_ns::Float32
    q::Float32           # 1 for a full avalanche; crosstalk adds integers; afterpulses < 1
    kind::Int8           # 0 photon, 1 crosstalk, 2 afterpulse, 3 dark
end

const AV_PHOTON = Int8(0)
const AV_CROSSTALK = Int8(1)
const AV_AFTERPULSE = Int8(2)
const AV_DARK = Int8(3)

"Mean number of neighbours one fired cell fires, given the probability p of at least one extra cell."
branching_mean(p::Real) = -log(1 - p)

"""
Number of extra crosstalk avalanches for one primary, by branching: each fired cell fires
a Poisson number of neighbours of mean lambda = -ln(1 - p), so that the probability of at
least one extra cell is p, and each of those does the same. The total number of cells is
Borel distributed, P(n) = exp(-lambda n) (lambda n)^(n - 1) / n!, of mean 1 / (1 - lambda).
"""
function n_crosstalk(rng::AbstractRNG, p::Float64)
    lam = branching_mean(p)
    k, pending = 0, 1
    while pending > 0 && k < 200
        born = rand_poisson(rng, lam * pending)
        k += born
        pending = born
    end
    return k
end

"""
Add the companions of a primary avalanche at time t: the crosstalk avalanches at the same
time, and for each avalanche an afterpulse with probability afterpulse, delayed by
Exp(ap_tau) and carrying the recharge fraction 1 - exp(-delay / recovery).
"""
function add_companions!(out::Vector{Avalanche}, rng::AbstractRNG, op::SipmOperation,
                         t::Float64, primary_kind::Int8)
    dev = op.dev
    nct = n_crosstalk(rng, op.crosstalk)
    for _ in 1:nct
        push!(out, Avalanche(Float32(t), 1f0, AV_CROSSTALK))
    end
    for _ in 1:(1 + nct)
        if rand(rng) < op.afterpulse
            delay = dev.ap_tau_ns * randexp(rng)
            push!(out, Avalanche(Float32(t + delay),
                                 Float32(1 - exp(-delay / dev.recovery_ns)), AV_AFTERPULSE))
        end
    end
    return out
end

"""
    respond(op, times_ns, lambdas_nm, rng) -> Vector{Avalanche}

The avalanche train of one channel for one event: the photon records of that channel
(times and wavelengths), thinned by the detection efficiency, each kept photon followed
by its companions; then the dark avalanches of the window, Poisson in number, uniform in
time, with their companions. Sorted in time.
"""
function respond(op::SipmOperation, times_ns::AbstractVector, lambdas_nm::AbstractVector,
                 rng::AbstractRNG)
    out = Avalanche[]
    dev = op.dev
    for (t, lam) in zip(times_ns, lambdas_nm)
        rand(rng) <= pde(dev, op.vov, lam) || continue
        push!(out, Avalanche(Float32(t), 1f0, AV_PHOTON))
        add_companions!(out, rng, op, Float64(t), AV_PHOTON)
    end
    t0, t1 = op.window_ns
    ndark = rand_poisson(rng, op.dark_hz * (t1 - t0) * 1e-9)
    for _ in 1:ndark
        t = t0 + (t1 - t0) * rand(rng)
        push!(out, Avalanche(Float32(t), 1f0, AV_DARK))
        add_companions!(out, rng, op, t, AV_DARK)
    end
    sort!(out; by = a -> a.t_ns)
    return out
end

"""
Mean charge per ns of the dark avalanches of one channel, in avalanche units: the dark
rate times the mean charge of a dark count with its crosstalk (1 / (1 - lambda)
avalanches, lambda the branching mean) and its afterpulses (each avalanche adds one with
probability afterpulse, of mean charge ap_tau / (ap_tau + recovery)).
"""
function dark_charge_per_ns(op::SipmOperation)
    dev = op.dev
    cluster = 1 / (1 - branching_mean(op.crosstalk))
    return op.dark_hz * 1e-9 * cluster * (1 + op.afterpulse * dev.ap_tau_ns / (dev.ap_tau_ns + dev.recovery_ns))
end

"Total charge of a train in avalanche units, and in photoelectron-equivalents of charge."
total_charge(train::Vector{Avalanche}) = sum(Float64(a.q) for a in train; init = 0.0)
