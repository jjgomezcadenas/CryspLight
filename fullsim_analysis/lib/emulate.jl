# Running a front-end chain on a subset of the decays of a photon file, for the scans:
# one chain per thread, SiPM and noise streams reseeded per decay, so the result does not
# depend on the thread count and two scans of the same decays differ only in the setting.

"""
    open_pes_run(cfg_path) -> (setup, pes, op)

The photon file of a front-end configuration, read whole (read_pes), with the
FrontendSetup of the configuration (load_frontend) and the SiPM operating point it
defines.
"""
function open_pes_run(cfg_path::AbstractString)
    st = load_frontend(cfg_path)
    pes = read_pes(st.pes_path)
    return (setup = st, pes = pes, op = st.op)
end

"""
    chain_maker(kind, settings, op::SipmOperation) -> Function

A function t -> chain building the front-end chain of the kind given ("citiroc",
"digitizer" or "radioroc") with its settings and the operating point op, its noise
stream seeded with t (one chain per thread).
"""
function chain_maker(kind::AbstractString, settings, op::SipmOperation)
    kind == "citiroc" && return t -> CitirocChain(settings, op, Xoshiro(t))
    kind == "digitizer" && return t -> DigitizerChain(settings, op, Xoshiro(t))
    kind == "radioroc" && return t -> RadiorocChain(settings, op, Xoshiro(t))
    error("front end $(kind) not known")
end

const APPLY = Dict("citiroc" => citiroc_event, "digitizer" => digitizer_event,
                   "radioroc" => radioroc_event)

"""
    emulate(pes, op, kind, settings, list; seed) -> (stored, triggered, ttrig, saturated, hg, lg, hits)

Apply the front end of the kind given, with its settings and the operating point op, to
the decays in list of the photon file pes (read_pes), each reseeded from (seed, decay) so
that the result does not depend on the threads: stored is nch x length(list) in avalanches
(the stored_pe of the Citiroc and Radioroc2 events, the charge in avalanches of the
digitizer ones), triggered and saturated (any channel) are Bool vectors, ttrig in ns;
hg and lg are the ADC counts of the two paths of the ASICs (zero for the digitizer), hits
the channels flagged by their own discriminator near the trigger.
"""
function emulate(pes, op::SipmOperation, kind::AbstractString, settings, list::AbstractVector{<:Integer};
                 seed::Integer = 5)
    nch = length(op.gain_factor)
    n = length(list)
    stored = zeros(Float32, nch, n); trig = falses(n); ttrig = fill(NaN32, n); sat = falses(n)
    hg = zeros(UInt16, nch, n); lg = zeros(UInt16, nch, n); hits = falses(nch, n)
    make = chain_maker(kind, settings, op)
    chains = [make(t) for t in 1:Threads.maxthreadid()]
    apply = APPLY[kind]
    Threads.@threads :static for i in 1:n
        k = list[i]
        ch = chains[Threads.threadid()]
        seed!(ch.rng, hash((seed, k, :noise)))
        trains = event_trains(op, event_pes(pes, k), nch, Xoshiro(hash((seed, k))))
        ev = apply(ch, trains)
        stored[:, i] = kind == "digitizer" ? ev.charge_pe : ev.stored_pe
        trig[i] = ev.triggered; ttrig[i] = ev.t_trigger; sat[i] = any(ev.saturated)
        kind == "digitizer" || (hg[:, i] = ev.hg; lg[:, i] = ev.lg)
        hits[:, i] = ev.hits
    end
    return (stored = stored, triggered = trig, ttrig = ttrig, saturated = sat, hg = hg, lg = lg, hits = hits)
end

"""
    event_lists(pes; nmax = 1500) -> (photopeak511, photopeak1275, compton, empty)

Index lists (Vector{Int}, decay numbers) of a photon file (read_pes) for the scans: the
full absorptions of a 511 keV gamma alone (more than 510 keV) and of the 1275 keV gamma
alone (more than 1273 keV), the decays depositing between 200 and 300 keV, and those
depositing nothing, each cut to its first nmax entries.
"""
function event_lists(pes; nmax::Int = 1500)
    ev = pes.events
    cut(v) = v[1:min(nmax, length(v))]
    (photopeak511 = cut(findall((ev.edep_511_kev .> 510) .& (ev.edep_1275_kev .== 0))),
     photopeak1275 = cut(findall((ev.edep_1275_kev .> 1273) .& (ev.edep_511_kev .== 0))),
     compton = cut(findall((ev.edep_kev .> 200) .& (ev.edep_kev .< 300))),
     empty = cut(findall(ev.edep_kev .== 0)))
end

"""
    with(s; kw...) -> typeof(s)

A copy of a keyword-constructed settings struct (CitirocSettings, DigitizerSettings,
RadiorocSettings) with the fields given as keywords replaced, in the units of the struct.
"""
with(s::T; kw...) where {T} = T(; (f => getfield(s, f) for f in fieldnames(T))..., kw...)

"""
    with_window(op::SipmOperation, w) -> SipmOperation

The operating point with another window w = (t0, t1) in ns for the dark avalanches.
"""
with_window(op::SipmOperation, w) =
    SipmOperation(op.dev, op.t_k, op.vov, op.vbr, op.gain, op.pde_peak, op.crosstalk, op.dark_hz,
                  op.afterpulse, op.gain_factor, (Float64(w[1]), Float64(w[2])))
