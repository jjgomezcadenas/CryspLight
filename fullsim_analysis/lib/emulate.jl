# Running a front-end chain on a subset of the decays of a photon file, for the scans:
# one chain per thread, SiPM and noise streams reseeded per decay, so the result does not
# depend on the thread count and two scans of the same decays differ only in the setting.

"The photon file of a front-end configuration, with the operating point it defines."
function open_pes_run(cfg_path::AbstractString)
    st = load_frontend(cfg_path)
    pes = read_pes(st.pes_path)
    return (setup = st, pes = pes, op = st.op)
end

"A function t -> chain for the kind and settings given (one chain per thread)."
function chain_maker(kind::AbstractString, settings, op::SipmOperation)
    kind == "citiroc" && return t -> CitirocChain(settings, op, Xoshiro(t))
    kind == "digitizer" && return t -> DigitizerChain(settings, op, Xoshiro(t))
    kind == "radioroc" && return t -> RadiorocChain(settings, op, Xoshiro(t))
    error("front end $(kind) not known")
end

const APPLY = Dict("citiroc" => citiroc_event, "digitizer" => digitizer_event,
                   "radioroc" => radioroc_event)

"""
    emulate(pes, op, kind, settings, list; seed) -> (stored, triggered, ttrig, saturated)

Apply the front end to the decays in list: stored is nch x length(list) in avalanches
(the stored_pe of the Citiroc and Radioroc2 events, the charge in avalanches of the
digitizer ones), triggered and saturated (any channel) are Bool vectors, ttrig in ns.
"""
function emulate(pes, op::SipmOperation, kind::AbstractString, settings, list::AbstractVector{<:Integer};
                 seed::Integer = 5)
    nch = length(op.gain_factor)
    n = length(list)
    stored = zeros(Float32, nch, n); trig = falses(n); ttrig = fill(NaN32, n); sat = falses(n)
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
    end
    return (stored = stored, triggered = trig, ttrig = ttrig, saturated = sat)
end

"""
Index lists of a photon file for the scans: photopeak511, photopeak1275, compton (200 to
300 keV) and empty decays, each at most nmax long.
"""
function event_lists(pes; nmax::Int = 1500)
    ev = pes.events
    cut(v) = v[1:min(nmax, length(v))]
    (photopeak511 = cut(findall((ev.edep_511_kev .> 510) .& (ev.edep_1275_kev .== 0))),
     photopeak1275 = cut(findall((ev.edep_1275_kev .> 1273) .& (ev.edep_511_kev .== 0))),
     compton = cut(findall((ev.edep_kev .> 200) .& (ev.edep_kev .< 300))),
     empty = cut(findall(ev.edep_kev .== 0)))
end

"A copy of a keyword-constructed settings struct with some fields replaced."
with(s::T; kw...) where {T} = T(; (f => getfield(s, f) for f in fieldnames(T))..., kw...)
