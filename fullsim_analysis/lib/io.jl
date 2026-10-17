# Reading an event file of the common layout (design document, Table "Structure of the
# event file"), measured or emulated, into one named tuple.

"""
    read_events(path) -> NamedTuple

Fields: path, tag, frontend, kind ("simulation" or "measurement"), run (the attributes of
/run as a Dict), nch, n, q (charge in avalanches, nch x n, Float64), charge (raw, as
stored), lg (low-gain counts or nothing), hits (nch x n, Bool), trig (n, Bool), ttrig (n),
flags (n, UInt8), x and y (channel centres, mm), chip, and mc (nothing for a measurement,
else edep, e511, e1275, nph (photons at the active areas), maps (8 x 8 x n photons),
ngam).
"""
function read_events(path::AbstractString)
    h5open(path, "r") do f
        a = attrs(f["run"])
        run = Dict{String,Any}(k => a[k] for k in keys(a))
        ev = f["events"]
        q = Float64.(read(ev["charge_pe"]))
        nch, n = size(q)
        mc = nothing
        if haskey(f, "mc")
            m = f["mc"]
            mc = (edep = read(m["edep_kev"]), e511 = read(m["edep_511_kev"]),
                  e1275 = read(m["edep_1275_kev"]), nph = read(m["n_photons_at_sipm"]),
                  maps = read(m["photon_maps"]), ngam = read(m["n_gammas"]))
        end
        (path = String(path), tag = String(run["tag"]), frontend = String(run["frontend"]),
         kind = String(get(run, "kind", "simulation")), run = run, nch = nch, n = n, q = q,
         charge = read(ev["charge"]),
         lg = haskey(ev, "charge_lg") ? read(ev["charge_lg"]) : nothing,
         hits = Bool.(read(ev["hits"])), trig = Bool.(read(ev["triggered"])),
         ttrig = read(ev["t_trigger_ns"]), flags = read(ev["flags"]),
         x = Float64.(read(f["channels"]["x_mm"])), y = Float64.(read(f["channels"]["y_mm"])),
         chip = read(f["channels"]["chip"]), mc = mc)
    end
end

"Decays tagged by the simulation as a full absorption of one 511 keV gamma alone."
select_photopeak511(e) = e.mc === nothing ? error("no simulated tags in $(e.tag)") :
    (e.mc.e511 .> 510) .& (e.mc.e1275 .== 0)
"Decays tagged as a full absorption of the 1275 keV gamma alone."
select_photopeak1275(e) = (e.mc.e1275 .> 1273) .& (e.mc.e511 .== 0)
"Decays with no energy deposit."
select_empty(e) = e.mc.edep .== 0
"Decays with a deposited energy in [lo, hi] keV."
select_compton(e, lo = 200, hi = 300) = (e.mc.edep .> lo) .& (e.mc.edep .< hi)
