# The SiPM and electronics stage of the full simulation: reads the photoelectron file of
# an optical run, applies the SiPM response and one front end (Citiroc or digitizer) to
# every decay, and writes the event file common to measurements and simulation (design
# document, Section "Files", Table "Structure of the event file"), with the group /mc of
# simulated quantities.
#
# Configuration (fullsim_runs/<campaign>/frontend_<tag>.toml):
# [run]       tag, pes (photoelectron file, relative to the repository), seed, n_events
#             (0 = all), output_dir
# [sipm]      device (data/<name>.toml), temperature_k, overvoltage_v, gain_spread,
#             window_ns = [t0, t1] for the dark avalanches
# [frontend]  type = "citiroc" | "digitizer", then the keys of CitirocSettings or
#             DigitizerSettings (names as in the structs)
# [channels]  chip_channels (channels per chip, 32 for the Citiroc, 64 for the others);
#             the acquisition channel number of geometric channel k is k - 1 by default

using Random: Xoshiro, seed!

struct FrontendSetup
    tag::String
    cfg::Dict{String,Any}
    cfg_text::String
    cfg_path::String
    pes_path::String
    seed::Int
    n_events::Int
    op::SipmOperation
    frontend::String
    settings::Any
    chip_channels::Int
    outdir::String
end

"Build a keyword struct from a TOML table, ignoring the keys that are not fields."
function settings_from(T::Type, tbl::AbstractDict)
    kw = Dict{Symbol,Any}()
    for (k, v) in tbl
        sym = Symbol(k)
        sym in fieldnames(T) || continue
        ft = fieldtype(T, sym)
        kw[sym] = ft <: Integer ? Int(v) :
                  ft <: AbstractFloat ? Float64(v) :
                  ft <: Vector{Float64} ? (v isa Number ? fill(Float64(v), 64) : Float64.(v)) : v
    end
    return T(; kw...)
end

function load_frontend(cfg_path::AbstractString; nx::Integer = 8, ny::Integer = 8)
    cfg_text = read(cfg_path, String)
    cfg = TOML.parse(cfg_text)
    run = cfg["run"]
    tag = get(run, "tag", splitext(basename(cfg_path))[1])
    repo = dirname(@__DIR__) |> dirname
    pes_path = isabspath(run["pes"]) ? run["pes"] : joinpath(repo, run["pes"])
    sp = cfg["sipm"]
    dev = SipmDevice(get(sp, "device", "sipm_s14160"))
    win = get(sp, "window_ns", [-1000.0, 10_000.0])
    op = SipmOperation(dev; t_k = sp["temperature_k"], vov = sp["overvoltage_v"],
                       n_channels = Int(nx * ny), gain_spread = get(sp, "gain_spread", 0.0),
                       seed = Int(get(run, "seed", 1)), window_ns = (win[1], win[2]))
    fe = cfg["frontend"]
    kind = fe["type"]
    settings = kind == "citiroc" ? settings_from(CitirocSettings, fe) :
               kind == "digitizer" ? settings_from(DigitizerSettings, fe) :
               kind == "radioroc" ? settings_from(RadiorocSettings, fe) :
               error("frontend type $(kind) not supported (citiroc, digitizer, radioroc)")
    chip = Int(get(get(cfg, "channels", Dict()), "chip_channels", kind == "citiroc" ? 32 : 64))
    outdir = joinpath(repo, get(run, "output_dir", "output"), tag)
    return FrontendSetup(tag, cfg, cfg_text, abspath(cfg_path), pes_path,
                         Int(get(run, "seed", 1)), Int(get(run, "n_events", 0)), op, kind,
                         settings, chip, outdir)
end

"Split the records of one event into the per-channel trains of the SiPM stage."
function event_trains(op::SipmOperation, e, nch::Int, rng::AbstractRNG)
    order = sortperm(e.channel)
    trains = [Avalanche[] for _ in 1:nch]
    i = 1
    n = length(order)
    while i <= n
        c = Int(e.channel[order[i]])
        j = i
        while j < n && e.channel[order[j + 1]] == c
            j += 1
        end
        idx = view(order, i:j)
        trains[c] = respond(op, view(e.t_ns, idx), view(e.lambda_nm, idx), rng)
        i = j + 1
    end
    # channels with no photon still carry their dark avalanches
    for c in 1:nch
        isempty(trains[c]) && (trains[c] = respond(op, Float32[], Float32[], rng))
    end
    return trains
end

"""
    run_fullsim_frontend(cfg_path; verbose) -> path of the event file

Apply the SiPM stage and the configured front end to every decay of the photoelectron
file and write output/<tag>/events.h5.
"""
function run_fullsim_frontend(cfg_path::AbstractString; verbose::Bool = true)
    st = load_frontend(cfg_path)
    pes = read_pes(st.pes_path)
    opt_cfg = TOML.parse(pes.attributes["config_toml"])
    nx, ny = size(pes.events.maps, 1), size(pes.events.maps, 2)
    nch = nx * ny
    ntot = length(pes.events.edep_kev)
    nev = st.n_events > 0 ? min(st.n_events, ntot) : ntot
    nt = Threads.maxthreadid()          # thread ids may exceed nthreads() (interactive pool)
    make_chain(t) = st.frontend == "citiroc" ? CitirocChain(st.settings, st.op, Xoshiro(t)) :
                    st.frontend == "digitizer" ? DigitizerChain(st.settings, st.op, Xoshiro(t)) :
                    RadiorocChain(st.settings, st.op, Xoshiro(t))
    chains = [make_chain(t) for t in 1:nt]
    chain = chains[1]
    apply = st.frontend == "citiroc" ? citiroc_event :
            st.frontend == "digitizer" ? digitizer_event : radioroc_event
    has_lg = st.frontend != "digitizer"

    mkpath(st.outdir)
    cp(st.cfg_path, joinpath(st.outdir, basename(st.cfg_path)); force = true)
    path = joinpath(st.outdir, "events.h5")
    # per-event outputs
    charge = has_lg ? zeros(UInt16, nch, nev) : zeros(Int32, nch, nev)
    lg = has_lg ? zeros(UInt16, nch, nev) : nothing
    qpe = zeros(Float32, nch, nev)
    hits = zeros(UInt8, nch, nev)
    ttrig = fill(NaN32, nev)
    triggered = zeros(UInt8, nev)
    flags = zeros(UInt8, nev)          # bit 1: a channel saturated; bit 2: no trigger
    # one event: deterministic in (seed, k) whatever the thread count, because the SiPM
    # stream and the noise stream of the chain are both reseeded from (seed, k)
    function one_event!(k::Int, ch)
        rng = Xoshiro(hash((st.seed, k)))
        seed!(ch.rng, hash((st.seed, k, :noise)))
        trains = event_trains(st.op, event_pes(pes, k), nch, rng)
        ev = apply(ch, trains)
        if has_lg
            charge[:, k] = ev.hg; lg[:, k] = ev.lg; qpe[:, k] = ev.stored_pe
        else
            charge[:, k] = ev.charge; qpe[:, k] = ev.charge_pe
        end
        hits[:, k] = UInt8.(ev.hits)
        ttrig[k] = ev.t_trigger
        triggered[k] = ev.triggered
        flags[k] = (any(ev.saturated) ? 0x01 : 0x00) | (ev.triggered ? 0x00 : 0x02)
        return
    end
    done = Threads.Atomic{Int}(0)
    t_all = @elapsed Threads.@threads :static for k in 1:nev
        one_event!(k, chains[Threads.threadid()])
        nd = Threads.atomic_add!(done, 1) + 1
        if verbose && nd % 20_000 == 0
            @printf("%-20s event %7d / %d\n", st.tag, nd, nev); flush(stdout)
        end
    end

    h5open(path, "w") do f
        # /run
        r = create_group(f, "run")
        a = attrs(r)
        a["tag"] = st.tag; a["kind"] = "simulation"; a["frontend"] = st.frontend
        a["frontend_config_toml"] = st.cfg_text
        a["optical_config_toml"] = pes.attributes["config_toml"]
        a["pes_file"] = st.pes_path
        a["code_version"] = code_version()
        a["crystal"] = opt_cfg["crystal"]["material"]
        a["temperature_k"] = st.op.t_k; a["overvoltage_v"] = st.op.vov
        a["breakdown_v"] = st.op.vbr; a["bias_v"] = st.op.vbr + st.op.vov
        a["n_events"] = nev; a["seed"] = st.seed; a["elapsed_s"] = t_all
        a["flags_key"] = "bit 1: a channel saturated; bit 2: no trigger"
        for fld in fieldnames(typeof(st.settings))
            v = getfield(st.settings, fld)
            a["fe_" * String(fld)] = v isa Vector ? (isempty(v) ? Float64[] : v) : v
        end
        # /channels: geometry of the optical run's readout
        ro = load_fullsim_readout(opt_cfg)
        cx, cy = ro.tiles ? channel_centres(ro) :
                 ([(i - 0.5) * ro.grid.pitch_mm for i in 1:nx], [(j - 0.5) * ro.grid.pitch_mm for j in 1:ny])
        ch = create_group(f, "channels")
        ch["channel"] = Int16.(1:nch)
        ch["x_mm"] = Float32[cx[(c - 1) % nx + 1] for c in 1:nch]
        ch["y_mm"] = Float32[cy[(c - 1) ÷ nx + 1] for c in 1:nch]
        ch["tile"] = Int8[(((c - 1) % nx) ÷ max(1, Int(ro.tile_n))) +
                          2 * (((c - 1) ÷ nx) ÷ max(1, Int(ro.tile_n))) + 1 for c in 1:nch]
        ch["chip"] = Int8[(c - 1) ÷ st.chip_channels for c in 1:nch]
        ch["chip_channel"] = Int8[(c - 1) % st.chip_channels for c in 1:nch]
        ch["acq_channel"] = Int16.(0:nch - 1)
        ch["status"] = fill(Int8(1), nch)      # 1 live, 0 dead, 2 noisy
        # /calib: the values used
        cb = create_group(f, "calib")
        unit = has_lg ? st.settings.adc_per_pe : chain.area_per_pe
        cb["gain_per_pe"] = Float32.(unit .* st.op.gain_factor)
        cb["gain_per_pe_unit"] = has_lg ? "ADC counts per avalanche, high gain" :
                                          "count-samples per avalanche"
        cb["pedestal"] = fill(Float32(has_lg ? st.settings.pedestal : 0), nch)
        ped_rms = st.frontend == "citiroc" ? st.settings.noise_pe * st.settings.adc_per_pe :
                  st.frontend == "radioroc" ? noise_at(st.settings, st.settings.tp_ns[1]) * st.settings.adc_per_pe :
                  chain.noise_charge
        cb["pedestal_rms"] = fill(Float32(ped_rms), nch)
        cb["dark_hz"] = fill(Float32(st.op.dark_hz), nch)
        cb["pde_peak"] = fill(Float32(st.op.pde_peak), nch)
        cb["crosstalk"] = fill(Float32(st.op.crosstalk), nch)
        cb["gain_electrons"] = Float32.(st.op.gain .* st.op.gain_factor)
        # /events
        ev = create_group(f, "events")
        write_chunked(ev, "charge", charge)
        lg !== nothing && write_chunked(ev, "charge_lg", lg)
        write_chunked(ev, "charge_pe", qpe)
        write_chunked(ev, "hits", hits)
        write_chunked(ev, "t_trigger_ns", ttrig)
        write_chunked(ev, "triggered", triggered)
        write_chunked(ev, "flags", flags)
        # /mc
        mc = create_group(f, "mc")
        pe = pes.events
        write_chunked(mc, "edep_kev", pe.edep_kev[1:nev])
        write_chunked(mc, "edep_511_kev", pe.edep_511_kev[1:nev])
        write_chunked(mc, "edep_1275_kev", pe.edep_1275_kev[1:nev])
        write_chunked(mc, "n_gammas", pe.n_gammas_in[1:nev])
        write_chunked(mc, "n_photons_emitted", pe.n_emitted[1:nev])
        write_chunked(mc, "n_photons_at_sipm", pe.n_detected[1:nev])
        write_chunked(mc, "photon_maps", pe.maps[:, :, 1:nev])
        ia = pes.interactions
        first = [ia.offset[k] + 1 for k in 1:nev]
        has = [ia.offset[k + 1] > ia.offset[k] for k in 1:nev]
        write_chunked(mc, "n_interactions", Int16[ia.offset[k + 1] - ia.offset[k] for k in 1:nev])
        write_chunked(mc, "x1_mm", Float32[has[k] ? ia.x_mm[first[k]] : NaN32 for k in 1:nev])
        write_chunked(mc, "y1_mm", Float32[has[k] ? ia.y_mm[first[k]] : NaN32 for k in 1:nev])
        write_chunked(mc, "z1_mm", Float32[has[k] ? ia.z_mm[first[k]] : NaN32 for k in 1:nev])
        write_chunked(mc, "e1_kev", Float32[has[k] ? ia.e_kev[first[k]] : 0f0 for k in 1:nev])
        write_chunked(mc, "kind1", Int8[has[k] ? ia.gamma_kind[first[k]] : 0 for k in 1:nev])
    end
    verbose && @printf("%-20s %d events, %.1f%% triggered, written %s (%.1f s)\n",
                       st.tag, nev, 100 * sum(triggered) / nev, path, t_all)
    return path
end

"Chunked dataset along the last axis (events)."
function write_chunked(g, name, A::AbstractArray)
    nd = ndims(A)
    n = size(A, nd)
    n == 0 && return (g[name] = A)
    ch = ntuple(i -> i < nd ? size(A, i) : min(4096, n), nd)
    d = create_dataset(g, name, eltype(A), size(A); chunk = ch)
    write(d, A)
    return d
end

"The Readout of an optical configuration (its TOML table), for the channel geometry."
function load_fullsim_readout(cfg::AbstractDict)
    rd = cfg["readout"]
    sz = Float32.(cfg["crystal"]["size_mm"])
    grid = SipmGrid(Float32(rd["pitch_mm"]), Int32(rd["nx"]), Int32(rd["ny"]))
    return get(rd, "tiles", false) ?
        Readout(grid; tiles = true, tile_n = Int(rd["tile_n"]), tile_pitch = rd["pitch_mm"],
                tile_edge = rd["tile_edge_mm"], active_w = rd["active_mm"],
                face_mm = (sz[1], sz[2]), sur_R = get(rd, "dead_reflectivity", 0.5),
                sur_specular = get(rd, "dead_specular", false)) :
        Readout(grid)
end
