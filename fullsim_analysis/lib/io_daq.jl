# Reading the files of the PET CAEN DAQ (pet-caen-daq, J. M. Benlloch) for the bench data:
# the decoded-event run file run_<id>.0000.h5 (schema 2), the run histograms
# run_<id>.histograms.h5, and the staircase and hold-delay scan artifacts. The layouts are
# those of docs/hdf5/*.md and backend/internal/hdf5store in that repository. The DAQ
# compresses with Blosc (filter 32001); H5Zblosc registers the filter on load.
#
# Values are returned as stored (raw ADC counts, raw clock ticks); no calibration here.

using H5Zblosc

const DAQ_EVENT_SCHEMA = 2

_daq_text(d) = String(UInt8.(read(d)))

"""
    daq_params(text) -> Dict{String,String}

The settings of a JANUS-format configuration text (the DAQ stores the submitted text in
/configuration/requested_janus): one entry per `Name value` line, comments after `#`
dropped, the value kept as written (e.g. "300 ns"). Indexed lines `Name[n]` are kept
under their own key; `daq_param` applies them.
"""
function daq_params(text::AbstractString)
    p = Dict{String,String}()
    for line in split(text, '\n')
        s = strip(first(split(line, '#'; limit = 2)))
        isempty(s) && continue
        parts = split(s; limit = 2)
        length(parts) == 2 || continue
        p[String(parts[1])] = String(strip(parts[2]))
    end
    p
end

"""
    daq_param(params, name; board = 0) -> String or nothing

The value of a setting for one board: the indexed line `name[board]` when present
(it overrides the common value, as in the DAQ and Janus), else the common `name`.
"""
daq_param(p::AbstractDict, name::AbstractString; board::Integer = 0) =
    get(p, "$(name)[$(board)]", get(p, String(name), nothing))

"""
    daq_value(params, name; board = 0) -> Float64 or nothing

A numeric setting with its unit stripped ("300 ns" -> 300.0, "45.4" -> 45.4); nothing
when absent or not numeric.
"""
function daq_value(p::AbstractDict, name::AbstractString; board::Integer = 0)
    v = daq_param(p, name; board = board)
    v === nothing && return nothing
    m = match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", v)
    m === nothing ? nothing : parse(Float64, m.match)
end

"""
    read_daq_run(path) -> NamedTuple

A decoded-event file of the DAQ (run_<id>.0000.h5, schema 2). Fields:

- `path`, `run_id`, `schema`;
- `obs`: the channel measurements of /events/spectroscopy/observations (sentinel rows
  dropped), as vectors of equal length: `event` (parent row, 0-based as stored),
  `trigger_id`, `timestamp` (raw ticks), `chain`, `node`, `channel`, `hg`, `lg`
  (raw ADC counts; meaningful where `has_hg` / `has_lg`), `toa`, `tot` (raw; meaningful
  where `has_timing`), `discriminator` (the latched charge-discriminator flag of the
  energy word, not a trigger flag);
- `nevents`: rows of /events/spectroscopy/events;
- `service`: the 1 Hz service records, `timestamp` (8 ns ticks), `fpga_t`, `board_t`,
  `detector_t`, `hv_t` (deg C), `hv_v` (V), `hv_i` (A), `hv_on`, `validity` (bit mask,
  check before use);
- `config`: `text` (the submitted JANUS text) and `params` (`daq_params` of it).
"""
function read_daq_run(path::AbstractString)
    h5open(path, "r") do f
        schema = Int(attrs(f)["schema_version"])
        schema == DAQ_EVENT_SCHEMA ||
            error("$(path): decoded-event schema $(schema), this reader expects $(DAQ_EVENT_SCHEMA)")
        rows = read(f["events/spectroscopy/observations"])
        rows = filter(r -> r.channel_valid != 0, rows)
        col(s, T) = T[getfield(r, s) for r in rows]
        obs = (event = col(:parent_row, Int), trigger_id = col(:trigger_id, UInt64),
               timestamp = col(:timestamp, UInt64), chain = col(:chain, Int),
               node = col(:node, Int), channel = col(:channel, Int),
               hg = col(:high_gain, Int), lg = col(:low_gain, Int),
               has_hg = col(:has_high_gain, Bool), has_lg = col(:has_low_gain, Bool),
               toa = col(:toa, Int), tot = col(:tot, Int), has_timing = col(:has_timing, Bool),
               discriminator = col(:discriminator, Bool))
        nevents = length(f["events/spectroscopy/events"])
        srows = read(f["events/service/events"])
        scol(s, T) = T[getfield(r, s) for r in srows]
        service = (timestamp = scol(:timestamp, UInt64), fpga_t = scol(:fpga_temperature_c, Float64),
                   board_t = scol(:board_temperature_c, Float64),
                   detector_t = scol(:detector_temperature_c, Float64),
                   hv_t = scol(:hv_temperature_c, Float64), hv_v = scol(:hv_voltage_v, Float64),
                   hv_i = scol(:hv_current_a, Float64), hv_on = scol(:hv_on, Bool),
                   validity = scol(:validity, UInt8))
        text = _daq_text(f["configuration/requested_janus"])
        (path = String(path), run_id = _daq_text(f["run/run_id"]), schema = schema, obs = obs,
         nevents = nevents, service = service, config = (text = text, params = daq_params(text)))
    end
end

"""
    daq_channel_values(run; gain = :hg, chain = 0, node = 0) -> Vector{Vector{Int}}

The raw values of one gain (`:hg` or `:lg`) of one board, one vector per channel
(index `ch + 1`), in event order. Rows without that gain are skipped.
"""
function daq_channel_values(run; gain::Symbol = :hg, chain::Integer = 0, node::Integer = 0)
    o = run.obs
    v = gain === :hg ? o.hg : gain === :lg ? o.lg : error("gain must be :hg or :lg")
    has = gain === :hg ? o.has_hg : o.has_lg
    out = [Int[] for _ in 1:64]
    for i in eachindex(v)
        (has[i] && o.chain[i] == chain && o.node[i] == node) || continue
        push!(out[o.channel[i] + 1], v[i])
    end
    out
end

"""
    daq_spectra(run; gain = :hg, chain = 0, node = 0, nbins = 16384) -> Matrix{Int}

Full-resolution spectra of one gain of one board: counts (nbins x 64), row `k + 1` is
ADC value `k`; values at or above `nbins` go to the last row.
"""
function daq_spectra(run; gain::Symbol = :hg, chain::Integer = 0, node::Integer = 0,
                     nbins::Integer = 16384)
    h = zeros(Int, nbins, 64)
    for (c, vals) in enumerate(daq_channel_values(run; gain = gain, chain = chain, node = node))
        for x in vals
            h[clamp(x, 0, nbins - 1) + 1, c] += 1
        end
    end
    h
end

"""
    read_daq_histograms(path) -> Dict{String,Vector}

The run histograms (run_<id>.histograms.h5, schema 2): for each family present
("pha_high", "pha_low", "toa", "tot") a vector of spectra, each a named tuple with
`chain`, `node`, `channel`, `minimum`, `bin_width` (axis units of the family),
`entries`, `underflow`, `overflow` and `counts` (Vector{UInt32}).
"""
function read_daq_histograms(path::AbstractString)
    h5open(path, "r") do f
        Int(attrs(f)["schema_version"]) == 2 || error("$(path): histogram schema is not 2")
        out = Dict{String,Vector}()
        for fam in keys(f["histograms"])
            g = f["histograms"][fam]
            index = read(g["spectra"]); bins = read(g["bins"])
            out[fam] = [(chain = Int(s.chain), node = Int(s.node), channel = Int(s.channel),
                         minimum = s.minimum, bin_width = s.bin_width, entries = Int(s.entries),
                         underflow = Int(s.underflow), overflow = Int(s.overflow),
                         counts = bins[s.bin_offset + 1:s.bin_offset + s.bin_count]) for s in index]
        end
        out
    end
end

"""
    read_daq_staircase(path) -> NamedTuple

A staircase artifact: `metadata` (the JSON text of the scan manifest), `threshold`
(the TD/QD coarse threshold of each point), `elapsed_s`, `t_or_rate`, `q_or_rate` (Hz),
and `rate` (64 x npoints, Hz, the rate of each channel's timing discriminator) and
`count` (64 x npoints).
"""
function read_daq_staircase(path::AbstractString)
    h5open(path, "r") do f
        g = f["staircase"]
        pts = read(g["points"]); ch = read(g["channel_measurements"])
        n = length(pts)
        rate = zeros(64, n); count = zeros(Int, 64, n)
        for r in ch
            rate[r.channel + 1, r.point_index + 1] = r.rate_cps
            count[r.channel + 1, r.point_index + 1] = r.count
        end
        (metadata = _daq_text(g["metadata_json"]), threshold = [Int(p.threshold) for p in pts],
         elapsed_s = [p.elapsed_nanoseconds * 1e-9 for p in pts],
         t_or_rate = [p.t_or_rate_cps for p in pts], q_or_rate = [p.q_or_rate_cps for p in pts],
         rate = rate, count = count)
    end
end

"""
    read_daq_hold_delay(path; codes_per_bin = 16) -> NamedTuple

A hold-delay scan artifact: `metadata` (JSON text), `delay_ns` (requested),
`effective_delay_ns` (read back), `events` (collected per point), `counts`
(512 x 64 x npoints, the high-gain spectrum of each channel at each point), `missing`
(64 x npoints, events without that channel) and `bin_low` (lower ADC edge of each of the
512 bins). The scan bins the 13-bit range in 16 ADC codes; with Range_14bit in the run
configuration it is 32, which the artifact does not record: pass `codes_per_bin = 32`.
"""
function read_daq_hold_delay(path::AbstractString; codes_per_bin::Integer = 16)
    h5open(path, "r") do f
        g = f["hold_delay"]
        pts = read(g["points"]); n = length(pts)
        counts = zeros(Int, 512, 64, n)
        for r in read(g["nonzero_hg_bins"])
            counts[r.bin + 1, r.channel + 1, r.point_index + 1] = r.count
        end
        missing = zeros(Int, 64, n)
        for r in read(g["missing_channels"])
            missing[r.channel + 1, r.point_index + 1] = r.count
        end
        (metadata = _daq_text(g["metadata_json"]), delay_ns = [Int(p.delay_ns) for p in pts],
         effective_delay_ns = [Int(p.effective_delay_ns) for p in pts],
         events = [Int(p.events_collected) for p in pts], counts = counts, missing = missing,
         bin_low = collect(0:511) .* codes_per_bin)
    end
end
