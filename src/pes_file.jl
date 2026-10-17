# The photoelectron file of the full simulation (design document, Section "Files"): one
# HDF5 file per run with, for every decay, the interactions of its gammas, the photon
# counts and one record per photoelectron (channel, time, wavelength), stored
# consecutively with offset arrays. Written in chunks of decays by the driver, so that a
# run of any length streams to disk; read back whole or per event.
#
# Layout
#   /events        edep_kev, edep_511_kev, edep_1275_kev, n_gammas_in, n_emitted,
#                  n_detected (one per decay), maps (nx x ny x n, integrated counts)
#   /interactions  offset (n + 1), x_mm, y_mm, z_mm, e_kev, t_ns, gamma_kind, proc
#   /pes           offset (n + 1), channel (UInt8), t_ns (Float32), lambda_nm (UInt16)
#   attributes     the run configuration (TOML text), the code version, n_decays, the
#                  PDE applied in the kernel (1 when the records are photons arriving at
#                  the active area, to be thinned by the detection efficiency later).

const PES_CHUNK = 1 << 16

struct PesWriter
    file::HDF5.File
    nx::Int
    ny::Int
    n_decays::Int                 # decays written so far
end

function _extendible(g, name, T, dims0::Tuple, chunk::Tuple)
    maxdims = ntuple(i -> i == length(dims0) ? -1 : dims0[i], length(dims0))
    create_dataset(g, name, T, (dims0, maxdims); chunk = chunk)
end

"""
    open_pes(path, nx, ny; cfg_text, pde_applied, lambda_nm, extra...) -> PesWriter

Create the file with empty extendible datasets and the run attributes.
"""
function open_pes(path::AbstractString, nx::Integer, ny::Integer;
                  cfg_text::AbstractString = "", pde_applied::Real = 1.0,
                  lambda_nm::Real = 0.0, attributes = Dict{String,Any}())
    nx, ny = Int(nx), Int(ny)
    f = h5open(path, "w")
    ev = create_group(f, "events")
    for (name, T) in (("edep_kev", Float32), ("edep_511_kev", Float32),
                      ("edep_1275_kev", Float32), ("n_gammas_in", Int8),
                      ("n_emitted", Int32), ("n_detected", Int32))
        _extendible(ev, name, T, (0,), (4096,))
    end
    _extendible(ev, "maps", UInt16, (nx, ny, 0), (nx, ny, 1024))
    ia = create_group(f, "interactions")
    _extendible(ia, "offset", Int64, (1,), (4096,))
    ia["offset"][1] = Int64(0)
    for (name, T) in (("x_mm", Float32), ("y_mm", Float32), ("z_mm", Float32),
                      ("e_kev", Float32), ("t_ns", Float32), ("gamma_kind", Int8),
                      ("proc", Int8))
        _extendible(ia, name, T, (0,), (4096,))
    end
    pe = create_group(f, "pes")
    _extendible(pe, "offset", Int64, (1,), (4096,))
    pe["offset"][1] = Int64(0)
    _extendible(pe, "channel", UInt8, (0,), (PES_CHUNK,))
    _extendible(pe, "t_ns", Float32, (0,), (PES_CHUNK,))
    _extendible(pe, "lambda_nm", UInt16, (0,), (PES_CHUNK,))
    a = attrs(f)
    a["config_toml"] = String(cfg_text)
    a["code_version"] = code_version()
    a["pde_applied"] = Float64(pde_applied)
    a["lambda_nm"] = Float64(lambda_nm)
    a["n_decays"] = 0
    a["channel_key"] = "channel = (j - 1) nx + i, i along x, j along y, 1-based"
    a["gamma_kind_key"] = "1 = 511 keV, 2 = 1275 keV"
    a["proc_key"] = "0 = photoelectric, 1 = Compton, 2 = below cut"
    for (k, v) in attributes
        a[k] = v
    end
    return PesWriter(f, Int(nx), Int(ny), 0)
end

function _append!(d, v::AbstractVector)
    n = length(d)
    isempty(v) && return
    HDF5.set_extent_dims(d, (n + length(v),))
    d[n + 1:n + length(v)] = v
    return
end

"Append the result of run_decays! (a chunk of consecutive decays) to the file."
function append_pes!(w::PesWriter, r::NamedTuple)
    ev, ia, pe = r.events, r.interactions, r.pes
    ev.first_decay == w.n_decays + 1 ||
        error("chunk starts at decay $(ev.first_decay), file holds $(w.n_decays)")
    n = length(ev.edep_kev)
    g = w.file["events"]
    for name in ("edep_kev", "edep_511_kev", "edep_1275_kev", "n_gammas_in",
                 "n_emitted", "n_detected")
        _append!(g[name], getfield(ev, Symbol(name)))
    end
    d = g["maps"]
    n0 = size(d, 3)
    HDF5.set_extent_dims(d, (w.nx, w.ny, n0 + n))
    d[:, :, n0 + 1:n0 + n] = ev.maps
    gi = w.file["interactions"]
    base = gi["offset"][end]
    _append!(gi["offset"], base .+ ia.offset[2:end])
    for name in ("x_mm", "y_mm", "z_mm", "e_kev", "t_ns", "gamma_kind", "proc")
        _append!(gi[name], getfield(ia, Symbol(name)))
    end
    gp = w.file["pes"]
    base = gp["offset"][end]
    _append!(gp["offset"], base .+ pe.offset[2:end])
    _append!(gp["channel"], pe.channel)
    _append!(gp["t_ns"], pe.t_ns)
    lam = attrs(w.file)["lambda_nm"]
    _append!(gp["lambda_nm"], fill(UInt16(round(lam)), length(pe.channel)))
    attrs(w.file)["n_decays"] = w.n_decays + n
    return PesWriter(w.file, w.nx, w.ny, w.n_decays + n)
end

close_pes(w::PesWriter) = close(w.file)

"Read a photoelectron file whole into the NamedTuple layout of run_decays!."
function read_pes(path::AbstractString)
    h5open(path, "r") do f
        rd(g, k) = read(f[g][k])
        a = attrs(f)
        (events = (edep_kev = rd("events", "edep_kev"),
                   edep_511_kev = rd("events", "edep_511_kev"),
                   edep_1275_kev = rd("events", "edep_1275_kev"),
                   n_gammas_in = rd("events", "n_gammas_in"),
                   n_emitted = rd("events", "n_emitted"),
                   n_detected = rd("events", "n_detected"),
                   maps = rd("events", "maps"), first_decay = 1),
         interactions = (offset = rd("interactions", "offset"),
                         x_mm = rd("interactions", "x_mm"), y_mm = rd("interactions", "y_mm"),
                         z_mm = rd("interactions", "z_mm"), e_kev = rd("interactions", "e_kev"),
                         t_ns = rd("interactions", "t_ns"),
                         gamma_kind = rd("interactions", "gamma_kind"),
                         proc = rd("interactions", "proc")),
         pes = (offset = rd("pes", "offset"), channel = rd("pes", "channel"),
                t_ns = rd("pes", "t_ns"), lambda_nm = rd("pes", "lambda_nm")),
         attributes = Dict(k => a[k] for k in keys(a)))
    end
end

"Records of event k (1-based) as views: (channel, t_ns, lambda_nm)."
function event_pes(r::NamedTuple, k::Integer)
    lo, hi = r.pes.offset[k] + 1, r.pes.offset[k + 1]
    return (channel = view(r.pes.channel, lo:hi), t_ns = view(r.pes.t_ns, lo:hi),
            lambda_nm = view(r.pes.lambda_nm, lo:hi))
end

"Git description of the working tree, or 'unknown' outside a repository."
function code_version()
    try
        return readchomp(`git -C $(dirname(@__DIR__)) describe --always --dirty --tags`)
    catch
        return "unknown"
    end
end
