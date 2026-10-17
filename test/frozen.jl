# Frozen output of the pedagogical path (full-simulation working plan, task 0.4).
#
# Three small deterministic runs of the code as it was at the tag `pedsim`, on the CPU
# backend, are reduced to SHA-256 digests of every output array and counter, and compared
# with the digests stored in test/frozen_checksums.toml. Any extension of the engine that
# is switched off in these configurations must leave every digest unchanged; a change in
# a digest means the old path no longer gives the old answer with the same random
# numbers.
#
# The cases:
#   point_reference  the point-source run of runs/point_bgo_teflon.toml, through the
#                    configuration reader and the CPU reference transport (run_photons!)
#   point_kernel     the same photons through the KernelAbstractions kernel on the CPU
#   gammas_bgo       the per-event gamma pipeline of runs/gammas_bgo.toml (run_events!)
#
# To regenerate the digests after a deliberate change of the pedagogical path:
#   julia --project=. --threads=8 test/frozen.jl --freeze
# and commit test/frozen_checksums.toml with the change that justified it.

using CryspLight
using CryspLight: load_setup
using SHA
using TOML

const FROZEN_FILE = joinpath(@__DIR__, "frozen_checksums.toml")
const REPO = dirname(@__DIR__)

digest(A::AbstractArray) = bytes2hex(sha256(reinterpret(UInt8, vec(Array(A)))))
digest(x) = bytes2hex(sha256(string(x)))

"Digests of an Accumulator: the two arrays and the counters as one tuple."
function digests(acc::Accumulator)
    Dict("counts" => digest(acc.counts),
         "first_ns" => digest(acc.first_ns),
         "counters" => digest((acc.ndet, acc.ndet_front, acc.nabs_bulk, acc.nabs_wall,
                               acc.nabs_sipm, acc.ncap, acc.nabs_sur, acc.nscat,
                               acc.sum_bounces_det)))
end

"Digests of the NamedTuple returned by run_events!, one per field."
digests(r::NamedTuple) = Dict(String(k) => digest(getfield(r, k)) for k in keys(r))

function case_point(kernel::Bool)
    st = load_setup(joinpath(REPO, "runs", "point_bgo_teflon.toml"))
    kw = (n_photons = 20_000, seed = st.seed, pos = st.pos, tau_ns = st.tau_ns,
          max_bounces = st.max_bounces)
    acc = kernel ? run_photons_ka!(st.box, st.op, Readout(st.grid), st.tb; kw...,
                                   ArrayT = Array) :
                   run_photons!(st.box, st.op, st.grid, st.tb; kw...)
    return digests(acc)
end

function case_gammas()
    cfgpath = joinpath(REPO, "runs", "gammas_bgo.toml")
    cfg = TOML.parsefile(cfgpath)
    mat = load_material(cfg["crystal"]["material"])
    wrap = load_material(cfg["wrap"]["material"])
    sipm = load_material(cfg["sipm"]["material"])
    sz = Float32.(cfg["crystal"]["size_mm"])
    box = Box((sz[1], sz[2], sz[3]))
    surf = cfg["surface"]
    op = OpticalParams(Float32(mat["n"]), Float32(cfg["sipm"]["coupling_n"]),
                       Float32(mat["abs_length_mm"]),
                       Float32(get(mat, "rayleigh_mm", Inf)),
                       Float32(wrap["reflectivity"]), wrap["model"] == "specular",
                       surface_has_gap(get(surf, "finish", "backpainted")),
                       Float32(deg2rad(surf["sigma_alpha_deg"])),
                       Float32(get(cfg["sipm"], "pde_override", sipm["pde"])))
    grid = SipmGrid(Float32(sipm["pitch_mm"]), Int32(sipm["nx"]), Int32(sipm["ny"]))
    tb = TimeBinning(Float32(cfg["binning"]["bin_ns"]),
                     Int32(round(cfg["binning"]["window_ns"] / cfg["binning"]["bin_ns"])))
    gmat = Gamma.make_material(mat["name"], cfg["gamma"]["density"],
                               joinpath(REPO, "data", cfg["gamma"]["xcom"]))
    pv = gamma_crystal(box.L, gmat)
    r = run_events!(box, op, Readout(grid), tb, pv, mat["yield_per_mev"];
                    n_events = 30, seed = Int(cfg["campaign"]["seed"]),
                    tau_ns = Float32(mat["tau_ns"]), batch_photons = 100_000,
                    ArrayT = Array)
    return digests(r)
end

frozen_cases() = Dict("point_reference" => case_point(false),
                      "point_kernel" => case_point(true),
                      "gammas_bgo" => case_gammas())

"Write the digests of the current code to the checksum file."
function freeze!(path::String = FROZEN_FILE)
    open(path, "w") do io
        println(io, "# SHA-256 digests of the pedagogical path; see test/frozen.jl.")
        println(io, "# Regenerate only with a deliberate change of that path:")
        println(io, "#   julia --project=. --threads=8 test/frozen.jl --freeze")
        TOML.print(io, frozen_cases())
    end
    return path
end

"Read the stored digests; nothing if the file does not exist."
read_frozen(path::String = FROZEN_FILE) = isfile(path) ? TOML.parsefile(path) : nothing

if abspath(PROGRAM_FILE) == (@__FILE__) && "--freeze" in ARGS
    println("frozen digests written to ", freeze!())
end
