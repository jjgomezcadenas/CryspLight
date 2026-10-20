# Material data: one TOML file per material under data/, selected by name from the run
# config. v0 is scalar (fixed wavelength); tabulated n(λ), Λ(λ), R(λ), PDE(λ) replace the
# scalars later without changing the interface.

const DATA_DIR = joinpath(dirname(@__DIR__), "data")

"""
    load_material(name; data_dir = DATA_DIR) -> Dict{String,Any}

The material file data_dir/<name>.toml (by default the data/ directory of the
repository) parsed as a Dict, with its values in the units of the file (lengths in mm,
times in ns, wavelengths in nm). An error when the file does not exist.
"""
function load_material(name::AbstractString; data_dir::AbstractString = DATA_DIR)
    path = joinpath(data_dir, name * ".toml")
    isfile(path) || error("material file not found: $path")
    return TOML.parsefile(path)
end

"""
    scintillation_times(mat::AbstractDict) -> (tau_ns, tau2_ns, frac1)

Scintillation time constants of a material Dict (load_material) as the keyword arguments
of the transport, a NamedTuple of Float32 with the times in ns. A file with only tau_ns
is a single component (tau2_ns = 0, frac1 = 1); one with tau2_ns and frac1 is a
two-component emission, frac1 being the weight of tau_ns. tau2_ns without frac1 is an
error.
"""
function scintillation_times(mat::AbstractDict)
    tau1 = Float32(mat["tau_ns"])
    haskey(mat, "tau2_ns") || return (tau_ns = tau1, tau2_ns = 0f0, frac1 = 1f0)
    haskey(mat, "frac1") || error("material $(mat["name"]): tau2_ns needs frac1")
    return (tau_ns = tau1, tau2_ns = Float32(mat["tau2_ns"]), frac1 = Float32(mat["frac1"]))
end
