# Material data: one TOML file per material under data/, selected by name from the run
# config. v0 is scalar (fixed wavelength); tabulated n(λ), Λ(λ), R(λ), PDE(λ) replace the
# scalars later without changing the interface.

const DATA_DIR = joinpath(dirname(@__DIR__), "data")

"Load data/<name>.toml as a Dict."
function load_material(name::AbstractString; data_dir::AbstractString = DATA_DIR)
    path = joinpath(data_dir, name * ".toml")
    isfile(path) || error("material file not found: $path")
    return TOML.parsefile(path)
end
