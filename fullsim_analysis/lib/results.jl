# Output of the scripts: fullsim_analysis/results/<campaign>/<study>/ with one HDF5 file
# for arrays and histograms and CSV files for tables. The Python plotters read these.

const RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))

"Directory of a study, created if needed."
function results_dir(campaign::AbstractString, study::AbstractString)
    d = joinpath(RESULTS_ROOT, campaign, study)
    mkpath(d)
    return d
end

"Open (create) the HDF5 file <study>.h5 of a study and run f on it."
function open_results(f::Function, campaign::AbstractString, study::AbstractString;
                      mode::AbstractString = "w")
    path = joinpath(results_dir(campaign, study), study * ".h5")
    h5open(f, path, mode)
    return path
end

"Provenance attributes on a group or file: code version, date, and any pairs given."
function stamp!(g, pairs::Pair...)
    a = attrs(g)
    a["code_version"] = code_version()
    a["written"] = string(now_utc())
    for (k, v) in pairs
        a[String(k)] = v
    end
    return g
end

now_utc() = Libc.strftime("%Y-%m-%dT%H:%M:%SZ", time())

"A histogram as two datasets, <name>_edges and <name>_counts."
function write_hist!(g, name::AbstractString, edges::AbstractVector, counts::AbstractVector)
    g[name * "_edges"] = Float64.(collect(edges))
    g[name * "_counts"] = Int64.(collect(counts))
    return g
end

"A vector dataset, Float64 unless integer."
write_vector!(g, name::AbstractString, v::AbstractVector) =
    (g[name] = eltype(v) <: Integer ? Int64.(collect(v)) : Float64.(collect(v)); g)

"A matrix dataset (channel maps, 8 x 8)."
write_matrix!(g, name::AbstractString, m::AbstractMatrix) = (g[name] = Float64.(collect(m)); g)

"""
    write_table(campaign, study, name, columns)

CSV table <name>.csv in the study directory; columns is a vector of name => vector pairs
of equal length. Numbers are written with full precision, NaN as empty.
"""
function write_table(campaign::AbstractString, study::AbstractString, name::AbstractString,
                     columns::Vector)
    path = joinpath(results_dir(campaign, study), name * ".csv")
    n = length(columns[1].second)
    all(length(c.second) == n for c in columns) || error("columns of unequal length")
    open(path, "w") do io
        println(io, join((c.first for c in columns), ","))
        for k in 1:n
            println(io, join((_cell(c.second[k]) for c in columns), ","))
        end
    end
    return path
end

_cell(v::AbstractFloat) = isfinite(v) ? string(v) : ""
_cell(v::Integer) = string(v)
_cell(v::Bool) = v ? "1" : "0"
_cell(v) = "\"" * replace(string(v), "\"" => "\"\"") * "\""
