# Output of the scripts: fullsim_analysis/results/<campaign>/<study>/ with one HDF5 file
# for arrays and histograms and CSV files for tables. The Python plotters read these.

const RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))

"""
    results_dir(campaign, study) -> String

The directory fullsim_analysis/results/<campaign>/<study>/ of a study, created if needed.
"""
function results_dir(campaign::AbstractString, study::AbstractString)
    d = joinpath(RESULTS_ROOT, campaign, study)
    mkpath(d)
    return d
end

"""
    open_results(f, campaign, study; mode = "w") -> String

Open the HDF5 file <study>.h5 in the study directory (created, or overwritten, with the
default mode "w") and run f on it, as in a do block. Returns the path of the file.
"""
function open_results(f::Function, campaign::AbstractString, study::AbstractString;
                      mode::AbstractString = "w")
    path = joinpath(results_dir(campaign, study), study * ".h5")
    h5open(f, path, mode)
    return path
end

"""
    stamp!(g, pairs...) -> g

Provenance attributes on an HDF5 group or file: code_version, the UTC date of writing,
and one attribute per name => value pair given.
"""
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

"""
    write_hist!(g, name, edges, counts) -> g

Write a histogram into the HDF5 group g as two datasets, <name>_edges (Float64) and
<name>_counts (Int64).
"""
function write_hist!(g, name::AbstractString, edges::AbstractVector, counts::AbstractVector)
    g[name * "_edges"] = Float64.(collect(edges))
    g[name * "_counts"] = Int64.(collect(counts))
    return g
end

"""
    write_vector!(g, name, v) -> g

Write the vector v into the HDF5 group g as the dataset name: Int64 for an integer
vector, Float64 otherwise.
"""
write_vector!(g, name::AbstractString, v::AbstractVector) =
    (g[name] = eltype(v) <: Integer ? Int64.(collect(v)) : Float64.(collect(v)); g)

"""
    write_matrix!(g, name, m) -> g

Write the matrix m (channel maps, 8 x 8) into the HDF5 group g as the Float64 dataset
name.
"""
write_matrix!(g, name::AbstractString, m::AbstractMatrix) = (g[name] = Float64.(collect(m)); g)

"""
    write_table(campaign, study, name, columns) -> String

Write the CSV table <name>.csv in the study directory and return its path; columns is a
vector of name => vector pairs of equal length. Numbers are written with full precision,
non-finite values as empty, Bool as 0/1 and anything else as a quoted string.
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

"""
    update_table(campaign, study, name, columns; key = "tag") -> String

Write the rows of `columns` into <name>.csv, replacing the rows of the existing file whose
`key` equals one of the new keys and keeping the others; columns are matched by name, and
a column missing on one side is left empty there. Lets several script runs (one per tag)
build one campaign table. Returns the path of the file.
"""
function update_table(campaign::AbstractString, study::AbstractString, name::AbstractString,
                      columns::Vector; key::AbstractString = "tag")
    path = joinpath(results_dir(campaign, study), name * ".csv")
    newnames = [c.first for c in columns]
    newkeys = Set(string.(columns[findfirst(==(key), newnames)].second))
    old_header = String[]; old_rows = Vector{Vector{String}}()
    if isfile(path)
        lines = filter(!isempty, readlines(path))
        old_header = split(lines[1], ",") .|> String
        for l in lines[2:end]
            f = split(l, ","; keepempty = true) .|> String
            k = strip(f[findfirst(==(key), old_header)], '"')
            k in newkeys || push!(old_rows, f)
        end
    end
    header = vcat(newnames, [h for h in old_header if !(h in newnames)])
    n = length(columns[1].second)
    open(path, "w") do io
        println(io, join(header, ","))
        for i in 1:n
            println(io, join((j <= length(columns) ? _cell(columns[j].second[i]) : "" for j in eachindex(header)), ","))
        end
        for f in old_rows
            println(io, join((begin k = findfirst(==(h), old_header); k === nothing ? "" : f[k] end for h in header), ","))
        end
    end
    return path
end
