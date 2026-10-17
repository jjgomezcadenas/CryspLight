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

"""
    update_table(campaign, study, name, columns; key = "tag")

Write the rows of `columns` into <name>.csv, replacing the rows of the existing file whose
`key` equals one of the new keys and keeping the others; columns are matched by name, and
a column missing on one side is left empty there. Lets several scripts runs (one per tag)
build one campaign table.
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
