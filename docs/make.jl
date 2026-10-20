# API reference of CryspLight, built locally (no deployment):
#
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
#   julia --project=docs docs/make.jl
#
# The HTML is written to docs/build/ (not tracked); open docs/build/index.html.

using Documenter
using CryspLight

# The analysis library is not a package: it is loaded by include, as the scripts do.
include(joinpath(@__DIR__, "..", "fullsim_analysis", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis

# checkdocs = :exports makes Documenter fail when a docstring of an exported name is left
# out of the pages, but it does not see exported names that have no docstring at all.
# Those are caught here, before the build; test/runtests.jl makes the same check for
# CryspLight. An exported name that is not defined in the module is an error too.
for mod in (CryspLight, FullsimAnalysis)
    exported = filter(n -> n !== nameof(mod), names(mod))
    undefined = filter(n -> !isdefined(mod, n), exported)
    isempty(undefined) ||
        error("exported names of $mod that are not defined: $(join(undefined, ", "))")
    undocumented = filter(n -> isdefined(mod, n) && !Base.Docs.hasdoc(mod, n), exported)
    isempty(undocumented) ||
        error("exported names of $mod without a docstring: $(join(undocumented, ", "))")
end

makedocs(;
    sitename = "CryspLight",
    authors = "J.J. Gómez-Cadenas",
    modules = [CryspLight, FullsimAnalysis],
    format = Documenter.HTML(; prettyurls = false, repolink = nothing, edit_link = nothing),
    remotes = nothing,
    checkdocs = :exports,
    doctest = true,
    warnonly = false,
    pages = [
        "Home" => "index.md",
        "Optical transport" => "optical.md",
        "Materials and configuration" => "config.md",
        "Gammas and events" => "events.md",
        "Photoelectron file and output" => "output.md",
        "Electronics" => "electronics.md",
        "Vendored gamma core" => "gamma.md",
        "Analysis library" => "analysis.md",
    ],
)
