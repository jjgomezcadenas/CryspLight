# Light-map figures from the per-event gamma outputs (tracked figure source: every
# committed image regenerates from this script + output/<tag>/events.h5).
#   julia --project=. scripts/plot_light_maps.jl output/gammas_csitl/events.h5 [more...]
# For each input file it writes to latex/figs/:
#   <tag>_maps.png  -- gallery: light maps by interaction type (photo/C1/C2) and depth
#   <tag>_doi.png   -- DOI handle: light-map RMS width vs true depth z1 (photoelectric)
using HDF5, Statistics, Printf
using CairoMakie

const PITCH = 6.0f0    # SiPM pitch [mm]; pixel centres at (i - 0.5) * PITCH

"Centroid and RMS radius [mm] of one nx x ny pe map."
function map_moments(m::AbstractMatrix)
    w = Float64.(m)
    s = sum(w)
    s == 0 && return (NaN, NaN, NaN)
    xs = [(i - 0.5) * PITCH for i in 1:size(m, 1), j in 1:size(m, 2)]
    ys = [(j - 0.5) * PITCH for i in 1:size(m, 1), j in 1:size(m, 2)]
    xc = sum(w .* xs) / s
    yc = sum(w .* ys) / s
    r2 = sum(w .* ((xs .- xc) .^ 2 .+ (ys .- yc) .^ 2)) / s
    return (xc, yc, sqrt(r2))
end

for path in (isempty(ARGS) ? ["output/gammas_csitl/events.h5",
                              "output/gammas_bgo/events.h5"] : ARGS)
    tag = h5open(f -> attrs(f)["tag"], path)
    maps, it, xyz1, e1, npe = h5open(path) do f
        read(f["maps"]), read(f["int_type"]), read(f["xyz1_mm"]),
        read(f["e1_kev"]), read(f["npe"])
    end
    Lz = maximum(xyz1[3, :])
    figdir = joinpath(dirname(dirname(abspath(path))), "..", "latex", "figs")
    mkpath(figdir)

    # ---- gallery: rows = interaction type, columns = shallow / mid / deep in z1 ----
    rows = [("photo", 0), ("C1", 1), ("C2", 2)]
    fig = Figure(size = (1050, 1000))
    for (ir, (lab, code)) in enumerate(rows)
        sel = findall(i -> it[i] == code && npe[i] > 0, eachindex(it))
        isempty(sel) && continue
        sel = sel[sortperm(xyz1[3, sel])]
        picks = [sel[max(1, round(Int, q * length(sel)))] for q in (0.1, 0.5, 0.9)]
        for (ic, ev) in enumerate(picks)
            ax = Axis(fig[ir, ic]; aspect = DataAspect(),
                      title = @sprintf("%s: z1 = %.1f mm, E1 = %.0f keV, %d pe",
                                       lab, xyz1[3, ev], e1[ev], npe[ev]),
                      titlesize = 13, xlabel = "SiPM column", ylabel = "SiPM row")
            heatmap!(ax, 1:8, 1:8, Float32.(maps[:, :, ev]); colormap = :viridis)
            scatter!(ax, [xyz1[1, ev] / PITCH + 0.5], [xyz1[2, ev] / PITCH + 0.5];
                     marker = :cross, color = :red, markersize = 14)
        end
    end
    Label(fig[0, :], "$tag: 8x8 light maps (pe after PDE); red cross = true (x1, y1)";
          fontsize = 16)
    save(joinpath(figdir, "$(tag)_maps.png"), fig)

    # ---- DOI handle: light-map RMS width vs true z1, photoelectric events ----
    sel = findall(i -> it[i] == 0 && npe[i] > 0, eachindex(it))
    widths = [map_moments(maps[:, :, i])[3] for i in sel]
    fig2 = Figure(size = (600, 450))
    ax = Axis(fig2[1, 1]; xlabel = "true depth z1 [mm]",
              ylabel = "light-map RMS radius [mm]",
              title = "$tag: photoelectric events (SiPMs at z = $(round(Lz, digits = 1)) mm side)")
    scatter!(ax, xyz1[3, sel], Float32.(widths); markersize = 4,
             color = (:steelblue, 0.5))
    save(joinpath(figdir, "$(tag)_doi.png"), fig2)
    println("$tag: $(length(sel)) photoelectric events -> ",
            joinpath(figdir, "$(tag)_maps.png"), " , ", "$(tag)_doi.png")
end
