# Numbers and distributions of the study at room temperature (docs_fullsim/na22/room_temperature_study.tex):
# BGO and CsI(Tl) read by the Citiroc 1A, the DT2745 digitiser (gain 4, 5 V) and the
# Radioroc2, six cases on the same decays of the two optical runs. For each case:
#   energy      the sum of the 64 channels in photoelectrons by the route of the
#               measurement (ADC values minus the zero level, over the gain, both from
#               the calibration on the dark counts; lib/calibrate.jl), calibrated in keV with the
#               511 and 1275 keV peaks; its spectrum and the Gaussian-plus-linear fit of
#               the 511 keV peak, with a bootstrap standard deviation of the resolution
#               that refits the two calibration peaks and the photopeak on every resample;
#   light       on the events in which one 511 keV gamma deposits its full energy, the
#               photons emitted, the photons reaching the SiPMs, the photoelectrons and
#               the fired cells (photoelectrons with their crosstalk and afterpulses),
#               from a replay of the SiPM response without dark counts, and the recorded
#               charge in photoelectron equivalents;
#   resolution  the relative width of each of these quantities on the same events (half
#               the 16-84% interval over the median, times 2.355), whose differences in
#               quadrature attribute the width to the light collection and detection,
#               to the SiPM (crosstalk and afterpulses) and to the readout (dark counts,
#               noise, shaping, zero level). The signed squared differences are kept, and a
#               paired bootstrap over the events gives each term a 95% interval; a term whose
#               interval includes zero is reported as an upper limit, not as a contribution;
#   channels    the brightest and the least illuminated channel of each of those events.
# Writes results/<campaign>/<set>/<set>.h5 (one group per case) and <set>.csv, where <set> is
# room_temperature (the six cases above, the default) or bgo_250k (BGO at 250 K read by the
# digitiser and the Radioroc2).
#   julia --project=. --threads=8 fullsim_analysis/scripts/room_temperature_study.jl <campaign> [n_replay] [set] [saved]
# With the word saved among the arguments the script does not replay the SiPM response: it
# reads the per-event arrays of the existing <set>.h5 (photoelectrons, fired cells, recorded
# energy) and the event files, recomputes the resolution and the decomposition with their
# bootstrap intervals, adds them to the groups of the HDF5 file and to the columns of the CSV
# table, and leaves every other column as it was.
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, HDF5, Statistics, Printf, TOML
using Random: Xoshiro

const ROOM = [
    ("bgo_citiroc", "fullsim_runs/2026-10_ambient/frontend_bgo_citiroc.toml"),
    ("bgo_digitiser", "fullsim_runs/2026-10_ambient/frontend_bgo_digitizer_g4.toml"),
    ("bgo_radioroc", "fullsim_runs/2026-xx_radioroc/frontend_bgo_ambient_radioroc.toml"),
    ("csitl_citiroc", "fullsim_runs/2026-10_ambient/frontend_csitl_citiroc.toml"),
    ("csitl_digitiser", "fullsim_runs/2026-10_ambient/frontend_csitl_digitizer_g4.toml"),
    ("csitl_radioroc", "fullsim_runs/2026-xx_radioroc/frontend_csitl_ambient_radioroc.toml"),
]
const BGO250 = [
    ("bgo250_digitiser", "fullsim_runs/2026-10_bgo250k/frontend_bgo_250k_digitizer_g4.toml"),
    ("bgo250_radioroc", "fullsim_runs/2026-10_bgo250k/frontend_bgo_250k_radioroc.toml"),
]
const SETS = Dict("room_temperature" => ROOM, "bgo_250k" => BGO250)
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const NBOOT_TERMS = 1000      # paired bootstrap of the decomposition
const NBOOT_FWHM = 200        # bootstrap of the resolution with the full calibration refitted

saved = "saved" in ARGS
args = filter(!=("saved"), ARGS)
campaign = args[1]
nreplay = length(args) >= 2 ? parse(Int, args[2]) : 3000
set = length(args) >= 3 ? args[3] : "room_temperature"
const CASES = SETS[set]
const TERMS = ("statistics", "sipm", "readout")

"Relative width in FWHM units: half the 16-84% interval over the median, times 2.355."
relwidth(x) = 2.355 * (quantile(x, 0.84) - quantile(x, 0.16)) / 2 / median(x)

"""
Sum over channels of the zero level of the low gain above the electronic pedestal, in
photoelectrons: what the peak detectors store from the dark pulses and the noise (zero
for the digitiser).
"""
function pedestal_sum(e, cal)
    e.frontend == "digitizer" && return 0.0
    base = Float64(e.run["fe_pedestal"])
    return sum((cal.lg_pedestal .- base) ./ (cal.gain .* cal.lg_over_hg))
end

"The energy scale and the 511 keV fit of a vector of channel sums: (fwhm, f511, p511, p1275, slope)."
function energy_fit(tot)
    p511 = find_peak511(tot); p1275 = find_peak1275(tot, p511)
    slope = (p1275.mean - p511.mean) / (1275 - 511)
    ekev = 511 .+ (tot .- p511.mean) ./ slope
    f511 = peak_fit(ekev, 400.0, 620.0)
    return (fwhm = 2.355 * f511.sigma / f511.mean, f511 = f511, p511 = p511, p1275 = p1275,
            slope = slope, ekev = ekev)
end

"""
Bootstrap of the energy resolution: nboot resamples of the channel sums of the triggered
events, the two calibration peaks and the 511 keV peak refitted on each. Returns the
standard deviation and the 2.5 and 97.5% quantiles of the resolution, and the number of
resamples whose fits failed (left out).
"""
function resolution_bootstrap(tot; nboot::Int = NBOOT_FWHM, seed::Integer = 2026)
    rng = Xoshiro(seed)
    vals = Float64[]
    n = length(tot)
    for _ in 1:nboot
        f = energy_fit(tot[rand(rng, 1:n, n)])
        isfinite(f.fwhm) && push!(vals, f.fwhm)
    end
    return (sd = std(vals), lo = quantile(vals, 0.025), hi = quantile(vals, 0.975),
            failed = nboot - length(vals))
end

"""
Paired bootstrap of the decomposition of the width: nboot resamples of the photopeak
events, the same events taken in the three arrays, and on each the three signed squared
differences (photoelectrons minus intrinsic, fired cells minus photoelectrons, recorded
minus fired cells). Returns per term the signed squared difference of the sample, the 2.5
and 97.5% quantiles of its bootstrap distribution and the fraction of negative resamples,
and the resampled values (nboot x 3).
"""
function decomposition_bootstrap(npe, ncell, recs, intrinsic; nboot::Int = NBOOT_TERMS, seed::Integer = 71)
    sq(a, b, c) = (relwidth(a)^2 - intrinsic^2, relwidth(b)^2 - relwidth(a)^2, relwidth(c)^2 - relwidth(b)^2)
    n = length(npe)
    rng = Xoshiro(seed)
    boot = zeros(nboot, 3)
    for b in 1:nboot
        ix = rand(rng, 1:n, n)
        boot[b, :] .= sq(npe[ix], ncell[ix], recs[ix])
    end
    point = sq(npe, ncell, recs)
    terms = [(sq = point[k], lo = quantile(boot[:, k], 0.025), hi = quantile(boot[:, k], 0.975),
              fneg = mean(boot[:, k] .< 0)) for k in 1:3]
    return (terms = terms, boot = boot)
end

"""
Add to the row and to the HDF5 group the resolution with its bootstrap and the three terms
with their signed squares, 95% intervals (in width units, the lower end zero when the
interval includes zero: an upper limit) and fractions of negative resamples.
"""
function bootstrap_row!(row, g, tot, npe, ncell, recs, intrinsic)
    rb = resolution_bootstrap(tot)
    row["fwhm511_boot_sd"] = rb.sd; row["fwhm511_boot_lo"] = rb.lo; row["fwhm511_boot_hi"] = rb.hi
    row["fwhm511_boot_failed"] = rb.failed
    db = decomposition_bootstrap(npe, ncell, recs, intrinsic)
    for (k, name) in enumerate(TERMS)
        t = db.terms[k]
        row["term_$(name)"] = sqrt(max(t.sq, 0.0))
        row["term_$(name)_sq"] = t.sq
        row["term_$(name)_lo"] = sqrt(max(t.lo, 0.0))
        row["term_$(name)_hi"] = sqrt(max(t.hi, 0.0))
        row["term_$(name)_fneg"] = t.fneg
        row["term_$(name)_upper_limit"] = t.lo <= 0
    end
    a = attrs(g)
    for k in ("fwhm511_boot_sd", "fwhm511_boot_lo", "fwhm511_boot_hi")
        a[k] = row[k]
    end
    haskey(g, "bootstrap_terms_sq") && delete_object(g, "bootstrap_terms_sq")
    g["bootstrap_terms_sq"] = db.boot
    attrs(g["bootstrap_terms_sq"])["columns"] = "squared width differences: statistics (pe - intrinsic), sipm (cells - pe), readout (recorded - cells); one row per resample"
    return row
end

"One line per term: the point value, the interval and whether it is an upper limit."
function print_terms(case, row)
    @printf("%-16s FWHM %.2f%% (bootstrap sd %.2f, 95%% %.2f to %.2f)", case, 100row["fwhm511"],
            100row["fwhm511_boot_sd"], 100row["fwhm511_boot_lo"], 100row["fwhm511_boot_hi"])
    for name in TERMS
        sq, lo, hi = row["term_$(name)_sq"], row["term_$(name)_lo"], row["term_$(name)_hi"]
        if row["term_$(name)_upper_limit"]
            @printf("; %s %+.2f%%^2 -> below %.2f%% (%.0f%% of resamples negative)", name, 1e4sq, 100hi, 100row["term_$(name)_fneg"])
        else
            @printf("; %s %.2f%% (%.2f to %.2f)", name, 100sqrt(sq), 100lo, 100hi)
        end
    end
    println()
end

"Read a CSV table written by write_table into name => column (strings; empty for missing)."
function read_csv_table(path)
    lines = filter(!isempty, readlines(path))
    header = String.(split(lines[1], ","))
    cols = Dict(h => String[] for h in header)
    for l in lines[2:end]
        f = String.(split(l, ","; keepempty = true))
        for (h, v) in zip(header, f)
            push!(cols[h], strip(v, '"'))
        end
    end
    return header, cols
end

cols = ["case", "tag", "frontend", "crystal", "overvoltage_v", "gain_electrons", "pde_peak", "crosstalk",
        "afterpulse", "dark_hz", "window_ns", "dark_in_window", "pedestal_sum", "failed_gain_fits", "keV_per_unit", "offset_kev",
        "ratio_1275_511", "fwhm511", "fwhm511_err", "fwhm511_boot_sd", "fwhm511_boot_lo", "fwhm511_boot_hi",
        "fwhm511_boot_failed", "peak1275_fwhm", "photopeak_events", "replayed",
        "photons_emitted", "photons_at_sipm", "photoelectrons", "fired_cells", "recorded_pe", "collection",
        "detection", "pe_per_emitted", "recorded_per_pe", "intrinsic", "width_pe", "width_cells",
        "width_recorded", "term_statistics", "term_sipm", "term_readout",
        [("term_$(t)_$(s)") for t in TERMS for s in ("sq", "lo", "hi", "fneg", "upper_limit")]...,
        "brightest_median", "brightest_p99", "least_median"]
bootcols = vcat(["fwhm511", "fwhm511_err", "fwhm511_boot_sd", "fwhm511_boot_lo", "fwhm511_boot_hi", "fwhm511_boot_failed",
                 "term_statistics", "term_sipm", "term_readout"],
                [("term_$(t)_$(s)") for t in TERMS for s in ("sq", "lo", "hi", "fneg", "upper_limit")])

rows = Dict{String,Any}[]
if saved
    # recompute the resolution and the decomposition from the saved arrays and the event files
    path = joinpath(results_dir(campaign, set), set * ".h5")
    isfile(path) || error("no saved results at $path")
    h5open(path, "r+") do h5
        for (case, cfg) in CASES
            haskey(h5, case) || (println("no group ", case, " in ", path); continue)
            st = load_frontend(joinpath(REPO, cfg))
            e = read_events(joinpath(st.outdir, "events.h5"))
            q, cal = measured_charge(e, campaign)
            tot = vec(sum(q[:, e.trig]; dims = 1))
            ef = energy_fit(tot)
            g = h5[case]
            npe = read(g["photoelectrons"]); ncell = read(g["fired_cells"]); recs = read(g["recorded_kev"])
            mat = TOML.parsefile(joinpath(REPO, "data", e.run["crystal"] * ".toml"))
            intrinsic = Float64(get(get(mat, "scintillation", mat), "intrinsic_fwhm_511", 0.05))
            row = Dict{String,Any}("case" => case, "fwhm511" => ef.fwhm,
                                   "fwhm511_err" => ef.fwhm / sqrt(2 * max(ef.f511.n, 1)))
            bootstrap_row!(row, g, tot, npe, ncell, recs, intrinsic)
            print_terms(case, row)
            push!(rows, row)
        end
    end
    # the CSV: the existing columns kept, the bootstrap columns added or replaced
    csv = joinpath(results_dir(campaign, set), set * ".csv")
    header, old = read_csv_table(csv)
    nrow = length(old["case"])
    for c in bootcols
        haskey(old, c) || (old[c] = fill("", nrow))
    end
    for row in rows
        i = findfirst(==(row["case"]), old["case"])
        i === nothing && error("case $(row["case"]) not in $csv")
        for c in bootcols
            v = row[c]
            old[c][i] = v isa Bool ? (v ? "1" : "0") : v isa AbstractFloat ? (isfinite(v) ? string(v) : "") : string(v)
        end
    end
    order = [c for c in cols if haskey(old, c)]
    append!(order, [h for h in header if !(h in order)])
    println("written ", write_table(campaign, set, set, [c => old[c] for c in order]))
else
    open_results(campaign, set; mode = "w") do h5
        for (case, cfg) in CASES
            st = load_frontend(joinpath(REPO, cfg))
            e = read_events(joinpath(st.outdir, "events.h5"))
            pes = read_pes(st.pes_path)
            op = st.op
            q, cal = measured_charge(e, campaign)
            tot = vec(sum(q[:, e.trig]; dims = 1))
            ef = energy_fit(tot)
            p511, p1275, slope, ekev, f511 = ef.p511, ef.p1275, ef.slope, ef.ekev, ef.f511
            fwhm = ef.fwhm
            dfwhm = fwhm / sqrt(2 * max(f511.n, 1))

            # the events of one fully absorbed 511 keV gamma, triggered
            pk = findall(select_photopeak511(e) .& e.trig)
            qpk = q[:, pk]
            rec = vec(sum(qpk; dims = 1))
            # replay of the SiPM response on the first nreplay of them, without dark counts
            sub = pk[1:min(nreplay, length(pk))]
            op_light = SipmOperation(op.dev, op.t_k, op.vov, op.vbr, op.gain, op.pde_peak, op.crosstalk,
                                     0.0, op.afterpulse, op.gain_factor, op.window_ns)
            npe = zeros(length(sub)); ncell = zeros(length(sub))
            Threads.@threads :static for i in eachindex(sub)
                tr = event_trains(op_light, event_pes(pes, sub[i]), e.nch, Xoshiro(hash((st.seed, sub[i], :light))))
                npe[i] = sum(count(a -> a.kind == 0, t) for t in tr)
                ncell[i] = sum(sum(Float64(a.q) for a in t; init = 0.0) for t in tr)
            end
            nemit = Float64.(e.mc.nemit[sub]); nph = Float64.(e.mc.nph[sub])
            # recorded charge on the calibrated energy scale: keV, and photoelectron equivalents
            # at 511 keV after the residual offset of the scale is removed
            recs = 511 .+ (vec(sum(q[:, sub]; dims = 1)) .- p511.mean) ./ slope
            mat = TOML.parsefile(joinpath(REPO, "data", e.run["crystal"] * ".toml"))
            intrinsic = Float64(get(get(mat, "scintillation", mat), "intrinsic_fwhm_511", 0.05))
            w_light, w_cell, w_rec = relwidth(npe), relwidth(ncell), relwidth(recs)
            s = st.settings
            window = e.frontend == "digitizer" ? s.gate_ns : s.hold_ns
            row = Dict{String,Any}(
                "case" => case, "tag" => e.tag, "frontend" => e.frontend, "crystal" => e.run["crystal"],
                "overvoltage_v" => op.vov, "gain_electrons" => op.gain, "pde_peak" => op.pde_peak,
                "crosstalk" => op.crosstalk, "afterpulse" => op.afterpulse, "dark_hz" => op.dark_hz,
                "window_ns" => window, "dark_in_window" => op.dark_hz * window * 1e-9 * e.nch,
                "pedestal_sum" => pedestal_sum(e, cal), "failed_gain_fits" => cal.failed_fits,
                "keV_per_unit" => 1 / slope, "offset_kev" => 511 - p511.mean / slope,
                "ratio_1275_511" => p1275.mean / p511.mean,
                "fwhm511" => fwhm, "fwhm511_err" => dfwhm, "peak1275_fwhm" => 2.355 * p1275.sigma / p1275.mean,
                "photopeak_events" => length(pk), "replayed" => length(sub),
                "photons_emitted" => mean(nemit), "photons_at_sipm" => mean(nph),
                "photoelectrons" => mean(npe), "fired_cells" => mean(ncell), "recorded_pe" => 511 * slope,
                "collection" => mean(nph) / mean(nemit), "detection" => mean(npe) / mean(nph),
                "pe_per_emitted" => mean(npe) / mean(nemit), "recorded_per_pe" => 511 * slope / mean(npe),
                "intrinsic" => intrinsic, "width_pe" => w_light, "width_cells" => w_cell, "width_recorded" => w_rec,
                "brightest_median" => median(brightest(qpk)), "brightest_p99" => quantile(brightest(qpk), 0.99),
                "least_median" => median(dimmest(qpk)))
            g = create_group(h5, case)
            stamp!(g, "config" => st.cfg_text, "frontend" => e.frontend, "tag" => e.tag)
            write_hist!(g, "spectrum_kev", histogram(ekev, 0.0, 2000.0, 400)...)
            attrs(g)["fit511_mean"] = f511.mean; attrs(g)["fit511_sigma"] = f511.sigma
            write_vector!(g, "photoelectrons", npe); write_vector!(g, "fired_cells", ncell)
            write_vector!(g, "recorded_kev", recs); write_vector!(g, "photons_emitted", nemit)
            write_vector!(g, "photons_at_sipm", nph)
            b = brightest(qpk); d = dimmest(qpk)
            write_hist!(g, "brightest", histogram(b, 0.0, quantile(b, 0.999), 100)...)
            write_hist!(g, "least", histogram(d, min(0.0, minimum(d)), quantile(d, 0.999), 80)...)
            bootstrap_row!(row, g, tot, npe, ncell, recs, intrinsic)
            push!(rows, row)
            @printf("%-16s FWHM %.2f%% ± %.2f (1275/511 %.3f, offset %.0f keV); emitted %.0f, at SiPMs %.0f, photoelectrons %.0f, cells %.0f, recorded %.0f; widths pe %.1f%%, cells %.1f%%, recorded %.1f%%\n",
                    case, 100fwhm, 100dfwhm, row["ratio_1275_511"], row["offset_kev"], mean(nemit), mean(nph),
                    mean(npe), mean(ncell), 511 * slope, 100w_light, 100w_cell, 100w_rec)
            print_terms(case, row)
            flush(stdout)
        end
    end
    println("written ", write_table(campaign, set, set, [c => [r[c] for r in rows] for c in cols]))
end
