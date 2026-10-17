# Trigger threshold scan of a front end on the photon file of a front-end configuration:
# for each threshold (and trigger mode) the fraction of 511 keV photopeak decays, of
# Compton decays (200 to 300 keV) and of decays without a deposit that trigger; the last
# one, divided by the window over which dark avalanches were generated, is the dark
# trigger rate. Writes results/<campaign>/trigger_scan/trigger_scan.csv (appending the
# rows of this configuration). Plot with fullsim_analysis/plots/trigger_scan.py.
#   julia --project=. --threads=8 fullsim_analysis/scripts/trigger_scan.jl <campaign> <frontend config> [n_events]
# The thresholds and modes scanned depend on the front end (table below).
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, Statistics, Printf

campaign, cfg = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1500
r = open_pes_run(cfg)
st, pes, op = r.setup, r.pes, r.op
lists = event_lists(pes; nmax = nmax)
tag = st.tag
window_ns = op.window_ns[2] - op.window_ns[1]
base = st.settings

# the scan: (mode, thresholds, settings constructor)
scans = if st.frontend == "citiroc"
    [("single", [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0],
      t -> with(base; threshold_pe = t, majority = 1)),
     ("majority2", [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0],
      t -> with(base; threshold_pe = t, majority = 2))]
elseif st.frontend == "digitizer"
    thr = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0, 250.0, 300.0, 400.0]
    [("majority2", thr, t -> with(base; threshold_counts = t, majority = 2)),
     ("single", thr, t -> with(base; threshold_counts = t, majority = 1))]
else
    [("tq", [10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0],
      t -> with(base; trigger = "tq", tq_threshold_pe = t)),
     ("t2", [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
      t -> with(base; trigger = "t2", t2_threshold_pe = t)),
     ("majority2_t1", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      t -> with(base; trigger = "majority", t1_threshold_pe = t, majority = 2))]
end

rows = (tag = String[], frontend = String[], mode = String[], threshold = Float64[],
        efficiency_511 = Float64[], efficiency_200_300 = Float64[], dark_per_window = Float64[],
        window_ns = Float64[], dark_khz = Float64[])
for (mode, thresholds, make) in scans, t in thresholds
    s = make(t)
    f(list) = mean(emulate(pes, op, st.frontend, s, list).triggered)
    e511, ec, ed = f(lists.photopeak511), f(lists.compton), f(lists.empty)
    push!(rows.tag, tag); push!(rows.frontend, st.frontend); push!(rows.mode, mode)
    push!(rows.threshold, t); push!(rows.efficiency_511, e511); push!(rows.efficiency_200_300, ec)
    push!(rows.dark_per_window, ed); push!(rows.window_ns, window_ns)
    push!(rows.dark_khz, ed / window_ns * 1e6)
    @printf("%-28s %-12s threshold %6.1f: 511 keV %.3f, 200-300 keV %.3f, dark per window %.3f (%.1f kHz)\n",
            tag, mode, t, e511, ec, ed, ed / window_ns * 1e6)
    flush(stdout)
end
# append to the campaign table, replacing the rows of this tag
path = joinpath(results_dir(campaign, "trigger_scan"), "trigger_scan.csv")
old = isfile(path) ? filter(l -> !startswith(l, "\"" * tag * "\"") && !isempty(l), readlines(path)[2:end]) : String[]
write_table(campaign, "trigger_scan", "trigger_scan", [String(k) => getfield(rows, k) for k in keys(rows)])
if !isempty(old)
    open(path, "a") do io
        foreach(l -> println(io, l), old)
    end
end
println("written ", path)
