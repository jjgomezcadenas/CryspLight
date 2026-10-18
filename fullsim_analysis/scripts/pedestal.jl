# Pedestal of the peak detectors of an ASIC front end (Citiroc, Radioroc2): the value the
# chain stores in each channel when the SiPMs see no scintillation light, from decays
# with no energy deposit read with an external trigger at the decay time (the emulation
# of a random-trigger run). The peak detector keeps the largest dark pulse, crosstalk
# group or noise excursion within the hold, so the pedestal is positive; the digitizer,
# whose charge is baseline subtracted, has none. Per channel, in avalanches: the energy
# path (the stored value of the Citiroc, the low gain of the Radioroc2), the high gain
# and the low gain, each calibrated as the analysis calibrates the event file (pedestal
# and counts per avalanche of /calib).
# Writes results/<campaign>/pedestal/pedestal.h5 (one group per front-end tag: mean and
# rms per channel) and pedestal.csv (one row per tag).
#   julia --project=. --threads=8 fullsim_analysis/scripts/pedestal.jl <campaign> <frontend config> [n_empty]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, HDF5, Statistics, Printf

campaign, cfg = ARGS[1], ARGS[2]
nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2000
r = open_pes_run(cfg)
st, pes, op = r.setup, r.pes, r.op
kind, s, tag = st.frontend, st.settings, st.tag
kind in ("citiroc", "radioroc") || error("$(tag): pedestals are defined for the ASIC front ends only")
empty = event_lists(pes; nmax = nmax).empty
t = @elapsed res = emulate(pes, op, kind, with(s; trigger = "external", external_trigger_ns = 0.0), empty)
all(res.triggered) || error("external trigger missed events")

nch = size(res.stored, 1)
gf = op.gain_factor
hg = (Float64.(res.hg) .- s.pedestal) ./ (s.adc_per_pe .* gf)
lg = (Float64.(res.lg) .- s.pedestal) ./ (s.adc_per_pe * s.lg_ratio .* gf)
energy = kind == "radioroc" ? lg : Float64.(res.stored)
m(x) = vec(mean(x; dims = 2)); sd(x) = vec(std(x; dims = 2))

open_results(campaign, "pedestal"; mode = isfile(joinpath(results_dir(campaign, "pedestal"), "pedestal.h5")) ? "r+" : "w") do h5
    haskey(h5, tag) && delete_object(h5, tag)
    g = create_group(h5, tag)
    stamp!(g, "config" => st.cfg_text, "frontend" => kind, "n_empty" => length(empty),
           "energy_path" => kind == "radioroc" ? "low gain" : "stored value (high gain)")
    write_vector!(g, "energy_mean", m(energy)); write_vector!(g, "energy_rms", sd(energy))
    write_vector!(g, "hg_mean", m(hg)); write_vector!(g, "hg_rms", sd(hg))
    write_vector!(g, "lg_mean", m(lg)); write_vector!(g, "lg_rms", sd(lg))
    tot = vec(sum(energy; dims = 1))
    write_hist!(g, "energy_sum", histogram(tot, minimum(tot) - 1, maximum(tot) + 1, 100)...)
end
tot = vec(sum(energy; dims = 1))
@printf("%-32s %-8s %d empty decays (%.0f s): energy path per channel %.2f (rms %.2f), sum of 64 %.1f (rms %.1f); high gain %.2f, low gain %.2f per channel\n",
        tag, kind, length(empty), t, mean(energy), mean(sd(energy)), mean(tot), std(tot), mean(hg), mean(lg))
println("written ", update_table(campaign, "pedestal", "pedestal",
    ["tag" => [tag], "frontend" => [kind], "n_empty" => [length(empty)],
     "energy_per_channel" => [mean(energy)], "energy_rms_per_channel" => [mean(sd(energy))],
     "energy_sum_mean" => [mean(tot)], "energy_sum_rms" => [std(tot)],
     "hg_per_channel" => [mean(hg)], "lg_per_channel" => [mean(lg)]]))
