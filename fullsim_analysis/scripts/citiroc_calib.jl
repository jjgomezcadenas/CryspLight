# The calibration of the Citiroc 1A with the periodic trigger of the board and the peak
# detector (peak stretch), as on the bench (note 3 of the bench programme), studied on the
# model where the true gain is known (docs_fullsim/calib/citiroc_calib.tex). The SiPMs
# are dark (no light, only their dark avalanches with crosstalk and afterpulses) at room
# temperature; the board opens windows of fixed length at fixed times, independent of the
# signal, and every channel stores the maximum of its high-gain slow signal over each
# window (hg_periodic). The photoelectron ladder of the spectrum, numbered from the
# pedestal (photoelectron_ladder), gives an apparent gain, whose ratio to the true gain of
# the channel is the correction factor k.
#   set A  one channel, the ladder against the peaking time of the high gain (25, 50,
#          87.5 ns), the overvoltage (3, 4, 5 V) and the noise (0 to 2 times that of the
#          model) at a window of 200 ns, and against the window (96 to 496 ns) at 25 ns
#          and 5 V;
#   set B  64 channels with a 5% gain spread, at 25 ns, 3 and 5 V: k channel by channel;
#   set E  one channel at 20, -3 and -23 C (293, 270 and 250 K), 5 V of overvoltage at each
#          temperature (the bias follows the breakdown voltage), 25, 50 and 87.5 ns, window
#          200 ns, noise of the model; the records are scaled with the dark rate to give about
#          40 000 windows with a dark avalanche per configuration; spectrum, k and the split
#          of set D;
#   set F  64 channels with a 5% gain spread in the three operating scenarios: 20 C at 25 ns,
#          -3 C and -23 C at 87.5 ns, 5 V of overvoltage, window 200 ns; records for about
#          20 000 windows with a dark avalanche per channel; k channel by channel;
#   set D  the spectrum of one channel split by the arrival of its avalanches relative to
#          the window (none within 400 ns before it or in it; only before it; only in it
#          with time to reach their maximum before it closes; only in it but too late for
#          that; a mixture), at 25 and 87.5 ns, 5 V, without and with the noise of the
#          model: the origin of each part of the spectrum.
# The statistical error of k is the spread of the fits on four disjoint quarters of the
# windows divided by two.
# Writes results/calib/citiroc/citiroc.h5 (one group per configuration: the spectrum in
# units of the true gain from the pedestal, the fitted peaks, the channel values of set B)
# and the tables citiroc_A.csv and citiroc_B.csv.
#   julia --project=. --threads=8 fullsim_analysis/scripts/citiroc_calib.jl [n_records]
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
using CryspLight, HDF5, Statistics, Printf
import Random
using Random: Xoshiro

const NREC = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000
const PEDESTAL = 200.0
const ADC_PER_PE = 10.0
const NOISE_MODEL = 0.1                   # rms of the high gain at 87.5 ns in the model, avalanches

"""
The spectrum of the periodic windows of one configuration: per channel, the stored high
gain in ADC counts (spread uniformly over each count) and the true gain.
"""
function ptrg_values(; tp, vov, noise_scale, window, nrec, nch = 1, gain_spread = 0.0, seed = 1,
                     records = 1:nrec, t_k = 293.15)
    op = SipmOperation(SipmDevice(); t_k = t_k, vov = vov, n_channels = nch, gain_spread = gain_spread,
                       seed = seed, window_ns = (-2000.0, 6000.0))
    s = CitirocSettings(tp_ns = tp, tp_lg_ns = 87.5, noise_pe = NOISE_MODEL * noise_scale, pedestal = PEDESTAL,
                        adc_per_pe = ADC_PER_PE, t_start_ns = -200.0, t_end_ns = 6000.0, dt_ns = 1.0)
    ch = CitirocChain(s, op, Xoshiro(seed))
    vals = [Float64[] for _ in 1:nch]
    for i in records
        Random.seed!(ch.rng, hash((seed, i, :noise)))
        rng = Xoshiro(hash((seed, i)))
        trains = [respond(op, Float32[], Float32[], rng) for _ in 1:nch]
        v = hg_periodic(ch, trains, window)
        for c in 1:nch
            append!(vals[c], v[c])
        end
    end
    jit = Xoshiro(hash((seed, :jitter, first(records))))
    for c in 1:nch
        vals[c] .+= rand(jit, length(vals[c])) .- 0.5
    end
    return (vals = vals, true_gain = ADC_PER_PE .* op.gain_factor, dark_rate = op.dark_hz,
            crosstalk = op.crosstalk, noise_tp = CryspLight.noise_at(s, tp))
end

"The ladder of one channel, with the statistical error of k from four quarters."
function ladder_k(x, g)
    lad = photoelectron_ladder(x, ADC_PER_PE; pedestal = PEDESTAL)
    q = [photoelectron_ladder(x[j:4:end], ADC_PER_PE; pedestal = PEDESTAL).gain / g for j in 1:4]
    q = filter(isfinite, q)
    err = length(q) >= 3 ? std(q) / 2 : NaN
    # the one-photoelectron peak alone, from the pedestal: its fitted position over the true
    # gain, with the error from the quarters
    p1(y) = peak_fit_window(y, PEDESTAL + 0.5ADC_PER_PE, PEDESTAL + 1.5ADC_PER_PE, 0.3ADC_PER_PE).mean
    k1 = (p1(x) - PEDESTAL) / g
    q1 = filter(isfinite, [(p1(x[j:4:end]) - PEDESTAL) / g for j in 1:4])
    return (lad = lad, k = lad.gain / g, k_err = err, zero_pe = (lad.zero - PEDESTAL) / g,
            f_signal = mean(x .> PEDESTAL + 0.5g), k1 = k1, k1_err = length(q1) >= 3 ? std(q1) / 2 : NaN)
end

const T_BEFORE = 400.0                    # ns before the window opening that count as "before"
const CATEGORIES = ("none", "before", "inside", "late", "mixed")

"""
The windows of one channel classified by the arrival times of its avalanches, with their
stored values: none in [t - T_BEFORE, t + window]; only before the opening; only inside
and early enough to reach the maximum (arrival <= t + window - t_peak, t_peak the time
from an avalanche to the maximum of its shaped signal); only inside but later; or a mixture.
"""
function ptrg_decomposed(; tp, vov, noise_scale, window, nrec, seed = 3, t_k = 293.15)
    op = SipmOperation(SipmDevice(); t_k = t_k, vov = vov, n_channels = 1, seed = seed,
                       window_ns = (-2000.0, 6000.0))
    s = CitirocSettings(tp_ns = tp, tp_lg_ns = 87.5, noise_pe = NOISE_MODEL * noise_scale, pedestal = PEDESTAL,
                        adc_per_pe = ADC_PER_PE, t_start_ns = -200.0, t_end_ns = 6000.0, dt_ns = 1.0)
    ch = CitirocChain(s, op, Xoshiro(seed))
    # time of the maximum of the shaped signal of one avalanche (no noise, no dark level)
    fill!(ch.x, 0.0); ch.x[sample_index(ch.grid, 0.0)] = 1.0
    copyto!(ch.y, ch.x); shape!(ch.y, ch.slow)
    t_peak = sample_time(ch.grid, argmax(ch.y))
    starts = CryspLight.periodic_windows(s, window, 2tp)
    out = Dict(c => Float64[] for c in CATEGORIES)
    for i in 1:nrec
        Random.seed!(ch.rng, hash((seed, i, :noise)))
        tr = respond(op, Float32[], Float32[], Xoshiro(hash((seed, i))))
        v = hg_periodic(ch, [tr], window)[1]
        ts = [Float64(a.t_ns) for a in tr]
        for (j, t) in enumerate(starts)
            nb = count(x -> t - T_BEFORE <= x < t, ts)
            ne = count(x -> t <= x <= t + window - t_peak, ts)
            nl = count(x -> t + window - t_peak < x <= t + window, ts)
            c = nb + ne + nl == 0 ? "none" : (ne + nl == 0) ? "before" : (nb + nl == 0) ? "inside" :
                (nb + ne == 0) ? "late" : "mixed"
            push!(out[c], v[j] + rand(ch.rng) - 0.5)
        end
    end
    return (vals = out, t_peak = t_peak, dark_rate = op.dark_hz)
end

configs_D = [(tp = tp, vov = 5.0, noise_scale = n, window = 200.0) for tp in (25.0, 87.5) for n in (0.0, 1.0)]

# set E: records for about 40 000 windows with a dark avalanche (windows per record: the
# record of 6200 ns over window + gap)
function nrec_E(t_k, tp; target = 40_000)
    r = SipmOperation(SipmDevice(); t_k = t_k, vov = 5.0, n_channels = 1).dark_hz
    return ceil(Int, target / ((1 - exp(-r * 200e-9)) * floor(6200 / (200 + 2tp))))
end
configs_F = [(tp = 25.0, vov = 5.0, noise_scale = 1.0, window = 200.0, t_k = 293.15),
             (tp = 87.5, vov = 5.0, noise_scale = 1.0, window = 200.0, t_k = 270.15),
             (tp = 87.5, vov = 5.0, noise_scale = 1.0, window = 200.0, t_k = 250.15)]
configs_E = [(tp = tp, vov = 5.0, noise_scale = 1.0, window = 200.0, t_k = tk)
             for tk in (293.15, 270.15, 250.15) for tp in (25.0, 50.0, 87.5)]
nameE(c) = @sprintf("T%.0f_tp%g", c.t_k, c.tp)

configs_A = vcat([(tp = tp, vov = v, noise_scale = n, window = 200.0)
                  for tp in (25.0, 50.0, 87.5) for v in (3.0, 4.0, 5.0) for n in (0.0, 0.5, 1.0, 1.5, 2.0)],
                 [(tp = 25.0, vov = 5.0, noise_scale = 1.0, window = w) for w in (96.0, 296.0, 400.0, 496.0)])
configs_B = [(tp = 25.0, vov = v, noise_scale = 1.0, window = 200.0) for v in (3.0, 5.0)]
name(c) = @sprintf("tp%g_v%g_n%g_w%g", c.tp, c.vov, c.noise_scale, c.window)

t0 = time()
res_A = Vector{Any}(undef, length(configs_A))
Threads.@threads :dynamic for i in eachindex(configs_A)
    cfg = configs_A[i]
    r = ptrg_values(; cfg..., nrec = NREC)
    res_A[i] = (cfg = cfg, r = r, fit = ladder_k(r.vals[1], r.true_gain[1]))
end
# set B: 64 channels; the records of each configuration are split in chunks over the threads
# (the random streams depend on the record, not on the chunk)
chunks = [(i, j) for i in eachindex(configs_B) for j in 1:8]
parts = Vector{Any}(undef, length(chunks))
Threads.@threads :dynamic for m in eachindex(chunks)
    i, j = chunks[m]
    parts[m] = ptrg_values(; configs_B[i]..., nrec = 2000, nch = 64, gain_spread = 0.05, seed = 2,
                           records = (250(j - 1) + 1):(250j))
end
res_B = map(eachindex(configs_B)) do i
    ps = [parts[m] for m in eachindex(chunks) if chunks[m][1] == i]
    vals = [reduce(vcat, [p.vals[c] for p in ps]) for c in 1:64]
    r = (ps[1]..., vals = vals)
    (cfg = configs_B[i], r = r, fits = [ladder_k(r.vals[c], r.true_gain[c]) for c in 1:64])
end
res_E = Vector{Any}(undef, length(configs_E))
Threads.@threads :dynamic for i in eachindex(configs_E)
    cfg = configs_E[i]
    n = nrec_E(cfg.t_k, cfg.tp)
    r = ptrg_values(; cfg..., nrec = n, seed = 4)
    d = ptrg_decomposed(; cfg..., nrec = n, seed = 4)
    res_E[i] = (cfg = cfg, nrec = n, r = r, fit = ladder_k(r.vals[1], r.true_gain[1]), d = d)
end
# set F: 64 channels, the records of each scenario split in 16 chunks over the threads
nchunk = 16
chunksF = [(i, j) for i in eachindex(configs_F) for j in 1:nchunk]
partsF = Vector{Any}(undef, length(chunksF))
Threads.@threads :dynamic for m in eachindex(chunksF)
    i, j = chunksF[m]
    n = nrec_E(configs_F[i].t_k, configs_F[i].tp; target = 20_000)
    lo = div((j - 1) * n, nchunk) + 1; hi = div(j * n, nchunk)
    partsF[m] = ptrg_values(; configs_F[i]..., nrec = n, nch = 64, gain_spread = 0.05, seed = 5, records = lo:hi)
end
res_F = map(eachindex(configs_F)) do i
    ps = [partsF[m] for m in eachindex(chunksF) if chunksF[m][1] == i]
    vals = [reduce(vcat, [p.vals[c] for p in ps]) for c in 1:64]
    r = (ps[1]..., vals = vals)
    (cfg = configs_F[i], nrec = nrec_E(configs_F[i].t_k, configs_F[i].tp; target = 20_000), r = r,
     fits = [ladder_k(r.vals[c], r.true_gain[c]) for c in 1:64])
end
partsF = nothing
res_D = Vector{Any}(undef, length(configs_D))
Threads.@threads :dynamic for i in eachindex(configs_D)
    res_D[i] = (cfg = configs_D[i], r = ptrg_decomposed(; configs_D[i]..., nrec = NREC))
end
@printf("%d configurations in %.0f s\n", length(configs_A) + length(configs_B) + length(configs_D) + length(configs_E) + length(configs_F), time() - t0)

open_results("calib", "citiroc") do h5
    stamp!(h5, "n_records" => NREC, "pedestal" => PEDESTAL, "adc_per_pe" => ADC_PER_PE,
           "noise_model_pe_87ns" => NOISE_MODEL)
    for x in res_A
        g = create_group(h5, "A_" * name(x.cfg))
        stamp!(g, "tp_ns" => x.cfg.tp, "vov" => x.cfg.vov, "noise_scale" => x.cfg.noise_scale,
               "window_ns" => x.cfg.window, "dark_rate_hz" => x.r.dark_rate, "crosstalk" => x.r.crosstalk)
        write_hist!(g, "spectrum_pe", histogram((x.r.vals[1] .- PEDESTAL) ./ x.r.true_gain[1], -0.5, 4.5, 500)...)
        write_vector!(g, "peaks_pe", (x.fit.lad.peaks .- PEDESTAL) ./ x.r.true_gain[1])
        write_vector!(g, "cells", Float64.(x.fit.lad.cells))
    end
    for x in res_B
        g = create_group(h5, "B_" * name(x.cfg))
        stamp!(g, "tp_ns" => x.cfg.tp, "vov" => x.cfg.vov, "window_ns" => x.cfg.window, "gain_spread" => 0.05)
        write_vector!(g, "true_gain", x.r.true_gain)
        write_vector!(g, "k", [f.k for f in x.fits])
        write_vector!(g, "k_err", [f.k_err for f in x.fits])
        write_vector!(g, "k1", [f.k1 for f in x.fits])
        write_vector!(g, "k1_err", [f.k1_err for f in x.fits])
        write_vector!(g, "zero_pe", [f.zero_pe for f in x.fits])
        allpe = reduce(vcat, [(x.r.vals[c] .- PEDESTAL) ./ x.r.true_gain[c] for c in 1:64])
        write_hist!(g, "spectrum_pe_all", histogram(allpe, -0.5, 4.5, 500)...)
    end
    for x in res_F
        g = create_group(h5, "F_" * nameE(x.cfg))
        stamp!(g, "tp_ns" => x.cfg.tp, "vov" => x.cfg.vov, "t_k" => x.cfg.t_k, "window_ns" => x.cfg.window,
               "records" => x.nrec, "gain_spread" => 0.05, "dark_rate_hz" => x.r.dark_rate)
        write_vector!(g, "true_gain", x.r.true_gain)
        for f in (:k, :k_err, :k1, :k1_err, :zero_pe)
            write_vector!(g, String(f), [getfield(y, f) for y in x.fits])
        end
        allpe = reduce(vcat, [(x.r.vals[c] .- PEDESTAL) ./ x.r.true_gain[c] for c in 1:64])
        write_hist!(g, "spectrum_pe_all", histogram(allpe, -0.5, 4.5, 500)...)
    end
    for x in res_E
        g = create_group(h5, "E_" * nameE(x.cfg))
        stamp!(g, "tp_ns" => x.cfg.tp, "vov" => x.cfg.vov, "t_k" => x.cfg.t_k, "window_ns" => x.cfg.window,
               "records" => x.nrec, "dark_rate_hz" => x.r.dark_rate, "t_peak_ns" => x.d.t_peak)
        write_hist!(g, "spectrum_pe", histogram((x.r.vals[1] .- PEDESTAL) ./ x.r.true_gain[1], -0.5, 4.5, 500)...)
        write_vector!(g, "peaks_pe", (x.fit.lad.peaks .- PEDESTAL) ./ x.r.true_gain[1])
        for c in CATEGORIES
            write_hist!(g, "spectrum_" * c, histogram((x.d.vals[c] .- PEDESTAL) ./ ADC_PER_PE, -0.5, 4.5, 500)...)
        end
    end
    for x in res_D
        g = create_group(h5, "D_" * name(x.cfg))
        stamp!(g, "tp_ns" => x.cfg.tp, "vov" => x.cfg.vov, "noise_scale" => x.cfg.noise_scale,
               "window_ns" => x.cfg.window, "t_peak_ns" => x.r.t_peak, "t_before_ns" => T_BEFORE,
               "dark_rate_hz" => x.r.dark_rate)
        for c in CATEGORIES
            write_hist!(g, "spectrum_" * c, histogram((x.r.vals[c] .- PEDESTAL) ./ ADC_PER_PE, -0.5, 4.5, 500)...)
        end
    end
end

col(f, xs) = [f(x) for x in xs]
pk(x, n) = (j = findfirst(==(n), x.fit.lad.cells); j === nothing ? NaN : (x.fit.lad.peaks[j] - PEDESTAL) / x.r.true_gain[1])
println("written ", write_table("calib", "citiroc", "citiroc_A", [
    "tp_ns" => col(x -> x.cfg.tp, res_A), "vov" => col(x -> x.cfg.vov, res_A),
    "noise_scale" => col(x -> x.cfg.noise_scale, res_A), "noise_pe" => col(x -> x.r.noise_tp, res_A),
    "window_ns" => col(x -> x.cfg.window, res_A), "dark_rate_hz" => col(x -> x.r.dark_rate, res_A),
    "crosstalk" => col(x -> x.r.crosstalk, res_A), "windows" => col(x -> length(x.r.vals[1]), res_A),
    "f_signal" => col(x -> x.fit.f_signal, res_A), "k" => col(x -> x.fit.k, res_A),
    "k_err" => col(x -> x.fit.k_err, res_A), "zero_pe" => col(x -> x.fit.zero_pe, res_A),
    "peak1_pe" => col(x -> pk(x, 1), res_A), "peak2_pe" => col(x -> pk(x, 2), res_A),
    "peak3_pe" => col(x -> pk(x, 3), res_A), "k1" => col(x -> x.fit.k1, res_A),
    "k1_err" => col(x -> x.fit.k1_err, res_A)]))
println("written ", write_table("calib", "citiroc", "citiroc_B", [
    "vov" => col(x -> x.cfg.vov, res_B), "channels_fitted" => col(x -> count(f -> isfinite(f.k), x.fits), res_B),
    "k_median" => col(x -> median(filter(isfinite, [f.k for f in x.fits])), res_B),
    "k_rms" => col(x -> std(filter(isfinite, [f.k for f in x.fits])), res_B),
    "k_err_median" => col(x -> median(filter(isfinite, [f.k_err for f in x.fits])), res_B),
    "k1_fitted" => col(x -> count(f -> isfinite(f.k1), x.fits), res_B),
    "k1_median" => col(x -> median(filter(isfinite, [f.k1 for f in x.fits])), res_B),
    "k1_rms" => col(x -> std(filter(isfinite, [f.k1 for f in x.fits])), res_B),
    "k1_err_median" => col(x -> median(filter(isfinite, [f.k1_err for f in x.fits])), res_B)]))
fin(v) = filter(isfinite, v)
println("written ", write_table("calib", "citiroc", "citiroc_F", [
    "t_k" => col(x -> x.cfg.t_k, res_F), "tp_ns" => col(x -> x.cfg.tp, res_F),
    "records" => col(x -> x.nrec, res_F), "dark_rate_hz" => col(x -> x.r.dark_rate, res_F),
    "windows_per_channel" => col(x -> length(x.r.vals[1]), res_F),
    "k_fitted" => col(x -> count(y -> isfinite(y.k), x.fits), res_F),
    "k_median" => col(x -> median(fin([y.k for y in x.fits])), res_F),
    "k_rms" => col(x -> std(fin([y.k for y in x.fits])), res_F),
    "k_err_median" => col(x -> median(fin([y.k_err for y in x.fits])), res_F),
    "k1_fitted" => col(x -> count(y -> isfinite(y.k1), x.fits), res_F),
    "k1_median" => col(x -> median(fin([y.k1 for y in x.fits])), res_F),
    "k1_rms" => col(x -> std(fin([y.k1 for y in x.fits])), res_F),
    "k1_err_median" => col(x -> median(fin([y.k1_err for y in x.fits])), res_F)]))
for x in res_F
    k = fin([y.k for y in x.fits]); k1 = fin([y.k1 for y in x.fits])
    @printf("F %-12s %6d records: ladder in %d channels, k %.3f rms %.3f; 1-pe peak in %d channels, k1 %.3f rms %.3f\n",
            nameE(x.cfg), x.nrec, length(k), median(k), std(k), length(k1), median(k1), std(k1))
end
frac(x, c) = length(x.d.vals[c]) / sum(length(v) for v in values(x.d.vals))
println("written ", write_table("calib", "citiroc", "citiroc_E", [
    "t_k" => col(x -> x.cfg.t_k, res_E), "tp_ns" => col(x -> x.cfg.tp, res_E),
    "vov" => col(x -> x.cfg.vov, res_E), "records" => col(x -> x.nrec, res_E),
    "windows" => col(x -> length(x.r.vals[1]), res_E), "dark_rate_hz" => col(x -> x.r.dark_rate, res_E),
    "f_signal" => col(x -> x.fit.f_signal, res_E), "k" => col(x -> x.fit.k, res_E),
    "k_err" => col(x -> x.fit.k_err, res_E), "k1" => col(x -> x.fit.k1, res_E),
    "k1_err" => col(x -> x.fit.k1_err, res_E),
    ("f_" * c => col(x -> frac(x, c), res_E) for c in CATEGORIES)...]))
for x in res_E
    @printf("E %-12s %6d records  dark %4.0f kHz  f_signal %.3f  k1 %.3f ± %.3f  k %.3f ± %.3f  %s\n", nameE(x.cfg), x.nrec,
            x.r.dark_rate / 1e3, x.fit.f_signal, x.fit.k1, x.fit.k1_err, x.fit.k, x.fit.k_err,
            join((@sprintf("%s %.3f", c, frac(x, c)) for c in CATEGORIES), ", "))
end
for x in res_D
    n = sum(length(v) for v in values(x.r.vals))
    @printf("D %-22s t_peak %.0f ns: %s\n", name(x.cfg), x.r.t_peak,
            join((@sprintf("%s %.3f", c, length(x.r.vals[c]) / n) for c in CATEGORIES), ", "))
end
for x in res_A
    @printf("%-24s noise %.3f pe  f_signal %.3f  k1 %.3f ± %.3f  k %.3f ± %.3f  zero %+.3f pe  peaks %s\n", name(x.cfg), x.r.noise_tp,
            x.fit.f_signal, x.fit.k1, x.fit.k1_err, x.fit.k, x.fit.k_err, x.fit.zero_pe, string(round.((x.fit.lad.peaks .- PEDESTAL) ./ x.r.true_gain[1]; digits = 3)))
end
