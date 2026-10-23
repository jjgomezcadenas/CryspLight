# Common-mode analysis of the 32 channels of a Citiroc, shared by noise_components.jl and
# noise_steps.jl. For one chip of one run, x_i(e) is the high (or low) gain of channel i in
# event e:
#   d_i(e)  = x_i(e) - median_e x_i                  deviation from the pedestal
#   N(e)    = #{i : |d_i(e)| > OUTLIER}              events with N >= OUTLIER_CHANNELS are
#                                                    jumps of the whole chip, set aside
#   CM(e)   = (1/32) sum_i d_i(e)                    common mode of the event
#   total   = median_i rms_e d_i
#   incoh   = median_i rms_e (d_i - CM) * sqrt(32/31) (subtracting the mean removes 1/32 of
#                                                    the channel's own noise)
#   coh^2   = var_e CM - incoh^2 / 32                (the mean keeps 1/32 of the incoherent
#                                                    variance)
#   b_i     = cov_e(d_i, CM) / var_e CM              how much common mode each channel takes;
#                                                    the split assumes b_i = 1
using Statistics

const OUTLIER = 60        # counts, five times the pedestal rms of the high gain
const OUTLIER_CHANNELS = 10

# board number from chain and node, as in the manifest topology
function board_numbers(text)
    re = r"\"board\":\s*(\d+),\s*\"chain\":\s*(\d+),\s*\"node\":\s*(\d+)"
    Dict((parse(Int, m[2]), parse(Int, m[3])) => parse(Int, m[1]) for m in eachmatch(re, text))
end

# trigger ids and events x 32 matrix of one gain of one chip, rows ordered by trigger id
function chip_matrix(o, sel, gain)
    tids = sort(unique(o.trigger_id[sel]))
    row = Dict(t => k for (k, t) in enumerate(tids))
    v = gain === :hg ? o.hg : o.lg
    M = fill(NaN, length(tids), 32)
    for i in sel
        M[row[o.trigger_id[i]], o.channel[i] % 32 + 1] = v[i]
    end
    tids, M
end

# measurements of one chip (0: channels 0-31, 1: 32-63) of the board at (chain, node)
chip_rows(o, chain, node, chip) =
    findall(i -> o.chain[i] == chain && o.node[i] == node && o.channel[i] ÷ 32 == chip && o.has_hg[i],
            eachindex(o.hg))

function components(M)
    D0 = M .- median(M; dims = 1)
    nout = vec(sum(abs.(D0) .> OUTLIER; dims = 2))
    out = nout .>= OUTLIER_CHANNELS
    D = D0[.!out, :]
    cm = vec(mean(D; dims = 2))
    total = median(vec(std(D; dims = 1)))
    inc = median(vec(std(D .- cm; dims = 1))) * sqrt(32 / 31)
    coh = sqrt(max(var(cm) - inc^2 / 32, 0.0))
    slope = [cov(D[:, i], cm) / var(cm) for i in 1:32]
    (; D0, nout, out, D, cm, total, coh, inc, slope)
end
