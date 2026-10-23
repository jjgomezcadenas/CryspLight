# Common-mode analysis of the 32 channels of a Citiroc, the method of the noise note
# (docs_fullsim/model_data/noise.tex), for bench and simulated pedestal runs alike. For one
# chip of one run, x_i(e) is the high (or low) gain of channel i in event e. Two kinds of
# anomaly are separated from the noise, each with its own variable: a channel that departs
# from the rest of its chip in one event (outlier of a channel), and a chip that moves as a
# whole (jump, seen in the common mode).
#   d_i(e)    = x_i(e) - median_e x_i                deviation from the pedestal
#   r(e)      = median_i d_i(e)                      reference of the event, which a single
#                                                    channel cannot move
#   s_i       = 1.4826 * MAD_e (d_i - r)             robust incoherent width of channel i
#   z_i(e)    = (d_i(e) - r(e)) / s_i                pull of channel i in event e
#   O_i(e)    = |z_i(e)| > CHANNEL_SIGMA             outlier of channel i in event e: left
#                                                    out of the common mode and the widths,
#                                                    counted per channel
#   CM(e)     = mean of d_i(e) over the channels that are not outliers   common mode
#   J(e)      = |CM(e) - median_e CM| > JUMP_SIGMA * 1.4826 * MAD_e CM   jump of the chip:
#                                                    event set aside
# On the events kept, without the outliers:
#   rms_i   = rms_e d_i                              noise of channel i
#   total   = median_i rms_i
#   incoh   = median_i rms_e (d_i - CM) * sqrt(32/31) (subtracting the mean removes 1/32 of
#                                                    the channel's own noise)
#   coh^2   = var_e CM - incoh^2 / 32                (the mean keeps 1/32 of the incoherent
#                                                    variance)
#   b_i     = cov_e(d_i, CM) / var_e CM              how much common mode each channel takes;
#                                                    the split assumes b_i = 1
const CHANNEL_SIGMA = 5.0
const JUMP_SIGMA = 6.0

madn(v) = 1.4826 * median(abs.(v .- median(v)))

# board number from chain and node, as in the manifest topology
function daq_board_numbers(text)
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

function chip_noise(M)
    ne, nc = size(M)
    D0 = M .- median(M; dims = 1)
    ref = vec(median(D0; dims = 2))
    s_i = [madn(D0[:, i] .- ref) for i in 1:nc]
    Z = (D0 .- ref) ./ s_i'                           # pull of each channel in each event
    O = abs.(Z) .> CHANNEL_SIGMA
    cm_all = [mean(D0[e, j] for j in 1:nc if !O[e, j]) for e in 1:ne]
    cm_centre, cm_width = median(cm_all), madn(cm_all)
    jump = abs.(cm_all .- cm_centre) .> JUMP_SIGMA * cm_width
    keep = .!jump
    D = D0[keep, :]
    Ok = O[keep, :]
    cm = cm_all[keep]
    rmsok(v, ok) = std(v[ok])
    rms = [rmsok(D[:, i], .!Ok[:, i]) for i in 1:nc]
    total = median(rms)
    inc = median([rmsok(D[:, i] .- cm, .!Ok[:, i]) for i in 1:nc]) * sqrt(nc / (nc - 1))
    coh = sqrt(max(var(cm) - inc^2 / nc, 0.0))
    slope = [(ok = .!Ok[:, i]; cov(D[ok, i], cm[ok]) / var(cm[ok])) for i in 1:nc]
    outliers = vec(sum(Ok; dims = 1))
    (; D0, Z, O, cm_all, jump, cut = (cm_centre, JUMP_SIGMA * cm_width), D, cm, rms, total, coh,
       inc, slope, outliers)
end
