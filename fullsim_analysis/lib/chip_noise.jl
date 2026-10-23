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
#   b_i       = cov_e(d_i, CM) / var_e CM            how much common mode each channel takes;
#                                                    the split assumes b_i = 1. Channels with
#                                                    b_i < BLIND_SLOPE do not see their SiPM
#                                                    (blind): CM is computed a second time
#                                                    without them, as for the outliers, and
#                                                    the widths below are over the others
#   J(e)      = |CM(e) - median_e CM| > JUMP_SIGMA * 1.4826 * MAD_e CM   jump of the chip:
#                                                    event set aside
# The outlier cut assumes a symmetric noise: with a long HoldDelay the stored maximum has a
# longer tail upwards, so the callers switch it off (outliers = false) in those runs.
# On the events kept, without the outliers:
#   rms_i   = rms_e d_i                              noise of channel i
#   total   = median_i rms_i                         over the n channels that are not blind
#   incoh   = median_i rms_e (d_i - CM) * sqrt(n/(n-1)) (subtracting the mean removes 1/n of
#                                                    the channel's own noise)
#   coh^2   = var_e CM - incoh^2 / n                 (the mean keeps 1/n of the incoherent
#                                                    variance)
const CHANNEL_SIGMA = 5.0
const JUMP_SIGMA = 5.0
const BLIND_SLOPE = 0.75   # a channel taking less common mode than this does not see its SiPM

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

function chip_noise(M; outliers::Bool = true, blind = nothing)
    ne, nc = size(M)
    D0 = M .- median(M; dims = 1)
    ref = vec(median(D0; dims = 2))
    s_i = [madn(D0[:, i] .- ref) for i in 1:nc]
    Z = (D0 .- ref) ./ s_i'                           # pull of each channel in each event
    O = outliers ? abs.(Z) .> CHANNEL_SIGMA : falses(ne, nc)
    rmsok(v, ok) = std(v[ok])
    # common mode over the channels in `use` that are not outliers, jumps, and the slope of
    # every channel against it on the events kept
    function pass(use)
        cm_all = [mean(D0[e, j] for j in 1:nc if use[j] && !O[e, j]) for e in 1:ne]
        centre, width = median(cm_all), madn(cm_all)
        jump = abs.(cm_all .- centre) .> JUMP_SIGMA * width
        keep = .!jump
        cm = cm_all[keep]
        slope = [(ok = .!O[keep, i]; cov(D0[keep, i][ok], cm[ok]) / var(cm[ok])) for i in 1:nc]
        (; cm_all, centre, width, jump, keep, cm, slope)
    end
    first = pass(trues(nc))
    blind = blind === nothing ? first.slope .< BLIND_SLOPE : blind
    p = pass(.!blind)                                 # blind channels left out of CM
    D, Ok, cm = D0[p.keep, :], O[p.keep, :], p.cm
    rms = [rmsok(D[:, i], .!Ok[:, i]) for i in 1:nc]
    good = findall(.!blind)
    n = length(good)
    total = median(rms[good])
    inc = median([rmsok(D[:, i] .- cm, .!Ok[:, i]) for i in good]) * sqrt(n / (n - 1))
    coh = sqrt(max(var(cm) - inc^2 / n, 0.0))
    outliers = vec(sum(Ok; dims = 1))
    (; D0, Z, O, cm_all = p.cm_all, jump = p.jump, cut = (p.centre, JUMP_SIGMA * p.width), D,
       cm, rms, total, coh, inc, slope = p.slope, blind, outliers)
end
