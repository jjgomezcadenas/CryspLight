# Philox4x32-10 counter-based RNG (Salmon et al., SC'11). Stateless per (key, counter):
# each photon owns a stream keyed by (seed, photon id), so results are bit-reproducible
# independent of thread scheduling, and the Metal kernel can reproduce the same draws.

const PHILOX_M0 = 0xD2511F53
const PHILOX_M1 = 0xCD9E8D57
const PHILOX_W0 = 0x9E3779B9
const PHILOX_W1 = 0xBB67AE85

@inline function philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in 1:10
        p0 = UInt64(PHILOX_M0) * c0
        p1 = UInt64(PHILOX_M1) * c2
        hi0 = (p0 >>> 32) % UInt32; lo0 = p0 % UInt32
        hi1 = (p1 >>> 32) % UInt32; lo1 = p1 % UInt32
        c0 = hi1 ⊻ c1 ⊻ k0
        c1 = lo1
        c2 = hi0 ⊻ c3 ⊻ k1
        c3 = lo0
        if r < 10
            k0 += PHILOX_W0
            k1 += PHILOX_W1
        end
    end
    return (c0, c1, c2, c3)
end

"""
A per-photon stream of uniforms: key = (seed, seed>>32), counter = (photon id lo, hi,
block index, 0). One Philox call yields four UInt32s, buffered.
"""
mutable struct PhiloxStream
    key::NTuple{2,UInt32}
    id_lo::UInt32
    id_hi::UInt32
    block::UInt32
    buf::NTuple{4,UInt32}
    i::Int
end

function PhiloxStream(seed::Integer, photon_id::Integer)
    s = UInt64(seed); p = UInt64(photon_id)
    PhiloxStream((s % UInt32, (s >>> 32) % UInt32),
                 p % UInt32, (p >>> 32) % UInt32,
                 0x00000000, (0x0, 0x0, 0x0, 0x0), 5)
end

@inline function next_u32!(s::PhiloxStream)
    if s.i > 4
        s.buf = philox4x32((s.id_lo, s.id_hi, s.block, 0x00000000), s.key)
        s.block += 0x00000001
        s.i = 1
    end
    x = s.buf[s.i]
    s.i += 1
    return x
end

"Uniform Float32 in (0, 1] — safe as the argument of log."
@inline randu(s::PhiloxStream) = ((next_u32!(s) >>> 8) + 1.0f0) * Float32(exp2(-24))
