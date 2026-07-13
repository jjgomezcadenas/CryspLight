# GPU twin of transport.jl via KernelAbstractions: one work-item per photon, per-photon
# output records (no atomics — reduced to an Accumulator on the host). The random-draw
# ORDER mirrors transport.jl exactly, so on the CPU backend the kernel is bit-identical
# to the reference path (enforced in the tests). On Metal, transcendental ulp
# differences may flip rare edge-case photons — statistical agreement is enforced there.
# Idioms follow RecoCrysp: isbits scalars/NTuples as kernel args (config structs are
# unpacked host-side), Int32 indices, Float32 arithmetic, @Const on read-only arrays,
# backend selected from the array type, no manual workgroup sizing.

using KernelAbstractions
const KA = KernelAbstractions

# ---- functional (isbits) Philox stream: same sequence as the mutable PhiloxStream ----

struct PhiloxState
    key::NTuple{2,UInt32}
    id_lo::UInt32
    id_hi::UInt32
    block::UInt32
    buf::NTuple{4,UInt32}
    i::Int32
end

@inline PhiloxState(seed::UInt64, pid::UInt64) =
    PhiloxState((seed % UInt32, (seed >>> 32) % UInt32),
                pid % UInt32, (pid >>> 32) % UInt32,
                0x00000000, (0x00000000, 0x00000000, 0x00000000, 0x00000000), Int32(5))

@inline function next_u32(st::PhiloxState)
    if st.i > Int32(4)
        buf = philox4x32((st.id_lo, st.id_hi, st.block, 0x00000000), st.key)
        return buf[1], PhiloxState(st.key, st.id_lo, st.id_hi, st.block + 0x00000001,
                                   buf, Int32(2))
    else
        # explicit branches: no dynamic tuple indexing on the device
        x = st.i == Int32(2) ? st.buf[2] : (st.i == Int32(3) ? st.buf[3] : st.buf[4])
        return x, PhiloxState(st.key, st.id_lo, st.id_hi, st.block, st.buf,
                              st.i + Int32(1))
    end
end

"Uniform Float32 in (0, 1], functional: returns (value, new state)."
@inline function frandu(st::PhiloxState)
    x, st2 = next_u32(st)
    return ((x >>> 8) + 1.0f0) * Float32(exp2(-24)), st2
end

# ---- device-side samplers: bodies mirror optics.jl/generation.jl draw-for-draw ----

@inline function fisotropic(st)
    u1, st = frandu(st)
    uz = 1f0 - 2f0 * u1
    r = sqrt(max(0f0, 1f0 - uz * uz))
    u2, st = frandu(st)
    phi = 2f0 * Float32(pi) * u2
    return r * cos(phi), r * sin(phi), uz, st
end

@inline function fgaussian(st)
    u1, st = frandu(st)
    u2, st = frandu(st)
    return sqrt(-2f0 * log(u1)) * cos(2f0 * Float32(pi) * u2), st
end

@inline function flambertian(st, face::Int32)
    u1, st = frandu(st)
    ct = sqrt(u1)
    s = sqrt(max(0f0, 1f0 - ct * ct))
    u2, st = frandu(st)
    phi = 2f0 * Float32(pi) * u2
    a = s * cos(phi); b = s * sin(phi)
    if face == FACE_XP
        return -ct, a, b, st
    elseif face == FACE_XM
        return ct, a, b, st
    elseif face == FACE_YP
        return a, -ct, b, st
    elseif face == FACE_YM
        return a, ct, b, st
    elseif face == FACE_BACK
        return a, b, -ct, st
    else
        return a, b, ct, st
    end
end

@inline function frayleigh(st, ux, uy, uz)
    mu = 0f0
    while true
        u1, st = frandu(st)
        mu = 1f0 - 2f0 * u1
        u2, st = frandu(st)
        2f0 * u2 <= 1f0 + mu * mu && break
    end
    s = sqrt(max(0f0, 1f0 - mu * mu))
    u3, st = frandu(st)
    phi = 2f0 * Float32(pi) * u3
    cb = s * cos(phi); sb = s * sin(phi)
    if abs(uz) < 0.9f0
        inv = 1f0 / sqrt(ux * ux + uy * uy)
        t1x = -uy * inv; t1y = ux * inv; t1z = 0f0
    else
        inv = 1f0 / sqrt(uy * uy + uz * uz)
        t1x = 0f0; t1y = -uz * inv; t1z = uy * inv
    end
    t2x = uy * t1z - uz * t1y
    t2y = uz * t1x - ux * t1z
    t2z = ux * t1y - uy * t1x
    vx = mu * ux + cb * t1x + sb * t2x
    vy = mu * uy + cb * t1y + sb * t2y
    vz = mu * uz + cb * t1z + sb * t2z
    inv = 1f0 / sqrt(vx * vx + vy * vy + vz * vz)
    return vx * inv, vy * inv, vz * inv, st
end

@inline function fsurface(st, ux, uy, uz, face::Int32, n_crystal, sigma_alpha)
    a, sgn = face_frame(face)
    Nx = a == 1 ? sgn : 0f0; Ny = a == 2 ? sgn : 0f0; Nz = a == 3 ? sgn : 0f0
    for _ in 1:20
        g, st = fgaussian(st)
        alpha = sigma_alpha * abs(g)
        alpha >= 1.2f0 && continue
        ca = cos(alpha); sa = sin(alpha)
        u, st = frandu(st)
        phi = 2f0 * Float32(pi) * u
        b = sa * cos(phi); c = sa * sin(phi)
        Nfx = a == 1 ? ca * sgn : b
        Nfy = a == 2 ? ca * sgn : (a == 1 ? b : c)
        Nfz = a == 3 ? ca * sgn : c
        cosi = ux * Nfx + uy * Nfy + uz * Nfz
        cosi <= 0f0 && continue
        tir, R = fresnel(n_crystal, 1f0, cosi)
        refl = tir
        if !tir
            u, st = frandu(st)
            refl = u < R
        end
        if refl
            rx = ux - 2f0 * cosi * Nfx
            ry = uy - 2f0 * cosi * Nfy
            rz = uz - 2f0 * cosi * Nfz
            if rx * Nx + ry * Ny + rz * Nz < 0f0
                return rx, ry, rz, false, st
            end
        else
            return ux, uy, uz, true, st
        end
    end
    r = specular(ux, uy, uz, face)
    return r[1], r[2], r[3], false, st
end

# ---- the kernel: status/idx/time/bounces record per photon ----

# NB: Metal caps kernel arguments at 31 buffer bindings and each scalar counts as one —
# so the parameters travel as the isbits structs OpticalParams/Readout, not unpacked.
@kernel function transport_kernel!(status, idx, tdet, bnc,
                                   @Const(px), @Const(py), @Const(pz), npos,
                                   L, op, ro, tau, seed, maxb)
    i = @index(Global)
    st = PhiloxState(seed, UInt64(i))
    box = Box(L)
    n_crystal = op.n_crystal; n_coupling = op.n_coupling
    abs_len = op.abs_len_mm; ray_len = op.rayleigh_mm
    wrap_R = op.wrap_R; wrap_spec = op.wrap_specular; air_gap = op.air_gap
    sigma_alpha = op.sigma_alpha; pde = op.pde
    disc = ro.disc; dcx = ro.cx; dcy = ro.cy; dr2 = ro.r2
    sur_R = ro.sur_R; sur_spec = ro.sur_specular; two_sided = ro.two_sided
    pitch = ro.grid.pitch_mm; nx = ro.grid.nx; ny = ro.grid.ny
    @inbounds begin
        j = Int32(1) + Int32((i - 1) % npos)
        x = px[j]; y = py[j]; z = pz[j]
        ux, uy, uz, st = fisotropic(st)
        t = 0f0
        if tau > 0f0
            u, st = frandu(st)
            t = -tau * log(u)
        end
        inv_v = n_crystal / C_MM_NS
        bounces = Int32(0)
        result = Int32(0)          # 0 = still flying
        out_idx = Int16(0)
        while result == Int32(0)
            d, face = wall_hit(x, y, z, ux, uy, uz, box)
            d_abs = Inf32
            if isfinite(abs_len)
                u, st = frandu(st)
                d_abs = -abs_len * log(u)
            end
            d_ray = Inf32
            if isfinite(ray_len)
                u, st = frandu(st)
                d_ray = -ray_len * log(u)
            end
            if d_abs < d && d_abs <= d_ray
                result = STATUS_ABS_BULK
            elseif d_ray < d
                x += d_ray * ux; y += d_ray * uy; z += d_ray * uz
                t += d_ray * inv_v
                ux, uy, uz, st = frayleigh(st, ux, uy, uz)
                bounces += Int32(1)
                bounces >= maxb && (result = STATUS_CAP)
            else
                x = clamp(x + d * ux, 0f0, L[1])
                y = clamp(y + d * uy, 0f0, L[2])
                z = clamp(z + d * uz, 0f0, L[3])
                a = face_axis(face)
                if a == 1
                    x = face == FACE_XP ? L[1] : 0f0
                elseif a == 2
                    y = face == FACE_YP ? L[2] : 0f0
                else
                    z = face == FACE_BACK ? L[3] : 0f0
                end
                t += d * inv_v
                cosi = a == 1 ? abs(ux) : (a == 2 ? abs(uy) : abs(uz))

                if face == FACE_BACK || (two_sided && face == FACE_FRONT)
                    tir, R = fresnel(n_crystal, n_coupling, cosi)
                    refl = tir
                    if !tir
                        u, st = frandu(st)
                        refl = u < R
                    end
                    if refl
                        ux, uy, uz = specular(ux, uy, uz, face)
                    elseif !disc || (x - dcx)^2 + (y - dcy)^2 <= dr2
                        u, st = frandu(st)
                        if u <= pde
                            ii = min(nx, Int32(1) + unsafe_trunc(Int32, x / pitch))
                            jj = min(ny, Int32(1) + unsafe_trunc(Int32, y / pitch))
                            lin = (jj - Int32(1)) * nx + ii
                            out_idx = face == FACE_FRONT ? Int16(-lin) : Int16(lin)
                            result = STATUS_DETECTED
                        else
                            result = STATUS_ABS_SIPM
                        end
                    else
                        u, st = frandu(st)
                        if u <= sur_R
                            if sur_spec
                                ux, uy, uz = specular(ux, uy, uz, face)
                            else
                                ux, uy, uz, st = flambertian(st, face)
                            end
                        else
                            result = STATUS_ABS_SUR
                        end
                    end
                elseif !air_gap
                    u, st = frandu(st)
                    if u <= wrap_R
                        if wrap_spec
                            ux, uy, uz = specular(ux, uy, uz, face)
                        else
                            ux, uy, uz, st = flambertian(st, face)
                        end
                    else
                        result = STATUS_ABS_WALL
                    end
                else
                    transmitted = false
                    if sigma_alpha > 0f0
                        ux, uy, uz, transmitted, st =
                            fsurface(st, ux, uy, uz, face, n_crystal, sigma_alpha)
                    else
                        tir, R = fresnel(n_crystal, 1f0, cosi)
                        refl = tir
                        if !tir
                            u, st = frandu(st)
                            refl = u < R
                        end
                        if refl
                            ux, uy, uz = specular(ux, uy, uz, face)
                        else
                            transmitted = true
                        end
                    end
                    if transmitted
                        u, st = frandu(st)
                        if u <= wrap_R
                            if wrap_spec
                                ux, uy, uz = specular(ux, uy, uz, face)
                            else
                                ux, uy, uz, st = flambertian(st, face)
                            end
                        else
                            result = STATUS_ABS_WALL
                        end
                    end
                end
                if result == Int32(0)
                    bounces += Int32(1)
                    bounces >= maxb && (result = STATUS_CAP)
                end
            end
        end
        status[i] = UInt8(result)
        idx[i] = out_idx
        tdet[i] = t
        bnc[i] = bounces
    end
end

"""
KernelAbstractions twin of run_photons!: same interface, one extra argument ArrayT
selecting the backend by array type (Array = CPU threads, Metal.MtlArray = GPU).
Returns the same Accumulator, reduced on the host from per-photon records.
"""
function run_photons_ka!(box::Box, op::OpticalParams, ro::Readout, tb::TimeBinning;
                         n_photons::Int, seed::Integer,
                         pos::Union{Nothing,NTuple{3,Float32}} = nothing,
                         positions::Union{Nothing,Vector{NTuple{3,Float32}}} = nothing,
                         tau_ns::Float32, max_bounces::Int = 100_000,
                         ArrayT = Array)
    plist = positions === nothing ? [something(pos)] : positions
    px = ArrayT(Float32[p[1] for p in plist])
    py = ArrayT(Float32[p[2] for p in plist])
    pz = ArrayT(Float32[p[3] for p in plist])
    status = ArrayT(zeros(UInt8, n_photons))
    idx = ArrayT(zeros(Int16, n_photons))
    tdet = ArrayT(zeros(Float32, n_photons))
    bnc = ArrayT(zeros(Int32, n_photons))

    backend = KA.get_backend(px)
    transport_kernel!(backend)(
        status, idx, tdet, bnc, px, py, pz, Int32(length(plist)), box.L,
        op, ro, Float32(tau_ns), UInt64(seed), Int32(max_bounces);
        ndrange = n_photons)
    KA.synchronize(backend)

    hs = Array(status); hi = Array(idx); ht = Array(tdet); hb = Array(bnc)
    acc = Accumulator(ro.grid, tb)
    nx = Int(ro.grid.nx)
    for k in 1:n_photons
        s = Int32(hs[k])
        if s == STATUS_DETECTED
            lin = Int(hi[k])
            lin < 0 && (acc.ndet_front += 1; lin = -lin)
            ii = (lin - 1) % nx + 1
            jj = (lin - 1) ÷ nx + 1
            t = ht[k]
            bin = 1 + unsafe_trunc(Int, t / tb.bin_ns)
            bin = bin > tb.nbins ? Int(tb.nbins) + 1 : bin
            acc.counts[ii, jj, bin] += UInt32(1)
            t < acc.first_ns[ii, jj] && (acc.first_ns[ii, jj] = t)
            acc.ndet += 1
            acc.sum_bounces_det += hb[k]
        elseif s == STATUS_ABS_BULK
            acc.nabs_bulk += 1
        elseif s == STATUS_ABS_WALL
            acc.nabs_wall += 1
        elseif s == STATUS_ABS_SIPM
            acc.nabs_sipm += 1
        elseif s == STATUS_ABS_SUR
            acc.nabs_sur += 1
        else
            acc.ncap += 1
        end
    end
    return acc
end
