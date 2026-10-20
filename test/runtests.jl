using Test
using Statistics
using HDF5
using CryspLight
using Metal
using Random: Xoshiro
import Random
using CryspLight: face_axis, specular, emission_time

"The operating point without dark counts, for chains fed with hand-made avalanche trains
(the chains remove the mean level of the dark counts they expect)."
nodark(o) = SipmOperation(o.dev, o.t_k, o.vov, o.vbr, o.gain, o.pde_peak, o.crosstalk, 0.0,
                          o.afterpulse, o.gain_factor, o.window_ns)

const GRID = SipmGrid(6f0, Int32(8), Int32(8))
const TB = TimeBinning(100f0, Int32(40))

@testset "Configuration validation" begin
    @test surface_has_gap("backpainted")
    @test !surface_has_gap("frontpainted")
    @test_throws ErrorException surface_has_gap("back-painted")
end

"Idealized params: no bulk absorption or scattering, perfect wrap, air gap, PDE = 1."
ideal(n_crystal; wrap_R = 1f0, spec = false, abs_len = Inf32, ray = Inf32, gap = true,
      sigma = 0f0, pde = 1f0) =
    OpticalParams(Float32(n_crystal), 1.45f0, Float32(abs_len), Float32(ray),
                  Float32(wrap_R), spec, gap, Float32(sigma), Float32(pde))

@testset "Philox RNG" begin
    s1 = PhiloxStream(42, 7)
    s2 = PhiloxStream(42, 7)
    @test [randu(s1) for _ in 1:100] == [randu(s2) for _ in 1:100]  # deterministic
    s3 = PhiloxStream(42, 8)
    @test randu(s3) != randu(PhiloxStream(43, 7))                    # streams differ
    u = [randu(PhiloxStream(1, i)) for i in 1:200_000]
    @test all(x -> 0f0 < x <= 1f0, u)
    @test isapprox(mean(u), 0.5; atol = 3e-3)                        # 4 sigma ~ 2.6e-3
    @test isapprox(std(u), sqrt(1 / 12); atol = 3e-3)
end

@testset "Geometry: slab distances" begin
    box = Box((48f0, 48f0, 37.2f0))
    d, f = wall_hit(24f0, 24f0, 18.6f0, 1f0, 0f0, 0f0, box)
    @test d ≈ 24f0 && f == FACE_XP
    d, f = wall_hit(24f0, 24f0, 18.6f0, 0f0, -1f0, 0f0, box)
    @test d ≈ 24f0 && f == FACE_YM
    d, f = wall_hit(24f0, 24f0, 18.6f0, 0f0, 0f0, 1f0, box)
    @test d ≈ 37.2f0 - 18.6f0 && f == FACE_BACK
    u = 1f0 / sqrt(2f0)
    d, f = wall_hit(0f0, 0f0, 0f0, u, 0f0, u, box)                   # diagonal: z wall first
    @test f == FACE_BACK && d ≈ 37.2f0 * sqrt(2f0)
end

@testset "Fresnel" begin
    # normal incidence: R = ((n1-n2)/(n1+n2))^2
    tir, R = fresnel(1.95f0, 1.45f0, 1f0)
    @test !tir && isapprox(R, ((1.95 - 1.45) / (1.95 + 1.45))^2; rtol = 1e-5)
    # TIR beyond the critical angle (crystal -> air, theta_c = 30.9 deg for n = 1.95)
    tir, _ = fresnel(1.95f0, 1f0, cos(deg2rad(40f0)))
    @test tir
    tir, _ = fresnel(1.95f0, 1f0, cos(deg2rad(20f0)))
    @test !tir
    # grazing incidence: R -> 1
    _, R = fresnel(1.95f0, 1.45f0, 1f-4)
    @test R > 0.999f0
    # symmetry: R identical from both sides at normal incidence
    _, Ra = fresnel(1.45f0, 1.95f0, 1f0)
    @test isapprox(Ra, ((1.95 - 1.45) / (1.95 + 1.45))^2; rtol = 1e-5)
end

@testset "Sampling distributions" begin
    s = PhiloxStream(7, 1)
    dirs = [isotropic_dir(s) for _ in 1:200_000]
    @test isapprox(mean(d[3] for d in dirs), 0; atol = 5e-3)          # <uz> = 0
    @test isapprox(mean(d[3]^2 for d in dirs), 1 / 3; atol = 3e-3)    # <uz^2> = 1/3
    @test all(d -> isapprox(sum(abs2, d), 1; atol = 1e-5), dirs[1:1000])

    lam = [lambertian_dir(s, FACE_BACK) for _ in 1:200_000]           # inward = -z
    @test all(d -> d[3] < 0, lam)
    @test isapprox(mean(-d[3] for d in lam), 2 / 3; atol = 3e-3)      # <cos> = 2/3

    lam = [lambertian_dir(s, FACE_XM) for _ in 1:1000]                # inward = +x
    @test all(d -> d[1] > 0, lam)

    dt = [emission_time(s, 800f0) for _ in 1:200_000]
    @test isapprox(mean(dt), 800; rtol = 0.01)
    @test emission_time(s, 0f0) == 0f0

    # Rayleigh phase function about +z: E[mu] = 0, E[mu^2] = 2/5 for (1 + mu^2)
    mus = [rayleigh_dir(s, 0f0, 0f0, 1f0)[3] for _ in 1:200_000]
    @test isapprox(mean(mus), 0; atol = 6e-3)
    @test isapprox(mean(abs2, mus), 0.4; atol = 5e-3)
    @test all(abs(sum(abs2, rayleigh_dir(s, 0.6f0, 0.64f0, 0.48f0)) - 1) < 1e-5
              for _ in 1:1000)

    # UNIFIED lobe: sigma_alpha -> 0 reduces to ideal specular reflection (TIR regime:
    # incidence 60 deg from the x normal, far beyond the 30.9 deg critical angle)
    u = (0.5f0, 0.5f0, sqrt(0.5f0))
    rx, ry, rz, tr = surface_interact(s, u..., FACE_XP, 1.95f0, 1f-6)
    @test !tr && isapprox(rx, -u[1]; atol = 1e-3) && isapprox(ry, u[2]; atol = 1e-3)
    # rough facet: any reflected direction must still point back into the crystal
    res = [surface_interact(s, u..., FACE_XP, 1.95f0, Float32(deg2rad(12))) for _ in 1:2000]
    @test all(r -> r[4] || r[1] < 0f0, res)
    # roughness opens an escape from TIR: some transmissions must now occur
    @test any(r -> r[4], res)
end

@testset "Transport: conservation and analytic limits" begin
    boxc = Box((48f0, 48f0, 37.2f0))
    pos = (24f0, 24f0, 18.6f0)

    # exact photon conservation with physical parameters
    op = ideal(1.95f0; wrap_R = 0.99f0, abs_len = 500f0, pde = 0.4f0)
    acc = run_photons!(boxc, op, GRID, TB; n_photons = 100_000, seed = 1, pos = pos,
                       tau_ns = 800f0)
    @test acc.ndet + acc.nabs_bulk + acc.nabs_wall + acc.nabs_sipm + acc.ncap == 100_000
    @test sum(acc.counts) == acc.ndet

    # Idealized Teflon (R = 1, no absorption, PDE = 1): TIR preserves the direction
    # components, so directions super-critical on ALL faces never reach the wrap and are
    # trapped forever (-> cap); nothing is ever absorbed, and at least the direct grease
    # escape cone P(|uz| > cos(48.0 deg)) = 1 - cos(asin(1.45/1.95)) is detected.
    nl = 50_000
    acc = run_photons!(boxc, ideal(1.95f0), GRID, TB; n_photons = nl, seed = 2,
                       pos = pos, tau_ns = 0f0, max_bounces = 20_000)
    @test acc.ndet + acc.ncap == nl
    @test acc.nabs_bulk == acc.nabs_wall == acc.nabs_sipm == 0
    pcone = 1 - cos(asin(1.45 / 1.95))
    @test acc.ndet / nl > pcone - 4 * sqrt(pcone * (1 - pcone) / nl)
    @test acc.ndet < nl

    # Specular box preserves |u| components: the same limit detects EXACTLY the
    # escape-cone fraction 1 - cos(theta_c'), theta_c' = asin(1.45/2.15) = 42.4 deg (BGO)
    boxb = Box((48f0, 48f0, 22.4f0))
    n = 200_000
    acc = run_photons!(boxb, ideal(2.15f0; spec = true), GRID, TB; n_photons = n,
                       seed = 3, pos = (24f0, 24f0, 11.2f0), tau_ns = 0f0,
                       max_bounces = 20_000)
    p = 1 - cos(asin(1.45 / 2.15))
    @test isapprox(acc.ndet / n, p; atol = 4 * sqrt(p * (1 - p) / n))
    @test acc.ncap + acc.ndet == n                                    # nothing absorbed

    # bulk absorption only (black walls via wrap_R = 0, source aimed nowhere special):
    # detected + wall + bulk must still close
    op = ideal(1.95f0; wrap_R = 0f0, abs_len = 100f0)
    acc = run_photons!(boxc, op, GRID, TB; n_photons = 50_000, seed = 4, pos = pos,
                       tau_ns = 0f0)
    @test acc.ndet + acc.nabs_bulk + acc.nabs_wall + acc.nabs_sipm + acc.ncap == 50_000
    @test acc.nabs_bulk > 0 && acc.nabs_wall > 0

    # Rayleigh restores ergodicity: pure scattering (no absorption), perfect wrap,
    # PDE = 1 -> NO direction class stays trapped; everything is eventually detected
    acc = run_photons!(boxc, ideal(1.95f0; ray = 100f0), GRID, TB; n_photons = 20_000,
                       seed = 5, pos = pos, tau_ns = 0f0)
    @test acc.ndet == 20_000
    @test acc.nscat > 0

    # surface roughness unlocks trapped classes: sigma_alpha > 0 must detect more than
    # the ideally polished surface in the same idealized-Teflon limit
    n0 = run_photons!(boxc, ideal(1.95f0), GRID, TB; n_photons = 30_000, seed = 6,
                      pos = pos, tau_ns = 0f0, max_bounces = 20_000).ndet
    n1 = run_photons!(boxc, ideal(1.95f0; sigma = Float32(deg2rad(6))), GRID, TB;
                      n_photons = 30_000, seed = 6, pos = pos, tau_ns = 0f0,
                      max_bounces = 20_000).ndet
    @test n1 > n0

    # front-painted (contact) Lambertian with R = 1: no Fresnel gate, no trapping ->
    # everything detected
    acc = run_photons!(boxc, ideal(1.95f0; gap = false), GRID, TB; n_photons = 20_000,
                       seed = 7, pos = pos, tau_ns = 0f0)
    @test acc.ndet == 20_000

    # conservation with every process on at once
    op = OpticalParams(1.95f0, 1.45f0, 500f0, 400f0, 0.98f0, false, true,
                       Float32(deg2rad(6)), 0.4f0)
    acc = run_photons!(boxc, op, GRID, TB; n_photons = 100_000, seed = 8, pos = pos,
                       tau_ns = 800f0)
    @test acc.ndet + acc.nabs_bulk + acc.nabs_wall + acc.nabs_sipm + acc.ncap == 100_000
    @test sum(acc.counts) == acc.ndet
end

@testset "Readout: disc aperture, surround, source scan" begin
    box = Box((48f0, 48f0, 37.2f0))
    op = ideal(1.95f0)
    tb = TB
    # disc of radius 10 about the centre; photon straight down at the centre -> detected
    ro = Readout(GRID; center = (24, 24), radius = 10, sur_R = 0)
    acc = Accumulator(GRID, tb)
    s = PhiloxStream(21, 1)
    @test propagate_photon!(acc, box, op, ro, tb, s,
                            24f0, 24f0, 0f0, 0f0, 0f0, 1f0, 0f0) == STATUS_DETECTED
    # straight down far outside the disc, absorbing surround -> killed at the surround
    @test propagate_photon!(acc, box, op, ro, tb, s,
                            4f0, 4f0, 0f0, 0f0, 0f0, 1f0, 0f0) == STATUS_ABS_SUR
    @test acc.nabs_sur == 1

    # conservation with a reflective (Lambertian) surround and physical parameters
    ro = Readout(GRID; center = (24, 24), radius = 10, sur_R = 0.95f0)
    opb = ideal(1.79f0; wrap_R = 0.9f0, abs_len = 300f0)
    acc = run_photons!(box, opb, ro, tb; n_photons = 50_000, seed = 22,
                       pos = (24f0, 24f0, 18f0), tau_ns = 0f0)
    @test total_terminated(acc) == 50_000
    @test acc.nabs_sur > 0 && acc.ndet > 0

    # multi-position source: deterministic, conserving, and photons reach both ends
    ps = [(24f0, 24f0, 5f0), (24f0, 24f0, 30f0)]
    a1 = run_photons!(box, opb, ro, tb; n_photons = 20_000, seed = 23,
                      positions = ps, tau_ns = 0f0)
    a2 = run_photons!(box, opb, ro, tb; n_photons = 20_000, seed = 23,
                      positions = ps, tau_ns = 0f0)
    @test total_terminated(a1) == 20_000
    @test a1.ndet == a2.ndet && a1.counts == a2.counts

    # two-sided readout: photons straight up/down are detected on the respective faces
    ro2 = Readout(GRID; two_sided = true)
    acc = Accumulator(GRID, tb)
    s = PhiloxStream(31, 1)
    @test propagate_photon!(acc, box, ideal(1.95f0), ro2, tb, s,
                            24f0, 24f0, 18f0, 0f0, 0f0, 1f0, 0f0) == STATUS_DETECTED
    @test propagate_photon!(acc, box, ideal(1.95f0), ro2, tb, s,
                            24f0, 24f0, 18f0, 0f0, 0f0, -1f0, 0f0) == STATUS_DETECTED
    @test acc.ndet == 2 && acc.ndet_front == 1
    # by symmetry a centred source splits evenly between the two faces
    acc = run_photons!(box, opb, ro2, tb; n_photons = 50_000, seed = 32,
                       pos = (24f0, 24f0, 18.6f0), tau_ns = 0f0)
    @test total_terminated(acc) == 50_000
    @test isapprox(acc.ndet_front / acc.ndet, 0.5; atol = 0.02)
end

@testset "KernelAbstractions kernel: CPU bit-parity, Metal agreement" begin
    box = Box((48f0, 48f0, 37.2f0))
    tb = TB
    # every process on at once: absorption + Rayleigh + rough surface + wrap + PDE +
    # disc aperture + reflective surround + scintillation time
    op = OpticalParams(1.95f0, 1.45f0, 500f0, 400f0, 0.98f0, false, true,
                       Float32(deg2rad(6)), 0.4f0)
    ro = Readout(GRID; center = (24, 24), radius = 20, sur_R = 0.9f0)
    kw = (n_photons = 100_000, seed = 8, pos = (24f0, 24f0, 18.6f0), tau_ns = 800f0)
    a = run_photons!(box, op, ro, tb; kw...)
    b = run_photons_ka!(box, op, ro, tb; kw..., ArrayT = Array)
    # the KA kernel mirrors the reference draw-for-draw: bit-identical on the CPU backend
    @test a.counts == b.counts
    @test a.first_ns == b.first_ns
    @test (a.ndet, a.nabs_bulk, a.nabs_wall, a.nabs_sipm, a.nabs_sur, a.ncap) ==
          (b.ndet, b.nabs_bulk, b.nabs_wall, b.nabs_sipm, b.nabs_sur, b.ncap)
    @test a.sum_bounces_det == b.sum_bounces_det && a.ndet_front == b.ndet_front

    # two-sided variant
    ro2 = Readout(GRID; two_sided = true)
    a2 = run_photons!(box, op, ro2, tb; kw...)
    b2 = run_photons_ka!(box, op, ro2, tb; kw..., ArrayT = Array)
    @test a2.counts == b2.counts && a2.ndet_front == b2.ndet_front

    # Metal (if available): exact conservation, statistical agreement (transcendental
    # ulp differences may flip rare edge photons — a few per million)
    if Metal.functional()
        g = run_photons_ka!(box, op, ro, tb; kw..., ArrayT = Metal.MtlArray)
        @test total_terminated(g) == kw.n_photons
        p = a.ndet / kw.n_photons
        @test abs(g.ndet - a.ndet) < 5 * sqrt(kw.n_photons * p * (1 - p)) + 10
    end
end

@testset "Vendored gamma core and event pipeline" begin
    data = joinpath(dirname(@__DIR__), "data")
    csi = Gamma.make_material("CsI", 4.51, joinpath(data, "xcom_CSI.csv"))
    bgo = Gamma.make_material("BGO", 7.13, joinpath(data, "xcom_BGO.csv"))

    # attenuation length of CsI at 511 keV ~ 2.44 cm (PTCryspMC test value)
    @test isapprox(Gamma.mfp(csi, 0.511), 2.44; rtol = 0.02)
    @test Gamma.mfp(bgo, 0.511) < Gamma.mfp(csi, 0.511)    # BGO stops harder
    # iodine K-edge (33.17 keV): photoelectric jump
    below = Gamma.sigma_macro(csi, 0.03310)[2]
    above = Gamma.sigma_macro(csi, 0.03325)[2]
    @test above > 2 * below
    # pair production closed at 511 keV, open at 10 MeV
    @test Gamma.sigma_macro(csi, 0.511)[3] == 0.0
    @test Gamma.sigma_macro(csi, 10.0)[3] > 0.0

    # vacuum: straight line, single escape record, full energy out
    L = (48f0, 48f0, 37.2f0)
    pvv = gamma_crystal(L, Gamma.vacuum_material())
    rng = Xoshiro(1)
    tr = Gamma.propagate_photon(0.511, (0.0, 0.0, -1.86), (0.0, 0.0, 1.0), pvv, rng)
    @test tr.escaped && length(tr.recs) == 1 && tr.recs[1].process == :escape
    @test tr.E == 0.511

    # deposits in the crystal frame: inside the box, energies sum to <= 511 keV,
    # times increase with depth
    pv = gamma_crystal(L, csi)
    nin = 0
    for ev in 1:200
        deps = gamma_deposits(pv, L, (24f0, 24f0, 0f0), (0.0, 0.0, 1.0), Xoshiro(ev))
        isempty(deps) && continue
        nin += 1
        @test all(d -> 0 <= d[1] <= 48 && 0 <= d[2] <= 48 && 0 <= d[3] <= 37.2, deps)
        @test sum(d -> d[4], deps) <= 511.001f0
        @test all(d -> 0 <= d[5] < 1.0f0, deps)          # sub-ns transit times
        # process tags: every deposit before the last is a Compton recoil; the last
        # is Compton (photon escaped), photoelectric, or a below-cut absorption
        @test all(d -> d[6] == CryspLight.DEP_COMPTON, deps[1:end-1])
        @test deps[end][6] in (CryspLight.DEP_PHOTO, CryspLight.DEP_COMPTON,
                               CryspLight.DEP_BELOWCUT)
    end
    @test nin > 100    # 2X0 of CsI interacts most of the time

    # Poisson sampler moments
    ps = [rand_poisson(Xoshiro(i), 20.0) for i in 1:20_000]
    @test isapprox(mean(ps), 20; rtol = 0.02) && isapprox(var(ps), 20; rtol = 0.05)

    # end-to-end mini run: conservation and sane photoelectron scale
    box = Box(L)
    op = OpticalParams(1.79f0, 1.45f0, 3115f0, 346f0, 0.99f0, false, true,
                       Float32(deg2rad(1.3)), 0.4f0)
    tbz = TimeBinning(100f0, Int32(80))
    r = run_events!(box, op, Readout(GRID), tbz, pv, 54_000;
                    n_events = 40, seed = 3, tau_ns = 1000f0,
                    batch_photons = 500_000)
    edep, npe, maps = r.edep, r.npe, r.maps
    @test length(edep) == 40 && size(maps) == (8, 8, 40)
    for ev in 1:40
        @test sum(Int, maps[:, :, ev]) == npe[ev]
        @test (edep[ev] == 0) == (npe[ev] == 0)
        if edep[ev] > 510                                # photopeak: pe ~ Y*E*eps*pde
            @test 5000 < npe[ev] < 15000
        end
        # truth record consistency
        @test r.e1[ev] + r.e2[ev] + r.er[ev] ≈ 511f0 atol = 0.01f0
        @test (r.int_type[ev] == -1) == (edep[ev] == 0)
        if r.int_type[ev] == 0                           # direct photoelectric
            @test r.n_int[ev] == 1
            @test r.e1[ev] ≈ 511f0 atol = 0.01f0
            @test abs(r.er[ev]) < 0.01f0
        end
        if r.int_type[ev] >= 0
            @test 0 <= r.xyz1[1, ev] <= 48 && 0 <= r.xyz1[3, ev] <= 37.2f0
            @test r.n_int[ev] >= 1 && r.e1[ev] > 0
        end
        # first-pe times finite exactly where the map counts, and positive
        for jj in 1:8, ii in 1:8
            @test isfinite(r.tmin[ii, jj, ev]) == (maps[ii, jj, ev] > 0)
            maps[ii, jj, ev] > 0 && @test r.tmin[ii, jj, ev] > 0
        end
    end
end

@testset "Detection record: indices, binning, first time" begin
    box = Box((48f0, 48f0, 37.2f0))
    op = ideal(1.95f0)
    # photon straight down +z from just above a known SiPM centre: lands in (2, 7)
    acc = Accumulator(GRID, TB)
    s = PhiloxStream(11, 1)
    st = propagate_photon!(acc, box, op, GRID, TB, s,
                           9f0, 39f0, 0f0, 0f0, 0f0, 1f0, 0f0)
    @test st == STATUS_DETECTED
    @test acc.counts[2, 7, 1] == 1                                    # first 100 ns bin
    texp = 37.2f0 * 1.95f0 / 299.792458f0
    @test isapprox(acc.first_ns[2, 7], texp; rtol = 1e-4)
    # overflow bin: emission far beyond the window
    st = propagate_photon!(acc, box, op, GRID, TB, s,
                           9f0, 39f0, 0f0, 0f0, 0f0, 1f0, 1f6)
    @test acc.counts[2, 7, end] == 1
end

@testset "Tiled readout layout (fullsim 0.5)" begin
    box = Box((48f0, 48f0, 22.4f0))
    grid = SipmGrid(6.2f0, Int32(8), Int32(8))
    ro = Readout(grid; tiles = true, face_mm = (48, 48), sur_R = 0)
    # S14161-6050HS-04 tiles are 25.0 mm; two of them overhang the 48 mm face by 1 mm
    @test ro.tile_w ≈ 25f0 atol = 1e-5
    @test ro.x0 ≈ -1f0 atol = 1e-5
    cx, cy = channel_centres(ro)
    @test cx ≈ 24 .+ [-21.8, -15.6, -9.4, -3.2, 3.2, 9.4, 15.6, 21.8] atol = 1e-4
    @test cy == cx
    # channel assignment: centres, the seam, the margins, the overhang
    @test active_channel(ro, Float32(cx[5]), Float32(cy[3])) == (true, 5, 3)
    @test active_channel(ro, 24f0, 24f0)[1] == false            # seam between tiles
    @test active_channel(ro, 24.3f0, 10f0)[1] == true           # 0.1 mm inside channel 5
    @test active_channel(ro, 24.3f0, 10f0)[2] == 5
    @test active_channel(ro, Float32(cx[1] + 3.1), 10f0)[1] == false   # 0.2 mm margin
    @test active_channel(ro, 0f0, 10f0) == (true, 1, 2)         # crystal edge, overhanging channel
    @test active_channel(ro, 48f0, 46f0) == (true, 8, 8)
    @test tile_channel(-0.5f0, ro, Int32(8)) == 0
    # active fraction of the 48 x 48 footprint: 46.4^2 / 48^2 = 0.9344
    n = 0; na = 0
    for x in 0.005f0:0.01f0:48f0, y in 0.005f0:0.01f0:48f0
        n += 1; na += active_channel(ro, x, y)[1]
    end
    @test isapprox(na / n, (46.4 / 48)^2; atol = 1e-3)
    # the full-face readout is unchanged by the new fields
    ro0 = Readout(grid)
    @test !ro0.tiles && active_channel(ro0, 24f0, 24f0) == (true, 4, 4)
    @test_throws ErrorException Readout(grid; tiles = true)
    @test_throws ErrorException Readout(SipmGrid(6.2f0, Int32(6), Int32(8)); tiles = true,
                                        face_mm = (48, 48))

    # transport: straight down at a channel centre is detected in that channel; at the
    # seam with an absorbing dead area it is killed there
    op = ideal(2.15f0)
    acc = Accumulator(grid, TB)
    s = PhiloxStream(41, 1)
    @test propagate_photon!(acc, box, op, ro, TB, s, Float32(cx[6]), Float32(cy[2]),
                            0f0, 0f0, 0f0, 1f0, 0f0) == STATUS_DETECTED
    @test acc.counts[6, 2, 1] == 1
    @test propagate_photon!(acc, box, op, ro, TB, s, 24f0, 24f0,
                            0f0, 0f0, 0f0, 1f0, 0f0) == STATUS_ABS_SUR
    # conservation with an absorbing dead area and physical parameters
    opb = OpticalParams(2.15f0, 1.42f0, 20000f0, 2222f0, 0.99f0, false, true,
                        Float32(deg2rad(1.3)), 0.4f0)
    kw = (n_photons = 100_000, seed = 51, pos = (24f0, 24f0, 11.2f0), tau_ns = 300f0)
    a = run_photons!(box, opb, ro, TB; kw...)
    @test total_terminated(a) == kw.n_photons && a.nabs_sur > 0 && a.ndet > 0
    @test sum(a.counts) == a.ndet
    # no absorption anywhere + Lambertian dead area (R = 1): nothing is absorbed; the
    # directions super-critical on all faces stay trapped (cap), as on the full face
    rol = Readout(grid; tiles = true, face_mm = (48, 48), sur_R = 1)
    al = run_photons!(box, ideal(2.15f0), rol, TB; n_photons = 20_000, seed = 52,
                      pos = (24f0, 24f0, 11.2f0), tau_ns = 0f0)
    @test al.ndet + al.ncap == 20_000 && al.ndet > 0
    @test al.nabs_sur == al.nabs_sipm == al.nabs_wall == al.nabs_bulk == 0
    # the dead area costs light: detected fraction below the full-face readout
    rod = Readout(grid; tiles = true, face_mm = (48, 48), sur_R = 0.5)
    ad = run_photons!(box, opb, rod, TB; kw...)
    a0 = run_photons!(box, opb, Readout(SipmGrid(6f0, Int32(8), Int32(8))), TB; kw...)
    @test ad.ndet < a0.ndet
    # kernel parity on the CPU with the tiled layout, and Metal agreement
    b = run_photons_ka!(box, opb, rod, TB; kw..., ArrayT = Array)
    @test ad.counts == b.counts && ad.first_ns == b.first_ns
    @test (ad.ndet, ad.nabs_sur, ad.nabs_sipm, ad.nabs_bulk) ==
          (b.ndet, b.nabs_sur, b.nabs_sipm, b.nabs_bulk)
    if Metal.functional()
        g = run_photons_ka!(box, opb, rod, TB; kw..., ArrayT = Metal.MtlArray)
        @test total_terminated(g) == kw.n_photons
        p = ad.ndet / kw.n_photons
        @test abs(g.ndet - ad.ndet) < 5 * sqrt(kw.n_photons * p * (1 - p)) + 10
    end
end

@testset "Coupling layer and SiPM window (fullsim 0.6)" begin
    n1, n2, n3 = 2.15f0, 1.42f0, 1.57f0
    # the window helper reproduces T2 / (1 - R1 R2) at the refracted angle
    for deg in (0f0, 20f0, 35f0, 40f0)
        cosi = cos(deg2rad(deg))
        cos2 = sqrt(1f0 - (n1 / n2)^2 * (1f0 - cosi^2))
        _, R2 = fresnel(n2, n3, cos2)
        _, R1 = fresnel(n2, n1, cos2)
        pexp = (1 - R2) / (1 - R1 * R2)
        s = PhiloxStream(61, Int(deg) + 1)
        N = 400_000
        k = count(_ -> window_pass(s, n1, n2, n3, cosi), 1:N)
        @test isapprox(k / N, pexp; atol = 4 * sqrt(pexp * (1 - pexp) / N) + 1e-6)
    end
    # a window of the same index as the coupling layer transmits everything
    s = PhiloxStream(62, 1)
    @test all(_ -> window_pass(s, n1, n2, n2, 0.8f0), 1:1000)
    # the nine-argument constructor means no window
    op9 = OpticalParams(n1, n2, 20000f0, 2222f0, 0.99f0, false, true, 0.0227f0, 0.4f0)
    @test op9.n_window == 0f0
    # the window costs well below a percent of the light (0.3% at normal incidence)
    box = Box((48f0, 48f0, 22.4f0))
    ro = Readout(SipmGrid(6.2f0, Int32(8), Int32(8)); tiles = true, face_mm = (48, 48),
                 sur_R = 0.5)
    opw = OpticalParams(n1, n2, 20000f0, 2222f0, 0.99f0, false, true, 0.0227f0, 1f0, n3)
    op0 = OpticalParams(n1, n2, 20000f0, 2222f0, 0.99f0, false, true, 0.0227f0, 1f0)
    kw = (n_photons = 200_000, seed = 63, pos = (24f0, 24f0, 11.2f0), tau_ns = 0f0)
    aw = run_photons!(box, opw, ro, TB; kw...)
    a0 = run_photons!(box, op0, ro, TB; kw...)
    @test total_terminated(aw) == kw.n_photons
    @test 0 < a0.ndet - aw.ndet < 0.01 * a0.ndet
    # parity of the two transport versions with the window on
    b = run_photons_ka!(box, opw, ro, TB; kw..., ArrayT = Array)
    @test aw.counts == b.counts && aw.first_ns == b.first_ns && aw.ndet == b.ndet
    if Metal.functional()
        g = run_photons_ka!(box, opw, ro, TB; kw..., ArrayT = Metal.MtlArray)
        @test total_terminated(g) == kw.n_photons
        p = aw.ndet / kw.n_photons
        @test abs(g.ndet - aw.ndet) < 5 * sqrt(kw.n_photons * p * (1 - p)) + 10
    end
end

@testset "Two-component scintillation and the fullsim materials (fullsim 0.7)" begin
    s = PhiloxStream(71, 1)
    dt = [emission_time(s, 730f0, 3340f0, 0.64f0) for _ in 1:400_000]
    @test isapprox(mean(dt), 0.64 * 730 + 0.36 * 3340; rtol = 0.01)
    # tau2 = 0 draws exactly as the single component
    s1 = PhiloxStream(72, 1); s2 = PhiloxStream(72, 1)
    @test [emission_time(s1, 300f0) for _ in 1:1000] ==
          [emission_time(s2, 300f0, 0f0, 1f0) for _ in 1:1000]
    # the material files of the full simulation
    amb = load_material("bgo_ambient"); cold = load_material("bgo_250k")
    csi = load_material("csitl_ambient")
    @test scintillation_times(amb) == (tau_ns = 300f0, tau2_ns = 60f0, frac1 = 0.9f0)
    @test scintillation_times(cold) == (tau_ns = 840f0, tau2_ns = 0f0, frac1 = 1f0)
    @test scintillation_times(csi) == (tau_ns = 730f0, tau2_ns = 3340f0, frac1 = 0.64f0)
    @test amb["n"] == cold["n"] == load_material("bgo")["n"]
    @test csi["abs_length_mm"] == load_material("csitl")["abs_length_mm"]
    @test_throws ErrorException scintillation_times(Dict("name" => "x", "tau_ns" => 1.0,
                                                         "tau2_ns" => 2.0))
    # transport with two components: the time distribution of the detected photons has
    # the two-component mean (plus the ns of flight), and both versions agree bit for bit
    box = Box((48f0, 48f0, 37.2f0))
    op = OpticalParams(1.79f0, 1.42f0, 3115f0, 346f0, 0.99f0, false, true, 0.0227f0, 1f0,
                       1.57f0)
    ro = Readout(SipmGrid(6.2f0, Int32(8), Int32(8)); tiles = true, face_mm = (48, 48),
                 sur_R = 0.5)
    tbl = TimeBinning(100f0, Int32(400))
    kw = (n_photons = 100_000, seed = 73, pos = (24f0, 24f0, 18.6f0),
          scintillation_times(csi)...)
    a = run_photons!(box, op, ro, tbl; kw...)
    b = run_photons_ka!(box, op, ro, tbl; kw..., ArrayT = Array)
    @test a.counts == b.counts && a.first_ns == b.first_ns
    tcent = [(k - 0.5) * 100 for k in 1:400]
    hist = [sum(a.counts[:, :, k]) for k in 1:400]
    tmean = sum(hist .* tcent) / sum(hist)
    @test isapprox(tmean, 0.64 * 730 + 0.36 * 3340; rtol = 0.03)   # binned, ~1% of overflow
    # the event pipeline accepts the two components
    data = joinpath(dirname(@__DIR__), "data")
    gmat = Gamma.make_material("CsI", 4.51, joinpath(data, "xcom_CSI.csv"))
    r = run_events!(box, op, ro, tbl, gamma_crystal(box.L, gmat), csi["yield_per_mev"];
                    n_events = 5, seed = 74, scintillation_times(csi)...,
                    batch_photons = 200_000)
    @test length(r.npe) == 5 && all(r.npe .>= 0)
end

@testset "Na-22 source and the per-decay pipeline (fullsim 0.8)" begin
    rng = Xoshiro(81)
    decs = [na22_decay(rng) for _ in 1:100_000]
    @test all(d -> d[1].kind == CryspLight.GAMMA_1275 && d[1].e_mev == 1.2745, decs)
    @test isapprox(count(d -> length(d) == 3, decs) / 100_000, 0.903; atol = 0.004)
    d3 = first(d for d in decs if length(d) == 3)
    @test all(d3[2].dir .≈ .-d3[3].dir) && d3[2].e_mev == d3[3].e_mev == 0.511
    L = (48f0, 48f0, 22.4f0); src = (24.0, 24.0, -30.0)
    e, p = entry_point(src, (0.0, 0.0, 1.0), L)
    @test e == (24f0, 24f0, 0f0) && p ≈ 30
    @test entry_point(src, (0.0, 0.0, -1.0), L) === nothing
    @test entry_point(src, (1.0, 0.0, 0.0), L) === nothing
    dc = (23.9, 23.9, 30.0) ./ sqrt(2 * 23.9^2 + 30.0^2)        # just inside a corner
    e, p = entry_point(src, dc, L)
    @test isapprox(e[1], 47.9; atol = 1e-3) && isapprox(e[2], 47.9; atol = 1e-3) &&
          isapprox(e[3], 0; atol = 1e-3)
    @test p ≈ sqrt(2 * 23.9^2 + 30.0^2)
    @test entry_point(src, (24.0, 24.0, 30.0) ./ 45.3, L) === nothing   # the corner itself
    # geometric acceptance of the front face from 30 mm: 12.8% of the sphere
    acc = face_acceptance(48, 48, 30)
    @test isapprox(acc, 0.1276; atol = 1e-3)
    N = 400_000
    nin = count(_ -> entry_point(src, isotropic_dir64(rng), L) !== nothing, 1:N)
    @test isapprox(nin / N, acc; atol = 4 * sqrt(acc * (1 - acc) / N))

    # the per-decay pipeline on ambient BGO, CPU backend
    box = Box(L)
    mat = load_material("bgo_ambient")
    op = OpticalParams(2.15f0, 1.42f0, 20000f0, 2222f0, 0.99f0, false, true, 0.0227f0,
                       0.5f0, 1.57f0)
    ro = Readout(SipmGrid(6.2f0, Int32(8), Int32(8)); tiles = true, face_mm = (48, 48),
                 sur_R = 0.5)
    tbd = TimeBinning(100f0, Int32(40))
    data = joinpath(dirname(@__DIR__), "data")
    pv = gamma_crystal(L, Gamma.make_material("BGO", 7.13, joinpath(data, "xcom_BGO.csv")))
    kw = (source_mm = src, seed = 83, batch_photons = 300_000, ArrayT = Array,
          scintillation_times(mat)...)
    nd = 1500
    r = run_decays!(box, op, ro, tbd, pv, mat["yield_per_mev"]; n_decays = nd, kw...)
    ev, ia, pe = r.events, r.interactions, r.pes
    @test length(ev.edep_kev) == nd && size(ev.maps) == (8, 8, nd)
    @test all(ev.edep_kev .<= 1786.5f0)
    @test ev.edep_kev ≈ ev.edep_511_kev .+ ev.edep_1275_kev
    fint = count(>(0), ev.edep_kev) / nd
    @test 0.12 < fint < 0.26          # one or two gammas of the decay, 13% acceptance each
    @test any(ev.n_gammas_in .>= 2)   # decays with both a 511 and the 1275 keV gamma
    @test ia.offset[1] == 0 && ia.offset[end] == length(ia.x_mm) && issorted(ia.offset)
    @test pe.offset[1] == 0 && pe.offset[end] == length(pe.channel) && issorted(pe.offset)
    for k in 1:nd
        @test pe.offset[k + 1] - pe.offset[k] == ev.n_detected[k] == sum(Int, ev.maps[:, :, k])
        @test (ia.offset[k + 1] > ia.offset[k]) == (ev.edep_kev[k] > 0)
        @test ev.n_detected[k] <= ev.n_emitted[k]
    end
    @test all(ia.gamma_kind .∈ Ref((Int8(1), Int8(2))))
    @test all(ia.proc .∈ Ref((Int8(0), Int8(1), Int8(2))))
    k = findfirst(>(0), ev.n_detected)
    m = zeros(Int, 8, 8)
    for c in pe.channel[pe.offset[k] + 1:pe.offset[k + 1]]
        m[(c - 1) % 8 + 1, (c - 1) ÷ 8 + 1] += 1
    end
    @test m == Int.(ev.maps[:, :, k])
    tfl = 30 / 299.792458
    @test all(ia.t_ns .>= tfl - 1e-4) && all(pe.t_ns .>= tfl - 1e-4)
    # photopeak scale: 511 keV x 8000 ph/MeV x collection ~0.8 x PDE 0.5 ~ 1500 pe
    pk = ev.n_detected[(ev.edep_511_kev .> 510) .& (ev.edep_1275_kev .== 0)]
    @test !isempty(pk) && 1000 < mean(pk) < 2000
    # the intrinsic resolution broadens the emitted light of photopeak decays: with 20%
    # FWHM and the same seed the deposits are identical and the photon counts differ
    rs = run_decays!(box, op, ro, tbd, pv, mat["yield_per_mev"]; n_decays = nd,
                     intrinsic_fwhm = 0.20, kw...)
    @test rs.events.edep_kev == ev.edep_kev
    sel = (ev.edep_511_kev .> 510) .& (ev.edep_1275_kev .== 0)
    @test std(rs.events.n_emitted[sel]) > 2 * std(ev.n_emitted[sel])
    @test isapprox(mean(rs.events.n_emitted[sel]), mean(ev.n_emitted[sel]); rtol = 0.03)
    @test load_fullsim(joinpath(dirname(@__DIR__), "fullsim_runs", "2026-10_ambient",
                                "optical_bgo_ambient.toml")).intrinsic ==
          (intrinsic_fwhm = 0.05, intrinsic_exponent = 0.5)
    # two chunks with disjoint decay ranges reproduce the decays of the single call
    r1 = run_decays!(box, op, ro, tbd, pv, mat["yield_per_mev"]; n_decays = 800,
                     first_decay = 1, kw...)
    r2 = run_decays!(box, op, ro, tbd, pv, mat["yield_per_mev"]; n_decays = nd - 800,
                     first_decay = 801, kw...)
    @test vcat(r1.events.edep_kev, r2.events.edep_kev) == ev.edep_kev
    @test vcat(r1.events.n_emitted, r2.events.n_emitted) == ev.n_emitted
    @test vcat(r1.interactions.x_mm, r2.interactions.x_mm) == ia.x_mm
end

@testset "Photoelectron file and the fullsim configuration (fullsim 0.9)" begin
    repo = dirname(@__DIR__)
    st = load_fullsim(joinpath(repo, "fullsim_runs", "2026-10_ambient",
                               "optical_bgo_ambient.toml"))
    @test st.box.L == (48f0, 48f0, 22.4f0) && st.ro.tiles && st.ro.x0 ≈ -1f0
    @test st.op.n_coupling == 1.42f0 && st.op.n_window == 1.57f0 && st.op.pde == 1f0
    @test st.op.n_crystal == 2.15f0 && st.op.air_gap && st.op.sigma_alpha ≈ deg2rad(1.3f0)
    @test st.times == (tau_ns = 300f0, tau2_ns = 60f0, frac1 = 0.9f0)
    @test st.source_mm == (24.0, 24.0, -30.0) && st.yield_per_mev == 8000
    @test st.ro.sur_R == 0.5f0 && st.tag == "optical_bgo_ambient"

    # a short run through the writer, in two chunks, read back whole and per event
    tmp = mktempdir()
    path = joinpath(tmp, "pes.h5")
    kw = (source_mm = st.source_mm, seed = 91, batch_photons = 300_000, ArrayT = Array,
          st.times...)
    w = open_pes(path, 8, 8; cfg_text = st.cfg_text, pde_applied = 1.0,
                 lambda_nm = st.lambda_nm, attributes = Dict("tag" => "test"))
    r1 = run_decays!(st.box, st.op, st.ro, st.tb, st.pv, st.yield_per_mev;
                     n_decays = 600, first_decay = 1, kw...)
    r2 = run_decays!(st.box, st.op, st.ro, st.tb, st.pv, st.yield_per_mev;
                     n_decays = 400, first_decay = 601, kw...)
    w = append_pes!(w, r1)
    @test_throws ErrorException append_pes!(w, r1)       # wrong chunk order
    w = append_pes!(w, r2)
    @test w.n_decays == 1000
    close_pes(w)
    f = read_pes(path)
    @test f.attributes["n_decays"] == 1000 && f.attributes["tag"] == "test"
    @test f.attributes["pde_applied"] == 1.0 && f.attributes["lambda_nm"] == 480.0
    @test f.events.edep_kev == vcat(r1.events.edep_kev, r2.events.edep_kev)
    @test f.events.maps == cat(r1.events.maps, r2.events.maps; dims = 3)
    @test f.interactions.offset ==
          vcat(r1.interactions.offset, r1.interactions.offset[end] .+ r2.interactions.offset[2:end])
    @test f.interactions.e_kev == vcat(r1.interactions.e_kev, r2.interactions.e_kev)
    @test f.pes.offset == vcat(r1.pes.offset, r1.pes.offset[end] .+ r2.pes.offset[2:end])
    @test f.pes.channel == vcat(r1.pes.channel, r2.pes.channel)
    @test f.pes.t_ns == vcat(r1.pes.t_ns, r2.pes.t_ns)
    @test all(f.pes.lambda_nm .== 480) && length(f.pes.lambda_nm) == length(f.pes.channel)
    # re-binning: the maps rebuilt from the records equal the stored maps, event by event
    for k in 1:1000
        e = event_pes(f, k)
        m = zeros(UInt16, 8, 8)
        for c in e.channel
            m[(c - 1) % 8 + 1, (c - 1) ÷ 8 + 1] += 1
        end
        @test m == f.events.maps[:, :, k]
        @test length(e.t_ns) == f.events.n_detected[k]
    end
    # the driver end to end on a tiny configuration written in the temp dir
    cfgtxt = replace(st.cfg_text, r"n_decays = \d+" => "n_decays = 300",
                     r"chunk_decays = \d+" => "chunk_decays = 100",
                     r"\[run\]" => "[run]\noutput_dir = \"$(tmp)\"")
    cfgp = joinpath(tmp, "tiny_bgo.toml")
    write(cfgp, cfgtxt)
    out = run_fullsim_optical(cfgp; ArrayT = Array, verbose = false)
    @test isfile(out) && isfile(joinpath(tmp, "tiny_bgo", "tiny_bgo.toml"))
    g = read_pes(out)
    @test g.attributes["n_decays"] == 300 && length(g.events.edep_kev) == 300
    @test g.pes.offset[end] == length(g.pes.channel) == sum(g.events.n_detected)
    @test haskey(g.attributes, "code_version") && haskey(g.attributes, "elapsed_s")
    # the driver, chunked by 100, reproduces the gamma-level content of a direct call
    rd = run_decays!(st.box, st.op, st.ro, st.tb, st.pv, st.yield_per_mev;
                     n_decays = 300, seed = st.seed, source_mm = st.source_mm,
                     batch_photons = 300_000, ArrayT = Array, st.times..., st.intrinsic...)
    @test g.events.edep_kev == rd.events.edep_kev
    @test g.events.n_emitted == rd.events.n_emitted
    @test g.attributes["seed"] == st.seed
end

@testset "SiPM device model (fullsim 0.10)" begin
    dev = SipmDevice()
    @test dev.name == "S14160-6050HS" && dev.n_cells == 14331
    @test breakdown_voltage(dev, 298.15) == 38.0
    @test isapprox(breakdown_voltage(dev, 196.0), 34.5; atol = 0.05)
    @test isapprox(gain(dev, 2.7), 2.5e6; rtol = 0.01)
    @test pde(dev, 2.7, 450) == 0.50 && pde(dev, 3.2, 450) == 0.54 && pde(dev, 5.0, 450) == 0.54
    @test isapprox(pde(dev, 1.7, 450), 0.37) && isapprox(pde(dev, 2.2, 450), 0.435)
    @test pde(dev, 3.0, 480) < pde(dev, 3.0, 450) && pde(dev, 3.0, 550) < pde(dev, 3.0, 480)
    @test isapprox(crosstalk(dev, 2.7), 0.07) && isapprox(crosstalk(dev, 3.7), 0.13)
    @test isapprox(crosstalk(dev, 3.0), 0.07 + 0.3 * 0.06; atol = 1e-6)
    @test dark_rate(dev, 298.15, 2.7) == 6.0e6
    @test isapprox(dark_rate(dev, 298.15 - 9, 2.7), 3.0e6)
    @test isapprox(dark_rate(dev, 298.15, 3.2), 6.0e6 * 0.54 / 0.50)
    @test dark_rate(dev, 196.0, 6.5) < 2e4            # kHz in the cold, as measured
    @test isapprox(afterpulse_prob(dev, 3.0), 0.01) && isapprox(afterpulse_prob(dev, 6.0), 0.04)
    # pulse: the 50 ns recharge filtered by 50 ohm x 2000 pF = 100 ns, peak 1 reached
    # 100 ln 2 = 69.3 ns after the avalanche, then a 100 ns fall
    ts = 0:0.1:800
    v = [pulse_shape(dev, t) for t in ts]
    @test isapprox(maximum(v), 1.0; atol = 1e-6)
    tp = ts[argmax(v)]
    @test isapprox(tp, 100 * log(2); atol = 0.2)
    @test pulse_shape(dev, 0) == 0 && pulse_shape(dev, -5) == 0
    @test isapprox(pulse_shape(dev, 500) / pulse_shape(dev, 400), exp(-1); rtol = 0.02)
    # the two constants play symmetric roles; 100 ohm doubles the input one
    p1 = physical_pulse(100, 2000, 25); p2 = physical_pulse(50, 2000, 50)
    @test isapprox(p1.fall, p2.fall) && isapprox(p1.t_peak, p2.t_peak)
    d100 = with_input(dev, 100)
    @test d100.pulse_fall_ns == 200 && d100.input_ohm == 100 && dev.input_ohm == 50
    @test physical_pulse(50, 2000, 25).fall == 50.0       # equal constants: finite limit

    # operating point and response statistics
    op = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 64,
                       window_ns = (-1000.0, 1000.0))
    @test isapprox(op.vbr, 38.0 - 0.034 * 5; atol = 1e-9) && isapprox(op.pde_peak, 0.524)
    @test op.gain_factor == ones(64)
    ops = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 64, gain_spread = 0.05,
                        seed = 3)
    @test isapprox(mean(ops.gain_factor), 1; atol = 0.03) && std(ops.gain_factor) > 0.02
    # thinning: with crosstalk, afterpulsing and dark counts switched off through a
    # device copy, the kept fraction is the PDE at 480 nm
    nodark = SipmOperation(SipmDevice(merge(load_material("sipm_s14160"),
        Dict("device" => merge(load_material("sipm_s14160")["device"],
             Dict("dark_hz" => 0.0, "afterpulse_prob" => 0.0,
                  "crosstalk" => [0.0, 0.0, 0.0, 0.0])))));
        t_k = 293.15, vov = 3.0, n_channels = 1)
    rng = Xoshiro(101)
    n = 200_000
    tr = respond(nodark, zeros(Float32, n), fill(480f0, n), rng)
    p480 = pde(dev, 3.0, 480)
    @test isapprox(length(tr) / n, p480; atol = 4 * sqrt(p480 / n))
    @test all(a -> a.kind == 0 && a.q == 1, tr)
    # crosstalk alone: mean total charge per kept photon is 1 / (1 - lambda), lambda = -ln(1 - p)
    noct = SipmOperation(SipmDevice(merge(load_material("sipm_s14160"),
        Dict("device" => merge(load_material("sipm_s14160")["device"],
             Dict("dark_hz" => 0.0, "afterpulse_prob" => 0.0)))));
        t_k = 293.15, vov = 3.0, n_channels = 1)
    tr = respond(noct, zeros(Float32, n), fill(450f0, n), rng)
    nph = count(a -> a.kind == 0, tr)
    pct = crosstalk(dev, 3.0)
    @test isapprox(total_charge(tr) / nph, 1 / (1 + log(1 - pct)); rtol = 0.02)
    # branching: no extra cell with probability 1 - p, and the Borel tail
    nk = [CryspLight.n_crosstalk(rng, 0.195) for _ in 1:400_000]
    @test isapprox(mean(nk .== 0), 0.805; atol = 0.003)
    lam = -log(1 - 0.195)
    borel(n) = exp(-lam * n) * (lam * n)^(n - 1) / factorial(n)
    @test isapprox(mean(nk .== 4), borel(5); rtol = 0.1)
    # dark counts alone over the default window of 20 us: Poisson, mean rate x length
    opd = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 1)
    only_dark = respond(opd, Float32[], Float32[], rng)
    @test !isempty(only_dark) && all(a -> a.kind in (1, 2, 3), only_dark)
    nd = [count(a -> a.kind == 3, respond(opd, Float32[], Float32[], rng)) for _ in 1:400]
    mu = dark_rate(dev, 293.15, 3.0) * 20_000e-9
    @test isapprox(mean(nd), mu; atol = 4 * sqrt(mu / 400))
    @test all(a -> -10_000 <= a.t_ns <= 10_000, only_dark)
    # afterpulses: delayed, reduced, about 1% of the avalanches at 3 V
    tr = respond(op, zeros(Float32, n), fill(450f0, n), rng)
    ap = filter(a -> a.kind == 2, tr)
    nav = count(a -> a.kind != 2 && a.kind != 3, tr)
    @test isapprox(length(ap) / nav, 0.01; rtol = 0.15)
    @test all(a -> 0 < a.q < 1 && a.t_ns > 0, ap)
    @test isapprox(mean(a.t_ns for a in ap), 100; rtol = 0.1)
    @test issorted(a.t_ns for a in tr)
    # reproducible by seed
    t1 = respond(op, Float32[1, 2, 3], Float32[480, 480, 480], Xoshiro(5))
    t2 = respond(op, Float32[1, 2, 3], Float32[480, 480, 480], Xoshiro(5))
    @test t1 == t2
end

@testset "Shaping and the Citiroc chain (fullsim 0.11)" begin
    dev = SipmDevice()
    dt = 0.5
    # the device pulse on the grid peaks at 69 ns with a unit peak per avalanche
    sh = Shaper(dev, 87.5, dt)
    y = zeros(4000); y[1] = 1.0
    device_pulse!(y, sh)
    @test isapprox(maximum(y), 1.0; atol = 1e-9)
    @test isapprox((argmax(y) - 1) * dt, 100 * log(2); atol = 1.0)
    @test isapprox(y[round(Int, 500 / dt) + 1] / y[round(Int, 400 / dt) + 1], exp(-1); rtol = 0.02)
    # the shaped single avalanche: unit peak, at about tp after the device peak
    y = zeros(4000); y[1] = 1.0
    shape!(y, sh)
    @test isapprox(maximum(y), 1.0; atol = 1e-9)
    @test 170 < (argmax(y) - 1) * dt < 260     # 87.5 ns shaping of the 69/100 ns pulse
    # linearity: two avalanches give twice the peak
    y2 = zeros(4000); y2[1] = 2.0
    shape!(y2, sh)
    @test isapprox(maximum(y2), 2.0; atol = 1e-9)
    # shaped noise has the requested rms
    rng = Xoshiro(111)
    gen! = noise_generator(sh, 0.1, 4000, rng)
    # (on long series: the sample rms of a short, strongly correlated series is biased low)
    @test isapprox(std(gen!(zeros(2_000_000))), 0.1; rtol = 0.03)
    # the noise scale does not depend on the stream that built the generator: two
    # generators reseeded alike give the same series
    ra = Xoshiro(1); rb = Xoshiro(2)
    ga = noise_generator(sh, 0.1, 4000, ra); gb = noise_generator(sh, 0.1, 4000, rb)
    Random.seed!(ra, 5); Random.seed!(rb, 5)
    @test ga(zeros(500)) == gb(zeros(500))

    op = nodark(SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 64))
    s = CitirocSettings(dt_ns = 0.5, t_start_ns = -200.0, t_end_ns = 6000.0)
    ch = CitirocChain(s, op, Xoshiro(112))
    # the stored fraction of the design document (69/100 ns pulse, 87.5 ns peaking, voltage
    # preamplifier: the differentiation suppresses the slow part of the train)
    @test isapprox(stored_fraction(ch, 300), 0.29; atol = 0.02)
    @test isapprox(stored_fraction(ch, 730; tau2_ns = 3340, frac1 = 0.64), 0.10; atol = 0.01)
    @test isapprox(stored_fraction(ch, 2720), 0.041; atol = 0.005)
    # behind a charge-sensitive preamplifier the same chain keeps more (0.49 for 300 ns)
    chq = CitirocChain(CitirocSettings(dt_ns = 0.5, t_start_ns = -200.0, t_end_ns = 6000.0,
                                       voltage_preamp = false), op, Xoshiro(112))
    @test isapprox(stored_fraction(chq, 300), 0.49; atol = 0.02)
    # a voltage-sensitive chain does not pass a constant input
    shv = Shaper(dev, 87.5, dt; voltage = true)
    yc = ones(20000); shape!(yc, shv)
    @test abs(yc[end]) < 1e-6 && abs(maximum(yc[end-100:end])) < 1e-6
    @test isapprox(stored_fraction(ch, 1e-3), 1.0; atol = 1e-6)      # delta: everything

    # one avalanche at t = 0 in channel 5, nothing else, no noise: trigger on that
    # channel, the stored value one avalanche, the ADC at pedestal + counts per pe
    quiet = CitirocChain(CitirocSettings(noise_pe = 0.0, fast_noise_pe = 0.0),
                         nodark(SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 64)),
                         Xoshiro(113))
    trains = [Avalanche[] for _ in 1:64]
    trains[5] = [Avalanche(0f0, 1f0, Int8(0))]
    e = citiroc_event(quiet, trains)
    @test e.triggered && 0 < e.t_trigger < 60
    @test e.hits[5] && count(e.hits) == 1
    @test isapprox(e.stored_pe[5], 1.0; atol = 1e-3)
    @test e.hg[5] == round(UInt16, 200 + 10) && all(e.hg[c] == 200 for c in 1:64 if c != 5)
    @test e.lg[5] == 201
    @test !any(e.saturated)
    # below threshold: no trigger, nothing stored
    half = [Avalanche[] for _ in 1:64]; half[5] = [Avalanche(0f0, 0.3f0, Int8(0))]
    e0 = citiroc_event(quiet, half)
    @test !e0.triggered && isnan(e0.t_trigger) && all(e0.hg .== 0)
    # external trigger at a fixed time: recorded even below threshold, the peak
    # detector keeps the subthreshold avalanche
    ext = CitirocChain(CitirocSettings(noise_pe = 0.0, fast_noise_pe = 0.0, trigger = "external",
                                       external_trigger_ns = -10.0), op, Xoshiro(115))
    ee = citiroc_event(ext, half)
    @test ee.triggered && ee.t_trigger == -10f0
    @test isapprox(ee.stored_pe[5], 0.3; atol = 1e-3) && ee.stored_pe[1] == 0
    # majority 2 within 50 ns: one channel is not enough, two within the window are
    maj = CitirocChain(CitirocSettings(noise_pe = 0.0, fast_noise_pe = 0.0, majority = 2),
                       op, Xoshiro(114))
    @test !citiroc_event(maj, trains).triggered
    t2 = deepcopy(trains); t2[9] = [Avalanche(20f0, 1f0, Int8(0))]
    @test citiroc_event(maj, t2).triggered
    t3 = deepcopy(trains); t3[9] = [Avalanche(200f0, 1f0, Int8(0))]
    @test !citiroc_event(maj, t3).triggered
    @test trigger_time([5.0, 20.0, NaN, 300.0], 2, 50.0) == 20.0
    @test isnan(trigger_time([5.0, 300.0], 2, 50.0))
    # an early isolated crossing does not veto a later coincidence of the same channel
    @test trigger_time([[-1000.0, 0.0], [2.0]], 2, 50.0) == 2.0
    @test isnan(trigger_time([[-1000.0, 0.0], Float64[]], 2, 50.0))   # one channel is not two
    tlate = deepcopy(trains); tlate[1] = [Avalanche(-1000f0, 3f0, Int8(0)), Avalanche(0f0, 3f0, Int8(0))]
    tlate[9] = [Avalanche(0f0, 3f0, Int8(0))]
    maj3 = CitirocChain(CitirocSettings(noise_pe = 0.0, fast_noise_pe = 0.0, majority = 2,
                                        threshold_pe = 2.0, t_start_ns = -1200.0), op, Xoshiro(118))
    et = citiroc_event(maj3, tlate)
    @test et.triggered && 0 < et.t_trigger < 80      # the rise of the pulse delays the crossing
    # an exponential train of 1000 avalanches (tau 300 ns) stores about 270 of them
    rngt = Xoshiro(115)
    tr = [Avalanche(Float32(-300 * log(rand(rngt))), 1f0, Int8(0)) for _ in 1:1000]
    big = [Avalanche[] for _ in 1:64]; big[1] = tr
    eb = citiroc_event(quiet, big)
    @test eb.triggered && isapprox(eb.stored_pe[1], 270; rtol = 0.1)
    # track and hold at the hold time stores less than the peak detector
    th = CitirocChain(CitirocSettings(noise_pe = 0.0, fast_noise_pe = 0.0,
                                      peak_detector = false, hold_ns = 500.0), op, Xoshiro(116))
    @test citiroc_event(th, big).stored_pe[1] < eb.stored_pe[1]
    # saturation: more than the input-charge ceiling flags and clamps
    @test isapprox(quiet.pe_ceiling, 400 / (1.602e-7 * 0.93e6 * 3.0); rtol = 1e-3)
    huge = [Avalanche[] for _ in 1:64]
    huge[2] = [Avalanche(0f0, Float32(quiet.pe_ceiling * 1.2), Int8(0))]
    eh = citiroc_event(quiet, huge)
    @test eh.saturated[2] && eh.stored_pe[2] <= quiet.pe_ceiling + 1e-3
    # a long train whose total exceeds the ceiling but whose stored amplitude does not is
    # not saturated
    slowtr = [Avalanche[] for _ in 1:64]
    slowtr[2] = [Avalanche(Float32(-1000 * log(rand(rngt))), Float32(quiet.pe_ceiling / 500), Int8(0)) for _ in 1:1000]
    es2 = citiroc_event(quiet, slowtr)
    @test !es2.saturated[2] && es2.stored_pe[2] < quiet.pe_ceiling
    # with noise and the peak detector, an empty channel stores a positive bias
    noisy = CitirocChain(CitirocSettings(hold_ns = 300.0), op, Xoshiro(117))
    vals = [citiroc_event(noisy, trains).stored_pe[20] for _ in 1:200]
    @test 0.05 < mean(vals) < 0.5
    # a high gain at full scale is not a saturation: the low gain measures the channel
    big2 = [Avalanche[] for _ in 1:64]
    big2[1] = [Avalanche(Float32(-300 * log(rand(rngt))), 1f0, Int8(0)) for _ in 1:1500]
    eb2 = citiroc_event(quiet, big2)
    @test eb2.hg[1] == 4095 && !eb2.saturated[1] && isapprox(eb2.stored_pe[1], 405; rtol = 0.1)
    # separate shaping of the two gains: short high gain, low gain at 87.5 ns keeps the
    # stored fraction of 87.5 ns
    two = CitirocChain(CitirocSettings(noise_pe = 0.0, fast_noise_pe = 0.0, tp_ns = 25.0,
                                       tp_lg_ns = 87.5, dt_ns = 0.5), op, Xoshiro(119))
    e2g = citiroc_event(two, big)
    @test isapprox(Float64(e2g.lg[1]) - 200, 270; rtol = 0.1)
    @test Float64(e2g.hg[1]) - 200 < 10 * (Float64(e2g.lg[1]) - 200)
    @test citiroc_event(two, trains).lg[5] == 201
    # self-triggered calibration: a single avalanche held at its maximum gives the
    # pedestal plus one gain; the delay is the time from the crossing to the maximum
    @test CryspLight.noise_at(CitirocSettings(), 87.5) == 0.1
    @test isapprox(CryspLight.noise_at(CitirocSettings(), 21.875), 0.2)
    d = hold_at_peak(two)
    @test 20 < d < 80
    hv = hg_at_crossings(two, trains, d)
    @test hv[5] == [210.0] && all(isempty(hv[c]) for c in 1:64 if c != 5)
    @test only(hg_at_crossings(two, trains, d - 45)[5]) < 210
    # with the peak detector a late hold keeps the maximum, while the track and hold has
    # passed it
    @test only(hg_at_crossings(two, trains, d + 45)[5]) < 210
    @test hg_at_crossings(two, trains, d + 45; peak_detector = true)[5] == [210.0]
end

@testset "Digitizer chain on the avalanche train (fullsim 0.11b)" begin
    dev = SipmDevice()
    @test pulse_cdf(dev, 0) == 0 && pulse_cdf(dev, -3) == 0
    @test isapprox(pulse_cdf(dev, 4000), 1.0; atol = 1e-9)
    @test 0.1 < pulse_cdf(dev, 60) < 0.4                       # 69 ns to the peak
    @test isapprox(window_fraction(dev, 0, -100, 4000), 1.0; atol = 1e-9)
    @test window_fraction(dev, 500, 0, 100) == 0                  # avalanche after the window
    @test isapprox(window_fraction(dev, 0, 0, 30) + window_fraction(dev, 0, 30, 4000), 1.0; atol = 1e-9)

    op = nodark(SipmOperation(dev; t_k = 293.15, vov = 5.0, n_channels = 64))   # 43 V ambient
    quiet = DigitizerChain(DigitizerSettings(noise_counts = 0.0, majority = 1,
                                             threshold_counts = 10.0), op, Xoshiro(121))
    @test isapprox(quiet.peak_counts_per_pe, 16.0)
    # the charge of one avalanche is the area of its sampled pulse (count-samples)
    yq = zeros(Float64, 4000); yq[1] = 1.0; device_pulse!(yq, quiet.pulse)
    @test isapprox(quiet.area_per_pe, 16.0 * sum(yq) * quiet.grid.dt / 8.0; rtol = 1e-6)
    @test quiet.n_gate == 125 && quiet.n_base == 125 && quiet.noise_charge == 0
    # one avalanche at t = 0 in channel 7: triggers (16 > 10), its whole area in the gate,
    # which starts 50 ns before the trigger
    trains = [Avalanche[] for _ in 1:64]
    trains[7] = [Avalanche(0f0, 1f0, Int8(0))]
    e = digitizer_event(quiet, trains)
    @test e.triggered && -10 <= e.t_trigger <= 60 && e.hits[7] && count(e.hits) == 1
    @test isapprox(e.charge_pe[7], 1.0; atol = 0.02) && e.charge[7] == round(Int32, quiet.area_per_pe * e.charge_pe[7])
    @test all(e.charge[c] == 0 for c in 1:64 if c != 7)
    @test isapprox(e.peak_counts[7], 16; rtol = 0.05)
    # a small pulse in the baseline window (half an avalanche, below the threshold) is
    # subtracted with the length ratio of the gate to the baseline window
    tb = deepcopy(trains); tb[8] = [Avalanche(-500f0, 0.5f0, Int8(3))]
    eb = digitizer_event(quiet, tb)
    @test isapprox(eb.charge_pe[8], -0.5; atol = 0.02)
    # 60 counts threshold at 16 counts per avalanche: one or two avalanches do not
    # trigger, a pile-up of eight at once does
    amb = DigitizerChain(DigitizerSettings(noise_counts = 0.0, majority = 1), op, Xoshiro(122))
    @test !digitizer_event(amb, trains).triggered
    t2 = deepcopy(trains); t2[7] = [Avalanche(0f0, 2f0, Int8(0))]
    @test !digitizer_event(amb, t2).triggered
    t8 = deepcopy(trains); t8[7] = [Avalanche(0f0, 8f0, Int8(0))]
    @test digitizer_event(amb, t8).triggered
    # majority 2 within 100 ns
    maj = DigitizerChain(DigitizerSettings(noise_counts = 0.0, majority = 2), op, Xoshiro(123))
    @test !digitizer_event(maj, t8).triggered
    t82 = deepcopy(t8); t82[30] = [Avalanche(50f0, 8f0, Int8(0))]
    @test digitizer_event(maj, t82).triggered
    t83 = deepcopy(t8); t83[30] = [Avalanche(500f0, 8f0, Int8(0))]
    @test !digitizer_event(maj, t83).triggered
    # integrated noise: 4 counts rms per sample over 125 + 125^2/125 samples
    noisy = DigitizerChain(DigitizerSettings(majority = 1, threshold_counts = 10.0), op, Xoshiro(124))
    @test isapprox(noisy.noise_charge, 4 * sqrt(250.0))
    qs = [digitizer_event(noisy, t8).charge[20] for _ in 1:400]
    @test isapprox(std(qs), 4 * sqrt(250.0); rtol = 0.15) && abs(mean(qs)) < 3 * 63 / 20
    # an exponential train keeps its whole charge in a gate longer than the decay, and
    # a fraction of it otherwise (a third for 300 ns, the 100 ns pulse delaying the charge)
    rngt = Xoshiro(125)
    tr = [Avalanche(Float32(-300 * log(rand(rngt))), 1f0, Int8(0)) for _ in 1:2000]
    big = [Avalanche[] for _ in 1:64]; big[1] = tr
    long = DigitizerChain(DigitizerSettings(noise_counts = 0.0, majority = 1, gate_ns = 4000.0),
                          op, Xoshiro(126))
    el = digitizer_event(long, big)
    @test el.triggered && isapprox(el.charge_pe[1], 2000; rtol = 0.01)
    short = DigitizerChain(DigitizerSettings(noise_counts = 0.0, majority = 1, gate_ns = 300.0),
                           op, Xoshiro(127))
    es = digitizer_event(short, big)
    @test 0.25 < es.charge_pe[1] / 2000 < 0.45
    # external trigger: threshold 0 places the gate at the given time without crossings
    ext = DigitizerChain(DigitizerSettings(noise_counts = 0.0, threshold_counts = 0.0,
                                           external_trigger_ns = 0.0), op, Xoshiro(129))
    ee = digitizer_event(ext, trains)
    @test ee.triggered && ee.t_trigger == 0 && isapprox(ee.charge_pe[7], 1.0; atol = 0.02)
    @test digitizer_event(ext, [Avalanche[] for _ in 1:64]).triggered
    # saturation of the input range at a high gain
    sat = DigitizerChain(DigitizerSettings(noise_counts = 0.0, majority = 1, gain = 100.0,
                                           threshold_counts = 100.0), op, Xoshiro(128))
    @test digitizer_event(sat, t8).saturated[7] == false
    thuge = deepcopy(trains); thuge[7] = [Avalanche(0f0, 30f0, Int8(0))]   # 48000 counts
    @test digitizer_event(sat, thuge).saturated[7]
    # majority 2: an isolated dark pulse at -1000 ns in channel 1 does not veto the
    # later coincidence of channels 1 and 2 at 0
    dmaj = DigitizerChain(DigitizerSettings(noise_counts = 0.0, gain = 4.0, threshold_counts = 480.0,
                                            majority = 2), op, Xoshiro(130))
    tl = [Avalanche[] for _ in 1:64]
    tl[1] = [Avalanche(-1000f0, 12f0, Int8(0)), Avalanche(0f0, 12f0, Int8(0))]
    tl[2] = [Avalanche(0f0, 12f0, Int8(0))]
    @test digitizer_event(dmaj, tl).triggered
end

@testset "Front-end stage and the event file (fullsim 0.12)" begin
    repo = dirname(@__DIR__)
    tmp = mktempdir()
    st = load_fullsim(joinpath(repo, "fullsim_runs", "2026-10_ambient", "optical_bgo_ambient.toml"))
    cfgtxt = replace(st.cfg_text, r"n_decays = \d+" => "n_decays = 400",
                     r"chunk_decays = \d+" => "chunk_decays = 200",
                     r"\[run\]" => "[run]\noutput_dir = \"$(tmp)\"")
    write(joinpath(tmp, "tiny.toml"), cfgtxt)
    pes_path = run_fullsim_optical(joinpath(tmp, "tiny.toml"); ArrayT = Array, verbose = false)
    # thresholds high enough that the dark pulses with crosstalk do not trigger: at
    # ambient temperature a dark avalanche with four or more crosstalk companions occurs
    # about once per event window at 3 V, and twice at 5 V
    for (kind, extra) in (("citiroc", "tp_ns = 87.5\nhold_ns = 300.0\nthreshold_pe = 8.0\nmajority = 1"),
                          ("digitizer", "gain = 1.0\nthreshold_counts = 130.0\nmajority = 2\ngate_ns = 1000.0"))
        fcfg = """
        [run]
        tag = "fe_$(kind)"
        pes = "$(pes_path)"
        seed = 7
        output_dir = "$(tmp)"
        [sipm]
        temperature_k = 293.15
        overvoltage_v = $(kind == "citiroc" ? 3.0 : 5.0)
        gain_spread = 0.05
        window_ns = [-3200.0, 3000.0]
        [frontend]
        type = "$(kind)"
        $(extra)
        """
        fp = joinpath(tmp, "fe_$(kind).toml")
        write(fp, fcfg)
        fs = load_frontend(fp)
        @test fs.frontend == kind && fs.op.vov == (kind == "citiroc" ? 3.0 : 5.0)
        @test fs.chip_channels == (kind == "citiroc" ? 32 : 64)
        kind == "citiroc" && @test fs.settings.threshold_pe == 8.0 && fs.settings.tp_ns == 87.5
        kind == "digitizer" && @test fs.settings.majority == 2 && fs.settings.gate_ns == 1000.0
        out = run_fullsim_frontend(fp; verbose = false)
        @test isfile(out)
        h5open(out, "r") do f
            @test attrs(f["run"])["frontend"] == kind && attrs(f["run"])["n_events"] == 400
            @test attrs(f["run"])["crystal"] == "bgo_ambient"
            @test read(f["channels"]["channel"]) == Int16.(1:64)
            @test isapprox(read(f["channels"]["x_mm"])[5], 24 + 3.2; atol = 1e-3)
            @test read(f["channels"]["chip"])[33] == (kind == "citiroc" ? 1 : 0)
            @test length(read(f["calib"]["gain_per_pe"])) == 64
            q = read(f["events"]["charge"])
            @test size(q) == (64, 400)
            trig = read(f["events"]["triggered"])
            edep = read(f["mc"]["edep_kev"])
            @test length(trig) == 400 && edep == read_pes(pes_path).events.edep_kev
            # photopeak events trigger; decays without a deposit almost never do
            pk = edep .> 510
            @test count(pk) > 0 && mean(trig[pk]) > 0.8
            @test mean(trig[edep .== 0]) < 0.2
            # the stored charge of photopeak events, in avalanches, is on the expected scale
            qpe = read(f["events"]["charge_pe"])
            tot = vec(sum(qpe; dims = 1))
            good = pk .& (trig .== 1)
            if kind == "citiroc"
                @test 600 < median(tot[good]) < 1100         # 3253 x 0.52 x 0.49 ~ 830, plus dark
                @test size(read(f["events"]["charge_lg"])) == (64, 400)
            else
                @test 1400 < median(tot[good]) < 2200        # 3253 x 0.54 x ~1 ~ 1750 + dark
            end
            @test all(read(f["events"]["hits"])[:, .!Bool.(trig)] .== 0)
            @test haskey(f["mc"], "photon_maps") && size(read(f["mc"]["photon_maps"])) == (8, 8, 400)
        end
    end
end


@testset "Radioroc2 chain (fullsim 0.15)" begin
    dev = SipmDevice()
    op = nodark(SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 64))
    tdev = with_pulse(dev, 5.0, 20.0)
    @test tdev.pulse_rise_ns == 5 && tdev.pulse_fall_ns == 20 && tdev.n_cells == dev.n_cells
    # no noise and no ceiling, except where the ceiling is the subject
    quiet(; lg_ceiling_pe = 1e9, kw...) =
        RadiorocChain(RadiorocSettings(; noise_pe_100ns = 0.0, time_noise_pe = 0.0,
                                       lg_ceiling_pe = lg_ceiling_pe, kw...), op, Xoshiro(151))
    # stored fractions of the design document table for the slow shaping times
    ch = quiet(tp_ns = fill(1800.0, 64))
    @test isapprox(stored_fraction(ch, 300), 0.97; atol = 0.02)
    @test isapprox(stored_fraction(ch, 2720), 0.55; atol = 0.03)
    @test isapprox(stored_fraction(ch, 1000), 0.81; atol = 0.03)
    ch3 = quiet(tp_ns = fill(300.0, 64))
    @test isapprox(stored_fraction(ch3, 300), 0.70; atol = 0.03)
    ch12 = quiet(tp_ns = fill(1200.0, 64))
    @test isapprox(stored_fraction(ch12, 300), 0.94; atol = 0.02)
    # per-channel peaking times: one calibrated shaper per distinct value
    tps = vcat(fill(300.0, 32), fill(1800.0, 32))
    chm = quiet(tp_ns = tps)
    @test length(chm.shapers) == 2
    @test stored_fraction(chm, 2720; channel = 1) < stored_fraction(chm, 2720; channel = 40)
    # noise scaling
    s0 = RadiorocSettings()
    @test noise_at(s0, 100) == 0.1 && isapprox(noise_at(s0, 1800), 0.1 * sqrt(100 / 1800))
    # ceilings for this gain (2.79e6): 2000 x 1e6 / gain = 717 avalanches on the low gain
    # and a tenth, 72, on the high gain at the default gain ratio
    chc = quiet(lg_ceiling_pe = 2000.0)
    @test isapprox(chc.lg_ceiling, 2000e6 / op.gain) && isapprox(chc.hg_ceiling, 200e6 / op.gain)

    # one avalanche: T1 fires, T2 (3 pe) and TQ (50 pe) do not; the trigger on T1 stores
    # one avalanche on the high gain, pedestal + 10 counts
    trains = [Avalanche[] for _ in 1:64]
    trains[5] = [Avalanche(0f0, 1f0, Int8(0))]
    e1 = radioroc_event(quiet(trigger = "t1"), trains)
    @test e1.triggered && e1.hits[5] && count(e1.hits) == 1
    @test isapprox(e1.stored_pe[5], 1.0; atol = 1e-3) && e1.hg[5] == 210 && e1.lg[5] == 201
    @test !radioroc_event(quiet(trigger = "t2"), trains).triggered
    # self-triggered calibration on T1 with the peak detector up to the maximum
    chs = quiet(tp_ns = fill(25.0, 64), tp_lg_ns = fill(1800.0, 64))
    dr = hold_at_peak(chs)
    @test 60 < dr < 160              # 25 ns shaping of the 69/100 ns pulse
    @test hg_at_crossings(chs, trains, dr)[5] == [210.0]
    # crossings within the hold of the previous one give no value
    @test length(CryspLight.held_at_crossings(collect(1.0:100.0), [0.0, 5.0, 60.0], CryspLight.TimeGrid(0.0, 1.0, 100), 30.0, true)) == 2
    @test !radioroc_event(quiet(trigger = "tq"), trains).triggered
    # a 1000-avalanche exponential train (tau 300 ns) in one channel: the charge trigger
    # fires, the stored value is about 0.67 of it at 300 ns shaping and 0.97 at 1.8 us;
    # a time threshold T2 at 3 pe fires on the pile-up as well
    rngt = Xoshiro(152)
    tr = [Avalanche(Float32(-300 * log(rand(rngt))), 1f0, Int8(0)) for _ in 1:1000]
    big = [Avalanche[] for _ in 1:64]; big[1] = tr
    eq3 = radioroc_event(quiet(trigger = "tq", tp_ns = fill(300.0, 64)), big)
    @test eq3.triggered && isapprox(eq3.stored_pe[1], 670; rtol = 0.1)
    eq18 = radioroc_event(quiet(trigger = "tq", tp_ns = fill(1800.0, 64), hold_ns = 3000.0), big)
    @test eq18.triggered && isapprox(eq18.stored_pe[1], 970; rtol = 0.05)
    @test eq18.t_trigger > eq3.t_trigger        # the slower shaper crosses TQ later
    @test radioroc_event(quiet(trigger = "t2"), big).triggered
    # majority of T1 within 50 ns: two channels needed
    @test !radioroc_event(quiet(trigger = "majority"), trains).triggered
    t2 = deepcopy(trains); t2[9] = [Avalanche(20f0, 1f0, Int8(0))]
    @test radioroc_event(quiet(trigger = "majority"), t2).triggered
    # above the high-gain ceiling the stored value comes from the low-gain path and the
    # event is not saturated; above the low-gain ceiling it is
    huge = [Avalanche[] for _ in 1:64]
    huge[2] = [Avalanche(0f0, Float32(chc.hg_ceiling * 1.5), Int8(0))]
    eh = radioroc_event(quiet(trigger = "t1", lg_ceiling_pe = 2000.0), huge)
    @test !eh.saturated[2] && isapprox(eh.stored_pe[2], 1.5 * chc.hg_ceiling; rtol = 1e-3)
    @test eh.hg[2] == round(UInt16, 200 + 10 * chc.hg_ceiling)
    @test isapprox(eh.lg[2], 200 + 10 * 0.1 * chc.hg_ceiling * 1.5; atol = 1.5)
    huge[2] = [Avalanche(0f0, Float32(chc.lg_ceiling * 1.2), Int8(0))]
    eh2 = radioroc_event(quiet(trigger = "t1", lg_ceiling_pe = 2000.0), huge)
    @test eh2.saturated[2] && eh2.stored_pe[2] <= chc.lg_ceiling + 1e-3
    # the settings reader accepts a scalar peaking time and a vector
    rs = settings_from(RadiorocSettings, Dict("tp_ns" => 600.0, "trigger" => "tq", "hold_ns" => 800))
    @test rs.tp_ns == fill(600.0, 64) && rs.hold_ns == 800.0 && rs.trigger == "tq"
    rv = settings_from(RadiorocSettings, Dict("tp_ns" => collect(1.0:64.0)))
    @test rv.tp_ns == collect(1.0:64.0)
    # the steady current of the dark counts gives no offset: with the slow shaping, the
    # value held at a random time on empty records averages to zero, early or late in the
    # record (the grid starts four peaking times before the record)
    opd = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 4, window_ns = (-9000.0, 6000.0))
    held(hold) = begin
        chd = RadiorocChain(RadiorocSettings(; noise_pe_100ns = 0.0, time_noise_pe = 0.0, tp_ns = fill(1800.0, 4),
                                             trigger = "external", external_trigger_ns = 0.0,
                                             peak_detector = false, hold_ns = hold, t_end_ns = 5000.0),
                            opd, Xoshiro(152))
        rd = Xoshiro(153)
        [radioroc_event(chd, [respond(opd, Float32[], Float32[], rd) for _ in 1:4]).stored_pe[1] for _ in 1:400]
    end
    early, late = held(0.0), held(4000.0)
    @test abs(mean(early)) < 4 * std(early) / sqrt(400) && abs(mean(late)) < 4 * std(late) / sqrt(400)
    @test isapprox(std(early), std(late); rtol = 0.2)
end

# the analysis library (fullsim_analysis/lib), for the calibration on the dark counts
include(joinpath(dirname(@__DIR__), "fullsim_analysis", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis

@testset "Dark-count calibration: ladder, dark baseline, matched filter (fullsim 0.16)" begin
    dev = SipmDevice()
    # the level removed with the mean dark current: zero behind the voltage preamplifier of
    # the Citiroc, the dark charge per sample times the area over peak of the shaped single
    # avalanche behind the charge preamplifier of the Radioroc2 (1.47 cells at room
    # temperature and 100 ns, 0.05 at 250 K), and 2.0 cells with the 100 ohm input
    opw = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 1, window_ns = (-9600.0, 6000.0))
    opc = SipmOperation(dev; t_k = 250.0, vov = 3.0, n_channels = 1, window_ns = (-9600.0, 6000.0))
    rr(op; kw...) = RadiorocChain(RadiorocSettings(; tp_ns = [100.0], tp_lg_ns = [1800.0], noise_pe_100ns = 0.0,
                                                   time_noise_pe = 0.0, t_end_ns = 6000.0, kw...), op, Xoshiro(1))
    chw, chc = rr(opw), rr(opc)
    @test isapprox(dark_baseline(chw), 1.4736; atol = 0.002)
    @test isapprox(dark_baseline(chc), 0.0531; atol = 0.001)
    @test abs(dark_baseline(CitirocChain(CitirocSettings(tp_ns = 25.0, tp_lg_ns = 87.5, t_end_ns = 3000.0), opw, Xoshiro(1)))) < 1e-4
    sh = chw.shapers[100.0]
    y = ones(20_000); shape!(y, sh)
    @test isapprox(dark_baseline(sh, opw), y[end] * CryspLight.dark_charge_per_ns(opw) * sh.dt; rtol = 1e-9)
    # the input resistance is a setting of the run ([sipm] input_ohm): 100 ohm doubles the
    # fall of the pulse, moves the hold of the calibration from 229 to 268 ns and the
    # removed level to 2.0 cells, and leaves the cold stored fraction at 1.8 us almost unchanged
    tmp = mktempdir()
    repo = dirname(@__DIR__)
    cfg = read(joinpath(repo, "fullsim_runs", "2026-10_bgo250k", "frontend_bgo_250k_radioroc.toml"), String)
    write(joinpath(tmp, "r100.toml"), replace(cfg, "device = \"sipm_s14160\"" => "device = \"sipm_s14160\"\ninput_ohm = 100.0"))
    st100 = load_frontend(joinpath(tmp, "r100.toml"))
    @test st100.op.dev.input_ohm == 100 && st100.op.dev.pulse_fall_ns == 200 && isapprox(st100.op.dev.pulse_rise_ns, 92.4; atol = 0.1)
    @test load_frontend(joinpath(repo, "fullsim_runs", "2026-10_bgo250k", "frontend_bgo_250k_radioroc.toml")).op.dev.input_ohm == 50
    op100 = SipmOperation(st100.op.dev; t_k = 293.15, vov = 3.0, n_channels = 1, window_ns = (-9600.0, 6000.0))
    ch100 = rr(op100)
    @test hold_at_peak(chw) == 229 && hold_at_peak(ch100) == 268
    @test isapprox(dark_baseline(ch100), 2.0095; atol = 0.002)
    slow(ch) = stored_fraction(RadiorocChain(RadiorocSettings(tp_ns = [1800.0], noise_pe_100ns = 0.0, time_noise_pe = 0.0,
                                                              t_end_ns = 12000.0), ch.op, Xoshiro(1)), 840.0)
    opw12 = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 1, window_ns = (-15000.0, 12000.0))
    op100_12 = SipmOperation(st100.op.dev; t_k = 293.15, vov = 3.0, n_channels = 1, window_ns = (-15000.0, 12000.0))
    f50 = stored_fraction(RadiorocChain(RadiorocSettings(tp_ns = [1800.0], noise_pe_100ns = 0.0, time_noise_pe = 0.0, t_end_ns = 12000.0), opw12, Xoshiro(1)), 840.0)
    f100 = stored_fraction(RadiorocChain(RadiorocSettings(tp_ns = [1800.0], noise_pe_100ns = 0.0, time_noise_pe = 0.0, t_end_ns = 12000.0), op100_12, Xoshiro(1)), 840.0)
    @test isapprox(f50, 0.843; atol = 0.002) && isapprox(f100, 0.846; atol = 0.002)
    # the stored fractions quoted in the notes: Citiroc at 87.5 ns with the BGO mixture
    # (90% of 300 ns, 10% of 60 ns) 0.33, CsI(Tl) 0.10, cold BGO 0.12; Radioroc2 at 1.8 us
    # 0.97, 0.71, 0.84
    cit = CitirocChain(CitirocSettings(tp_ns = 87.5, noise_pe = 0.0, fast_noise_pe = 0.0, t_end_ns = 8000.0), opw12, Xoshiro(1))
    @test isapprox(stored_fraction(cit, 300.0; tau2_ns = 60.0, frac1 = 0.9), 0.326; atol = 0.005)
    @test isapprox(stored_fraction(cit, 730.0; tau2_ns = 3340.0, frac1 = 0.64), 0.101; atol = 0.005)
    @test isapprox(stored_fraction(cit, 840.0), 0.123; atol = 0.005)
    r18 = RadiorocChain(RadiorocSettings(tp_ns = [1800.0], noise_pe_100ns = 0.0, time_noise_pe = 0.0, t_end_ns = 12000.0), opw12, Xoshiro(1))
    @test isapprox(stored_fraction(r18, 300.0; tau2_ns = 60.0, frac1 = 0.9), 0.971; atol = 0.005)
    @test isapprox(stored_fraction(r18, 730.0; tau2_ns = 3340.0, frac1 = 0.64), 0.708; atol = 0.005)

    # the controlled example: isolated clusters of 1, 2 and 3 cells, no noise, no dark
    # avalanche in the record, hold at the peak; the held value is n minus the removed
    # level, so the one-cell pulse sits below the pedestal at room temperature
    held(ch, n) = begin
        t1, _, _, hg, _ = CryspLight.radioroc_waves(ch, [[Avalanche(0f0, Float32(n), Int8(3))]])
        only(CryspLight.held_at_crossings(hg[1], t1[1], ch.grid, hold_at_peak(ch), true))
    end
    hw = [held(chw, n) for n in 1:3]; hc = [held(chc, n) for n in 1:3]
    @test all(isapprox.(hw, (1:3) .- dark_baseline(chw); atol = 1e-6))
    @test all(isapprox.(hc, (1:3) .- dark_baseline(chc); atol = 1e-6))
    @test isapprox(200 + 10hw[1], 195.26; atol = 0.01) && isapprox(200 + 10hc[1], 209.47; atol = 0.01)

    # the same clusters embedded in stationary dark histories (noise 0.1 pe, grid 2 ns):
    # the ladder fitted on the self-triggered spectrum numbers its peaks from the expected
    # zero, pedestal - gain x baseline, and the fitted gain and zero follow; the three
    # isolated held values above receive their number of cells, and so do the embedded
    # clusters with no other avalanche within 300 ns, in the median and mostly one by one
    function dark_spectrum(op, nrec)
        s = RadiorocSettings(tp_ns = [100.0], tp_lg_ns = [1800.0], noise_pe_100ns = 0.1, time_noise_pe = 0.1,
                             t_end_ns = 6000.0, dt_ns = 2.0)
        ch = RadiorocChain(s, op, Xoshiro(1))
        delay = hold_at_peak(ch)
        vals = Float64[]; lab = Int[]; labval = Float64[]
        for i in 1:nrec
            Random.seed!(ch.rng, i)
            tr = respond(op, Float32[], Float32[], Xoshiro(i))
            n = 1 + i % 3
            isolated = all(a -> abs(a.t_ns) > 300, tr)
            push!(tr, Avalanche(0f0, Float32(n), Int8(0))); sort!(tr; by = a -> a.t_ns)
            t1, _, _, hg, _ = CryspLight.radioroc_waves(ch, [tr])
            append!(vals, CryspLight.held_at_crossings(hg[1], t1[1], ch.grid, delay, true))
            j = findfirst(t -> abs(t) <= 10, t1[1])
            if isolated && j !== nothing
                k0 = sample_index(ch.grid, t1[1][j]); k1 = sample_index(ch.grid, t1[1][j] + delay)
                push!(lab, n); push!(labval, maximum(view(hg[1], k0:k1)))
            end
        end
        rng = Xoshiro(9)
        adc = clamp.(round.(200 .+ 10 .* vals), 0, 4095) .+ rand(rng, length(vals)) .- 0.5
        return (adc = adc, baseline = dark_baseline(ch), lab = lab, labadc = 200 .+ 10 .* labval)
    end
    opw2 = SipmOperation(dev; t_k = 293.15, vov = 3.0, n_channels = 1, window_ns = (-8600.0, 6000.0))
    opc2 = SipmOperation(dev; t_k = 250.0, vov = 3.0, n_channels = 1, window_ns = (-8600.0, 6000.0))
    for (op, nrec, hiso, fmin) in ((opw2, 2000, hw, 0.5), (opc2, 1500, hc, 0.9))
        d = dark_spectrum(op, nrec)
        lad = photoelectron_ladder(d.adc, 10.0; pedestal = 200.0, baseline = d.baseline)
        @test isfinite(lad.gain) && lad.cells == [1, 2, 3]
        @test isapprox(lad.gain, 10.0; rtol = 0.1)
        @test abs((lad.zero - 200) / lad.gain + d.baseline) < 0.3
        cells(v) = round.(Int, (v .- lad.zero) ./ lad.gain)
        @test cells(200 .+ 10 .* hiso) == [1, 2, 3]
        for n in 1:3
            sel = d.lab .== n
            @test count(sel) > 20
            @test cells(median(d.labadc[sel])) == n
            @test mean(cells(d.labadc[sel]) .== n) > fmin
        end
        println("T $(op.t_k): N $(length(d.adc)) gain $(lad.gain) zero $(lad.zero) expected $(200 - 10d.baseline); isolated labelled ", [count(d.lab .== n) for n in 1:3],
                " right ", [round(mean(cells(d.labadc[d.lab .== n]) .== n); digits = 2) for n in 1:3])
        # numbered from the pedestal instead (baseline 0), the room-temperature spectrum
        # would put the two-cell peak first, 0.4 gain from the pedestal: it is rejected,
        # not fitted with a zero one gain too high; at 250 K the two agree
        lad0 = photoelectron_ladder(d.adc, 10.0; pedestal = 200.0)
        if op.t_k > 280
            @test isnan(lad0.gain)
        else
            @test isapprox(lad0.zero, lad.zero; atol = 0.1lad.gain) && isapprox(lad0.gain, lad.gain; rtol = 0.02)
        end
    end
    # a spectrum with peaks at the wrong spacing (the ladder of the noise-0.7 case of the
    # review: 5.9 counts apart) is rejected rather than fitted
    rngx = Xoshiro(21)
    fake = vcat([205.9 .+ 5.9k .+ 1.2 .* randn(rngx, 4000 ÷ (k + 1)) for k in 0:3]...)
    @test isnan(photoelectron_ladder(fake, 10.0; pedestal = 200.0, baseline = 1.47).gain)
    # and a clean ladder at the nominal spacing from the expected zero is fitted
    good = vcat([195.3 .+ 10k .+ 1.0 .* randn(rngx, 6000 ÷ (k + 1)) for k in 0:3]...)
    lg = photoelectron_ladder(good, 10.0; pedestal = 200.0, baseline = 1.47)
    @test lg.cells == [1, 2, 3] && isapprox(lg.gain, 10.0; atol = 0.2) && isapprox(lg.zero, 185.3; atol = 0.5)

    # the periodic trigger (PTRG) with the peak detector, as in the bench calibration
    # (note 3): windows of 200 ns at fixed times; without noise or dark counts an empty
    # window stores the pedestal, a single avalanche whose maximum falls inside one window
    # stores one gain, and one whose maximum falls after the window closes stores less
    opq = nodark(SipmOperation(dev; t_k = 293.15, vov = 5.0, n_channels = 2, window_ns = (-9600.0, 6000.0)))
    sq = CitirocSettings(tp_ns = 25.0, tp_lg_ns = 87.5, noise_pe = 0.0, fast_noise_pe = 0.0,
                         t_start_ns = -200.0, t_end_ns = 3000.0, dt_ns = 0.5)
    chq = CitirocChain(sq, opq, Xoshiro(3))
    starts = CryspLight.periodic_windows(sq, 200.0, 50.0)
    @test starts[1] == -200 && all(diff(starts) .== 250) && starts[end] + 200 <= 3000 && starts[end] + 450 > 3000
    tq = sample_time(chq.grid, argmax(begin
        bin_train!(chq.x, chq.grid, [Avalanche(0f0, 1f0, Int8(0))]); copyto!(chq.y, chq.x); shape!(chq.y, chq.slow); chq.y
    end))                                          # time of the maximum of an avalanche at t = 0
    inside = [[Avalanche(Float32(starts[3] + 100 - tq), 1f0, Int8(0))], Avalanche[]]
    late = [[Avalanche(Float32(starts[3] + 200 + 30 - tq), 1f0, Int8(0))], Avalanche[]]
    vi = hg_periodic(chq, inside, 200.0; gap = 50.0)
    @test length(vi[1]) == length(starts) && all(vi[2] .== 200)
    @test vi[1][3] == 210 && count(vi[1] .> 200) == 1
    @test 200 < hg_periodic(chq, late, 200.0; gap = 50.0)[1][3] < 210
    # with the dark counts at 5 V and room temperature at 25 ns, the spectrum of the windows
    # is the ladder of the bench: a peak at the pedestal (no avalanche), then one, two and
    # three cells, each with a sharp upper edge and a tail below it (avalanches that
    # arrived before the window). Without noise the ladder numbered from the pedestal gives
    # the gain; the noise of the model raises the maximum of a small pulse more than that of
    # a large one, so that it compresses the ladder and the fitted spacing falls below the
    # gain (5% at 25 ns): the same procedure on the bench and in the simulation
    opd5 = SipmOperation(dev; t_k = 293.15, vov = 5.0, n_channels = 1, window_ns = (-9600.0, 6000.0))
    function ptrg_ladder(noise)
        ch = CitirocChain(CitirocSettings(tp_ns = 25.0, tp_lg_ns = 87.5, noise_pe = noise, t_start_ns = -200.0,
                                          t_end_ns = 6000.0), opd5, Xoshiro(5))
        pv = Float64[]
        for i in 1:1500
            Random.seed!(ch.rng, i)
            append!(pv, hg_periodic(ch, [respond(opd5, Float32[], Float32[], Xoshiro(i))], 200.0)[1])
        end
        pv .+= rand(Xoshiro(9), length(pv)) .- 0.5
        return (frac = mean(pv .> 205), lad = photoelectron_ladder(pv, 10.0; pedestal = 200.0), n = length(pv))
    end
    p0, pn = ptrg_ladder(0.0), ptrg_ladder(0.1)
    println("PTRG 200 ns, 25 ns, 5 V: $(pn.n) windows, fraction above half a cell $(round(pn.frac; digits = 3)); ",
            "gain without noise $(round(p0.lad.gain; digits = 3)), with noise $(round(pn.lad.gain; digits = 3)), zero $(round(pn.lad.zero; digits = 2))")
    @test 0.5 < pn.frac < 0.8
    @test p0.lad.cells == [1, 2, 3] && isapprox(p0.lad.gain, 10.0; rtol = 0.03) && abs(p0.lad.zero - 200) < 1.5
    @test pn.lad.cells == [1, 2, 3] && 0.9 < pn.lad.gain / p0.lad.gain < 0.99 && abs(pn.lad.zero - 200) < 3.5

    # the matched filter of the digitiser calibration: the local maximum over the alignments
    # exceeds the amplitude of an isolated unit pulse by 4 to 6% depending on the sampling
    # phase; normalised, it gives 1.00 on average over the phases
    opd = nodark(SipmOperation(dev; t_k = 250.0, vov = 5.0, n_channels = 1, window_ns = (-10000.0, 20000.0)))
    chd = DigitizerChain(DigitizerSettings(gain = 4.0, noise_counts = 0.0), opd, Xoshiro(1))
    mf = matched_filter_template(chd)
    @test length(mf.tpl) == 10 && maximum(mf.tpl) == 1.0 && abs(mf.tpl[1]) < 1e-9
    @test length(mf.responses) == 4 && all(1.03 .< mf.responses .< 1.08)
    @test isapprox(mf.norm, 1.0536; atol = 0.002)
    g, step = chd.grid, 4
    resp = Float64[]
    for phase in 0:step-1
        bin_train!(chd.x, g, [Avalanche(Float32(2phase), 1f0, Int8(0))]); device_pulse!(chd.x, chd.pulse)
        yf = matched_filter(chd.x[1:step:end], mf.tpl, mf.nrm, mf.norm)
        amps = filter_amplitudes(yf, 0.3)
        @test length(amps) == 1
        push!(resp, amps[1])
    end
    @test isapprox(mean(resp), 1.0; atol = 0.005)
    @test maximum(resp) - minimum(resp) < 0.03
    println("matched filter: responses ", round.(mf.responses; digits = 4), " norm ", round(mf.norm; digits = 4),
            "; normalised ", round.(resp; digits = 4), " mean ", round(mean(resp); digits = 4), " spread ", round(maximum(resp) - minimum(resp); digits = 4))
    # two cells give two, and the amplitude scales with the gain of the channel
    bin_train!(chd.x, g, [Avalanche(0f0, 2f0, Int8(0))]); device_pulse!(chd.x, chd.pulse)
    @test isapprox(only(filter_amplitudes(matched_filter(chd.x[1:step:end], mf.tpl, mf.nrm, mf.norm), 0.3)), 2resp[1]; rtol = 1e-9)
end

include("frozen.jl")

@testset "Frozen output of the pedagogical path" begin
    stored = read_frozen()
    @test stored !== nothing      # run `julia --project=. test/frozen.jl --freeze` once
    if stored !== nothing
        current = frozen_cases()
        for (case, fields) in stored, (field, hex) in fields
            @test haskey(current, case) && current[case][field] == hex
        end
        @test Set(keys(current)) == Set(keys(stored))
    end
end

@testset "Bench DAQ files: run, histograms, staircase and hold-delay scan" begin
    # Files written by pet-caen-daq (main 75086b2): a simulated four-board run with the
    # production configuration and test pulses, and two scan artifacts with known contents.
    fx = joinpath(@__DIR__, "fixtures", "daq")
    r = read_daq_run(joinpath(fx, "run_1.0000.h5"))
    @test r.run_id == "1" && r.schema == 2 && r.nevents == 4
    @test length(r.obs.hg) == 256 && sort(unique(r.obs.chain)) == 0:3
    # one row per channel and board: 64 channels of board 0, one event each
    @test all(==(1), length.(daq_channel_values(r; chain = 0)))
    @test sum(daq_spectra(r; chain = 0)) == 64
    # the run histograms hold the same values: bin of 2 codes, one entry each
    H = read_daq_histograms(joinpath(fx, "run_1.histograms.h5"))
    @test sort(collect(keys(H))) == ["pha_high", "pha_low", "toa", "tot"]
    s0 = only(filter(s -> s.chain == 0 && s.channel == 0, H["pha_high"]))
    hg0 = only(daq_channel_values(r; chain = 0)[1])
    @test s0.entries == 1 && s0.counts[floor(Int, (hg0 - s0.minimum) / s0.bin_width) + 1] == 1
    # configuration: indexed lines override the common value for their board
    p = r.config.params
    @test daq_param(p, "TD_CoarseThreshold"; board = 0) == "181"
    @test daq_param(p, "TD_CoarseThreshold"; board = 1) == "183"
    @test daq_value(p, "HoldDelay") == 300.0
    # the service records carry the monitored HV, here equal to the requested HV_Vbias
    @test length(r.service.hv_v) == 8 && all(r.service.validity .== 0x7f)
    @test r.service.hv_v[1] ≈ daq_value(p, "HV_Vbias")
    st = read_daq_staircase(joinpath(fx, "staircase.h5"))
    @test st.threshold == [300, 250, 200] && st.t_or_rate == [200.0, 400.0, 600.0]
    @test st.rate[1:3, 2] == [20.0, 22.0, 24.0] && st.count[64, 3] == 83
    hd = read_daq_hold_delay(joinpath(fx, "hold-delay.h5"))
    @test hd.delay_ns == [0, 8, 16] && hd.events == [100, 100, 100]
    @test hd.counts[11, 1, 1] == 60 && hd.counts[42, 1, 2] == 40 && hd.counts[21, 6, 3] == 9
    @test vec(sum(hd.counts[:, 1, :]; dims = 1)) == [100, 100, 100]
    @test hd.missing[64, :] == [0, 1, 2] && hd.bin_low[2] == 16
end
