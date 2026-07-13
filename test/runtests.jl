using Test
using Statistics
using CryspLight
using Metal
using Random: Xoshiro
using CryspLight: face_axis, specular, emission_time

const GRID = SipmGrid(6f0, Int32(8), Int32(8))
const TB = TimeBinning(100f0, Int32(40))

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
    edep, npe, maps = run_events!(box, op, Readout(GRID), tbz, pv, 54_000;
                                  n_events = 40, seed = 3, tau_ns = 1000f0,
                                  batch_photons = 500_000)
    @test length(edep) == 40 && size(maps) == (8, 8, 40)
    for ev in 1:40
        @test sum(Int, maps[:, :, ev]) == npe[ev]
        @test (edep[ev] == 0) == (npe[ev] == 0)
        if edep[ev] > 510                                # photopeak: pe ~ Y*E*eps*pde
            @test 5000 < npe[ev] < 15000
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
