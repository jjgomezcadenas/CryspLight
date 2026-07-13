# Surface/bulk realism scan for CsI + Teflon (point source at centre, PDE = 1,
# delta emission time): what sigma_alpha (UNIFIED lobe), the front-painted contact mode,
# Rayleigh scattering, and Lambda_abs each do to the light collection.
# Reproduces the table in latex/crysp_light_metal.tex (Sec. First application).
#   julia --project=. --threads=auto scripts/study_surface.jl
using CryspLight, Printf

box = Box((48f0, 48f0, 37.2f0)); pos = (24f0, 24f0, 18.6f0)
grid = SipmGrid(6f0, Int32(8), Int32(8)); tb = TimeBinning(100f0, Int32(40))
mk(; abs = 500f0, ray = Inf32, gap = true, sig = 0.0) =
    OpticalParams(1.95f0, 1.45f0, abs, ray, 0.99f0, false, gap,
                  Float32(deg2rad(sig)), 1f0)

cases = [
    ("gap, polished (v0 baseline)",       mk()),
    ("gap, sigma_alpha 1.3 deg",          mk(sig = 1.3)),
    ("gap, sigma_alpha 6 deg",            mk(sig = 6.0)),
    ("gap, sigma_alpha 12 deg",           mk(sig = 12.0)),
    ("contact (frontpainted) Lambertian", mk(gap = false)),
    ("gap, polished + Rayleigh 300 mm",   mk(ray = 300f0)),
    ("gap, 1.3 deg + Rayleigh 300 mm",    mk(sig = 1.3, ray = 300f0)),
    ("gap, polished, Lambda_abs 5 m",     mk(abs = 5000f0)),
    ("gap, 1.3 deg, Lambda_abs 5 m",      mk(abs = 5000f0, sig = 1.3)),
]

for (name, op) in cases
    acc = run_photons!(box, op, grid, tb; n_photons = 1_000_000, seed = 1,
                       pos = pos, tau_ns = 0f0)
    @printf("%-36s det %6.2f%%  bulk %6.2f%%  wall %6.2f%%  <b|det> %5.1f\n",
            name, 100acc.ndet / 1e6, 100acc.nabs_bulk / 1e6, 100acc.nabs_wall / 1e6,
            acc.sum_bounces_det / max(acc.ndet, 1))
end
