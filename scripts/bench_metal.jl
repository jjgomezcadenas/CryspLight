# Throughput benchmark: reference CPU path vs the KernelAbstractions kernel on the CPU
# backend and on Metal. CRYSP cold-CsI configuration (calibrated bulk), delta emission.
# Methodology per RecoCrysp: explicit warm-up (absorbs JIT), best-of-N timing.
#   julia --project=. --threads=auto scripts/bench_metal.jl [n_photons]
using CryspLight, Metal, Printf

n = isempty(ARGS) ? 4_000_000 : parse(Int, ARGS[1])
box = Box((48f0, 48f0, 37.2f0))
op = OpticalParams(1.95f0, 1.45f0, 3000f0, 333.3f0, 0.99f0, false, true,
                   Float32(deg2rad(1.3)), 1f0)
grid = SipmGrid(6f0, Int32(8), Int32(8)); tb = TimeBinning(100f0, Int32(80))
ro = Readout(grid)
kw = (seed = 1, pos = (24f0, 24f0, 18.6f0), tau_ns = 0f0)

function bench(name, f, n; reps = 5)
    f(min(n, 100_000))                       # warm-up: JIT + Metal pipeline compile
    best = minimum((@elapsed f(n)) for _ in 1:reps)
    @printf("%-28s %8.3f s   %6.1f Mphotons/s\n", name, best, n / best / 1e6)
    return best
end

println("CRYSP CsI config, $n photons, $(Threads.nthreads()) CPU threads\n")
t1 = bench("reference (threads)", m -> run_photons!(box, op, ro, tb; n_photons = m, kw...), n)
t2 = bench("KA kernel, CPU backend", m -> run_photons_ka!(box, op, ro, tb; n_photons = m,
                                                          kw..., ArrayT = Array), n)
t3 = bench("KA kernel, Metal", m -> run_photons_ka!(box, op, ro, tb; n_photons = m,
                                                    kw..., ArrayT = Metal.MtlArray), n)
@printf("\nMetal speedup: %.1fx vs reference, %.1fx vs KA-CPU\n", t1 / t3, t2 / t3)
