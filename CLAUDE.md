# CLAUDE.md — CryspLight: GPU (Metal) optical-photon transport in scintillating crystals

Orientation for any Claude Code session on this repo.

## Working style

Ask questions plainly, in prose — not as multiple-choice "shopping list" menus (avoid the
AskUserQuestion option-list format). State what you need to know directly.

**Terminal output: never emit blue-rendering text; use bold.** JJ reads responses in a terminal
where markdown links, `inline code` (backticks), and file:line references all render blue. Do not
use any of them in responses — no backticks, no code fences for listings, no clickable links. Emphasise
with **bold** and write paths, filenames, config keys, and code identifiers as plain text.

**Figures: always from a tracked tool.** Every figure that goes into a doc (the LaTeX notes,
`docs/`) must be produced by a checked-in script that reads the data and writes the image — never
with a throwaway inline/ad-hoc command. The script and the generated image are both committed, so
any figure regenerates from the repo.

**Results: always state where they live.** When reporting any result, give the paths —
figures, metrics, dataframes, configs — so JJ can open them directly.

**Papers: don't read PDFs directly — they are heavy on context (each page is an image). Ask JJ
first: he can usually provide a .txt of the paper. Only fall back to reading the PDF (few, targeted
pages) if no text version is available.**

## Purpose

Simulate the scintillation-light transport in a wrapped CsI/BGO crystal read out by an 8×8 SiPM
matrix, with the photon propagation running on the GPU (Apple Metal). The gamma transport
(511 keV photons → energy deposits) comes from the vendored PTCrysp engine; this repo starts from
the deposits.

## The notes — read these first after a context compact

Five LaTeX notes in `latex/` carry the full record; together with this file they are
sufficient to resume work without prior context:

1. `crysp_light_metal.tex` — **design**: problem statement, the optical model
   (UNIFIED surfaces, bulk split), the calibrated material table, GPU design + achieved
   performance.
2. `csitl_calib.tex` — **calibrations**: CsI(Tl) (Brose ladder + Knyazev), cold CsI
   (CRYSP measurement; adopted f = 0.9 split, 62–83% bracket + breaker measurements),
   BGO (Ding cube + Gironnet + Mao bound). Every parameter's provenance.
3. `crysplight_soft.tex` — **software**: architecture, RNG/reproducibility, the
   KernelAbstractions kernel, the bit-parity contract, Metal gotchas, benchmarks.
4. `crysplight_e2e.tex` — **end-to-end**: the vendored gamma stage, the per-event
   pipeline, first physics for CsI(Tl) and BGO with analytic cross-checks, and the
   per-event truth / CNN-dataset layer (format, encodings, first light-map figures).
5. `crysplight_cnn.tex` — **reconstruction**: CryspNet architecture (pedagogical),
   exact D4 augmentation, Anger/MLP baselines, millimetric (x,y,z) results for both
   crystals; figures pulled via graphicspath from analysis/results/cnn.

## Status (2026-07-15) and next steps

Everything below is BUILT, tested (9 test sets), committed and pushed: optical
transport (CPU reference + bit-identical KA/Metal kernel, 33 Mphotons/s), three
calibrated materials, three calibration campaigns, the vendored gamma core, the
per-event pipeline (runs/gammas_{csitl,bgo}.toml → output/<tag>/events.h5).
CRYSP baselines (PDE=1): cold CsI 83.0%, BGO 88.0%, CsI(Tl) 88.8%. End-to-end
photopeak at wavelength-correct PDE: CsI(Tl) 9798 pe, BGO 2832 pe — both on
calibration-chain expectations. The CNN-dataset layer is in: per-event truth
(xyz1/e1, xyz2/e2, er = 511−e1−e2, int_type −1/0/X = none/photo/X-Compton, n_int),
the 8×8 first-pe time matrix (tmin_ns, Inf = no hit), flat PDE 0.40 for both
crystals (BGO photopeak 2518 pe), all in events.h5 chunked along events —
column-major means h5py reads (n,8,8)/(n,3), PyTorch-ready with no transposes.
Light-map figures from scripts/plot_light_maps.jl (tracked; images in latex/figs).
The Python analysis layer (analysis/, same repo) holds true_vars.py (control
plots + containment stats from the 50k runs) and the CNN reconstruction
(analysis/cnn: dataset.py with exact D4 augmentation — self-test via python3
analysis/cnn/dataset.py; model.py CryspNet ~0.6M-param residual ConvNet + MLP +
Anger baselines; train.py driven by analysis/cnn/runs/cnn_{csitl,bgo}.toml).
Trained on the 500k runs (torch MPS). Idealized (truth-selected photo): test
RMSE x/y/z = 1.21/1.24/1.11 mm CsI(Tl), 0.84/0.84/0.92 mm BGO; depth linear over
the FULL crystal. Phase 1 Compton (realistic 511 +- 2 sigma window on smeared
energy, FWHM 6%/10%, configs cnn_*_win.toml): global p68 = 2.0/2.0/2.1 mm CsI(Tl)
(26.8% of events beyond 5 mm), 1.1/1.1/1.3 mm BGO (9.7%); photo class survives
the mixture nearly intact; tail = soft-recoil Compton (first site outshone by
the second). All results in analysis/results/cnn; note updated. Next candidates:
phase 2 = six-output two-site regression (x1..z2, coincident-target convention
for single-site; min-over-orderings loss as fallback; predicted separation =
observable quality flag); two-channel input (tmin matrix) for timing-aided
reconstruction;
response layer (intrinsic resolution ~5.3% FWHM + SiPM excess noise) for
realistic spectra; per-event time histograms; Brose phase-2 frustum; cold-CsI
gamma runs.

## Layout (PTCryspMC conventions)

- `src/` — the Julia package: Philox RNG (`philox.jl`), box geometry (`geometry.jl`),
  Fresnel/specular/Lambertian surface optics (`optics.jl`), photon generation
  (`generation.jl`), the reference bounce loop + accumulators (`transport.jl`), the
  **GPU twin** (`kernel.jl` — one KernelAbstractions kernel, functional Philox,
  per-photon records, no atomics; `run_photons_ka!` with `ArrayT = Array` for the CPU
  backend or `Metal.MtlArray` for the GPU), config reading (`config.jl`), HDF5 output
  (`output.jl`), material loading (`materials.jl`).
- **Kernel invariant**: `kernel.jl` mirrors `transport.jl` draw-for-draw — on the CPU
  backend it is BIT-IDENTICAL to the reference (tested). Any physics change must be made
  in BOTH files, and the parity test will catch a slip. Metal gotchas: ≤31 kernel-arg
  buffers (pass isbits structs, not unpacked scalars), Float32/Int32 only, no dynamic
  tuple indexing, no nscat tracking in the KA path. Measured: 33 Mphotons/s on Metal =
  6.2× the 8-thread CPU path (`scripts/bench_metal.jl`).
- `src/gamma/` — the **vendored** 511 keV gamma-transport core from PTCryspMC.jl @
  d4982db (see `src/gamma/VENDORED.md` for the file map and adaptations). Internal
  conventions cm/MeV/Float64; the mm/keV/Float32 bridge is `src/gamma_interface.jl`
  (adds per-deposit times = path/c). The per-event pipeline (gammas → deposits →
  batched optical transport → per-event maps) is `src/events.jl` (`run_events!`);
  application `scripts/shoot_gammas.jl` + `runs/gammas_{csitl,bgo}.toml`.
- `runs/` — **tracked TOML run configs** (the parameter source of truth; tag = filename).
- `output/` — **gitignored results**, one dir per tag: `output/<tag>/light.h5` + a copy of
  the config.
- `data/` — one TOML per material (csi, bgo, teflon, esr, sipm_s14160). Scalar optical
  properties for now; tabulated curves replace them without interface changes.
- `analysis/` — the **Python** analysis layer (same git repo): control-plot and
  ML-side scripts reading `output/<tag>/events.h5` (h5py sees event-axis-first
  arrays, PyTorch-ready). `true_vars.py` — interaction-truth control plots +
  stats, filtered to fully contained events (edep > 510.5 keV, NOT er == 0 which
  would drop contained C3+); committed images in `analysis/results/true_info/`.
- `scripts/` — user drivers (`run_point_source.jl`, `study_surface.jl`) and the
  calibration campaigns, one per anchor measurement, each with its `runs/*.toml`:
  `csitl_calib.jl` (Brose wrapping ladder), `soleti_calib.jl` (CRYSP cold-CsI 3×3×20),
  `ding_calib.jl` (BGO 6 mm cube, two-sided readout); `test/` — `Pkg.test`, compulsory
  for every step of the simulation chain; `latex/` — the notes; `papers/` — reference
  papers (PDF + txt; read the txt, not the PDF).

Run: `julia --project=. --threads=auto scripts/run_point_source.jl runs/<cfg>.toml`
Test: `julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'`

## Conventions and gotchas

- **PDE is applied on the fly** in the kernel; outputs are detected photoelectrons
  (8×8×Nt counts in 100 ns bins + 8×8 first-photoelectron times). Sensor and PDE version
  are stamped in the HDF5 attrs.
- Scalar validation optics: CsI 100k ph/MeV, τ=800 ns, λ=350 nm, n=1.95; BGO 15k ph/MeV,
  τ=1500 ns, λ=550 nm, n=2.15; grease n=1.45. Λ_abs (500/1000 mm) are placeholders to tune.
- Surface model = full Geant4 UNIFIED: back-painted (air gap, default) vs front-painted
  (contact) finishes, plus the σ_α specular lobe (micro-facet Gaussian). Baseline configs:
  backpainted, sigma_alpha_deg = 1.3 (measured polish). Rayleigh scattering is a separate
  bulk process (`rayleigh_mm` in the crystal TOML, ∞ if absent), phase ∝ 1+cos²θ.
- Kernel invariants the tests enforce: exact photon conservation across the five terminal
  statuses; the idealized specular (ESR, R=1, σ_α=0) limit detects exactly 1−cos θc′;
  polished + air gap traps direction classes super-critical on all faces (TIR preserves
  |u| components) — physics, not a bug; Rayleigh restores ergodicity (pure scattering +
  perfect wrap ⇒ all detected); contact-Lambertian R=1 detects all; survival draws use `<=`
  so R=1 never absorbs (randu hits 1.0 with p=2⁻²⁴).
- Key sensitivity (scripts/study_surface.jl, table in the note): σ_α and Λ_abs interact —
  calibrate them jointly against data; Λ_abs alone barely helps a polished crystal.
- Julia threading gotcha (bit us once): inside `Threads.@threads` bodies, never assign a
  variable name that is also assigned in the enclosing scope — it gets boxed and shared
  across threads and silently races.
