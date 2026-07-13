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

**Papers: don't read PDFs directly — they are heavy on context (each page is an image). Ask JJ
first: he can usually provide a .txt of the paper. Only fall back to reading the PDF (few, targeted
pages) if no text version is available.**

## Purpose

Simulate the scintillation-light transport in a wrapped CsI/BGO crystal read out by an 8×8 SiPM
matrix, with the photon propagation running on the GPU (Apple Metal). The gamma transport
(511 keV photons → energy deposits) comes from the vendored PTCrysp engine; this repo starts from
the deposits. Three notes in `latex/`: `crysp_light_metal.tex` (problem statement and the
optical model), `csitl_calib.tex` (calibrations: CsI(Tl), cold CsI, BGO — parameter
provenance), `crysplight_soft.tex` (software: architecture, the KernelAbstractions
implementation, bit-parity contract, performance). Read those for physics and design;
this file records layout and conventions.

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
- `runs/` — **tracked TOML run configs** (the parameter source of truth; tag = filename).
- `output/` — **gitignored results**, one dir per tag: `output/<tag>/light.h5` + a copy of
  the config.
- `data/` — one TOML per material (csi, bgo, teflon, esr, sipm_s14160). Scalar optical
  properties for now; tabulated curves replace them without interface changes.
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
