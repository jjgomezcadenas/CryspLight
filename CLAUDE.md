# CLAUDE.md — CryspLight: GPU (Metal) optical-photon transport in scintillating crystals

Orientation for any Claude Code session on this repo.

## Working style

Ask questions plainly, in prose — not as multiple-choice "shopping list" menus (avoid the
AskUserQuestion option-list format). State what you need to know directly.

**Terminal output: never emit blue-rendering text; use bold.** JJ reads responses in a terminal
where markdown links, `inline code` (backticks), and file:line references all render blue. Do not
use any of them in responses — no backticks, no code fences for listings, no clickable links. Emphasise
with **bold** and write paths, filenames, config keys, and code identifiers as plain text.

**Writing in the notes and design documents.** Connected prose with verbs, not comma lists of
noun phrases: name what the items are and tie them to the physics ("The main parameters are the
number of photoelectrons per MeV ---and from there the efficiency to turn photons into
photoelectrons---, the energy resolution, the scintillation lifetime, ..."). Beware the reflex
of replacing a vague sentence by a catalogue of concrete nouns: it is the same error. No slogans,
metaphors or flourishes ("every ingredient is known", "sits on the bench", "everything between X
and Y"); state the plain fact. Lists only to organise substance (numbered stages with bold names,
each item a full sentence). Describe outputs concretely; no undefined shorthand ("truth",
"pipeline", "words"). Write for an external reader: no git tags, repository layout or "existing
code" in the physics text; the relation to the code goes in an Implementation section or the
working plan. Introduce things before using them. Do not fix open implementation choices. After
a rewrite check for repetition with the neighbouring paragraph. Correct JJ's typos and stray
punctuation on the fly; never rephrase his content without asking. Detailed record in memory:
feedback-writing-notes-style.

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
   exact D4 augmentation, Anger/MLP baselines, idealized + realistic-window results,
   two-Gaussian response model, the ordering theorem (backscatter enrichment, the
   255.5 keV step), two-site studies and the information limit of the 8×8 map, the
   reconstructability classifier and the acceptance dial (A2/A1 0.5→0.04 in z);
   figures via graphicspath from analysis/results.

## Status (2026-10-28) and next steps

Two layers in this repo. The **pedagogical simulation** (tag `pedsim`, everything in the
five notes of `latex/`) is closed: optical transport (CPU reference + bit-identical
KA/Metal kernel), three calibrated materials, the vendored gamma core, the per-event
pipeline, the CNN-dataset layer and the CryspNet reconstruction with its resolution
recipe (details in the notes; nothing pending there). The **full simulation and the bench
comparison** (since 16 Oct 2026) is the live work; its source of truth is the working plan
`docs_fullsim/plan/working_plan.tex` (tasks, decisions, log) and the design document
`docs_fullsim/model/fullsim_design.tex`. Built and tested: tiled readout, grease window,
ambient/cold materials, Na-22 decays, photoelectron file, SiPM model, three emulated
front ends (Citiroc 1A, DT2745 digitizer, Radioroc2), the first emulated campaign, the
room-temperature study note, the Citiroc calibration note (periodic trigger), the API
reference (`docs/`, Documenter).

**Bench (Donostia, Citiroc 1A on A5202 boards, JMB's DAQ).** Sessions of 21, 22, 23 and
27 Oct 2026; raw data in the gitignored `benchdata/<day>/`, logbooks in
`docs_fullsim/bench/logbooks/`, findings F1-F14 in `fullsim_analysis/results/bench/findings.md`,
daily reports in `docs_fullsim/bench/reports/`, talks in `docs_fullsim/bench/talks/`. The
session of 27 Oct (boards 1, 2, 4, CAEN adapters, firmware 0xa7070800; DAQ numbers 0, 1, 2)
covered the three calibration menus and has one report each (menu 1 pedestal noise, menu 2
threshold and hold delay, menu 3 single-photoelectron calibration) and the talk of 28 Oct.
Frozen numbers: pedestal noise 12.3 counts per channel at 25 ns (6.2 coherent per board,
10.6 incoherent), the same on five boards and three sessions; gain 165-178 counts/pe at
43 V = 4.1 V over breakdown (38.8-39.0 V), 41-42 counts/pe/V; noise 0.07 pe at the three
shapings; gain ratio 87.5/25 ns = 1.006; threshold edge 179-182 DAC, 20 DAC/pe at 43 V.
The noise note `docs_fullsim/model_data/noise.tex` ("Noise Model for CRYSP") holds the
method, the two cases and the status for the model.

**Next.** Menu 4 (low-gain calibration from the ratio of the two paths on the first crystal
runs at 87.5 ns); the model fit of the noise (series, parallel, 1/f + coherent per board);
bench requests pending: F12 (entries at 0 above breakdown), F13 (temperature sensor of
board 1), F14 (hold-delay lag of the DAQ), board 3 with the CAEN adapters, a pedestal run
with the HV off, the oscilloscope probe of the slow shaper. Then the crystal campaigns of the
plan (ambient BGO and CsI(Tl), cold BGO, Radioroc2).

**Report style (agreed 28 Oct).** Reports and talks: schematic (tables, formulas, itemize),
"Dishes served" first and "Summary of results" last, figures explained bullet by bullet,
expectations stated against the model, only the day's data, no out-of-scope remarks, no
defensive sentences, authors JJ and Laura Navarro with "Assisted by CLAUDE".

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
- **Full simulation layer:** `src/frontend/` (SiPM model and the three front ends),
  `fullsim_runs/<campaign>/` (configurations), `fullsim_analysis/` (Julia library `lib/`,
  one script per study in `scripts/`, bench scripts in `scripts/bench/`, Python plotters in
  `plots/`, committed results in `results/`), `docs_fullsim/` (model, calib, na22, plan,
  reviews, model_data = the noise note, bench = logbooks, reports, talks), `bench/` (the
  bench notes in Spanish: menus 0-3), `benchdata/` (gitignored raw bench data, README tracked),
  `docs/` (API reference). Boards: the DAQ numbers boards by (chain, node); the bench numbers
  differ (27 Oct: DAQ 0, 1, 2 = boards 1, 2, 4), see `benchdata/README.md`.

Supported platform: Apple Silicon macOS, Julia 1.11 or newer (the committed manifest
was generated with Julia 1.12.6). Metal is a required package dependency; simulations
fall back to the KernelAbstractions CPU backend when Metal is not functional. The
Python analysis requires Python 3.11 or newer and the exact packages in
analysis/requirements.txt. Public setup and quick-start instructions are in README.md.

Instantiate: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
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
