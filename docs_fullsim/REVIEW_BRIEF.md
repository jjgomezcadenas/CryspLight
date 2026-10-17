# Independent review of the CryspLight full simulation

You are reviewing work done by another model with the author (J.J. Gómez-Cadenas, CRYSP group) over 16 to 18 October 2026. Your job is to find what is wrong, unjustified or untested, not to confirm it. Assume nothing in the documents is correct until you have checked it against the code, the datasheets or a calculation of your own.

## Context in five lines

CryspLight is a Julia optical Monte Carlo (Metal GPU) for scintillation light in a 48 x 48 mm2 BGO or CsI(Tl) crystal read by 64 SiPMs (four Hamamatsu S14161-6050HS-04 tiles). The "full simulation", on branches fullsim and radioroc (radioroc checked out), extends it from ideal photons to what three front ends record: the Citiroc 1A ASIC, the CAEN DT2745 digitiser and the Radioroc2 ASIC. Bench runs with BGO and CsI(Tl) at room temperature start the week of 20 October 2026, with the digitiser first. The emulation is being used now to choose the bench settings, so an error here becomes a wrong instruction to the experimenter.

## Read in this order

1. docs_fullsim/fullsim_design.tex: the physics specification. Sections Electronics, Response of the SiPM, Files, Observables, Parameters.
2. docs_fullsim/working_plan.tex: tasks, decisions with their reasons, open questions, log. The table "First emulated campaign" and the section "The first study" carry the current numerical claims.
3. /Users/jjgomezcadenas/Projects/BGOTSCAN/docs/digitiser_run_instructions_and_procedure.tex: the instructions that will go to the experimenter. Highest priority.
4. Code, new since tag pedsim (git diff pedsim..radioroc --stat): src/source.jl, src/decays.jl, src/pes_file.jl, src/fullsim_config.jl, src/frontend/*.jl, the tiled readout and window changes in src/transport.jl and src/kernel.jl, data/*_ambient.toml, data/bgo_cold.toml, data/sipm_s14160.toml, test/runtests.jl, test/frozen.jl.
5. Analysis: fullsim_analysis/lib/*.jl, fullsim_analysis/scripts/*.jl, fullsim_analysis/plots/*.py, and the outputs under fullsim_analysis/results/.
6. Primary sources, as text: papers/citiroc1a-datasheet.txt, papers/radioroc2-datasheet.txt, papers/s14160_s14161_series_kapd1064e.txt, papers/um8717-2740-2745-digitizers.txt, papers/bgo-1505.00165.txt. Read the .txt files, not the PDFs.

## Where to put the stress

These are the places the authors themselves consider weakest, in order of consequence.

1. The digitiser instructions. Check every number in the settings table against the manual and the emulation: counts per photoelectron (3.2 per volt of overvoltage at gain 1 comes from an earlier cold run, not a datasheet), the noise acceptance criterion of 8 counts at gain 4, the thresholds of 290 and 120 counts, the DC offset headroom, record length and pre-trigger, the gate opening 300 ns before the trigger (not yet confirmed by the gate scan). Is the procedure executable as written by someone who was not in the conversation? Are there undefined terms?
2. The trigger conclusions. The claim that dark pulses with crosstalk companions, not single dark photoelectrons, set the ambient thresholds, and the resulting thresholds for each front end (fullsim_analysis/scripts/trigger_scan.jl, src/frontend/citiroc.jl trigger_time, src/frontend/sipm.jl n_crosstalk). Is the crosstalk model (geometric, simultaneous) adequate for this conclusion? Is the dark window and the per-window to rate conversion right? The cold digitiser run triggered only 4.6% of decays at the ambient threshold; is the explanation given in the plan log correct?
3. The shaping model. src/frontend/shaping.jl implements the shaper as n + 1 integrators, arguing that the preamplifier integrates the SiPM current so the output for one avalanche is the CR-RC^n step response. Is that right for the Citiroc (voltage preamplifier, Cin 15 pF) and for the Radioroc2 (charge preamplifier)? The stored fractions 0.39 / 0.15 / 0.07 (Citiroc) and 0.67 / 0.94 / 0.97, 0.55 (Radioroc2) depend on it. Check sipm_pulse! normalisation and the calibration of a unit avalanche.
4. The digitiser emulation without waveforms (src/frontend/digitizer.jl): the claim that the gated, baseline-subtracted charge equals the sum over avalanches of area times window weight, plus one Gaussian of rms sigma times sqrt(Ng + Ng^2/Nb). Verify the formula and the trigger on the 8 ns grid. The noise model is white; the bench noise is not.
5. Radioroc2 assumptions: the ceiling interpretation (2000 photoelectrons at 1e6 on the low gain, a tenth on the high gain, scaling inversely with preamplifier gain), the noise scaling with the peaking time, the time-preamplifier pulse (5/20 ns, invented), the decision to use the charge trigger TQ. The dynamic-range study says the high-gain path saturates in 14 to 47% of 511 keV events; check it.
6. Physics inputs with weak provenance: the PDE spectral factor in data/sipm_s14160.toml (read approximately from a datasheet figure), the intrinsic resolution as one Gaussian per decay scaling as (511/E)^0.5, the BGO yields at ambient (7600) and cold (15000) as midpoints, afterpulsing 1% at 3 V scaling as Vov^2, the dead-area reflectivity 0.5.
7. Statistics and fitting: the Gaussian plus linear fit in fullsim_analysis/lib/fits.jl, the resolution numbers it produces versus the tagged-decay widths, the 1275/511 ratio biases attributed to the peak-detector pedestal.
8. Reproducibility and the non-regression contract: test/frozen.jl and the claim that the pedagogical path is bit-identical; the per-event reseeding in the threaded front-end stage; the chunked decay streams (block size in src/decays.jl).
9. The text, against the author's writing rules in CLAUDE.md (section "Writing in the notes and design documents"): connected prose, no noun catalogues, no flourishes, written for an external reader, no undefined terms. Flag violations; do not rewrite.

## How to check

- Run the tests: julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'. They take about ten minutes.
- Do your own calculations in a scratch directory, outside the repository. Do not modify tracked files, do not commit, do not push.
- Large outputs exist under output/ (pes.h5 and events.h5 per run); read them with h5py or HDF5.jl rather than rerunning the GPU stage.
- Some scans may still be running when you start: the cold-BGO trigger scans (digitiser and Radioroc2), the digitiser gate scans (ambient BGO, ambient CsI(Tl), cold BGO) and the Radioroc2 shaping scans (the three crystals). Their outputs go to fullsim_analysis/results/<campaign>/trigger_scan, gate_scan and shaping_scan; treat those directories as incomplete if their files changed in the last hour.

## What not to report

Stay at the level of the decisions the work supports: bench settings, the physics model, the numbers in the documents, the tests. Leave out:

- Detail that cannot change a setting, a number in the documents or a conclusion: numerical precision far below the statistical errors, alternative but equivalent formulations, style of code that works.
- Points outside the scope: the pedagogical simulation before tag pedsim, the vendored gamma code, general software-engineering advice, features nobody asked for.
- Remarks without context: an objection to a parameter whose provenance is already stated as an assumption in the documents is a finding only if you show that a plausible alternative value changes a conclusion, and by how much.
- Lists of minor items. If you have more than about fifteen findings, keep the fifteen that matter most and drop the rest.

A good finding is one the author would act on before the bench runs.

## Report

One Markdown file, written to docs_fullsim/review_<your model>_<date>.md, then stop. For each finding:

- **Where:** file and line, or document and section.
- **Claim:** what the code or text asserts.
- **Problem:** what is wrong, with the evidence (a calculation, a datasheet line, a failing check).
- **Severity:** blocks the bench instructions / changes a number in the documents / cosmetic.
- **Confidence:** verified, plausible, or suspicion.

Start the report with a five-line verdict on the digitiser instructions alone. Keep findings you could not verify separate from verified ones. Do not pad the report with things that are fine.
