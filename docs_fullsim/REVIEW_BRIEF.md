# Independent review of the CryspLight full simulation

You are reviewing work done by another model with the author (J.J. Gómez-Cadenas, CRYSP group) over 16 to 18 October 2026 (brief updated on 18 October, evening). Your job is to find what is wrong, unjustified or untested, not to confirm it. Assume nothing in the documents is correct until you have checked it against the code, the datasheets or a calculation of your own.

## Context in five lines

CryspLight is a Julia optical Monte Carlo (Metal GPU) for scintillation light in a 48 x 48 mm2 BGO or CsI(Tl) crystal read by 64 SiPMs (four Hamamatsu S14161-6050HS-04 tiles). The "full simulation", on branches fullsim and radioroc (radioroc checked out), extends it from ideal photons to what three front ends record: the Citiroc 1A ASIC, the CAEN DT2745 digitiser and the Radioroc2 ASIC. Bench runs with BGO and CsI(Tl) at room temperature start the week of 20 October 2026, with the digitiser first. The simulation is being used now to choose the bench settings, so an error here becomes a wrong setting on the bench. Its first result is a note on the six cases at room temperature (two crystals, three readouts).

## What the note is, and what is known

The note docs_fullsim/room_temperature_study.tex is a simulation made before any measurement with this set-up. Its purpose is to propose operating conditions for the bench (overvoltage, digitiser gain, thresholds, shaping, integration windows) and to predict what the measurements will show. Its settings are proposals, not established facts: the bench will first measure the inputs they depend on and the settings will be revised if the measurements differ. The inputs fall in three classes.

- Fixed by the hardware or the datasheets: the geometry, the SiPM breakdown, gain and detection efficiency curves (Hamamatsu), the ASIC shaping times, gains, ranges and hold delays (Weeroc datasheets), the digitiser range and sampling (CAEN manual).
- Taken from earlier measurements of the group or the literature, stated as such in the design document: the light yields, the decay times, the intrinsic resolution of 5% FWHM, the crosstalk and dark-count curves against overvoltage, the digitiser scale of 3.2 counts per photoelectron per volt at gain 1.
- Assumed, to be measured in the first hour of the bench runs: the digitiser noise of 4 counts per sample at every gain (the gain-4 proposal depends on it), the noise of the ASICs, the dark-count rate and crosstalk at the actual temperature (the thresholds depend on them), the ratio of the Radioroc2 shaper gains at 300 ns and 1.8 us.

Judge the note as what it is. That an input is unmeasured is known and is not a finding. A finding is a conclusion that the simulation does not support given its inputs (a code error, a wrong calculation, a misread datasheet, an analysis that does not measure what it claims), or an input whose plausible alternative value would change a proposed setting or a conclusion, with the size of the effect computed.

## Read in this order

1. docs_fullsim/room_temperature_study.tex: the note on BGO and CsI(Tl) at room temperature with the Citiroc 1A, the DT2745 at gain 4 and 5 V, and the Radioroc2. It carries the current numerical claims (resolutions, photoelectrons, calibration, trigger). Highest priority.
2. docs_fullsim/fullsim_design.tex: the physics specification. Sections Electronics, Response of the SiPM, Files, Observables, Parameters.
3. docs_fullsim/working_plan.tex: tasks, decisions with their reasons (several were revised on 18 October: read the dated entries), open questions, log, and the results of the front-end comparison.
4. Code, new since tag pedsim (git diff pedsim..radioroc --stat): src/source.jl, src/decays.jl, src/pes_file.jl, src/fullsim_config.jl, src/frontend/*.jl, the tiled readout and window changes in src/transport.jl and src/kernel.jl, data/*_ambient.toml, data/bgo_cold.toml, data/sipm_s14160.toml, test/runtests.jl, test/frozen.jl.
5. Analysis: fullsim_analysis/lib/*.jl, fullsim_analysis/scripts/*.jl (the note uses pedestal.jl, calibration.jl, trigger_scan.jl, compare.jl and room_temperature_study.jl), fullsim_analysis/plots/*.py, and the outputs under fullsim_analysis/results/2026-10_ambient/ (room_temperature/, calibration/, pedestal/, trigger_scan/, compare/).
6. Primary sources, as text: papers/citiroc1a-datasheet.txt, papers/radioroc2-datasheet.txt, papers/s14160_s14161_series_kapd1064e.txt, papers/um8717-2740-2745-digitizers.txt, papers/bgo-1505.00165.txt. Read the .txt files, not the PDFs.

## Where to put the stress

These are the places the authors themselves consider weakest, in order of consequence.

1. The digitiser operating point. It changed on 18 October from gain 1 to gain 4 at 5 V of overvoltage, because simulated dark-pulse waveforms showed no fittable single-photoelectron peak at gain 1 with 4 counts of noise per sample and 4.4 MHz of dark counts per channel at 5 V (5.5 MHz of fired cells with crosstalk). The dark rate scales in the model with the detection efficiency, hence barely with the overvoltage (4.3 MHz at 3 V); that scaling is an assumption worth checking. Check the claim (fullsim_analysis/scripts/calibration.jl, digitiser branch) and its premises: 3.2 counts per photoelectron per volt at gain 1 (from an earlier cold run, not a datasheet), a noise of 4 counts that does not grow with the gain, the 1 V range at gain 4 against a largest pulse of 0.6 V; the matched filter (template, baseline of 4 samples, local maxima); the threshold of 480 counts with majority 2; the integration window of 1 us (BGO) and 3 us (CsI(Tl)) from 50 ns before the trigger.
2. The calibration of the ASICs from dark counts. The Citiroc self-triggers at half a photoelectron and holds at the peaking time; the fitted gain is 0.885 of the true one, attributed to the hold falling after the maximum. The Radioroc2 shows no single-photoelectron peak at 1.8 us (dark pile-up) and is calibrated at 300 ns, with the gain transferred through the ratio of the shaper gains, which the bench would measure with charge injection. Is the selection of channels (discriminator within 2 ns of the trigger) realistic for the chips? Does the Radioroc2 chip offer a charge-injection input (papers/radioroc2-datasheet.txt)? The acquisition board has not been chosen (the design document assumes the Weeroc evaluation board as a placeholder), so its access to that input is unknown and not a finding.
3. The peak-detector pedestal and the energy scale. The ASICs store dark pulses even without light (99 to 785 photoelectron equivalents over 64 channels); the analysis subtracts a pedestal measured on empty decays with an external trigger and then calibrates with the 511 and 1275 keV peaks, which absorbs a residual offset (64 keV for the BGO Radioroc2). Is the resolution so defined (fullsim_analysis/scripts/compare.jl, room_temperature_study.jl) the quantity the bench will measure? Is the external-trigger simulation of a random trigger right (src/frontend/citiroc.jl, radioroc.jl)?
4. The resolution decomposition and the light chain of the note (room_temperature_study.jl): the replay of the SiPM response without dark counts on the photopeak events, the robust width (16 to 84% interval), the attribution by differences in quadrature to intrinsic, photoelectron statistics, crosstalk and afterpulses, and readout. Is the decomposition sound, and are the statements of the note about the origin of the readout term (shaping fraction for the Citiroc, dark counts in the window for the digitiser) supported by it?
5. The trigger conclusions. The claim that dark pulses with crosstalk companions, not single dark photoelectrons, set the ambient thresholds, and the resulting thresholds (8 and 7 photoelectron equivalents for the Citiroc, 7.5 with majority 2 for the digitiser, a charge trigger at 40 for the Radioroc2, which keeps only 72% of the BGO deposits between 200 and 300 keV). Is the crosstalk model (geometric, simultaneous) adequate? Are the dark-rate bounds quoted in the note (below 0.3 kHz from 1500 empty decays) right?
6. The shaping model. src/frontend/shaping.jl implements the shaper as n + 1 integrators, arguing that the preamplifier integrates the SiPM current so the output for one avalanche is the CR-RC^n step response. Is that right for the Citiroc (voltage preamplifier, Cin 15 pF) and for the Radioroc2 (charge preamplifier)? The recorded fractions of the note (0.46 and 0.17 of the photoelectrons with the Citiroc, 1.05 and 0.74 with the Radioroc2) depend on it.
7. The digitiser simulation without waveforms (src/frontend/digitizer.jl): the claim that the gated, baseline-subtracted charge equals the sum over cells of area times window weight, plus one Gaussian of rms sigma times sqrt(Ng + Ng^2/Nb). The noise model is white; the bench noise is not.
8. Radioroc2 assumptions: the ceiling interpretation (2000 photoelectrons at 1e6 on the low gain, a tenth on the high gain), the noise scaling with the peaking time, the time-preamplifier pulse (5/20 ns, invented), the overvoltage of 2 V for CsI(Tl) chosen to keep the low gain below its ceiling.
9. Physics inputs with weak provenance: the PDE spectral factor in data/sipm_s14160.toml, the single emission wavelength per crystal (480 and 550 nm, no spectrum), the intrinsic resolution as one Gaussian per decay scaling as (511/E)^0.5, the BGO yield at ambient (7600 per MeV), afterpulsing 1% at 3 V scaling as Vov^2, the dead-area reflectivity 0.5.
10. Statistics and fitting: fullsim_analysis/lib/fits.jl (peak_fit was changed on 18 October to fit around the mode with all events, after it cut a peak at the edge of its range), the per-channel gain fits and their robust summary in calibration.jl.
11. Reproducibility: test/frozen.jl and the claim that the pedagogical path is bit-identical; the per-event reseeding in the threaded front-end stage.
12. The text of room_temperature_study.tex, against the author's writing rules in CLAUDE.md (section "Writing in the notes and design documents"): connected prose, no noun catalogues, no flourishes, written for an external reader, no undefined terms, British English in the note. Flag violations; do not rewrite.

## How to check

- Run the tests: julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'. They take about ten minutes.
- Do your own calculations in a scratch directory, outside the repository. Do not modify tracked files, do not commit, do not push.
- Large outputs exist under output/ (pes.h5 and events.h5 per run); read them with h5py or HDF5.jl rather than rerunning the GPU stage.
- No scan is running. The results of the note are complete under fullsim_analysis/results/2026-10_ambient/; the cold campaign (2026-xx_digitizer_cold) and the early Radioroc2 scans (2026-xx_radioroc) are older and outside the scope of this review except where the note uses them.
- Every figure of the note regenerates with python3 fullsim_analysis/plots/room_temperature_study.py 2026-10_ambient and python3 fullsim_analysis/plots/geometry.py (arguments in its docstring).

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
- **Severity:** changes a bench setting / changes a number or conclusion in the documents / cosmetic.
- **Confidence:** verified, plausible, or suspicion.

Start the report with a five-line verdict on the main conclusions of the note. Keep findings you could not verify separate from verified ones. Do not pad the report with things that are fine.
