# Independent review of the CryspLight full simulation (brief of 20 October 2026)

You are reviewing a simulation built by another model with the author (J.J. Gómez-Cadenas, CRYSP group). Your job is to find what is wrong, unjustified or untested, not to confirm it. Assume nothing in the documents is correct until you have checked it against the code, the datasheets, the papers or a calculation of your own. A previous review (docs_fullsim/review_gpt-6_2026-10-18.md) covered the state of 18 October; most of what it examined has since been rewritten, so do not rely on it except as a record of what was found then.

## What the simulation is

CryspLight simulates monolithic scintillating crystals, 48 x 48 mm2 and two radiation lengths thick, polished, wrapped in Teflon and coupled with optical grease to an 8 x 8 array of Hamamatsu S14160-6050HS SiPMs (6 x 6 mm2, about 2 nF each; four S14161-6050HS-04 tiles). It answers one question: how the performance of such a detector (light efficiency, photoelectrons per keV, energy resolution and, later, spatial resolution) depends on the crystal (BGO, CsI(Tl)), on the readout (CAEN DT2745 waveform digitiser, Citiroc 1A ASIC, Radioroc2 ASIC) and on the temperature. It has three models: the light transport in the crystal (GPU optical Monte Carlo, calibrated in earlier campaigns), the SiPM (detection efficiency, gain, branching crosstalk, afterpulses, dark counts, and the pulse that an avalanche gives at the input of the electronics) and each readout (shaping, noise, trigger, peak detection or sampling and integration, conversion, saturation). Each readout is calibrated on its own dark counts, as a system with many crystals would have to be.

The configurations are BGO and CsI(Tl) at room temperature with the three readouts, and BGO at 250 K with the digitiser and the Radioroc2. The cold case was BGO at 196 K until 20 October; it was moved to 250 K (see the defect below). The Citiroc is not simulated cold: its shaping keeps 0.12 of a 0.84 us decay, so even with 1.69 times more light it stores 28% less than at room temperature.

## The defect corrected on 20 October, to be checked first

The BGO light yield and decay time against temperature were inconsistent with the source they cited. The simulation used 7600 ph/MeV at room temperature and 15 000 ph/MeV with 2.72 us at 196 K, a warm-to-cold light ratio of about 2.0. Gironnet et al., NIM A 594 (2008) 358 (papers/tau_dependence_BGO.txt), give a ratio of 2.33 between 196 K and 295 K. The inputs are now: 8000 ph/MeV and 300 ns (plus a 60 ns component with 10% of the light) at room temperature, the author's adopted reference; and at 250 K, 13 500 ph/MeV (Gironnet's curve scaled to 8000) and 0.84 us from their three-level model, Eq. (1). Their Fig. 1 was digitised by eye and their model evaluated in scripts/bgo_gironnet_table.py, which writes data/bgo_gironnet_T.csv. Check:

- the digitisation of Fig. 1 against the text (half the maximum at 250 K, factor 3.5 between 6 and 295 K) and the evaluation of Eq. (1) against their measured points (0.43 us at 295 K, 2.2 us at 200 K);
- whether the 0.84 us at 250 K is credible, given that their technique loses precision below 1 us (their 0.43 us at 295 K against 0.30 us in the literature);
- whether scaling the curve measured under alpha excitation to a gamma yield at room temperature is justified (they argue that the alpha/gamma ratio does not depend on temperature);
- the material files data/bgo_ambient.toml and data/bgo_250k.toml, and that nothing in the code or the documents still uses the old values.

The author's own measurement of 2.72 us at 196 K is set aside (probably a temperature misreading) and is not a finding.

## Read in this order

1. docs_fullsim/fullsim_design.tex: the specification of the models, their parameters and their provenance. The author is rewriting the abstract and Section 1 by hand; that text is a draft (it still says 196 K) and is not a finding. The model sections (Response of the SiPM, Electronics, Observables, Parameters) are current.
2. docs_fullsim/room_temperature_study.tex: the six cases at room temperature on 10 000 decays (uncertainty about 0.2 points on the resolutions). The BGO numbers were updated on 20 October to the 8000 ph/MeV inputs.
3. fullsim_analysis/results/2026-10_bgo250k/bgo_250k/bgo_250k.csv and the case figures there: the two cases at 250 K. No note has been written for them yet.
4. Code: src/frontend/sipm.jl (physical_pulse, with_input), src/frontend/shaping.jl (crrc_voltage!, Shaper with voltage), src/frontend/citiroc.jl, src/frontend/radioroc.jl, src/frontend/digitizer.jl, src/frontend/frontend_run.jl; data/sipm_s14160.toml; test/runtests.jl.
5. Analysis: fullsim_analysis/lib/fits.jl (photoelectron_ladder, peak_fit_window), lib/calibrate.jl, scripts/calibration.jl, scripts/trigger_scan.jl, scripts/room_temperature_study.jl, tools/pick_threshold.py, and the configurations in fullsim_runs/2026-10_ambient, 2026-xx_radioroc and 2026-10_bgo250k.
6. Primary sources, as text (do not read the PDFs): papers/citiroc1a-datasheet.txt, papers/radioroc2-datasheet.txt, papers/s14160_s14161_series_kapd1064e.txt, papers/um8717-2740-2745-digitizers.txt, papers/tau_dependence_BGO.txt, and the other .txt files in papers/.

## What changed on 19 and 20 October and has not been reviewed

1. **Pulse of the SiPM.** No longer two free time constants: the cell recharge (tau_rec = 50 ns, assumed) filtered by the terminal capacitance (2000 pF, datasheet) and the input resistance (50 ohm for all readouts), giving a difference of exponentials with peak at 69 ns and a 100 ns fall. The fast component through the quenching capacitance is neglected. Is this the right circuit for the 6050HS, and is 50 ohm right for the Radioroc2 and the digitiser?
2. **Citiroc preamplifiers.** Implemented as voltage-sensitive, AC-coupled (datasheet: "two parallel AC coupled voltage preamplifiers"): the CR-RC^2 shaper acts on the current pulse, so the CR differentiates it (crrc_voltage!). The Radioroc2 keeps the charge-sensitive form (n + 1 integrators). The stored fractions follow: 0.29 (BGO) and 0.10 (CsI(Tl)) for the Citiroc at 87.5 ns, against 0.49 and 0.20 for a charge preamplifier. Is the voltage-preamplifier model physically right, including the effect of the 50 ohm termination and the 2 nF on the preamplifier input?
3. **Noise of the ASICs.** 0.1 photoelectron rms on the high gain at 87.5 ns (Citiroc) and 100 ns (Radioroc2) at 3 V, scaled as (t_ref/t_p)^0.5; the low gain has the same noise in ADC counts. Derived from the Radioroc2 datasheet bound (1.6 fC with 100 pF) scaled by about 20 to 2 nF, about 30 fC or 0.07 photoelectron, rounded up. Is the scaling with capacitance and with peaking time defensible?
4. **Calibration of the ASICs on dark counts.** Every crossing of the discriminator at 0.5 photoelectron holds the high gain at the maximum of its pulse (the hold delay is the noise-free crossing-to-maximum delay), with a busy time; the peaks are the minima of the smoothed second derivative of the histogram, and the gain is the slope of the line through the 1, 2 and 3 photoelectron peaks. High-gain shaping 25 ns for the Citiroc (fitted/true 0.943, scatter 2.2%) and 100 ns for the Radioroc2 (0.959, 3.6%). The real A5202 board triggers on the OR of 64 channels with one common hold; the simulation holds each channel at its own crossing. Is the method sound, and does the idealisation change the conclusions? The low gain is calibrated through the high-to-low ratio, assumed known from charge injection; the injection charge is instantaneous while an avalanche is spread over 100 ns.
5. **Radioroc2 population at zero.** At room temperature the high-gain calibration spectrum of the Radioroc2 holds a population at zero, crossings of the time discriminator with no high-gain signal; at 250 K, with 30 times fewer dark counts, it is absent. What produces it?
6. **Digitiser calibration bias.** The matched-filter gain is 1.031 of the true one at room temperature and 1.056 at 250 K, the same for every channel. The energy scale absorbs it, but it enters the recorded photoelectron count. Why does it change with the dark rate?
7. **Thresholds.** Lowest threshold whose dark-trigger rate is below 1 kHz at 95% confidence, from 1500 empty decays. Room temperature: Citiroc 8 pe for BGO and CsI(Tl) (for BGO no threshold passes the rule; 8 gives 0.6 kHz and 76% efficiency between 200 and 300 keV), digitiser 704 counts (11 pe) for BGO and 576 (9 pe) for CsI(Tl) with majority 2, Radioroc2 charge trigger 30 (BGO) and 20 (CsI(Tl)). 250 K: digitiser 320 counts (5 pe), Radioroc2 10. Is 1500 empty decays enough for the bound, and is the crosstalk model (branching, Borel) adequate at the thresholds used?
8. **The 250 K configurations.** Digitiser at gain 4 and 5 V with a 4 us gate; Radioroc2 at 3 V with 1.8 us low-gain shaping and 3.56 us hold; dark counts from the model at 250 K (about 155 kHz per channel). Are the gate and hold appropriate for an 0.84 us decay, and is the dark-rate temperature law (factor 2 every 9 K) adequate down to 250 K?
9. **Resolution decomposition.** Replay of the SiPM response without dark counts on the photopeak events, widths as half the 16-84% interval, terms by differences in quadrature. At room temperature the BGO Radioroc2 resolution went from 9.15% to 8.33% when the yield went from 7600 to 8000 ph/MeV, more than the 5% light increase alone suggests; is that statistics of the 10k sample, or does it reveal something in the analysis?

## Current results (10 000 decays; FWHM at 511 keV, about +-0.2 points)

| Crystal, temperature | Citiroc 1A | DT2745 (gain 4, 5 V) | Radioroc2 |
|---|---|---|---|
| BGO, room temperature (8000 ph/MeV) | 11.2% | 8.9% | 8.3% |
| CsI(Tl), room temperature | 8.3% | 6.2% | 6.0% |
| BGO, 250 K (13 500 ph/MeV, 0.84 us) | not simulated | 7.1% | 6.7% |

Tables: fullsim_analysis/results/2026-10_ambient/room_temperature/room_temperature.csv, fullsim_analysis/results/2026-10_bgo250k/bgo_250k/bgo_250k.csv; calibration and trigger tables in the calibration/ and trigger_scan/ folders of each campaign.

## What is not a finding

- Inputs the bench measures this week: the noise of the ASICs with the SiPMs below breakdown, the hold delay and the shape of the shaped pulse, the single-photoelectron spectra, the dark rate and crosstalk against the threshold, and, with the digitiser, the height of one photoelectron at gains 1, 4 and 8 and the pulse shape. Unmeasured is known; a finding is a conclusion the simulation does not support given its inputs, or an input whose plausible alternative would change a conclusion, with the size of the effect computed.
- The draft abstract and Section 1 of fullsim_design.tex.
- The bench instruction notes in bench/ (written for the experimenter, out of scope).
- The spatial resolution, not yet implemented on the readout output.

## How to report

Write docs_fullsim/review_<model>_2026-10-2x.md: a few lines of summary, then one section per finding with where (file and line), what is wrong, the evidence (calculation, code reading, datasheet line), the consequence for a stated number or setting, and a proposed fix. Separate verified defects from plausible alternatives. Do not edit any other file in the repository; put scratch calculations outside it and say where.
