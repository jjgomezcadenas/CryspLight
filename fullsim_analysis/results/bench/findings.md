# Bench findings

Problems and anomalies found in the bench data (A5202 boards, Citiroc 1A, PET CAEN DAQ), their
cause and their solution. One entry per finding, numbered in order of discovery; an entry is
updated, not deleted, when the cause or the solution is found. Runs are DAQ run numbers; the
data are in benchdata/<YYYY-MM-DD>/ (not in git), the tables and figures in
fullsim_analysis/results/bench/<YYYY-MM-DD>/, produced by
fullsim_analysis/scripts/bench/inspect_run.jl, noise_components.jl and fullsim_analysis/plots/staircase.py;
the daily reports are in docs_fullsim/bench/. The requests that follow from the open entries
are listed at the end.
Boards are numbered as in the DAQ (board 0-4 = chain,node 0,0 / 0,1 / 1,0 / 1,1 / 1,2),
channels 0-63.

| # | Finding | Seen | Status |
|---|---|---|---|
| F1 | All values at full scale (8191) | 21 Oct, runs 1-9, hold scan 12 | solved (JMB) |
| F2 | HV_Vbias 40.7 V in the files, 36.5 V in the logbook | 21 Oct | explained |
| F3 | Logbook temperature was the board's | 21 Oct | solved |
| F4 | SiPM not seen: board 0 channels 15, 55, 57, 59, 61, 63 (staircases); 9 more predicted on boards 1, 3, 4 (common mode) | 21-22 Oct | open |
| F5 | Board 0: low dark rate in channels 0, 2, 8, 10 | 21 Oct, scans 10-11 | open |
| F6 | 0xDEADBEEF as a staircase count | 21 Oct, scan 11 | open |
| F7 | Boards 1 and 4: channels 0-31 read 0 in every event | 22 Oct, runs 26-34 | open |
| F8 | Scans without logbook rows or staircase.h5 | 21 Oct, scans 10-12 | open |
| F9 | Board 2 runs a different firmware | 22 Oct, runs 26-34 | open |
| F10 | Board 0: common-mode jumps of the whole board in 0.1-0.4% of the events | 22 Oct, runs 26-32 | open |
| F11 | Two pedestal peaks in the quick-look plots grouped by node | 22 Oct, runs 26-34 | explained |

## F1. All values at full scale

- **Seen:** 21 Oct, runs 1-9 and hold scan 12, board 0. Every measurement of every channel, in
  both gains, is 8191 (all 13 bits at one), independent of shaping, hold delay and gain. The
  noise of the electronics could not be measured. Evidence: 2026-10-21/runs.csv, channels.csv,
  report_2026-10-21.tex.
- **Cause:** not in the recorded configuration, which agreed with bench note 1. Fixed by JMB
  (logbook, 22 Oct); the cause has not been reported.
- **Status:** solved. Runs 26-34 (22 Oct) have no value at full scale.

## F2. HV_Vbias in the files and in the logbook

- **Seen:** 21 Oct. The files record HV_Vbias = 40.7 V (HV_IndivAdj = 0, HV_Adjust_Range =
  4.5 V); the logbook said 36.5 V. HV current 0.2-0.4 uA, i.e. SiPMs below breakdown.
- **Cause:** HV_Vbias is the output of the HV module; the individual adjustment with
  HV_IndivAdj = 0 subtracts 4.2 V, so the SiPMs see 36.5 V.
- **Status:** explained. From 22 Oct the logbook writes "40.7 -> 36.5".

## F3. Temperature in the logbook

- **Seen:** 21 Oct. The logbook gave 50-50.5 C, the temperature of the board; the detector was
  at 24.7-24.8 C.
- **Status:** solved. From 22 Oct the logbook gives the detector temperature, per board.

## F4. Channels that do not see their SiPM

- **Seen:** 21 Oct, scans 10 and 11, board 0, SiPMs above breakdown. Channels 15, 55, 57, 59,
  61 and 63 fall from the noise plateau (edge at 180-185 DAC) to zero within 10 DAC; 30 DAC
  above the edge they count 0-4 Hz, against a median of 6e5 Hz. Evidence:
  2026-10-21/staircase_11.png, staircase_11_channels.csv.
- **Cause:** the SiPMs are connected (known), so the signal is lost between SiPM and Citiroc:
  a bad contact or a fault in the connector or the adapter board.
- **Found also in the pedestal runs (22 Oct):** the common mode of a chip enters through the
  SiPMs, so a channel that does not see its SiPM does not receive it. The slope of each
  channel against the common mode of its chip (noise_components.jl, channel_slopes.csv) is
  0.02-0.6 for the six channels in every run, and above 0.85 for every channel of board 0
  with avalanches. With a cut at 0.75 the same criterion finds, on the other boards:
  board 1: 61, 63; board 3: 45, 47, 49, 60; board 4: 59, 61, 63; board 2: none. These are
  predictions for the staircases of the five boards. The odd channels 55-63 recur on boards
  0, 1 and 4. The pedestal rms alone does not separate them (0.82-0.87 of the board median
  against 0.81-0.89 for others). Evidence: 2026-10-22/bench_slopes.png, bench_noise.png.
  noise_components.jl prints the channels below the cut ("SiPM not seen"); inspect_run.jl
  no longer cuts on the rms.
- **Action:** staircases of all five boards (menu 2) to test the predictions; check the
  contacts, connectors and adapter boards of these channels.
- **Consequence:** these channels cannot be calibrated with the periodic trigger and will not
  record the crystal light.

## F5. Channels with low dark rate

- **Seen:** 21 Oct, staircases 10 and 11, board 0 (other boards not scanned). Channels 0, 2, 8 and 10 count 60-90 kHz
  30 DAC above the edge, 7-10 times below the median. Evidence as F4.
- **Not seen in the pedestal:** these channels see their SiPM (they count dark pulses); a low
  dark rate cannot show in a run below breakdown.
- **Cause:** unknown; even channels 0-10, again a connector or mapping pattern?
- **Action:** staircases of all boards, to see whether other boards have such channels.

## F6. 0xDEADBEEF in a staircase

- **Seen:** 21 Oct, scan 11, channel 61, threshold 490: 3735928559 counts (0xDEADBEEF), a
  sentinel, not a measurement. staircase.py marks such points invalid.
- **Action:** none for now (not asked; see Requests).

## F7. Half of boards 1 and 4 reads 0

- **Seen:** 22 Oct, runs 26-34. On boards 1 (chain 0, node 1) and 4 (chain 1, node 2),
  channels 0-31 give 0 in every event, high and low gain; channels 32-63 of the same boards are
  normal (pedestals 50-230 counts, rms about 12 counts at 25 ns). Channels 0-31 are one of the
  two Citirocs of the A5202. Evidence: 2026-10-22/channels.csv, bench_pedestals.png.
- **Not the cause:** Pedestal = 0 on all boards. It was the first suspicion, but the live
  channels have pedestals of 50-310 counts and essentially no value at 0, so the ADC floor
  clips nothing.
- **Not in the recorded configuration:** the effective configuration stored in the run files
  is the same for the silent chips as for the others: readout, charge and time discriminators
  enabled, preamplifiers on, gains 55, the same thresholds and shaping, and per-channel
  pedestal calibrations of 102-271 counts like those of the other chips.
- **Hypothesis:** the same chip on two boards, with a configuration identical to that of the
  chips that work, points to the readout (firmware or DAQ) rather than to two hardware faults,
  presumably software. To be confirmed by JMB.
- **Test:** read board 1 with Janus. If Janus sees channels 0-31, the problem is in the DAQ.
- **Action:** question to JMB. Once solved, repeat runs 26-34 (menu 1) for these 64 channels.

## F8. Scans without logbook rows

- **Seen:** 21 Oct. Scans 10, 11 (staircases) and 12 (hold delay) were taken in the evening
  (23:41 and 00:52 local time) and have no logbook rows; the manifests of the scans do not
  record the HV, and the staircase.h5 artifacts were not copied.
- **Action:** Laura to add the rows (time, settings, HV) and copy the artifacts.

## F9. Firmware of board 2

- **Seen:** 22 Oct, manifests of runs 26-34. Boards 0, 1, 3 and 4 run firmware 0xa1230708,
  board 2 runs 0xa7070800. Boards 3 and 4 also report a different acquisition state (16387,
  against 3 for the others) and service status 0x400d (0x000d for the others).
- **Cause:** unknown; not visibly related to F7 (board 2 reads normally, board 1 does not).
- **Action:** question to JMB: is the difference intended, could it be behind F7 or F10?

## F10. Common-mode jumps of board 0

- **Seen:** 22 Oct. First reported as board 0 being noisier at 25 ns (median high-gain rms
  15.9 counts in run 26, against 12.0-12.4 on boards 1-4). The robust width (interquartile
  range / 1.349) of board 0 is the same as on the other boards (11.9 counts): the excess is
  in the tails. These come from events in which the channels of a chip move together by
  60-230 counts, up or down, often in two consecutive triggers, and in most runs both chips
  of the board in the same number of events: 41 such events in run 26
  (0.4%), 10-28 per chip in runs 27-32, none in runs 33 and 34. The other boards have none
  (board 3: one event in runs 30-32). Evidence: 2026-10-22/noise_components.csv (column
  outlier_events; fullsim_analysis/scripts/bench/noise_components.jl).
- **Cause:** unknown. A whole-board shift points to the reference, the supply or the ground
  of board 0, or to a disturbance on its cable; board 0 is also the coolest (board 48-51 C
  against 54-57 C).
- **Consequence:** small for the pedestal runs (the events are set aside), but in data
  with the crystal such an event would look like a shift of the energy of all channels.
- **Action:** question to JMB and Laura (is anything different in the cabling or the power of
  board 0?); watch for it in the next runs.

## F11. Two pedestal peaks in the plots grouped by node

- **Seen:** 23 Oct, in Laura's quick-look plots of runs 26-28: two pedestal peaks in most
  channels of "board 0" and in channels 32-63 of "board 1", half of "board 2" empty, and only
  three boards.
- **Cause:** the plots group the measurements by node alone. With two chains (boards 0, 1 on
  chain 0, nodes 0, 1; boards 2, 3, 4 on chain 1, nodes 0, 1, 2), node 0 mixes boards 0 and 2,
  node 1 boards 1 and 3, node 2 is board 4 alone; the empty halves are F7. The peak positions
  agree channel by channel with the pedestals of each board. Each board alone has one
  pedestal per channel. Evidence: 2026-10-22/node_mixing.png, pedestal_spectra_board0-4.png
  (fullsim_analysis/scripts/bench/pedestal_spectra.jl, fullsim_analysis/plots/pedestal_spectra.py).
- **Status:** explained; not shorts on the adapter boards, no board missing.
- **Solution:** identify the board by (chain, node), numbered with the topology table of the
  run (manifest execution_identity.topology.boards, or configuration/effective/boards in the
  event file). Recipe in the report of 22 Oct.

## Requests

Open entries and what they need, by person. For Laura, menu 1 is complete; the next session
is menu 2 (bench note 2), with the corrections listed in the report of 22 Oct.

- **JMB:** F7 (silent chip on boards 1 and 4; Janus test), F10 (common-mode jumps of board 0:
  hardware or firmware?), F9 (firmware of board 2: intended, related to F7 or F10?). F6 is not
  asked: a single invalid point, flagged by staircase.py.
- **Laura:** F11 (group by chain and node in the quick-look plots); F4 and F5 (staircases of the five boards, in menu 2; contacts and connector of
  the board-0 channels), F8 (logbook rows of scans 10-12), F10 (cabling and power of board 0);
  after F7 is solved, repeat menu 1 for the 64 missing channels.
