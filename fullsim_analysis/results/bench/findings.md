# Bench findings

Problems and anomalies found in the bench data (A5202 boards, Citiroc 1A, PET CAEN DAQ), their
cause and their solution. One entry per finding, numbered in order of discovery; an entry is
updated, not deleted, when the cause or the solution is found. Runs are DAQ run numbers; the
data are in benchdata/<YYYY-MM-DD>/ (not in git), the tables and figures in
fullsim_analysis/results/bench/<YYYY-MM-DD>/, produced by
fullsim_analysis/scripts/bench/inspect_run.jl and fullsim_analysis/plots/staircase.py.
Boards are numbered as in the DAQ (board 0-4 = chain,node 0,0 / 0,1 / 1,0 / 1,1 / 1,2),
channels 0-63.

| # | Finding | Seen | Status |
|---|---|---|---|
| F1 | All values at full scale (8191) | 21 Oct, runs 1-9, hold scan 12 | solved (JMB) |
| F2 | HV_Vbias 40.7 V in the files, 36.5 V in the logbook | 21 Oct | explained |
| F3 | Logbook temperature was the board's | 21 Oct | solved |
| F4 | Board 0: no signal from the SiPM in channels 15, 55, 57, 59, 61, 63 | 21 Oct, scans 10-11 | open |
| F5 | Board 0: low dark rate in channels 0, 2, 8, 10 | 21 Oct, scans 10-11 | open |
| F6 | 0xDEADBEEF as a staircase count | 21 Oct, scan 11 | open |
| F7 | Boards 1 and 4: channels 0-31 read 0 in every event | 22 Oct, runs 26-34 | open |
| F8 | Scans without logbook rows or staircase.h5 | 21 Oct, scans 10-12 | open |
| F9 | Board 2 runs a different firmware | 22 Oct, runs 26-34 | open |
| F10 | Board 0 noisier than the others at 25 ns | 22 Oct, runs 26-34 | open |

## F1. All values at full scale

- **Seen:** 21 Oct, runs 1-9 and hold scan 12, board 0. Every measurement of every channel, in
  both gains, is 8191 (all 13 bits at one), independent of shaping, hold delay and gain. The
  noise of the electronics could not be measured. Evidence: 2026-10-21/runs.csv, channels.csv,
  report_2026-10-21.tex.
- **Cause:** not in the recorded configuration, which agreed with bench note 1. Fixed by JMB
  (logbook, 22 Oct); the cause has not been reported.
- **Status:** solved. Runs 26-34 (22 Oct) have no value at full scale.
- **Action:** ask JMB what the cause was, to record it here.

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

- **Seen:** 21 Oct, scans 10 and 11, board 0 (the only board scanned so far), SiPMs above
  breakdown. Channels 15, 55, 57, 59, 61 and 63 fall from the noise plateau (edge at 180-185
  DAC) to zero within 10 DAC; 30 DAC above the edge they count 0-4 Hz, against a median of
  6e5 Hz. Evidence: 2026-10-21/staircase_11.png, staircase_11_channels.csv.
- **Cause:** the SiPMs are connected (known), so the signal is lost between SiPM and Citiroc:
  a bad contact or a fault in the connector. Five of the six are the odd channels 55-63.
- **Pedestal noise as a hint, not a test:** a SiPM at the input adds capacitance and raises
  the noise of the preamplifier. In run 28 (22 Oct, 87.5 ns, below breakdown) the six channels
  are at 0.82-0.87 of the board-0 median rms, but channels with avalanches reach 0.81-0.89 on
  boards 0, 1, 3 and 4, so the noise does not separate them; at 25 ns they sit at 0.97-1.05.
  Histograms: 2026-10-22/bench_noise.png (fullsim_analysis/plots/bench_noise.py).
  inspect_run.jl flags only channels below 0.6 of the board median (a dead input); none on
  22 Oct.
- **Action:** staircases of all five boards; check the contacts and connector of these
  channels on board 0.
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
- **Action:** question to JMB.

## F7. Half of boards 1 and 4 reads 0

- **Seen:** 22 Oct, runs 26-34. On boards 1 (chain 0, node 1) and 4 (chain 1, node 2),
  channels 0-31 give 0 in every event, high and low gain; channels 32-63 of the same boards are
  normal (pedestals 50-230 counts, rms about 12 counts at 25 ns). Channels 0-31 are one of the
  two Citirocs of the A5202. Evidence: 2026-10-22/channels.csv.
- **Not the cause:** Pedestal = 0 on all boards. It was the first suspicion, but the live
  channels have pedestals of 50-310 counts and essentially no value at 0, so the ADC floor
  clips nothing.
- **Cause:** unknown: the first Citiroc of these two boards is not read, not configured, or
  its data are lost in the readout.
- **Action:** question to JMB and Laura.

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
- **Action:** question to JMB.

## F10. Board 0 noisier

- **Seen:** 22 Oct, runs 26-34. At 25 ns and HoldDelay 0 the median high-gain rms is 15.9
  counts on board 0 and 12.0-12.4 on boards 1-4; at 87.5 ns there is no difference (12.7
  against 12.9-13.1). Board 0
  also runs cooler (board 48-51 C against 54-57 C).
- **Cause:** unknown; board 0 may be the one connected to the SiPM array (to be confirmed).
- **Action:** confirm which board reads the array.
