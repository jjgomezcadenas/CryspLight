# benchdata

Raw files from the CRYSP bench (Donostia), as downloaded from the PET CAEN DAQ
(Runs tab, Artifacts) and from the oscilloscope. The files themselves are not in git;
only this README is.

Layout: one folder per measurement day, `benchdata/YYYY-MM-DD/`, holding the files as
they were downloaded (`run_N.0000.h5`, `run_N.histograms.h5`, `staircase.h5`,
`hold-delay.h5` renamed with their run number, oscilloscope CSV files) and a copy of the
day's logbook rows. Each file is described by its row in the Google Sheet logbook (bench
note 0): run number or file name, menu and step, settings, temperature, incidents.

Readers: fullsim_analysis/lib/io_daq.jl (`read_daq_run`, `read_daq_histograms`,
`read_daq_staircase`, `read_daq_hold_delay`). Studies of these data go in
fullsim_analysis/scripts/bench/, their results in fullsim_analysis/results/.

## Days

- 2026-10-21, 2026-10-22, 2026-10-23 (runs 42-50 with the red CAEN adapter boards; the test runs
  35-41 of that day in 2026-10-23_tests): five boards, DAQ board numbers 0-4 = chain,node
  0,0 / 0,1 / 1,0 / 1,1 / 1,2. Logbook rows in logbook.csv.
- 2026-10-27: the whole programme of bench note 0 on three boards. Copied from
  jjgomez@hyperion.sw.ehu.es:/data/lnavarro/26_10_caen_measurements_jj/ (the remote folder also
  holds the runs 1-50 of 21-23 Oct; runs 51-90 and 107 are not there). Three boards on one chain,
  DAQ board 0, 1, 2 = chain,node 0,0 / 0,1 / 0,2, all on firmware 0xa7070800: Laura's boards 1, 2
  and 4 (those with the upgraded FPGA), in that order, as the logbook states ("aunque pone boards
  1, 2 y 4 en el daq aparecen como 0, 1 y 2"; the detector temperatures per board agree, and so
  do the blind channels: DAQ 0 has 61 and 63 blind as board 1 on 23 Oct, DAQ 2 has 59, 61, 63 as
  predicted for board 4 on 22 Oct, DAQ 1 none, as board 2). Contents, by DAQ number and time
  (UTC; the logbook gives local time, UTC+1):
  - runs 91-99 (16:11-16:35): menu 1, pedestals below breakdown (HV_Vbias 40.7 -> 36.5 V),
    10 000 events, HoldDelay 0 at 25, 50, 87.5 ns (91-93), then 25 ns at HoldDelay 96, 200, 296,
    400, 496 ns (94-98) and HoldDelay 0 again (99);
  - scan-100 (16:52): menu 2 step 1, staircase on board 1 (DAQ 0), thresholds 150-300, 25 ns,
    HV_Vbias 47.2 -> 43 V (5 V of overvoltage);
  - hold-scan-101 to 106 (17:09-17:21): menu 2 steps 2-5, hold-delay scans on board 1 (DAQ 0) at
    43 V, threshold 185, 10 000 events per delay: 101 at 25 ns and 102, 103 at 50 ns with channel 0
    shooting (0-248 ns in steps of 8 for 101, 0-496 in steps of 16 for 102 and 103), then 104, 105,
    106 at 25 ns, 0-248 ns, with channels 20, 40 and 60 shooting;
  - runs 108-112 (17:37-17:50): menu 3, calibration with the periodic trigger (PtrgPeriod 1 ms,
    HoldDelay 200 ns, 50 000 events per board): HV_Vbias 47.2 V (5 V of overvoltage) at 25, 50 and
    87.5 ns (108-110), then 46.2 V (111) and 45.2 V (112) at 25 ns;
  - scan-113, 114, 115 (18:06-18:19): menu 3 step 4, staircases on board 1 (DAQ 0) at 25 ns:
    113 at 43 V (5 V of overvoltage), thresholds 184-385, 1-2 pe spacing 20 DAC by Laura's
    reading; 114 and 115 at 41 V (3 V), thresholds 150-300 and 190-275, spacing 12 DAC. The
    HoldDelay was 200 ns in 113 and 0 in 114-115 (Laura asks whether it matters).
  logbook.csv holds all the rows of the day (menus 1, 2 and 3), in three blocks as in the sheet.
