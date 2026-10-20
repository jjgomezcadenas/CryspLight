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
