# Analysis library

The analysis library, module `FullsimAnalysis` under fullsim_analysis/lib, is not part
of the package: the scripts load it with

```
include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
using .FullsimAnalysis
```

A study usually reads an event file with `read_events`, selects decays with the
`select_*` functions, builds the equalised sum and fits its peaks (`equalised_sum`,
`find_peak511`, `peak_fit`), draws channel maps (`occupancy_map`, `mean_map`,
`seed_peak_map`) and writes its arrays and tables under
fullsim_analysis/results/<campaign>/<study>/ (`open_results`, `write_hist!`,
`write_table`, `update_table`). The scans run a front end on chosen decays of a photon
file without writing an event file (`open_pes_run`, `event_lists`, `emulate`, `with`).
The calibration helpers fit the photoelectron ladder of the dark counts
(`photoelectron_ladder`, the matched filter of the digitizer) and convert the charges to
photoelectrons by the route of the measurement (`measured_charge`). The readers of the
bench DAQ files (`read_daq_run`, `read_daq_histograms`, `read_daq_staircase`,
`read_daq_hold_delay`) return the stored values in raw ADC counts and clock ticks.

## Event files and selections

```@autodocs
Modules = [FullsimAnalysis]
Pages = ["lib/io.jl"]
```

## Fits

```@autodocs
Modules = [FullsimAnalysis]
Pages = ["lib/fits.jl"]
```

## Spectra and maps

```@autodocs
Modules = [FullsimAnalysis]
Pages = ["lib/spectra.jl", "lib/maps.jl"]
```

## Results files

```@autodocs
Modules = [FullsimAnalysis]
Pages = ["lib/results.jl"]
```

## Emulation scans

```@autodocs
Modules = [FullsimAnalysis]
Pages = ["lib/emulate.jl"]
```

## Calibration

```@autodocs
Modules = [FullsimAnalysis]
Pages = ["lib/calibrate.jl"]
```

## Bench DAQ files

```@autodocs
Modules = [FullsimAnalysis]
Pages = ["lib/io_daq.jl"]
```
