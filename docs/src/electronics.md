# Electronics

The electronics stage turns the photon records of each channel into what the readout
stores. A `SipmDevice` read from its data file and a `SipmOperation` (temperature,
overvoltage, gain factors) define the sensor; `respond` thins the records with the
detection efficiency and adds crosstalk, afterpulses and dark counts, returning a train
of `Avalanche`s per channel (`event_trains` does this for all the channels of a decay).
The front ends work on a `TimeGrid`: a `Shaper` applies the device pulse and a CR-RC^n
shaper, calibrated to the peak of one avalanche, and `noise_generator` adds shaped
noise. Each front end has its settings, a chain built once per run and an event
function: `CitirocSettings`, `CitirocChain` and `citiroc_event` for the Citiroc 1A,
`RadiorocSettings`, `RadiorocChain` and `radioroc_event` for the Radioroc2,
`DigitizerSettings`, `DigitizerChain` and `digitizer_event` for the DT2745 digitizer.
The calibration helpers (`hg_at_crossings`, `hg_periodic`, `hold_at_peak`,
`stored_fraction`, `dark_baseline`) emulate the bench procedures. `load_frontend` reads
a front-end configuration into a `FrontendSetup`, and `run_fullsim_frontend` applies the
whole stage to a photoelectron file and writes the event file.

## SiPM

```@autodocs
Modules = [CryspLight]
Pages = ["src/frontend/sipm.jl"]
```

## Shaping

```@autodocs
Modules = [CryspLight]
Pages = ["src/frontend/shaping.jl"]
```

## Citiroc 1A

```@autodocs
Modules = [CryspLight]
Pages = ["src/frontend/citiroc.jl"]
```

## Radioroc2

```@autodocs
Modules = [CryspLight]
Pages = ["src/frontend/radioroc.jl"]
```

## DT2745 digitizer

```@autodocs
Modules = [CryspLight]
Pages = ["src/frontend/digitizer.jl"]
```

## Running the stage

```@autodocs
Modules = [CryspLight]
Pages = ["src/frontend/frontend_run.jl"]
```
