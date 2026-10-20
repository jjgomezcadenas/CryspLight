# CryspLight

CryspLight simulates the scintillation light of a wrapped CsI or BGO crystal read out by
a matrix of SiPMs, and the response of the sensors and of the readout electronics to it.
The optical photons are transported on the GPU (Apple Metal, through
KernelAbstractions) or on the CPU, with bit-identical results on the CPU backend. The
gamma transport that produces the energy deposits comes from a vendored copy of the
PTCryspMC.jl core.

A full simulation runs in two stages. The optical stage (`run_fullsim_optical`) shoots
Na-22 decays at a point source, transports their gammas through the crystal, converts
each energy deposit into scintillation photons and transports the photons to the SiPM
plane; it writes one photoelectron file per run, with the interactions of every decay
and one record (channel, time, wavelength) per photon reaching an active area. The
electronics stage (`run_fullsim_frontend`) reads that file, applies the SiPM response
(detection efficiency, crosstalk, afterpulses, dark counts) and one front end (Citiroc
1A, Radioroc2 or DT2745 digitizer), and writes an event file with the same layout as
the measured data. The analysis library reads event files, measured or simulated, and
writes the results of the studies.

Units follow the physics of each stage: lengths in mm, times in ns, energies in keV
(MeV inside the vendored gamma core, which also works in cm), wavelengths in nm, charges
in avalanches (photoelectrons) at the SiPM and in ADC counts or count-samples after the
electronics.

```@docs
CryspLight
```

## Pages

```@contents
Pages = ["optical.md", "config.md", "events.md", "output.md", "electronics.md",
         "gamma.md", "analysis.md"]
Depth = 1
```

## Building this reference

The reference is built locally with Documenter.jl from the docs/ environment:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

and the result is opened at docs/build/index.html. The build fails when an exported
name has no docstring or when a docstring is left out of these pages.
