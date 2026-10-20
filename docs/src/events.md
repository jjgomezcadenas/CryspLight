# Gammas and events

This block turns gammas into scintillation photons and the photons into per-event
outputs. `gamma_crystal` builds the volume of the vendored gamma core for the crystal,
and `gamma_deposits` transports one gamma through it and returns its energy deposits in
the crystal frame, with their times. Two pipelines use them. `run_events!` shoots 511 keV
gammas along the crystal axis and returns, per event, the light map, the first
photoelectron times and the truth of the first two interactions. `run_decays!` simulates
Na-22 decays at a point source outside the crystal: `na22_decay` gives the gammas of a
decay (`GammaRay`), `entry_point` finds where each meets the crystal, and the deposits
are converted into photons with `rand_poisson` and transported in batches with
`run_photons_ka!`; its output is what the photoelectron file stores.
`face_acceptance` gives the geometric acceptance of a face for comparison.

```@autodocs
Modules = [CryspLight]
Pages = ["src/gamma_interface.jl", "src/events.jl", "src/source.jl", "src/decays.jl"]
```
