# Materials and configuration

Every parameter of a run comes from TOML files: one file per material under data/
(crystal, wrap, SiPM), read with `load_material` and, for the scintillation times,
`scintillation_times`, and one run configuration that names the materials and sets the
geometry, surface finish, readout, source and run size. A point-source configuration
under runs/ is run end to end by `run_from_config`, which validates the finish with
`surface_has_gap` and writes the light map. A full-simulation optical configuration under
fullsim_runs/ is resolved by `load_fullsim` into a `FullsimSetup`, which
`run_fullsim_optical` uses to produce the photoelectron file.

```@autodocs
Modules = [CryspLight]
Pages = ["src/materials.jl", "src/config.jl", "src/fullsim_config.jl"]
```
