# Optical transport

The optical transport follows each scintillation photon from its emission point in the
crystal until it is detected at the SiPM plane or lost. Each photon draws from its own
counter-based random stream (`PhiloxStream`, `randu`), so the result does not depend on
the threads. The crystal is a `Box`, and `OpticalParams`, `Readout` (built on a
`SipmGrid`) and `TimeBinning` describe its optics, the readout face and the time binning
of the output. `run_photons!` generates photons at one or several points
(`isotropic_dir` and the emission time) and calls `propagate_photon!` for each, which
uses `wall_hit`, the surface optics (`fresnel`, `surface_interact`, `lambertian_dir`,
`window_pass`) and the bulk processes (`rayleigh_dir`) until a terminal status; the
tallies go into an `Accumulator`. `run_photons_ka!` is the same transport as one
KernelAbstractions kernel, for the CPU or Metal backends, and is the path used by the
event pipelines.

```@autodocs
Modules = [CryspLight]
Pages = ["src/philox.jl", "src/geometry.jl", "src/optics.jl", "src/generation.jl",
         "src/transport.jl", "src/kernel.jl"]
```
