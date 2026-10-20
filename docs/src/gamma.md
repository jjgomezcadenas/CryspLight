# Vendored gamma core

The submodule `CryspLight.Gamma` is a copy of the 511 keV photon-transport core of
PTCryspMC.jl at commit d4982db, kept as close to upstream as possible; src/gamma/VENDORED.md
lists, file by file, what was copied verbatim and what was trimmed. It works in cm, MeV
and Float64, with volumes centred on the origin. A `Material` is built from an XCOM table
with `make_material`, placed as a `Box` in a `LogicalVolume` and a `PhysicalVolume`, and
`propagate_photon` transports one gamma through it, returning its interactions. The
rest of CryspLight reaches the core only through `gamma_crystal` and `gamma_deposits`
(page Gammas and events), which convert to the crystal frame in mm and keV and add the
deposit times. The docstrings below are those of the vendored files, left as upstream
wrote them.

```@autodocs
Modules = [CryspLight.Gamma]
```
