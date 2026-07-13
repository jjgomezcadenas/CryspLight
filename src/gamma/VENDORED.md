# Vendored gamma-transport core

Source: PTCryspMC.jl (jjgomezcadenas), commit **d4982db**, vendored 2026-07-13.
Both repositories share the same author; no separate license file exists upstream.

| file here | upstream | changes |
|---|---|---|
| nist_data.jl | src/nist_data.jl | verbatim |
| sampling.jl | src/sampling.jl | verbatim |
| gtransport.jl | src/transport.jl | verbatim (renamed to avoid clash with the optical transport) |
| gmaterials.jl | src/materials.jl | trimmed: scintillation/readout fields and the materials.json/JSON layer dropped; `make_material(name, density, xcom_csv)` added |
| ggeometry.jl | src/geometry.jl | trimmed to Box + LogicalVolume/PhysicalVolume + slab method; Cylinder/Sphere/CylShell/Scanner not vendored |
| ../../data/xcom_CSI.csv, xcom_BGO.csv | data/ | verbatim (NIST XCOM exports) |

Internal conventions preserved: cm, MeV, Float64, origin-centred volumes, explicit
AbstractRNG everywhere. Physics scope: free path from XCOM cross sections,
Compton (Klein-Nishina, Butcher-Messel) with local recoil deposit, photoelectric
absorption, pair channel (closed below 1.022 MeV), 10 keV energy cut; coherent
scattering loaded but unused (as upstream); no electron transport.
The mm/keV/Float32 conversion and the per-deposit times (cumulative path / c) are
added OUTSIDE the vendored code, in src/gamma_interface.jl.

When updating: re-diff against upstream at the recorded commit before pulling changes.
