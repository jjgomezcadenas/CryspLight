# Photoelectron file and output

The photoelectron file holds the result of the optical stage of a full simulation: for
every decay, the interactions of its gammas, the photon counts and the integrated map,
and one record (channel, time, wavelength) per photon reaching an active area, stored
consecutively with offset arrays. The driver creates it with `open_pes`, appends each
chunk of decays returned by `run_decays!` with `append_pes!` and closes it with
`close_pes`; the electronics stage reads it back whole with `read_pes` and takes the
records of one decay with `event_pes`. `code_version` stamps the git description of the
code into the files. The point-source runs write their own light.h5 through
`run_from_config`.

```@autodocs
Modules = [CryspLight]
Pages = ["src/pes_file.jl", "src/output.jl"]
```
