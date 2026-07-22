# CryspLight

CryspLight simulates scintillation-light transport in wrapped CsI and BGO crystals
read out by an 8 x 8 SiPM matrix. It combines a Julia optical and 511 keV gamma
Monte Carlo with a Python/PyTorch reconstruction pipeline.

## Platform and prerequisites

The supported environment is Apple Silicon macOS with Julia 1.11 or newer. Metal.jl
is a required package dependency and provides GPU acceleration when functional; the
transport kernel can also run on the CPU using ordinary Julia arrays. Python 3.11 or
newer is required for the analysis programs.

## Installation

Instantiate the locked Julia environment from the repository root:

    julia --project=. -e 'using Pkg; Pkg.instantiate()'

Create and populate a Python virtual environment:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r analysis/requirements.txt

The committed Manifest.toml and exact Python requirements describe the environments
used to reproduce the simulation and analysis results. PyTorch wheels are platform
and Python-version specific; use a supported Python version if pip cannot resolve the
locked torch release.

## Quick start

Run the Julia tests:

    julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'

Run a point-source optical simulation:

    julia --project=. --threads=auto scripts/run_point_source.jl runs/point_csitl_teflon.toml

Generate an end-to-end gamma-event dataset:

    julia --project=. --threads=auto scripts/shoot_gammas.jl runs/gammas_csitl.toml

Train a reconstruction model after generating the configured event dataset:

    .venv/bin/python analysis/cnn/train.py analysis/cnn/runs/cnn_csitl.toml

Run configurations under runs/ are the source of truth. Simulation files are written
under output/<tag>/, which is intentionally ignored by Git. The gamma pipeline writes
events.h5 with event energy, SiPM count and first-hit-time maps, and first/second-site
truth. HDF5 arrays written by Julia are event-axis-first when read by h5py, so maps
arrive as (events, y, x) and positions as (events, 3).

## Repository guide

- src/ contains the Julia package, including the CPU reference transport, the
  KernelAbstractions CPU/Metal twin, and the vendored gamma core.
- test/runtests.jl checks analytical physics limits, conservation, RNG behavior,
  gamma transport, event output, and CPU/GPU agreement.
- analysis/ contains control plots and the PyTorch reconstruction pipeline.
- latex/ contains the design, calibration, software, end-to-end, and reconstruction
  notes. CLAUDE.md provides a detailed project status and developer orientation.
- data/ contains material and NIST XCOM inputs; scripts/ contains reproducible drivers.

## Reproducibility and scope

The simulation records run configuration, seed, material parameters, backend, and
relevant detector metadata in its outputs. The optical CPU and KernelAbstractions
implementations intentionally mirror one another draw-for-draw, with parity enforced
by tests. The physical assumptions and calibration provenance are documented in the
LaTeX notes.

This is research software. Validate its models and calibrations for your intended use.
See LICENSE for usage rights.
