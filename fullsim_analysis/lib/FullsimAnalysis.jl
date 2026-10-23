# Library of the full-simulation analysis (Julia). Scripts under fullsim_analysis/scripts/
# include this file; they compute and write files (HDF5 for arrays, CSV for tables) under
# fullsim_analysis/results/<campaign>/<study>/, and the Python scripts under
# fullsim_analysis/plots/ draw them. No plotting here.
#
#   include(joinpath(@__DIR__, "..", "lib", "FullsimAnalysis.jl"))
#   using .FullsimAnalysis
module FullsimAnalysis

using HDF5, Statistics, LinearAlgebra, Printf
using CryspLight
using Random: Xoshiro, seed!

include("io.jl")
include("fits.jl")
include("spectra.jl")
include("maps.jl")
include("results.jl")
include("emulate.jl")
include("calibrate.jl")
include("io_daq.jl")
include("chip_noise.jl")

export read_events, select_photopeak511, select_photopeak1275, select_empty, select_compton
export fit_gauss_linear, peak_fit, peak_fit_window, photoelectron_ladder
export equalised_sum, histogram, find_peak511, find_peak1275, peak_window, brightest, dimmest
export occupancy_map, mean_map, seed_peak_map, channel_grid
export results_dir, write_table, update_table, write_hist!, write_vector!, write_matrix!, open_results, stamp!
export open_pes_run, emulate, chain_maker, event_lists, with, with_window
export measured_charge, fitted_gains, matched_filter_template, matched_filter, filter_amplitudes
export daq_params, daq_param, daq_value, read_daq_run, daq_channel_values, daq_spectra
export read_daq_histograms, read_daq_staircase, read_daq_hold_delay
export daq_board_numbers, chip_rows, chip_matrix, chip_noise, CHANNEL_SIGMA, JUMP_SIGMA, BLIND_SLOPE

end
