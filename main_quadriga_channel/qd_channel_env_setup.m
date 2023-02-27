% setup the Quadriga channel parameters.

function [] = qd_channel_env_setup(n_ant, n_sub_carr, subcarrier_spacing, center_freq, distance, scenario)
% path to the Quadriga directory
addpath('C:\Program Files\QuaDriGa_2021.07.12_v2.6.1-0\quadriga_src');

warning ('off','all');

global n_sub_carr;
global bandwidth;

bandwidth = n_sub_carr * subcarrier_spacing;

global sim_params;
sim_params = qd_simulation_parameters;
sim_params.use_3GPP_baseline = 1;
sim_params.center_frequency = center_freq;
sim_params.show_progress_bars = false;

global layout;
layout =  qd_layout( sim_params );                              
layout.no_tx = 1;
antenna_array = qd_arrayant.generate('3gpp-3d', 1, n_ant, center_freq);
layout.tx_array = antenna_array;

layout.no_rx = 1;
layout.rx_position = [distance; 0 ; 1.5];
layout.set_scenario(scenario);

end