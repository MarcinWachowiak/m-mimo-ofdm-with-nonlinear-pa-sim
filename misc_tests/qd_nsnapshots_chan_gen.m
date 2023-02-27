% Generate and plot a number of channel snapshots.

close all
clear all
addpath('C:\Program Files\QuaDriGa_2021.07.12_v2.6.1-0\quadriga_src')

rng(1)

subcarrier_spacing = 15e3;
n_sub_carr = 4096;
bandwidth = n_sub_carr * subcarrier_spacing;
n_ant = 64;
center_freq = 3.5e9;
distance = 300; 
n_snapshots = 100;

sim_params = qd_simulation_parameters;
sim_params.use_3GPP_baseline = 1;
sim_params.center_frequency = center_freq;                

layout = qd_layout( sim_params );                              
layout.no_tx = 1;
antenna_array = qd_arrayant.generate('3gpp-3d', 1, n_ant, center_freq);
layout.tx_array = antenna_array;

layout.no_rx = 1;
track_len = (n_snapshots -1 ) / sim_params.samples_per_meter;
layout.rx_track = qd_track('linear', track_len, pi/2);
layout.rx_position = [distance; 0; 1.5];
interpolate_positions(layout.rx_track, sim_params.samples_per_meter);
layout.set_scenario('3GPP_3D_UMa_LOS');
% layout.visualize()

channel_builder = layout.init_builder;
gen_parameters(channel_builder);
channels = channel_builder.get_channels;

channels_fr = squeeze(channels.fr(bandwidth, n_sub_carr));

%%
sim_params.samples_per_meter
hold on;
for idx = 1:10
    plot(1:n_sub_carr ,10*log10(abs(channels_fr(1,:,10*idx))));
end
xlim([0, 4096]);
xlabel("Subcarrier idx [-]");
ylabel("Channel attenuation [dB]");
grid on;

hold off;



