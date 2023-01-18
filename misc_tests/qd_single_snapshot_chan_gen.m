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

sim_params = qd_simulation_parameters;
sim_params.use_3GPP_baseline = 1;
sim_params.center_frequency = center_freq;                

layout = qd_layout( sim_params );                              
layout.no_tx = 1;
antenna_array = qd_arrayant.generate('3gpp-3d', 1, n_ant, center_freq);
layout.tx_array = antenna_array;

layout.no_rx = 1;
layout.set_scenario('3GPP_3D_UMa_LOS');
% layout.visualize()
%%

figure;
hold on;
for idx = 1:10
    layout.rx_position = [distance; 0; 1.5];
    channels = layout.get_channels;
    channels_fr = squeeze(channels.fr(bandwidth, n_sub_carr));
    plot(1:n_sub_carr ,10*log10(abs(channels_fr(1,:,1))));
end
xlim([0, 4096]);
xlabel("Subcarrier idx [-]");
ylabel("Channel attenuation [dB]");
grid on;

hold off;

%% 
figure;
hold on;
for idx = 1:10
    layout.rx_position = [distance; 0; 1.5];
    channels = layout.get_channels;
    channels_fr = squeeze(channels.fr(bandwidth, n_sub_carr));
    plot(1:n_sub_carr ,10*log10(abs(channels_fr(1,:,1))));
end
xlim([0, 4096]);
xlabel("Subcarrier idx [-]");
ylabel("Channel attenuation [dB]");
grid on;

hold off;


