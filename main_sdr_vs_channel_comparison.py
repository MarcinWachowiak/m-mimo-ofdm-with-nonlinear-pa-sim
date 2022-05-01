# antenna array evaluation
# %%
import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch

import antenna_arrray
import channel
import distortion
import modulation
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import to_db, pts_on_circum, pts_on_semicircum
from matplotlib.ticker import MaxNLocator
# TODO: consider logger
set_latex_plot_style()
# %%
print("Multi antenna processing init!")
bit_rng = np.random.default_rng(4321)

ibo_val_db = 5

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=ibo_val_db, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))

my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=30, cord_y=30, cord_z=1.5, center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx.correct_constellation()
my_miso_chan = channel.MisoLosFd()

# %%
plot_full_circle = False

if plot_full_circle:
    n_points = 360
    rx_points = pts_on_circum(r=300, n=n_points)
else:
    n_points = 180
    rx_points = pts_on_semicircum(r=300, n=n_points)

radian_vals = np.radians(np.linspace(0, n_points, n_points + 1))

psd_nfft = 4096
n_samp_per_seg = 1024
n_snapshots = 10

# %%
# plot PSD for chosen point/angle
point_idx_psd = 78
n_ant_vec = [16, 32, 64, 128]

desired_sc_psd_at_angle_lst = []
distortion_sc_psd_at_angle_lst = []
rx_sig_at_point_clean = []
rx_sig_at_point_full = []
rx_sig_at_max_point_full = []
rx_sig_at_max_point_clean = []

for n_ant in n_ant_vec:
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    my_array = antenna_arrray.LinearArray(n_elements=n_ant, base_transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
    my_rx.set_position(cord_x=212, cord_y=212, cord_z=1.5)

    max_point_idx = int(np.degrees(np.arctan(my_rx.cord_y/my_rx.cord_x)))

    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, skip_attenuation=False)
    channel_mat_at_point_fd = my_miso_chan.get_channel_mat_fd()
    my_array.set_precoding_matrix(channel_mat_fd=channel_mat_at_point_fd, mr_precoding=True)
    my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

    psd_at_angle_desired = np.empty(radian_vals.shape)
    psd_at_angle_dist = np.empty(radian_vals.shape)
    for pt_idx, point in enumerate(rx_points):
        (x_cord, y_cord) = point
        my_rx.set_position(cord_x=x_cord, cord_y=y_cord, cord_z=1.5)
        # update channel matrix constant for a given point
        my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                      skip_attenuation=False)
        rx_ofdm_sc_accum = []
        clean_rx_ofdm_sc_accum = []

        for snap_idx in range(n_snapshots):

            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
            arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True, return_both=True)
            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd)
            rx_sig_td = utilities.to_time_domain(rx_sig_fd)
            rx_sc_ofdm_symb_fd = np.concatenate(
                (rx_sig_fd[-my_mod.n_sub_carr // 2:], rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))
            rx_sc_ofdm_symb_td = utilities.to_time_domain(rx_sc_ofdm_symb_fd)

            clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd)
            clean_rx_sig_td = utilities.to_time_domain(clean_rx_sig_fd)
            clean_sc_ofdm_symb_fd = np.concatenate(
                (clean_rx_sig_fd[-my_mod.n_sub_carr // 2:], clean_rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))
            clean_sc_ofdm_symb_td = utilities.to_time_domain(clean_sc_ofdm_symb_fd)

            rx_ofdm_sc_accum.append(rx_sc_ofdm_symb_td)
            clean_rx_ofdm_sc_accum.append(clean_sc_ofdm_symb_td)

        rx_sig_accum_arr = np.concatenate(rx_ofdm_sc_accum).ravel()
        clean_rx_sig_accum_arr = np.concatenate(clean_rx_ofdm_sc_accum).ravel()
        sc_ofdm_distortion_sig = rx_sig_accum_arr - my_rx.modem.alpha * clean_rx_sig_accum_arr

        dist_ofdm_symb_freq_arr, dist_ofdm_symb_psd_arr = welch(sc_ofdm_distortion_sig, fs=psd_nfft, nfft=psd_nfft,
                                                          nperseg=n_samp_per_seg, return_onesided=False)
        clean_ofdm_symb_freq_arr, clean_ofdm_symb_psd_arr = welch(clean_rx_sig_accum_arr, fs=psd_nfft, nfft=psd_nfft,
                                                            nperseg=n_samp_per_seg, return_onesided=False)

        psd_at_angle_desired[pt_idx] = to_db(np.sum(np.array(clean_ofdm_symb_psd_arr)))
        psd_at_angle_dist[pt_idx] = to_db(np.sum(np.array(dist_ofdm_symb_psd_arr)))

    desired_sc_psd_at_angle_lst.append(psd_at_angle_desired)
    distortion_sc_psd_at_angle_lst.append(psd_at_angle_dist)
    print("--- Computation time: %f ---" % (time.time() - start_time))

los_sdr_at_angle = np.subtract(np.array(desired_sc_psd_at_angle_lst), np.array(distortion_sc_psd_at_angle_lst))

desired_sc_psd_at_angle_lst = []
distortion_sc_psd_at_angle_lst = []

for n_ant in n_ant_vec:
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    my_array = antenna_arrray.LinearArray(n_elements=n_ant, base_transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
    my_miso_chan = channel.MisoTwoPathFd()

    my_rx.set_position(cord_x=212, cord_y=212, cord_z=1.5)
    max_point_idx = int(np.degrees(np.arctan(my_rx.cord_y/my_rx.cord_x)))

    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, skip_attenuation=False)
    channel_mat_at_point_fd = my_miso_chan.get_channel_mat_fd()
    my_array.set_precoding_matrix(channel_mat_fd=channel_mat_at_point_fd, mr_precoding=True)
    my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

    sc_psd_at_angle_desired = np.empty(radian_vals.shape)
    sc_psd_at_angle_dist = np.empty(radian_vals.shape)
    for pt_idx, point in enumerate(rx_points):
        # generate different channel for each point
        # precode only for single known point
        (x_cord, y_cord) = point
        my_rx.set_position(cord_x=x_cord, cord_y=y_cord, cord_z=1.5)
        # update channel matrix coefficients for new position
        my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                      skip_attenuation=False)

        rx_ofdm_sc_accum = []
        clean_rx_ofdm_sc_accum = []
        for snap_idx in range(n_snapshots):

            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
            arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True, return_both=True)

            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd)
            rx_sig_td = utilities.to_time_domain(rx_sig_fd)
            rx_sc_ofdm_symb_fd = np.concatenate(
                (rx_sig_fd[-my_mod.n_sub_carr // 2:], rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))
            rx_sc_ofdm_symb_td = utilities.to_time_domain(rx_sc_ofdm_symb_fd)

            clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd)
            clean_rx_sig_td = utilities.to_time_domain(clean_rx_sig_fd)
            clean_sc_ofdm_symb_fd = np.concatenate(
                (clean_rx_sig_fd[-my_mod.n_sub_carr // 2:], clean_rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))
            clean_sc_ofdm_symb_td = utilities.to_time_domain(clean_sc_ofdm_symb_fd)

            # calculate PSD at point for the last value of N antennas
            if pt_idx == point_idx_psd and n_ant == n_ant_vec[-1]:
                rx_sig_at_point_clean.append(clean_rx_sig_td)
                rx_sig_at_point_full.append(rx_sig_td)

            if pt_idx == max_point_idx and n_ant == n_ant_vec[-1]:
                rx_sig_at_max_point_clean.append(clean_rx_sig_td)
                rx_sig_at_max_point_full.append(rx_sig_td)

            rx_ofdm_sc_accum.append(rx_sc_ofdm_symb_td)
            clean_rx_ofdm_sc_accum.append(clean_sc_ofdm_symb_td)

        rx_ofdm_symb_accum_arr = np.concatenate(rx_ofdm_sc_accum).ravel()
        clean_ofdm_symb_accum_arr = np.concatenate(clean_rx_ofdm_sc_accum).ravel()
        sc_ofdm_distortion_sig = rx_ofdm_symb_accum_arr - my_rx.modem.alpha * clean_ofdm_symb_accum_arr

        dist_ofdm_symb_freq_arr, dist_ofdm_symb_psd_arr = welch(sc_ofdm_distortion_sig, fs=psd_nfft, nfft=psd_nfft,
                                                          nperseg=n_samp_per_seg, return_onesided=False)
        clean_ofdm_symb_freq_arr, clean_ofdm_symb_psd_arr = welch(clean_ofdm_symb_accum_arr, fs=psd_nfft, nfft=psd_nfft,
                                                            nperseg=n_samp_per_seg, return_onesided=False)

        sc_psd_at_angle_desired[pt_idx] = to_db(np.sum(np.array(clean_ofdm_symb_psd_arr)))
        sc_psd_at_angle_dist[pt_idx] = to_db(np.sum(np.array(dist_ofdm_symb_psd_arr)))

    desired_sc_psd_at_angle_lst.append(sc_psd_at_angle_desired)
    distortion_sc_psd_at_angle_lst.append(sc_psd_at_angle_dist)
    print("--- Computation time: %f ---" % (time.time() - start_time))

two_path_sdr_at_angle = np.subtract(np.array(desired_sc_psd_at_angle_lst), np.array(distortion_sc_psd_at_angle_lst))

precoding_point_idx = 45
desired_sc_psd_at_angle_lst = []
distortion_sc_psd_at_angle_lst = []

for n_ant in n_ant_vec:
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    my_array = antenna_arrray.LinearArray(n_elements=n_ant, base_transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5,
                                          cord_x=0, cord_y=0, cord_z=15)
    my_rx.set_position(cord_x=212, cord_y=212, cord_z=1.5)

    my_miso_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, seed=1234)

    chan_mat_at_point_fd = my_miso_chan.get_channel_mat_fd()
    # if n_ant != 1:
    my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point_fd, mr_precoding=True)
    my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

    sc_psd_at_angle_desired = np.empty(radian_vals.shape)
    sc_psd_at_angle_dist = np.empty(radian_vals.shape)
    for pt_idx, point in enumerate(rx_points):
        # generate different channel for each point
        if pt_idx == precoding_point_idx:
            my_miso_chan.set_channel_mat_fd(chan_mat_at_point_fd)
        else:
            my_miso_chan.reroll_channel_coeffs()

        rx_ofdm_sc_accum = []
        clean_rx_ofdm_sc_accum = []
        for snap_idx in range(n_snapshots):

            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
            arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True, return_both=True)

            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd)
            rx_sig_td = utilities.to_time_domain(rx_sig_fd)
            rx_sc_ofdm_symb_fd = np.concatenate(
                (rx_sig_fd[-my_mod.n_sub_carr // 2:], rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))
            rx_sc_ofdm_symb_td = utilities.to_time_domain(rx_sc_ofdm_symb_fd)

            clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd)
            clean_rx_sig_td = utilities.to_time_domain(clean_rx_sig_fd)
            clean_sc_ofdm_symb_fd = np.concatenate(
                (clean_rx_sig_fd[-my_mod.n_sub_carr // 2:], clean_rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))
            clean_sc_ofdm_symb_td = utilities.to_time_domain(clean_sc_ofdm_symb_fd)

            # calculate PSD at point for the last value of N antennas
            if pt_idx == point_idx_psd and n_ant == n_ant_vec[0]:
                rx_sig_at_point_clean.append(clean_rx_sig_td)
                rx_sig_at_point_full.append(rx_sig_td)

            if pt_idx == precoding_point_idx and n_ant == n_ant_vec[0]:
                rx_sig_at_max_point_clean.append(clean_rx_sig_td)
                rx_sig_at_max_point_full.append(rx_sig_td)

            rx_ofdm_sc_accum.append(rx_sc_ofdm_symb_td)
            clean_rx_ofdm_sc_accum.append(clean_sc_ofdm_symb_td)

        rx_ofdm_symb_accum_arr = np.concatenate(rx_ofdm_sc_accum).ravel()
        clean_ofdm_symb_accum_arr = np.concatenate(clean_rx_ofdm_sc_accum).ravel()
        sc_ofdm_distortion_sig = rx_ofdm_symb_accum_arr - my_rx.modem.alpha * clean_ofdm_symb_accum_arr

        dist_ofdm_symb_freq_arr, dist_ofdm_symb_psd_arr = welch(sc_ofdm_distortion_sig, fs=psd_nfft, nfft=psd_nfft,
                                                                nperseg=n_samp_per_seg, return_onesided=False)
        clean_ofdm_symb_freq_arr, clean_ofdm_symb_psd_arr = welch(clean_ofdm_symb_accum_arr, fs=psd_nfft, nfft=psd_nfft,
                                                                  nperseg=n_samp_per_seg, return_onesided=False)

        sc_psd_at_angle_desired[pt_idx] = to_db(np.sum(np.array(clean_ofdm_symb_psd_arr)))
        sc_psd_at_angle_dist[pt_idx] = to_db(np.sum(np.array(dist_ofdm_symb_psd_arr)))

    desired_sc_psd_at_angle_lst.append(sc_psd_at_angle_desired)
    distortion_sc_psd_at_angle_lst.append(sc_psd_at_angle_dist)
    print("--- Computation time: %f ---" % (time.time() - start_time))

rayleigh_sdr_at_angle = np.subtract(np.array(desired_sc_psd_at_angle_lst), np.array(distortion_sc_psd_at_angle_lst))


# %%
# plot signal to distortion ratio
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 5), sharex=True, gridspec_kw={'height_ratios': [1, 1, 2]})
deg_vals = np.rad2deg(radian_vals)
# LOS
for idx, n_ant in enumerate(n_ant_vec):
    ax1.plot(deg_vals, los_sdr_at_angle[idx], label=n_ant, linewidth=1.5)
ax1.grid(True)
ax1.set_title("a) LOS channel", fontsize=8)
ax1.set_ylabel("SDR [dB]", fontsize=8)
ax1.set_ylim([24, 28])
# Two-path
for idx, n_ant in enumerate(n_ant_vec):
    ax2.plot(deg_vals, two_path_sdr_at_angle[idx], label=n_ant, linewidth=1.5)
ax2.grid(True)
ax2.set_title("b) Two-path channel", fontsize=8)
ax2.set_ylabel("SDR [dB]", fontsize=8)
ax2.set_ylim([18, 32])
ax2.set_yticks([18, 26, 34])

# Rayleigh
for idx, n_ant in enumerate(n_ant_vec):
    ax3.plot(deg_vals, rayleigh_sdr_at_angle[idx], label=n_ant, linewidth=1.5)
ax3.grid(True)
ax3.set_title("c) Rayleigh channel", fontsize=8)
ax3.legend(title="Number of antennas:", ncol=len(n_ant_vec), loc=(0.05, -0.6), borderaxespad=0)
ax3.set_xlabel("Angle [°]", fontsize=8)
ax3.set_ylabel("SDR [dB]", fontsize=8)
ax3.set_xlim([0, 180])
ax3.set_ylim([24,50])
ax3.set_yticks([26, 32, 38, 44, 50])
ax3.set_xticks(np.linspace(0, 180, 7, endpoint=True))
fig.suptitle("Signal to distortion ratio")

# zoom on Rayleigh SDR peak
# inset axes....
axins = ax3.inset_axes([0.4, 0.3, 0.55, 0.6])
# sub region of the original image
for idx, n_ant in enumerate(n_ant_vec):
    axins.plot(deg_vals, rayleigh_sdr_at_angle[idx], label=n_ant, linewidth=1.5)

axins.tick_params(axis='x', labelsize=8)
axins.set_xlim(44, 46)
axins.set_xticks([44, 45, 46])

axins.tick_params(axis='y', labelsize=8)
axins.set_ylim(36, 48)
axins.set_yticks([36, 42, 48])
axins.grid()
# axins.set_xticklabels([])
# axins.set_yticklabels([])

ax3.indicate_inset_zoom(axins, edgecolor="black")


plt.tight_layout()
plt.savefig("figs/sdr_at_angle_ibo%d_%dto%dant_sweep.pdf" % (
my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
plt.show()

print("Finished processing!")
