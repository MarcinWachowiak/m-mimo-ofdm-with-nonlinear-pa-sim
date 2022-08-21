#!/usr/bin/env python
# antenna array evaluation
# %%
import os
import sys

sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime

import numpy as np
from scipy.signal import welch

import antenna_arrray
import channel
import distortion
import modulation
import transceiver
import utilities
from utilities import pts_on_semicircum

n_ant_val = int(sys.argv[1])
channel_type_str = str(sys.argv[2])

# %%
ibo_val_db = 3
n_snapshots = 10
n_points = 180 * 10
radial_distance = 300
precoding_angle = 45
sel_psd_angle = 78

sel_ptx_idx = int(n_points / 180 * sel_psd_angle)

# PSD plotting params
psd_nfft = 4096
n_samp_per_seg = 1024

rx_points = pts_on_semicircum(r=radial_distance, n=n_points)
radian_vals = np.radians(np.linspace(0, 180, n_points + 1))
# %%
my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=2048, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=ibo_val_db, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))

my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                cord_x=212,
                                cord_y=212, cord_z=1.5, center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))

my_miso_chan = channel.MisoLosFd()

# for channel_type_str in channel_type_lst:
#
#     desired_sig_power_per_nant = []
#     distortion_sig_power_per_nant = []
#     for n_ant_val in n_ant_vec:
start_time = time.time()
print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

rx_sig_at_sel_point_des = []
rx_sig_at_sel_point_dist = []
rx_sig_at_max_point_des = []
rx_sig_at_max_point_dist = []

bit_rng = np.random.default_rng(4321)
my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)

if channel_type_str == "los":
    my_miso_chan = channel.MisoLosFd()
elif channel_type_str == "two_path":
    my_miso_chan = channel.MisoTwoPathFd()
elif channel_type_str == "rayleigh":
    my_miso_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                          seed=1234)
else:
    raise ValueError('Unknown channel type!')

# precode for a single point
precoding_point_idx = int(n_points / 180 * precoding_angle)
(pt_x, pt_y) = rx_points[precoding_point_idx]
my_rx.set_position(cord_x=pt_x, cord_y=pt_y, cord_z=1.5)

if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan, channel.MisoTwoPathFd):
    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                  skip_attenuation=False)
else:
    my_miso_chan.reroll_channel_coeffs()

channel_mat_at_point_fd = my_miso_chan.get_channel_mat_fd()
my_array.set_precoding_matrix(channel_mat_fd=channel_mat_at_point_fd, mr_precoding=True)
my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

vk_mat = my_array.get_precoding_mat()
vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)
ibo_vec = 10 * np.log10(10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
ak_vect = np.expand_dims(ak_vect, axis=1)

desired_sig_pow_per_pt = []
distorted_sig_pow_per_pt = []
for pt_idx, point in enumerate(rx_points):
    (x_cord, y_cord) = point
    my_rx.set_position(cord_x=x_cord, cord_y=y_cord, cord_z=1.5)
    # update channel matrix constant for a given point
    if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan, channel.MisoTwoPathFd):
        my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                      skip_attenuation=False)
    else:
        if pt_idx == precoding_point_idx:
            my_miso_chan.set_channel_mat_fd(channel_mat_at_point_fd)
        else:
            my_miso_chan.reroll_channel_coeffs()

    desired_sig_pow_arr = np.zeros(n_snapshots)
    distortion_sig_pow_arr = np.zeros(n_snapshots)
    for snap_idx in range(n_snapshots):
        tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
        arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True,
                                                            return_both=True)

        rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd, sum=False)
        rx_sc_ofdm_symb_fd = np.concatenate(
            (rx_sig_fd[:, -my_mod.n_sub_carr // 2:], rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]), axis=1)
        # rx_sc_ofdm_symb_td = utilities.to_time_domain(rx_sc_ofdm_symb_fd)

        clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd, sum=False)

        clean_sc_ofdm_symb_fd = np.concatenate(
            (clean_rx_sig_fd[:, -my_mod.n_sub_carr // 2:], clean_rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
            axis=1)

        sc_ofdm_distortion_sig = np.subtract(rx_sc_ofdm_symb_fd, (ak_vect * clean_sc_ofdm_symb_fd))

        # calculate PSD at point for the last value of N antennas
        if pt_idx == sel_ptx_idx:
            # for PSD plotting take into consideration full BW not only SC
            desired_sig = np.sum(ak_vect * clean_rx_sig_fd, axis=0)
            distortion_sig = np.sum(np.subtract(rx_sig_fd, (ak_vect * clean_rx_sig_fd)), axis=0)
            rx_sig_at_sel_point_des.append(utilities.to_time_domain(desired_sig))
            rx_sig_at_sel_point_dist.append(utilities.to_time_domain(distortion_sig))

        if pt_idx == precoding_point_idx:
            # for PSD plotting take into consideration full BW not only SC
            desired_sig = np.sum(ak_vect * clean_rx_sig_fd, axis=0)
            distortion_sig = np.sum(np.subtract(rx_sig_fd, (ak_vect * clean_rx_sig_fd)), axis=0)
            rx_sig_at_max_point_des.append(utilities.to_time_domain(desired_sig))
            rx_sig_at_max_point_dist.append(utilities.to_time_domain(distortion_sig))

        desired_sig_pow_arr[snap_idx] = np.sum(
            np.power(np.abs(np.sum(ak_vect * clean_sc_ofdm_symb_fd, axis=0)), 2))
        distortion_sig_pow_arr[snap_idx] = np.sum(np.power(np.abs(np.sum(sc_ofdm_distortion_sig, axis=0)), 2))
        # calculate SDR on symbol basis
    # SUM OF SIGNAL POWERS TO AVOID SMALL VALUE ERRORS
    desired_sig_pow_per_pt.append(np.sum(desired_sig_pow_arr))
    distorted_sig_pow_per_pt.append(np.sum(distortion_sig_pow_arr))

# desired_sig_power_per_nant.append(desired_sig_pow_per_pt)
# distortion_sig_power_per_nant.append(distorted_sig_pow_per_pt)

rx_sig_at_max_point_des_arr = np.concatenate(rx_sig_at_max_point_des).ravel()
rx_sig_at_max_point_dist_arr = np.concatenate(rx_sig_at_max_point_dist).ravel()

rx_des_at_max_point_freq_arr, rx_des_at_max_point_psd = welch(rx_sig_at_max_point_des_arr, fs=psd_nfft,
                                                              nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                              return_onesided=False)
rx_dist_at_max_point_freq_arr, rx_dist_at_max_point_psd = welch(rx_sig_at_max_point_dist_arr, fs=psd_nfft,
                                                                nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                                return_onesided=False)

rx_sig_at_sel_point_des_arr = np.concatenate(rx_sig_at_sel_point_des).ravel()
rx_sig_at_sel_point_dist_arr = np.concatenate(rx_sig_at_sel_point_dist).ravel()

rx_des_at_sel_point_freq_arr, rx_des_at_sel_point_psd = welch(rx_sig_at_sel_point_des_arr, fs=psd_nfft,
                                                              nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                              return_onesided=False)
rx_dist_at_sel_point_freq_arr, rx_dist_at_sel_point_psd = welch(rx_sig_at_sel_point_dist_arr, fs=psd_nfft,
                                                                nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                                return_onesided=False)

psd_max_filename_str = "psd_mrt_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
    my_miso_chan, ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val)

data_lst_max = []
tmp_lst_max = [rx_des_at_max_point_freq_arr, rx_des_at_max_point_psd, rx_dist_at_max_point_freq_arr,
               rx_dist_at_max_point_psd]
for arr1 in tmp_lst_max:
    data_lst_max.append(arr1)
utilities.save_to_csv(data_lst=data_lst_max, filename=psd_max_filename_str)

# fig4, ax4 = plt.subplots(1, 1)
# sorted_des_rx_at_max_freq_arr, sorted_des_psd_at_max_arr = zip(
#     *sorted(zip(rx_des_at_max_point_freq_arr, rx_des_at_max_point_psd)))
# ax4.plot(np.array(sorted_des_rx_at_max_freq_arr), to_db(np.array(sorted_des_psd_at_max_arr)),
#          label="Desired")
# sorted_dist_rx_at_max_freq_arr, sorted_dist_psd_at_max_arr = zip(
#     *sorted(zip(rx_dist_at_max_point_freq_arr, rx_dist_at_max_point_psd)))
# ax4.plot(np.array(sorted_dist_rx_at_max_freq_arr), to_db(np.array(sorted_dist_psd_at_max_arr)),
#          label="Distorted")
#
# ax4.set_title("Power spectral density at angle %d$\degree$" % precoding_angle)
# ax4.set_xlabel("Subcarrier index [-]")
# ax4.set_ylabel("Power [dB]")
# ax4.legend(title="IBO = %d [dB]" % my_tx.impairment.ibo_db)
# ax4.grid()
# plt.tight_layout()
# plt.savefig("figs/beampatterns/%s.png" % psd_max_filename_str, dpi=600, bbox_inches='tight')
# # plt.show()
# plt.cla()
# plt.close()

psd_sel_filename_str = "psd_mrt_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
    my_miso_chan, ibo_val_db, n_points, n_snapshots, sel_psd_angle, n_ant_val)

data_lst_sel = []
tmp_lst_sel = [rx_des_at_sel_point_freq_arr, rx_des_at_sel_point_psd, rx_dist_at_sel_point_freq_arr,
               rx_dist_at_sel_point_psd]
for arr1 in tmp_lst_sel:
    data_lst_sel.append(arr1)
utilities.save_to_csv(data_lst=data_lst_sel, filename=psd_sel_filename_str)

# fig5, ax5 = plt.subplots(1, 1)
# sorted_des_rx_at_sel_freq_arr, sorted_des_psd_at_sel_arr = zip(
#     *sorted(zip(rx_des_at_sel_point_freq_arr, rx_des_at_sel_point_psd)))
# ax5.plot(np.array(sorted_des_rx_at_sel_freq_arr), to_db(np.array(sorted_des_psd_at_sel_arr)),
#          label="Desired")
# sorted_dist_rx_at_sel_freq_arr, sorted_dist_psd_at_sel_arr = zip(
#     *sorted(zip(rx_dist_at_sel_point_freq_arr, rx_dist_at_sel_point_psd)))
# ax5.plot(np.array(sorted_dist_rx_at_sel_freq_arr), to_db(np.array(sorted_dist_psd_at_sel_arr)),
#          label="Distorted")
#
# ax5.set_title("Power spectral density at angle %d$\degree$" % sel_psd_angle)
# ax5.set_xlabel("Subcarrier index [-]")
# ax5.set_ylabel("Power [dB]")
# ax5.legend(title="IBO = %d [dB]" % my_tx.impairment.ibo_db)
# ax5.grid()
# plt.tight_layout()
# plt.savefig("figs/beampatterns/%s.png" % psd_sel_filename_str, dpi=600, bbox_inches='tight')
# # plt.show()
# plt.cla()
# plt.close()

# %%
filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
    my_miso_chan, ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val)

data_lst_per_nant = []
tmp_total_lst = [desired_sig_pow_per_pt, distorted_sig_pow_per_pt]
for arr1 in tmp_total_lst:
    data_lst_per_nant.append(arr1)
utilities.save_to_csv(data_lst=data_lst_per_nant, filename=filename_str)

print("--- Computation time: %f ---" % (time.time() - start_time))

# # %%
# # plot beampatterns of desired signal
# fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(3.5, 3))
# ax1.set_theta_zero_location("E")
# plt.tight_layout()
# ax1.set_thetalim(0, np.pi)
# ax1.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
# ax1.yaxis.set_major_locator(MaxNLocator(5))
#
# dist_lines_lst = []
# for idx, n_ant in enumerate(n_ant_vec):
#     ax1.plot(radian_vals, to_db(desired_sig_power_per_nant[idx]), label=n_ant, linewidth=1.5)
# ax1.set_title("Desired signal PSD at angle [dB]", pad=-15)
# ax1.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center', borderaxespad=0)
# ax1.grid(True)
# plt.savefig("figs/beampatterns/%s_desired_signal_beampattern_ibo%d_angle%d_npoints%d_nsnap%d_nant%s.png" % (
#     my_miso_chan, ibo_val_db, precoding_angle, n_points, n_snapshots, '_'.join([str(val) for val in n_ant_vec])),
#             dpi=600, bbox_inches='tight')
# # plt.show()
# plt.cla()
# plt.close()
#
# # %%
# # plot beampatterns of distortion signal
# fig2, ax2 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(3.5, 3))
# ax2.set_theta_zero_location("E")
# ax2.set_thetalim(0, np.pi)
# ax2.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
# ax2.yaxis.set_major_locator(MaxNLocator(5))
#
# dist_lines_lst = []
# for idx, n_ant in enumerate(n_ant_vec):
#     ax2.plot(radian_vals, to_db(distortion_sig_power_per_nant[idx]), label=n_ant, linewidth=1.5)
# ax2.set_title("Distortion signal PSD at angle [dB]", pad=-15)
# ax2.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center', borderaxespad=0)
# ax2.grid(True)
# plt.savefig("figs/beampatterns/%s_distortion_signal_beampattern_ibo%d_angle%d_npoints%d_nsnap%d_nant%s.png" % (
#     my_miso_chan, ibo_val_db, precoding_angle, n_points, n_snapshots, '_'.join([str(val) for val in n_ant_vec])),
#             dpi=600, bbox_inches='tight')
#
# # plt.show()
# plt.cla()
# plt.close()
#
# # %%
# # plot signal to distortion ratio
# fig3, ax3 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(3.5, 3))
# ax3.set_theta_zero_location("E")
# ax3.set_thetalim(0, np.pi)
# ax3.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
#
# for idx, n_ant in enumerate(n_ant_vec):
#     ax3.plot(radian_vals,
#              to_db(np.array(desired_sig_power_per_nant[idx]) / np.array(distortion_sig_power_per_nant[idx])),
#              label=n_ant,
#              linewidth=1.5)
# ax3.set_title("Signal to distortion ratio at angle [dB]", pad=-15)
# ax3.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center', borderaxespad=0)
# ax3.grid(True)
# plt.tight_layout()
# plt.savefig("figs/beampatterns/%s_sdr_beampattern_ibo%d_angle%d_npoints%d_nsnap%d_nant%s.png" % (
#     my_miso_chan, ibo_val_db, precoding_angle, n_points, n_snapshots, '_'.join([str(val) for val in n_ant_vec])),
#             dpi=600, bbox_inches='tight')
#
# # plt.show()
# plt.cla()
# plt.close()
