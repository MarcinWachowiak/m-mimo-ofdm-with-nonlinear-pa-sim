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
import corrector
import distortion
import modulation
import noise
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, ebn0_to_snr, to_db

# TODO: consider logger

set_latex_plot_style()
# %%
print("Multi antenna processing init!")

my_mod = modulation.OfdmQamModem(constel_size=16, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=20, avg_symb_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=my_mod, impairment=my_distortion, center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx = transceiver.Transceiver(modem=my_mod, impairment=None, cord_x=100, cord_y=100, cord_z=1.5,
                                center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx.modem.correct_constellation(my_tx.impairment.ibo_db)

my_noise = noise.Awgn(snr_db=20, noise_p_dbm=-90, seed=1234)
my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))

# %%
# Upsample ratio eval, number of iterations fixed
print("Distortion IBO/TOI value:", my_distortion.ibo_db)
cnc_n_iter_val = 0
print("CNC N iterations:", cnc_n_iter_val)
cnc_n_upsamp_val = 4
print("CNC upsample factor:", cnc_n_upsamp_val)

# ebn0_arr = np.arange(0, 21, 2)
# print("Eb/n0 values:", ebn0_arr)
# snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_fft, my_mod.n_sub_carr, my_mod.constel_size)
# print("SNR values:", snr_arr)

noise_floor_vals = np.arange(-120, -119, 1)
print(noise_floor_vals)

# BER accuracy settings
bits_sent_max = int(1e6)
n_err_min = 1000

# sweep antenna count sweep
n_ant_vec = [1, 2, 4]

lambda_corr_estimate = []
bits_per_nant = []

total_tx_pow_dbm = 10
#%%
# channel estimation phase
my_array = antenna_arrray.LinearArray(n_elements=1, transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5,
                                      cord_x=0, cord_y=0, cord_z=15)
my_array.set_tx_power_lvls(tx_power_dbm=total_tx_pow_dbm, total=True)

my_miso_chan = channel.MisoLosFd()
my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, skip_attenuation=False)

chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point)

# estimate lambda correcting coefficient
# same seed is required
bit_rng = np.random.default_rng(4321)
for idx, noise_p_dbm in enumerate(noise_floor_vals):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    my_noise.noise_p_dbm = noise_p_dbm
    n_ofdm_symb = 1e3
    ofdm_symb_idx = 0
    lambda_numerator_vecs = []
    lambda_denominator_vecs = []
    while ofdm_symb_idx < n_ofdm_symb:
        tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
        tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

        rx_sig_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
        rx_ofdm_symbol = my_noise.process(rx_sig_fd, fixed_noise_power=True)

        clean_ofdm_symbol_fd = np.squeeze(clean_ofdm_symbol_fd)

        clean_nsc_ofdm_symb_fd = np.concatenate((clean_ofdm_symbol_fd[-my_mod.n_sub_carr // 2:], clean_ofdm_symbol_fd[1:(my_mod.n_sub_carr // 2) + 1]))
        rx_nsc_ofdm_symb_fd = np.concatenate((rx_ofdm_symbol[-my_mod.n_sub_carr // 2:], rx_ofdm_symbol[1:(my_mod.n_sub_carr // 2) + 1]))

        lambda_numerator_vecs.append(np.multiply(rx_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))
        lambda_denominator_vecs.append(np.multiply(clean_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))

        ofdm_symb_idx += 1

    # calculate lambda estimate
    lambda_num = np.average(np.vstack(lambda_numerator_vecs), axis=0)
    lambda_denum = np.average(np.vstack(lambda_denominator_vecs), axis=0)
    lambda_corr_estimate.append(lambda_num / lambda_denum)
    print("--- Computation time: %f ---" % (time.time() - start_time))

#%%
# def_constel_power_corr = np.sqrt(10 ** ((10 * np.log10(1000 * my_mod.avg_symbol_power) - total_tx_pow_dbm) / 10))

# transmission link evaluation phase
for n_ant_idx, n_ant in enumerate(n_ant_vec):

    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    my_array = antenna_arrray.LinearArray(n_elements=n_ant, transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5,
                                          cord_x=0, cord_y=0, cord_z=15)
    my_array.set_tx_power_lvls(tx_power_dbm=total_tx_pow_dbm, total=True)

    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, skip_attenuation=False)

    chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
    my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point)

    # same seed is required
    # calculate BER based on channel estimate
    bit_rng = np.random.default_rng(4321)
    bers = np.zeros([len(noise_floor_vals)])
    for idx, noise_p_dbm in enumerate(noise_floor_vals):
        my_noise.noise_p_dbm = noise_p_dbm
        n_err = 0
        bits_sent = 0
        while bits_sent < bits_sent_max and n_err < n_err_min:
            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
            tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

            # print("def dbm:", 10 * np.log10(1000*my_tx.modem.avg_symbol_power))
            # print("tx dbm:", my_array.array_elements[0].tx_power_dbm)
            # print("diff dbm", (10 * np.log10(1000*my_mod.avg_symbol_power) - my_array.array_elements[0].tx_power_dbm))
            #print(def_constel_power_corr)
            #print("tx pow", signal_power(my_array.array_elements[0].transmit(tx_bits, out_domain_fd=True, skip_dist=True)*def_constel_power_corr))
            #print("mod pow", signal_power(my_mod.modulate(tx_bits)))

            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
            rx_ofdm_symbol_fd = my_noise.process(rx_sig_fd, fixed_noise_power=True)

            # enchanced CNC reception
            # scale the RX signal to match the default constellation
            rx_bits = my_cnc_rx.receive(n_iters=cnc_n_iter_val, upsample_factor=cnc_n_upsamp_val,
                                         in_sig_fd=rx_ofdm_symbol_fd, channel_estimation_mat=np.abs(lambda_corr_estimate[idx])*n_ant)

            n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
            bits_sent += my_mod.n_bits_per_ofdm_sym
            n_err += n_bit_err
        bers[idx] = n_err / bits_sent
    bits_per_nant.append(bers)

    print("--- Computation time: %f ---" % (time.time() - start_time))
#
# # %%
# # plot beampatterns of desired signal
# fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
# plt.tight_layout()
# ax1.set_theta_zero_location("E")
#
# if plot_full_circle:
#     ax1.set_thetalim(-np.pi, np.pi)
#     ax1.set_xticks(np.pi / 180. * np.linspace(180, -180, 24, endpoint=False))
# else:
#     ax1.set_thetalim(0, np.pi)
#     ax1.set_xticks(np.pi / 180. * np.linspace(0, 180, 12, endpoint=False))
#
# dist_lines_lst = []
# for idx, n_ant in enumerate(n_ant_vec):
#     ax1.plot(radian_vals, desired_psd_at_angle_lst[idx], label=n_ant, linewidth=1.5)
# ax1.set_title("Desired signal PSD at angle [dB]")
# ax1.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center')
# ax1.grid(True)
# plt.savefig("figs/desired_signal_beampattern_ibo%d_%dto%dant_sweep.png" % (
#     my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
# # plot beampatterns of distortion signal
# fig2, ax2 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
# plt.tight_layout()
# ax2.set_theta_zero_location("E")
# if plot_full_circle:
#     ax2.set_thetalim(-np.pi, np.pi)
#     ax2.set_xticks(np.pi / 180. * np.linspace(180, -180, 24, endpoint=False))
# else:
#     ax2.set_thetalim(0, np.pi)
#     ax2.set_xticks(np.pi / 180. * np.linspace(0, 180, 12, endpoint=False))
#
# dist_lines_lst = []
# for idx, n_ant in enumerate(n_ant_vec):
#     ax2.plot(radian_vals, distortion_psd_at_angle_lst[idx], label=n_ant, linewidth=1.5)
# ax2.set_title("Distortion signal PSD at angle [dB]")
# ax2.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center')
# ax2.grid(True)
# plt.savefig("figs/distortion_signal_beampattern_ibo%d_%dto%dant_sweep.png" % (
#     my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
# # Desired vs distortion PSD beampattern comparison for given number of antennas
# fig3, ax3 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
# plt.tight_layout()
# ax3.set_theta_zero_location("E")
# if plot_full_circle:
#     ax3.set_thetalim(-np.pi, np.pi)
#     ax3.set_xticks(np.pi / 180. * np.linspace(180, -180, 24, endpoint=False))
# else:
#     ax3.set_thetalim(0, np.pi)
#     ax3.set_xticks(np.pi / 180. * np.linspace(0, 180, 12, endpoint=False))
#
# dist_lines_lst = []
# # select index to plot
# sel_idx = 3
# ax3.plot(radian_vals, desired_psd_at_angle_lst[sel_idx], label="Desired", linewidth=1.5)
# ax3.plot(radian_vals, distortion_psd_at_angle_lst[sel_idx], label="Distortion", linewidth=1.5)
# ax3.set_title("Power spectral density at angle [dB]")
# ax3.legend(title="N antennas = %d, signals:" % n_ant_vec[sel_idx], ncol=2, loc='lower center')
# ax3.grid(True)
# plt.savefig("figs/desired_vs_distortion_beampattern_ibo%d_%dant.png" % (my_tx.impairment.ibo_db, np.max(n_ant_vec)),
#             dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
# # plot signal to distortion ratio polar
# fig4, ax4 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
# plt.tight_layout()
# ax4.set_theta_zero_location("E")
# if plot_full_circle:
#     ax4.set_thetalim(-np.pi, np.pi)
#     ax4.set_xticks(np.pi / 180. * np.linspace(180, -180, 24, endpoint=False))
# else:
#     ax4.set_thetalim(0, np.pi)
#     ax4.set_xticks(np.pi / 180. * np.linspace(0, 180, 12, endpoint=False))
#
# for idx, n_ant in enumerate(n_ant_vec):
#     ax4.plot(radian_vals, desired_psd_at_angle_lst[idx] - distortion_psd_at_angle_lst[idx], label=n_ant, linewidth=1.5)
# ax4.set_title("Signal to distortion ratio at angle [dB]")
# ax4.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center')
# ax4.grid(True)
# plt.savefig("figs/sdr_at_angle_polar_ibo%d_%dto%dant_sweep.png" % (
#     my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
# # plot signal to distortion ratio cartesian
# fig5, ax5 = plt.subplots(1, 1)
# plt.tight_layout()
# for idx, n_ant in enumerate(n_ant_vec):
#     ax5.plot(np.degrees(radian_vals), desired_psd_at_angle_lst[idx] - distortion_psd_at_angle_lst[idx], label=n_ant,
#              linewidth=1.5)
#
# ax5.set_xlim([30, 60])
# ax5.set_xlabel("Angle [$\degree$]")
# ax5.set_ylabel("Signal to distortion ratio [dB]")
#
# ax5.set_title("Signal to distortion ratio at angle")
# ax5.legend(title="N antennas:")
# ax5.grid(True)
# plt.savefig("figs/sdr_at_angle_cartesian_ibo%d_%dto%dant_sweep.png" % (
#     my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
# # plot PSD at selected point/angle
# fig6, ax6 = plt.subplots(1, 1)
# sorted_clean_rx_at_point_freq_arr, sorted_clean_psd_at_point_arr = zip(
#     *sorted(zip(rx_clean_at_point_freq_arr, rx_clean_at_point_psd)))
# ax6.plot(np.array(sorted_clean_rx_at_point_freq_arr), to_db(np.array(sorted_clean_psd_at_point_arr)), label="Desired")
# sorted_dist_rx_at_point_freq_arr, sorted_dist_psd_at_point_arr = zip(
#     *sorted(zip(rx_dist_at_point_freq_arr, rx_dist_at_point_psd)))
# ax6.plot(np.array(sorted_dist_rx_at_point_freq_arr), to_db(np.array(sorted_dist_psd_at_point_arr)), label="Distorted")
#
# ax6.set_title("Power spectral density at angle %d$\degree$" % point_idx_psd)
# ax6.set_xlabel("Subcarrier index [-]")
# ax6.set_ylabel("Power [dB]")
# ax6.legend(title="IBO = %d [dB]" % my_tx.impairment.ibo_db)
# ax6.grid()
# plt.tight_layout()
# plt.savefig("figs/psd_at_angle_%ddeg_ibo%d_ant%d.png" % (point_idx_psd, my_tx.impairment.ibo_db, np.max(n_ant_vec)),
#             dpi=600,
#             bbox_inches='tight')
# plt.show()
#
# print("Finished processing!")
