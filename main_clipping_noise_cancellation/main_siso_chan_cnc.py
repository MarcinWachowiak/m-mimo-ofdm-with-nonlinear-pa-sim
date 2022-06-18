# SISO OFDM simulation with nonlinearity
# Clipping noise cancellation eval
# %%
import os, sys
sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import antenna_arrray
import channel
import corrector
import distortion
import modulation
import noise
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, ebn0_to_snr

set_latex_plot_style()

# %%
n_ant = 4
my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=2048, cp_len=128)
my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
# my_mod.plot_constellation()
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
my_array = antenna_arrray.LinearArray(n_elements=n_ant, base_transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                         cord_x=212, cord_y=212, cord_z=1.5,
                                         center_freq=int(3.5e9), carrier_spacing=int(15e3))
my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))

# my_miso_chan = channel.MisoTwoPathFd()
my_miso_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx, seed=1234)
my_noise = noise.Awgn(snr_db=10, seed=1234)
bit_rng = np.random.default_rng(4321)

ebn0_arr = np.arange(0, 31, 2)
print("Eb/n0 values:", ebn0_arr)
snr_arr = ebn0_arr
print("SNR values:", snr_arr)

if not isinstance(my_miso_chan, channel.RayleighMisoFd):
    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                  skip_attenuation=False)

chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)
agc_corr_vec = np.sqrt(np.sum(np.power(np.abs(chan_mat_at_point), 2), axis=0))

my_extended_cnc_rx = corrector.CncReceiverExtended(antenna_array=copy.deepcopy(my_array),
                                                   channel=copy.deepcopy(my_miso_chan))

# # DSP test
agc_corr_nsc = np.concatenate((agc_corr_vec[-my_mod.n_sub_carr // 2:], agc_corr_vec[1:(my_mod.n_sub_carr // 2) + 1]))
# chan_mat_nsc = np.hstack(
#     (chan_mat_at_point[:, -my_mod.n_sub_carr // 2:], chan_mat_at_point[:, 1:(my_mod.n_sub_carr // 2) + 1]))
# precoding_mat = my_array.array_elements[0].modem.precoding_mat
#
# tf_fd_precod_after_chan = chan_mat_nsc * precoding_mat
# tf_fd_after_agc = tf_fd_precod_after_chan / agc_corr_nsc

plot_psd = False
n_collected_snapshots = 100
psd_nfft = 128
n_samp_per_seg = 64

bits_sent_max = int(1e6)
n_err_min = 1000
convergence_epsilon = 0.001  # e.g. 0.1%
conv_ite_th = np.inf  # number of iterations after the convergence threshold is activated

# %%
# Number of CNC iterations eval, upsample ratio fixed
ibo_val_db = 0
my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
my_cnc_rx.impairment.set_ibo(ibo_val_db)
my_extended_cnc_rx.update_distortion(ibo_val_db=ibo_val_db)
print("Distortion IBO/TOI value:", ibo_val_db)
cnc_n_iters_lst = [1, 2, 3, 5, 12]
print("CNC number of iteration list:", cnc_n_iters_lst)
cnc_n_upsamp = 2
# Single CNC iteration is equal to standard reception without distortion compensation
cnc_n_iters_lst = np.insert(cnc_n_iters_lst, 0, 0)

abs_lambda = my_mod.calc_alpha(ibo_db=ibo_val_db)

include_clean_run = True
if include_clean_run:
    cnc_n_iters_lst = np.insert(cnc_n_iters_lst, 0, 0)

ber_per_dist, freq_arr, clean_ofdm_psd, distortion_psd, tx_ofdm_psd = ([] for i in range(5))

for run_idx, cnc_n_iter_val in enumerate(cnc_n_iters_lst):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if not (include_clean_run and run_idx == 0):
        my_standard_rx.modem.correct_constellation(ibo_val_db)
        my_cnc_rx.impairment.set_ibo(ibo_val_db)

    bers = np.zeros([len(snr_arr)])
    for idx, snr in enumerate(snr_arr):
        my_noise.snr_db = snr
        n_err = 0
        bits_sent = 0
        ite_cnt = 0
        while bits_sent < bits_sent_max and n_err < n_err_min:
            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
            tx_ofdm_symbol, clean_ofdm_symbol = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)
            if include_clean_run and run_idx == 0:
                rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol)
                rx_ofdm_symbol = my_noise.process(rx_ofdm_symbol, avg_sample_pow=my_mod.avg_symbol_power * np.mean(np.power(agc_corr_nsc,2)), disp_data=False)
            else:
                rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol)
                rx_ofdm_symbol = my_noise.process(rx_ofdm_symbol, avg_sample_pow=my_mod.avg_symbol_power * np.mean(np.power(agc_corr_nsc, 2)) * abs_lambda ** 2)
            # apply AGC
            rx_ofdm_symbol = rx_ofdm_symbol / agc_corr_vec

            if include_clean_run and run_idx == 0:
                # standard reception
                rx_ofdm_symbol = utilities.to_time_domain(rx_ofdm_symbol)
                rx_ofdm_symbol = np.concatenate((rx_ofdm_symbol[-my_mod.cp_len:], rx_ofdm_symbol))
                rx_bits = my_standard_rx.receive(rx_ofdm_symbol)
            else:
                # enchanced CNC reception
                # rx_bits = my_cnc_rx.receive(n_iters=cnc_n_iter_val, upsample_factor=cnc_n_upsamp,
                #                             in_sig_fd=rx_ofdm_symbol)
                rx_bits = my_extended_cnc_rx.receive(n_iters=cnc_n_iter_val, in_sig_fd=rx_ofdm_symbol)

            n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
            # check convergence
            # calc tmp ber
            if ite_cnt > conv_ite_th:
                prev_step_ber = n_err / bits_sent

            bits_sent += my_mod.n_bits_per_ofdm_sym
            n_err += n_bit_err
            curr_ber = n_err / bits_sent
            if ite_cnt > conv_ite_th and prev_step_ber != 0:
                rel_change = np.abs(curr_ber - prev_step_ber) / prev_step_ber
                if rel_change < convergence_epsilon:
                    break
            ite_cnt += 1

        bers[idx] = n_err / bits_sent
    ber_per_dist.append(bers)
    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log')
for idx, cnc_iter_val in enumerate(cnc_n_iters_lst):
    if include_clean_run:
        if idx == 0:
            ax1.plot(ebn0_arr, ber_per_dist[idx], label="No distortion")
        elif idx == 1:
            ax1.plot(ebn0_arr, ber_per_dist[idx], label="Standard RX")
        else:
            ax1.plot(ebn0_arr, ber_per_dist[idx], label="CNC NI = %d, J = %d" % (cnc_iter_val, cnc_n_upsamp))
    else:
        if idx == 0:
            ax1.plot(ebn0_arr, ber_per_dist[idx], label="Standard RX")
        else:
            ax1.plot(ebn0_arr, ber_per_dist[idx],
                     label="CNC NI = %d, J = %d" % (cnc_iter_val, cnc_n_upsamp))
# fix log scaling
ax1.set_title("Bit error rate, QAM %d, N_ANT = %d, IBO = %d [dB]" % (my_mod.constellation_size, n_ant, ibo_val_db))
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("BER")
ax1.grid()
ax1.legend()

plt.tight_layout()
plt.savefig("../figs/cnc_%s_nant%d_ber_ibo%d_niter%d_sweep_nupsamp%d.png" % (my_miso_chan, n_ant,
                                                                      ibo_val_db, np.max(cnc_n_iters_lst),
                                                                      cnc_n_upsamp), dpi=600, bbox_inches='tight')
plt.show()

#
# # %%
# # Upsample ratio eval, number of iterations fixed
# ibo_val_db = 5
# print("Distortion IBO/TOI value:", ibo_val_db)
# cnc_n_upsamp_lst = [2, 4, 8]
# cnc_n_upsamp_lst = np.insert(cnc_n_upsamp_lst, 0, 1)
#
# print("CNC upsample factors list:", cnc_n_upsamp_lst)
# cnc_n_iter_val = 4
#
# include_clean_run = True
# if include_clean_run:
#     cnc_n_upsamp_lst = np.insert(cnc_n_upsamp_lst, 0, 1)
#
# ber_per_dist, freq_arr, clean_ofdm_psd, distortion_psd, tx_ofdm_psd = ([] for i in range(5))
#
# start_time = time.time()
# for run_idx, cnc_upsample_val in enumerate(cnc_n_upsamp_lst):
#
#     if not (include_clean_run and run_idx == 0):
#         my_standard_rx.modem.correct_constellation(ibo_val_db)
#         my_tx.impairment.set_ibo(ibo_val_db)
#         my_cnc_rx.impairment.set_ibo(ibo_val_db)
#
#     bers = np.zeros([len(snr_arr)])
#
#     for idx, snr in enumerate(snr_arr):
#         my_noise.snr_db = snr
#         n_err = 0
#         bits_sent = 0
#         while bits_sent < bits_sent_max and n_err < n_err_min:
#             tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
#             tx_ofdm_symbol, clean_ofdm_symbol = my_tx.transmit(tx_bits, out_domain_fd=False, return_both=True)
#
#             if include_clean_run and run_idx == 0:
#                 rx_ofdm_symbol = my_noise.process(clean_ofdm_symbol, my_mod.avg_sample_power)
#             else:
#                 rx_ofdm_symbol = my_noise.process(tx_ofdm_symbol, my_mod.avg_sample_power)
#
#             if include_clean_run and run_idx == 0:
#                 # standard reception
#                 rx_bits = my_standard_rx.receive(rx_ofdm_symbol)
#             elif include_clean_run and run_idx == 1:
#                 # enchanced CNC reception
#                 # Change domain TD of RX signal to FD
#                 no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(rx_ofdm_symbol[my_cnc_rx.modem.cp_len:]),
#                                                  norm="ortho").numpy()
#                 rx_bits = my_cnc_rx.receive(n_iters=0, upsample_factor=cnc_upsample_val, in_sig_fd=no_cp_fd_sig_mat)
#
#             else:
#                 # enchanced CNC reception
#                 # Change domain TD of RX signal to FD
#                 no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(rx_ofdm_symbol[my_cnc_rx.modem.cp_len:]),
#                                                  norm="ortho").numpy()
#                 rx_bits = my_cnc_rx.receive(n_iters=cnc_n_iter_val, upsample_factor=cnc_upsample_val,
#                                             in_sig_fd=no_cp_fd_sig_mat)
#
#             n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
#
#             bits_sent += my_mod.n_bits_per_ofdm_sym
#             n_err += n_bit_err
#         bers[idx] = n_err / bits_sent
#     ber_per_dist.append(bers)
#
# print("--- Computation time: %f ---" % (time.time() - start_time))
#
# # %%
# fig1, ax1 = plt.subplots(1, 1)
# ax1.set_yscale('log')
# for idx, cnc_upsamp_val in enumerate(cnc_n_upsamp_lst):
#     if include_clean_run:
#         if idx == 0:
#             ax1.plot(ebn0_arr, ber_per_dist[idx], label="No distortion")
#         elif idx == 1:
#             ax1.plot(ebn0_arr, ber_per_dist[idx], label="Standard RX")
#         else:
#             ax1.plot(ebn0_arr, ber_per_dist[idx], label="CNC NI = %d, J = %d" % (cnc_n_iter_val, cnc_upsamp_val))
#     else:
#         if idx == 0:
#             ax1.plot(ebn0_arr, ber_per_dist[idx], label="Standard RX")
#         else:
#             ax1.plot(ebn0_arr, ber_per_dist[idx],
#                      label="CNC NI = %d, J = %d" % (cnc_n_iter_val, cnc_n_upsamp))
# # fix log scaling
# ax1.set_title("Bit error rate, QAM %d, IBO = %d [dB]" % (my_mod.constellation_size, ibo_val_db))
# ax1.set_xlabel("Eb/N0 [dB]")
# ax1.set_ylabel("BER")
# ax1.grid()
# ax1.legend()
#
# plt.tight_layout()
# plt.savefig("../figs/ber_soft_lim_siso_cnc_ibo%d_niter%d_nupsamp%d_sweep.png" % (
# my_tx.impairment.ibo_db, cnc_n_iter_val, np.max(cnc_n_upsamp_lst)), dpi=600, bbox_inches='tight')
# plt.show()
#
print("Finished execution!")
