# SISO OFDM simulation with nonlinearity
# Clipping noise cancellation eval
# %%
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import corrector
import distortion
import modulation
import noise
import transceiver
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, ebn0_to_snr

set_latex_plot_style()

# %%

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
# my_mod.plot_constellation()
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
# my_tx.impairment.plot_characteristics()

my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=None)
my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))

my_noise = noise.Awgn(snr_db=10, seed=1234)
bit_rng = np.random.default_rng(4321)

ebn0_arr = np.arange(0, 21, 1)
print("Eb/n0 values:", ebn0_arr)
snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_fft, my_mod.n_sub_carr, my_mod.constel_size)
print("SNR values:", snr_arr)

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
print("Distortion IBO/TOI value:", ibo_val_db)
cnc_n_iters_lst = [1, 2, 3, 4, 8, 16]
print("CNC number of iteration list:", cnc_n_iters_lst)
cnc_n_upsamp = 4
# Single CNC iteration is equal to standard reception without distortion compensation
cnc_n_iters_lst = np.insert(cnc_n_iters_lst, 0, 0)

include_clean_run = True
if include_clean_run:
    cnc_n_iters_lst = np.insert(cnc_n_iters_lst, 0, 0)

ber_per_dist, freq_arr, clean_ofdm_psd, distortion_psd, tx_ofdm_psd = ([] for i in range(5))

start_time = time.time()
for run_idx, cnc_n_iter_val in enumerate(cnc_n_iters_lst):

    if not (include_clean_run and run_idx == 0):
        my_standard_rx.modem.correct_constellation(ibo_val_db)
        my_tx.impairment.set_ibo(ibo_val_db)
        my_cnc_rx.impairment.set_ibo(ibo_val_db)

    bers = np.zeros([len(snr_arr)])
    for idx, snr in enumerate(snr_arr):
        my_noise.snr_db = snr
        n_err = 0
        bits_sent = 0
        ite_cnt = 0
        while bits_sent < bits_sent_max and n_err < n_err_min:
            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
            tx_ofdm_symbol, clean_ofdm_symbol = my_tx.transmit(tx_bits, out_domain_fd=False, return_both=True)

            if include_clean_run and run_idx == 0:
                rx_ofdm_symbol = my_noise.process(clean_ofdm_symbol, my_mod.avg_sample_power)
            else:
                rx_ofdm_symbol = my_noise.process(tx_ofdm_symbol, my_mod.avg_sample_power)

            if include_clean_run and run_idx == 0:
                # standard reception
                rx_bits = my_standard_rx.receive(rx_ofdm_symbol)
            else:
                # enchanced CNC reception
                # Change domain TD of RX signal to FD
                no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(rx_ofdm_symbol[my_cnc_rx.modem.cp_len:]),
                                                 norm="ortho").numpy()
                rx_bits = my_cnc_rx.receive(n_iters=cnc_n_iter_val, upsample_factor=cnc_n_upsamp,
                                            in_sig_fd=no_cp_fd_sig_mat)

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
ax1.set_title("Bit error rate, QAM %d, IBO = %d [dB]" % (my_mod.constellation_size, ibo_val_db))
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("BER")
ax1.grid()
ax1.legend()

plt.tight_layout()
plt.savefig("figs/ber_soft_lim_siso_cnc_ibo%d_niter%d_sweep_nupsamp%d.png" % (
    my_tx.impairment.ibo_db, np.max(cnc_n_iters_lst), cnc_n_upsamp), dpi=600, bbox_inches='tight')
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
# plt.savefig("figs/ber_soft_lim_siso_cnc_ibo%d_niter%d_nupsamp%d_sweep.png" % (
# my_tx.impairment.ibo_db, cnc_n_iter_val, np.max(cnc_n_upsamp_lst)), dpi=600, bbox_inches='tight')
# plt.show()
#
print("Finished execution!")
