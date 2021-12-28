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
from utilities import count_mismatched_bits, ebn0_to_snr, to_db, td_signal_power

# TODO: consider logger

set_latex_plot_style()
# %%
print("Multi antenna processing init!")
ibo_db_val = 5

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=ibo_db_val, avg_symb_pow=my_mod.avg_sample_power)
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
cnc_n_iter_val = 3
print("CNC N iterations:", cnc_n_iter_val)
cnc_n_upsamp_val = 4
print("CNC upsample factor:", cnc_n_upsamp_val)

# ebn0_arr = np.arange(0, 21, 2)
# print("Eb/n0 values:", ebn0_arr)
# snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_fft, my_mod.n_sub_carr, my_mod.constel_size)
# print("SNR values:", snr_arr)

noise_floor_vals = np.arange(-60, -91, -2)
print(noise_floor_vals)

# BER accuracy settings
bits_sent_max = int(1e6)
n_err_min = 1000

# sweep antenna count sweep
n_ant_vec = [1, 2, 4, 8, 16]

lambda_corr_estimate = []
bers_per_nant = []

total_tx_pow_dbm = 10
# %%
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

        clean_nsc_ofdm_symb_fd = np.concatenate(
            (clean_ofdm_symbol_fd[-my_mod.n_sub_carr // 2:], clean_ofdm_symbol_fd[1:(my_mod.n_sub_carr // 2) + 1]))
        rx_nsc_ofdm_symb_fd = np.concatenate(
            (rx_ofdm_symbol[-my_mod.n_sub_carr // 2:], rx_ofdm_symbol[1:(my_mod.n_sub_carr // 2) + 1]))

        lambda_numerator_vecs.append(np.multiply(rx_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))
        lambda_denominator_vecs.append(np.multiply(clean_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))

        ofdm_symb_idx += 1

    # calculate lambda estimate
    lambda_num = np.average(np.vstack(lambda_numerator_vecs), axis=0)
    lambda_denum = np.average(np.vstack(lambda_denominator_vecs), axis=0)
    lambda_corr_estimate.append(lambda_num / lambda_denum)
    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
# N ANTENNAS EVAL
for n_ant_idx, n_ant in enumerate(n_ant_vec):
    print_pow = True
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

            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
            rx_ofdm_symbol_fd = my_noise.process(rx_sig_fd, fixed_noise_power=True)

            # if print_pow and noise_p_dbm == noise_floor_vals[-1]:
            #     print("RX power:", to_db(td_signal_power(utilities.to_time_domain(rx_ofdm_symbol_fd)/np.sqrt(n_ant))))
            #     print_pow = False

            # correct for different TX power (in regard to default constellation values)
            rx_constel_pow_corr = np.sqrt(my_mod.avg_sample_power / (1e-3 * 10 ** (total_tx_pow_dbm / 10)))
            # correct for mimo gain too
            rx_ofdm_symbol_fd = (rx_ofdm_symbol_fd / np.sqrt(n_ant)) * rx_constel_pow_corr

            # enchanced CNC reception
            # scale the RX signal to match the default constellation
            rx_bits = my_cnc_rx.receive(n_iters=cnc_n_iter_val, upsample_factor=cnc_n_upsamp_val,
                                        in_sig_fd=rx_ofdm_symbol_fd,
                                        channel_estimation_mat=np.abs(lambda_corr_estimate[idx]))

            n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
            bits_sent += my_mod.n_bits_per_ofdm_sym
            n_err += n_bit_err

        bers[idx] = n_err / bits_sent
    bers_per_nant.append(bers)

    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log')
for idx, n_ant in enumerate(n_ant_vec):
    ax1.plot(noise_floor_vals, bers_per_nant[idx], label=n_ant)

plt.gca().invert_xaxis()
ax1.set_title("Bit error rate, QAM %d, PTX = %d [dBm], IBO = %d [dB]" % (my_mod.constellation_size, total_tx_pow_dbm, ibo_db_val))
ax1.set_xlabel("Noise floor [dBm]")
ax1.set_ylabel("BER")
ax1.grid()
ax1.legend(title="N antennas:")

plt.tight_layout()
plt.savefig(
    "figs/ber_soft_lim_miso_ptxtot%ddbm_cnc_ibo%d_niter%d_nupsamp%d_nant%d_sweep.png" % (
    total_tx_pow_dbm, ibo_db_val, cnc_n_iter_val, cnc_n_upsamp_val, n_ant_vec[-1]),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
# # %%

# %%
# CNC N ITERS EVAL
num_ant = 16
cnc_n_iter_lst = [0, 1, 2, 3, 4]
bers_per_niter = []
for n_iter_idx, n_iter in enumerate(cnc_n_iter_lst):
    print_pow = True
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    my_array = antenna_arrray.LinearArray(n_elements=num_ant, transceiver=my_tx, center_freq=int(3.5e9),
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

            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
            rx_ofdm_symbol_fd = my_noise.process(rx_sig_fd, fixed_noise_power=True)

            # if print_pow and noise_p_dbm == noise_floor_vals[-1]:
            #     print("RX power:", to_db(td_signal_power(utilities.to_time_domain(rx_ofdm_symbol_fd)/np.sqrt(n_ant))))
            #     print_pow = False

            # correct for different TX power (in regard to default constellation values)
            rx_constel_pow_corr = np.sqrt(my_mod.avg_sample_power / (1e-3 * 10 ** (total_tx_pow_dbm / 10)))
            # correct for mimo gain too
            rx_ofdm_symbol_fd = (rx_ofdm_symbol_fd / np.sqrt(num_ant)) * rx_constel_pow_corr

            # enchanced CNC reception
            # scale the RX signal to match the default constellation
            rx_bits = my_cnc_rx.receive(n_iters=n_iter, upsample_factor=cnc_n_upsamp_val,
                                        in_sig_fd=rx_ofdm_symbol_fd,
                                        channel_estimation_mat=np.abs(lambda_corr_estimate[idx]))

            n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
            bits_sent += my_mod.n_bits_per_ofdm_sym
            n_err += n_bit_err

        bers[idx] = n_err / bits_sent
    bers_per_niter.append(bers)

    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log')
for idx, n_iter in enumerate(cnc_n_iter_lst):
    ax1.plot(noise_floor_vals, bers_per_niter[idx], label=n_iter)

plt.gca().invert_xaxis()
ax1.set_title("Bit error rate, QAM %d, PTX = %d [dBm], IBO = %d [dB]" % (my_mod.constellation_size, total_tx_pow_dbm, ibo_db_val))
ax1.set_xlabel("Noise floor [dBm]")
ax1.set_ylabel("BER")
ax1.grid()
ax1.legend(title="CNC NI:")

plt.tight_layout()
plt.savefig(
    "figs/ber_soft_lim_miso_ptxtot%ddbm_cnc_ibo%d_niter%d_nupsamp%d_niter_sweep.png" % (
    total_tx_pow_dbm, ibo_db_val, cnc_n_iter_lst[-1], cnc_n_upsamp_val),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
# # %%
