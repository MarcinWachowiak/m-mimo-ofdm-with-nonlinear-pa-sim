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
from utilities import count_mismatched_bits, ebn0_to_snr, to_db, td_signal_power, to_time_domain
from scipy import interpolate
# TODO: consider logger

set_latex_plot_style()
# %%
print("Multi antenna processing init!")
# remember to copy objects not to avoid shared properties modifications!
# check modifications before copy and what you copy!
my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=5, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=None, cord_x=100, cord_y=100, cord_z=1.5,
                                center_freq=int(3.5e9), carrier_spacing=int(15e3))

my_miso_chan = channel.MisoTwoPathFd()
my_noise = noise.Awgn(snr_db=20, noise_p_dbm=-90, seed=1234)
my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))

# %%
# Upsample ratio eval, number of iterations fixed
# arbitrarly set params:
n_ant_val = 1
ebn0_db_vals = np.arange(10, 31, 2)

print("Eb/n0 values:", ebn0_db_vals)
snr_db_vals = ebn0_to_snr(ebn0_db_vals, my_mod.n_fft, my_mod.n_sub_carr, my_mod.constel_size)
print("SNR value:", snr_db_vals)

cnc_n_iter_vals = [0,1,2,4]
print("CNC N iterations:", cnc_n_iter_vals)
cnc_n_upsamp_val = 4
print("CNC upsample factor:", cnc_n_upsamp_val)

ibo_arr = np.arange(0, 6, 0.25)
print("IBO values:", ibo_arr)

# BER accuracy settings
bits_sent_max = int(1e6)
n_err_min = 1000

abs_lambda_per_ibo = np.zeros(len(ibo_arr))
ber_per_ibo_snr_iter = np.zeros((len(ibo_arr), len(snr_db_vals), len(cnc_n_iter_vals)))

my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5,
                                      cord_x=0, cord_y=0, cord_z=15)

my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, skip_attenuation=False)

chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)

# %%
# lambda estimation phase
for ibo_idx, ibo_val_db in enumerate(ibo_arr):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power,
                               channel_mat_fd=chan_mat_at_point)
    # correct avg sample power in nonlinearity after precoding

    # estimate lambda correcting coefficient
    # same seed is required
    bit_rng = np.random.default_rng(4321)
    dist_rx_pow_coeff = (np.abs(my_mod.calc_alpha(ibo_val_db))) ** 2

    n_ofdm_symb = 1e3
    ofdm_symb_idx = 0
    lambda_numerator_vecs = []
    lambda_denominator_vecs = []
    while ofdm_symb_idx < n_ofdm_symb:
        tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
        tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

        rx_sig_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
        rx_sig_clean_fd = my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol_fd)

        clean_nsc_ofdm_symb_fd = np.concatenate(
            (rx_sig_clean_fd[-my_mod.n_sub_carr // 2:], rx_sig_clean_fd[1:(my_mod.n_sub_carr // 2) + 1]))
        rx_nsc_ofdm_symb_fd = np.concatenate(
            (rx_sig_fd[-my_mod.n_sub_carr // 2:], rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))

        lambda_numerator_vecs.append(np.multiply(rx_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))
        lambda_denominator_vecs.append(np.multiply(clean_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))

        ofdm_symb_idx += 1

        # calculate lambda estimate
    lambda_num = np.average(np.vstack(lambda_numerator_vecs), axis=0)
    lambda_denum = np.average(np.vstack(lambda_denominator_vecs), axis=0)
    abs_lambda_per_ibo[ibo_idx] = np.abs(np.average(lambda_num / lambda_denum))

    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
# BER vs IBO eval
for ibo_idx, ibo_val_db in enumerate(ibo_arr):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power,
                               channel_mat_fd=chan_mat_at_point)
    for snr_idx, snr_val_db in enumerate(snr_db_vals):
        my_noise.snr_db = snr_val_db

        for iter_idx, cnc_iters_val in enumerate(cnc_n_iter_vals):
            # same seed is required
            # calculate BER based on channel estimate
            bit_rng = np.random.default_rng(4321)
            bers = np.zeros([len(cnc_n_iter_vals)])
            my_noise.snr_db = snr_val_db
            n_err = 0
            bits_sent = 0
            while bits_sent < bits_sent_max and n_err < n_err_min:
                tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
                tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

                rx_sig_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
                rx_ofdm_symbol_fd = my_noise.process(rx_sig_fd, avg_sample_pow=my_mod.avg_sample_power*(abs_lambda_per_ibo[ibo_idx]**2), fixed_noise_power=False)

                # enchanced CNC reception
                rx_bits = my_cnc_rx.receive(n_iters=cnc_iters_val, upsample_factor=cnc_n_upsamp_val,
                                            in_sig_fd=rx_ofdm_symbol_fd,
                                            lambda_estimation=abs_lambda_per_ibo[ibo_idx])

                n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
                bits_sent += my_mod.n_bits_per_ofdm_sym
                n_err += n_bit_err
            ber_per_ibo_snr_iter[ibo_idx, snr_idx, iter_idx] = n_err / bits_sent

    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
#%%
# extract SNR value providing given BER from collected data
req_ebn0_per_ibo = np.zeros((len(cnc_n_iter_vals), len(ibo_arr)))
target_ber = 1e-3
plot_once = False
for iter_idx, iter_val in enumerate(cnc_n_iter_vals):
    for ibo_idx, ibo_val in enumerate(ibo_arr):
        ber_per_ebn0 = ber_per_ibo_snr_iter[ibo_idx, :, iter_idx]
        # investigate better interpolation options
        interpol_func = interpolate.interp1d(ber_per_ebn0, ebn0_db_vals)
        if ibo_val == 2 and iter_val == 0:
            # ber vector
            if plot_once:
                fig1, ax1 = plt.subplots(1, 1)
                ax1.set_yscale('log')
                ax1.plot(ber_per_ebn0, ebn0_db_vals, label=iter_val)
                ax1.plot(ber_per_ebn0, interpol_func(ber_per_ebn0), label=iter_val)

                ax1.grid()
                plt.tight_layout()
                plt.show()
                print("Required Eb/No:", interpol_func(target_ber))

                fig2, ax2 = plt.subplots(1, 1)
                ax2.set_yscale('log')
                ax2.plot(ebn0_db_vals, ber_per_ebn0)

                plot_once = False
        try:
            req_ebn0_per_ibo[iter_idx, ibo_idx] = interpol_func(target_ber)
        except:
            # value not found in interpolation, replace with inf
            req_ebn0_per_ibo[iter_idx, ibo_idx] = np.inf


# %%
fig1, ax1 = plt.subplots(1, 1)
for ite_idx, ite_val in enumerate(cnc_n_iter_vals):
    #read by columns
    if ite_idx == 0:
        ite_val = "0 - standard"
    ax1.plot(ibo_arr, req_ebn0_per_ibo[ite_idx,:], label=ite_val)

ax1.set_title("BER = %1.1e, QAM %d, N ant = %d" % (target_ber, my_mod.constellation_size, n_ant_val))
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel("Eb/n0 [dB]")
ax1.grid()
ax1.legend(title="CNC N iterations")

plt.tight_layout()
plt.savefig(
    "figs/constant_ber%1.0e_req_ebn0_vs_ibo%dto%d_soft_lim_miso_cnc_%dqam_%dnant.png" % (target_ber, min(ibo_arr), max(ibo_arr), my_mod.constel_size, n_ant_val),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")

