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
from matplotlib.ticker import MaxNLocator

# TODO: consider logger

set_latex_plot_style()
# %%
print("Multi antenna processing init!")
# remember to copy objects not to avoid shared properties modifications!
# check modifications before copy and what you copy!
my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=0, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=100, cord_y=100, cord_z=1.5,
                                center_freq=int(3.5e9), carrier_spacing=int(15e3))
my_rx.correct_constellation()


# %%
n_ant_arr = [1, 2, 3, 4] # 16, 32, 64, 128]
print("N antennas values:", n_ant_arr)
ibo_arr = np.arange(0, 11.0, 1)
print("IBO values:", ibo_arr)

abs_lambda_per_nant_per_ibo = np.zeros((len(n_ant_arr), len(ibo_arr)))
abs_lambda_per_ibo_analytical = []

# get analytical value of alpha
for ibo_idx, ibo_val_db in enumerate(ibo_arr):
    abs_lambda_per_ibo_analytical.append(my_mod.calc_alpha(ibo_val_db))
# %%
# lambda estimation phase
for n_ant_idx, n_ant_val in enumerate(n_ant_arr):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    for ibo_idx, ibo_val_db in enumerate(ibo_arr):
        lambda_corr_estimate = []
        my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                              wav_len_spacing=0.5,
                                              cord_x=0, cord_y=0, cord_z=15)
        my_miso_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, seed=1234)

        chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
        my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)
        my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
        # correct avg sample power in nonlinearity after precoding

        # estimate lambda correcting coefficient
        # same seed is required
        bit_rng = np.random.default_rng(4321)
        n_ofdm_symb = 5e2
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
        abs_lambda_per_nant_per_ibo[n_ant_idx, ibo_idx] = (np.abs(np.average(lambda_num / lambda_denum)))
    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
fig1, ax1 = plt.subplots(1, 1)

# plot analytical
ax1.plot(ibo_arr, abs_lambda_per_ibo_analytical, label="Analytical")

for n_ant_idx, n_ant_val in enumerate(n_ant_arr):
    ax1.plot(ibo_arr, abs_lambda_per_nant_per_ibo[n_ant_idx, :], label=n_ant_val)


ax1.set_title("Constellation shrinking coefficient - alpha for N antennas [-]")
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel("Alpha [-]")
ax1.grid()
ax1.legend(title="N antennas:")

plt.tight_layout()
plt.savefig(
    "figs/constel_shrinking_coeff_nant%dto%d_ibo%1.1fto%1.1f.png" % (min(n_ant_arr), max(n_ant_arr), min(ibo_arr), max(ibo_arr)),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")

#%%
