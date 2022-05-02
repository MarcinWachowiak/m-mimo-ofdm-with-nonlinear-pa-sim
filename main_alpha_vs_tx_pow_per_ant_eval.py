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
my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=212, cord_y=212, cord_z=1.5,
                                center_freq=int(3.5e9), carrier_spacing=int(15e3))
my_rx.correct_constellation()


# %%
n_ant_val = 64
# averaging length
n_ofdm_symb = 1e2
print("N antennas values:", n_ant_val)
ibo_val_db = 5
print("IBO value:", ibo_val_db)

# %%
my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5,
                                      cord_x=0, cord_y=0, cord_z=15)
my_miso_rayleigh_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, seed=1234)
my_miso_los_chan = channel.MisoLosFd()
my_miso_two_path_chan = channel.MisoTwoPathFd()

my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, skip_attenuation=False)
my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, skip_attenuation=False)

#list of channel objects
chan_lst = [my_miso_rayleigh_chan, my_miso_two_path_chan, my_miso_los_chan]

lambda_per_nant_per_ibo = np.zeros((len(chan_lst), n_ant_val))
tx_pow_arr = np.zeros((len(chan_lst), n_ant_val))

for chan_idx, chan_obj in enumerate(chan_lst):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    tmp_chan_mat = chan_obj.get_channel_mat_fd()
    my_array.set_precoding_matrix(channel_mat_fd=tmp_chan_mat, mr_precoding=True)

    my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

    # estimate lambda correcting coefficient
    # same seed is required
    bit_rng = np.random.default_rng(4321)
    ofdm_symb_idx = 0
    lambda_numerator_vecs = []
    lambda_denominator_vecs = []
    ofdm_symb_pow_lst = []
    while ofdm_symb_idx < n_ofdm_symb:
        # reroll coeffs for each symbol for rayleigh chan
        # if chan_idx == 0:
        #     chan_obj.reroll_channel_coeffs()
        #     tmp_chan_mat = chan_obj.get_channel_mat_fd()
        #     my_array.set_precoding_matrix(channel_mat_fd=tmp_chan_mat, mr_precoding=True)
        #     my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

        tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
        tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

        clean_nsc_ofdm_symb_fd = np.concatenate(
            (clean_ofdm_symbol_fd[:, -my_mod.n_sub_carr // 2:], clean_ofdm_symbol_fd[:, 1:(my_mod.n_sub_carr // 2) + 1,]), axis=1)
        tx_nsc_ofdm_symb_fd = np.concatenate(
            (tx_ofdm_symbol_fd[:, -my_mod.n_sub_carr // 2:], tx_ofdm_symbol_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]), axis=1)

        # estimate lambda parameters for each antenna and compare in regard to the average
        lambda_numerator_vecs.append(np.multiply(tx_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))
        lambda_denominator_vecs.append(np.multiply(clean_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))
        ofdm_symb_pow_lst.append(np.sum(np.abs(clean_nsc_ofdm_symb_fd) ** 2, axis=1) / my_mod.n_fft)
        ofdm_symb_idx += 1

    #calculate the power of signal for each antenna
    tx_pow_arr[chan_idx, :] = np.average(ofdm_symb_pow_lst, axis=0)

    # calculate lambda estimate
    lambda_num = np.average(np.hstack(lambda_numerator_vecs), axis=1)
    lambda_denum = np.average(np.hstack(lambda_denominator_vecs), axis=1)
    lambda_per_nant_per_ibo[chan_idx, :] = np.abs(lambda_num / lambda_denum)
    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
fig1, ax1 = plt.subplots(1, 1)
# plot averaged power and lambda
avg_tx_pow = my_mod.avg_sample_power * my_array.get_avg_precoding_gain()
lambda_analytical = my_mod.calc_alpha(ibo_db=ibo_val_db)

ax1.plot(tx_pow_arr[0, :], lambda_per_nant_per_ibo[0, :], '.', label="Rayleigh")
ax1.plot(tx_pow_arr[1, :], lambda_per_nant_per_ibo[1, :], '.', label="Two-path")
ax1.plot(tx_pow_arr[2, :], lambda_per_nant_per_ibo[2, :], '.', label="LOS")
ax1.plot(avg_tx_pow, lambda_analytical, '.k', label="Analytical")

ax1.set_title(r"$\alpha$ coeff in relation to TX power [-]")
ax1.set_xlabel("TX power: $AVG\, |y_{k,n}|^2$[W]")
ax1.set_ylabel(r"$\alpha$ [-]")
ax1.grid()
ax1.legend(title="Channel:")

plt.tight_layout()
plt.savefig(
    "figs/alpha_vs_tx_power_per_ant%d_ibo%1.1f.png" % (n_ant_val, ibo_val_db),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")