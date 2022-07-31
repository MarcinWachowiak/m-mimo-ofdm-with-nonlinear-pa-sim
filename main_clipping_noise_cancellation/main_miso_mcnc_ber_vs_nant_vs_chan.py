# antenna array evaluation
# %%
import os
import sys

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
from plot_settings import set_latex_plot_style, reset_color_cycle
from utilities import count_mismatched_bits, ebn0_to_snr
import utilities

set_latex_plot_style()
# %%

n_ant_arr = [1, 2, 4, 8, 16, 32, 64, 128]
ebn0_db = 15
ibo_val_db = 0
cnc_n_iter_lst = [1, 2, 3, 4, 5, 6, 7, 8]
# standard RX
cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)

# print("Eb/n0 value:", ebn0_db)
# print("CNC N iterations:", cnc_n_iter_lst)
# print("IBO values:", ibo_arr)

# modulation
constel_size = 64
n_fft = 4096
n_sub_carr = 2048
cp_len = 128

# BER accuracy settings
bits_sent_max = int(1e7)
n_err_min = int(1e6)

# remember to copy objects not to avoid shared properties modifications!
# check modifications before copy and what you copy!
my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len)

my_distortion = distortion.SoftLimiter(ibo_db=0, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=212,
                                cord_y=212, cord_z=1.5,
                                center_freq=int(3.5e9), carrier_spacing=int(15e3))

bers_per_nant = []
for n_ant_val in n_ant_arr:
    my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5,
                                          cord_x=0, cord_y=0, cord_z=15)

    # channel type
    my_miso_los_chan = channel.MisoLosFd()
    my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                      skip_attenuation=False)
    my_miso_two_path_chan = channel.MisoTwoPathFd()
    my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                           skip_attenuation=False)

    my_miso_rayleigh_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                                   seed=1234)
    chan_lst = [my_miso_los_chan, my_miso_two_path_chan, my_miso_rayleigh_chan]
    bers_per_chan = []

    for my_miso_chan in chan_lst:
        start_time = time.time()
        print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        my_mcnc_rx = corrector.McncReceiver(copy.deepcopy(my_array), copy.deepcopy(my_miso_chan))

        chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
        my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)

        snr_val_db = ebn0_to_snr(ebn0_db, my_mod.n_sub_carr, my_mod.n_sub_carr, my_mod.constel_size)
        my_noise = noise.Awgn(snr_db=20, noise_p_dbm=-90, seed=1234)
        my_noise.snr_db = snr_val_db

        my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
        my_mcnc_rx.update_distortion(ibo_db=ibo_val_db)

        hk_mat = np.concatenate((chan_mat_at_point[:, -my_mod.n_sub_carr // 2:],
                                 chan_mat_at_point[:, 1:(my_mod.n_sub_carr // 2) + 1]), axis=1)
        vk_mat = my_array.get_precoding_mat()
        vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)
        hk_vk_agc = np.multiply(hk_mat, vk_mat)
        hk_vk_agc_avg_vec = np.sum(hk_vk_agc, axis=0)
        hk_vk_noise_scaler = np.mean(np.power(hk_vk_agc_avg_vec, 2))

        hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
        hk_vk_agc_nfft[-(n_sub_carr // 2):] = hk_vk_agc_avg_vec[0:n_sub_carr // 2]
        hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = hk_vk_agc_avg_vec[n_sub_carr // 2:]

        ibo_vec = 10 * np.log10(10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
        ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
        ak_vect = np.expand_dims(ak_vect, axis=1)

        ak_hk_vk_agc = ak_vect * hk_vk_agc
        ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)
        ak_hk_vk_noise_scaler = np.mean(np.power(np.abs(ak_hk_vk_agc_avg_vec), 2))

        ak_hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
        ak_hk_vk_agc_nfft[-(n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:n_sub_carr // 2]
        ak_hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[n_sub_carr // 2:]

        bit_rng = np.random.default_rng(4321)

        bers = np.zeros([len(cnc_n_iter_lst)])
        n_err = np.zeros([len(cnc_n_iter_lst)])
        bits_sent = np.zeros([len(cnc_n_iter_lst)])
        bers_per_ite = np.zeros(len(cnc_n_iter_lst))
        while True:
            ite_use_flags = np.logical_and((n_err < n_err_min), (bits_sent < bits_sent_max))
            if ite_use_flags.any() == True:
                curr_ite_lst = cnc_n_iter_lst[ite_use_flags]
            else:
                break

            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)

            tx_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=False)
            rx_ofdm_symbol_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
            rx_ofdm_symbol_fd = my_noise.process(rx_ofdm_symbol_fd,
                                                 avg_sample_pow=my_mod.avg_symbol_power * ak_hk_vk_noise_scaler,
                                                 disp_data=False)
            rx_ofdm_symbol_fd = np.divide(rx_ofdm_symbol_fd, ak_hk_vk_agc_nfft)

            # CNC reception
            rx_bits_per_iter_lst = my_mcnc_rx.receive(n_iters_lst=curr_ite_lst, in_sig_fd=rx_ofdm_symbol_fd)

            ber_idx = np.array(list(range(len(cnc_n_iter_lst))))
            act_ber_idx = ber_idx[ite_use_flags]
            for idx in range(len(rx_bits_per_iter_lst)):
                n_bit_err = count_mismatched_bits(tx_bits, rx_bits_per_iter_lst[idx])
                n_err[act_ber_idx[idx]] += n_bit_err
                bits_sent[act_ber_idx[idx]] += my_mod.n_bits_per_ofdm_sym

        for ite_idx in range(len(cnc_n_iter_lst)):
            bers_per_ite[ite_idx] = n_err[ite_idx] / bits_sent[ite_idx]
        bers_per_chan.append(bers_per_ite)

        print("--- Computation time: %f ---" % (time.time() - start_time))

    bers_per_nant.append(bers_per_chan)

# %%
# parse data for easier plotting
bers_per_chan_per_nite_per_n_ant = []
for chan_idx, chan_obj in enumerate(chan_lst):
    ber_per_ite_lst = []
    for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
        ber_per_nant = np.zeros(len(n_ant_arr))
        for nant_idx, n_ant_val in enumerate(n_ant_arr):
            ber_per_nant[nant_idx] = bers_per_nant[nant_idx][chan_idx][ite_idx]
        ber_per_ite_lst.append(ber_per_nant)
    bers_per_chan_per_nite_per_n_ant.append(ber_per_ite_lst)

fig1, ax1 = plt.subplots(1, 1)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=10)
ax1.set_xticks(n_ant_arr)
ax1.set_xticklabels(n_ant_arr)

chan_linestyles = ['o-', 's-', '*-']
for chan_idx, chan_obj in enumerate(chan_lst):
    for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
        ax1.plot(n_ant_arr, bers_per_chan_per_nite_per_n_ant[chan_idx][ite_idx], chan_linestyles[chan_idx],
                 fillstyle="none", label=ite_val)
    reset_color_cycle()

ax1.set_title(
    "BER vs N ant, MCNC, QAM %d, IBO = %d [dB], Eb/n0 = %d [dB], " % (my_mod.constellation_size, ibo_val_db, ebn0_db))
ax1.set_xlabel("N antennas [-]")
ax1.set_ylabel("BER")
ax1.grid(which='major', linestyle='-')
ax1.grid(which='minor', linestyle='--')

import matplotlib.lines as mlines

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

n_ite_legend = []
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[ite_idx], label=ite_val))
leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="lower left", ncol=1)
plt.gca().add_artist(leg1)

import matplotlib.lines as mlines

los = mlines.Line2D([0], [0], linestyle='none', marker="o", fillstyle="none", color='k', label='LOS')
twopath = mlines.Line2D([0], [0], linestyle='none', marker="s", fillstyle="none", color='k', label='Two-path')
rayleigh = mlines.Line2D([0], [0], linestyle='none', marker="*", fillstyle="none", color='k', label='Rayleigh')
leg2 = plt.legend(handles=[los, twopath, rayleigh], title="Channels:", loc="upper left")
plt.gca().add_artist(leg2)

plt.tight_layout()

filename_str = "ber_vs_nant_mcnc_nant%s_ebn0_%d_ibo%d_niter%s" % (
    '_'.join([str(val) for val in n_ant_arr]), ebn0_db, ibo_val_db, '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("figs/vm_worker_results/%s.png" % filename_str, dpi=600, bbox_inches='tight')
# plt.show()
plt.cla()
plt.close()

# %%
data_lst = []
data_lst.append(n_ant_arr)
for arr1 in bers_per_chan_per_nite_per_n_ant:
    for arr2 in arr1:
        data_lst.append(arr2)
utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

# read_data = utilities.read_from_csv(filename=filename_str)
print("Finished execution!")
