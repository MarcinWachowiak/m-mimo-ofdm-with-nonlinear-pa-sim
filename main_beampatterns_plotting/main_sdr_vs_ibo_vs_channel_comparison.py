# antenna array evaluation
# %%
import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import antenna_arrray
import channel
import distortion
import modulation
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import to_db, pts_on_circum, pts_on_semicircum

# TODO: consider logger
set_latex_plot_style()
# %%
print("Multi antenna processing init!")
bit_rng = np.random.default_rng(4321)

ibo_arr = np.arange(0, 8.0, 0.1)
print("IBO values:", ibo_arr)

n_ant_arr = [1,2,4]
print("N ANT values:", n_ant_arr)

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=3.0, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))

my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=212,
                                cord_y=212, cord_z=1.5, center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))

my_array = antenna_arrray.LinearArray(n_elements=np.min(n_ant_arr), base_transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
my_rx.set_position(cord_x=212, cord_y=212, cord_z=1.5)

my_miso_los_chan = channel.MisoLosFd()
my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                  skip_attenuation=False)
my_miso_two_path_chan = channel.MisoTwoPathFd()
my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                       skip_attenuation=False)
chan_lst = [my_miso_two_path_chan, my_miso_los_chan]

# %%
psd_nfft = 4096
n_samp_per_seg = 1024
n_snapshots = 30

# %%
# plot PSD for chosen point/angle
sdr_at_ibo_per_chan = []
# direct visibility channels - increasing antennas does not increase SDR - run for lowest value of n_ant_arr
for chan_idx, chan_obj in enumerate(chan_lst):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    my_miso_chan = chan_obj
    channel_mat_at_point_fd = my_miso_chan.get_channel_mat_fd()
    my_array.set_precoding_matrix(channel_mat_fd=channel_mat_at_point_fd, mr_precoding=True)
    sdr_at_ibo = np.zeros(len(ibo_arr))

    for ibo_idx, ibo_val_db in enumerate(ibo_arr):
        my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
        my_rx.update_distortion(ibo_db=ibo_val_db)

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

            rx_ofdm_sc_accum.append(rx_sc_ofdm_symb_fd)
            clean_rx_ofdm_sc_accum.append(clean_sc_ofdm_symb_fd)

        rx_sig_accum_arr = np.concatenate(rx_ofdm_sc_accum).ravel()
        clean_rx_sig_accum_arr = np.concatenate(clean_rx_ofdm_sc_accum).ravel()
        sc_ofdm_distortion_sig = rx_sig_accum_arr - my_rx.modem.alpha * clean_rx_sig_accum_arr

        # dist_ofdm_symb_freq_arr, dist_ofdm_symb_psd_arr = welch(sc_ofdm_distortion_sig, fs=psd_nfft, nfft=psd_nfft,
        #                                                         nperseg=n_samp_per_seg, return_onesided=False)
        # clean_ofdm_symb_freq_arr, clean_ofdm_symb_psd_arr = welch(clean_rx_sig_accum_arr, fs=psd_nfft, nfft=psd_nfft,
        #                                                           nperseg=n_samp_per_seg, return_onesided=False)
        # sdr_at_ibo[ibo_idx] = to_db(np.sum(clean_ofdm_symb_psd_arr)/np.sum(dist_ofdm_symb_psd_arr))

        sdr_at_ibo[ibo_idx] = to_db(
            np.sum(np.power(np.abs(clean_rx_sig_accum_arr), 2)) / np.sum(np.power(np.abs(sc_ofdm_distortion_sig), 2)))

    sdr_at_ibo_per_chan.append(sdr_at_ibo)
    print("--- Computation time: %f ---" % (time.time() - start_time))

# for rayleigh channel
sdr_at_ibo_per_rayleigh = []
for n_ant_idx, n_ant_val in enumerate(n_ant_arr):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)

    my_miso_rayleigh_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, seed=1234)
    my_miso_chan = my_miso_rayleigh_chan

    channel_mat_at_point_fd = my_miso_chan.get_channel_mat_fd()
    my_array.set_precoding_matrix(channel_mat_fd=channel_mat_at_point_fd, mr_precoding=True)
    sdr_at_ibo = np.zeros(len(ibo_arr))

    for ibo_idx, ibo_val_db in enumerate(ibo_arr):
        my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
        my_rx.update_distortion(ibo_db=ibo_val_db)

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

            rx_ofdm_sc_accum.append(rx_sc_ofdm_symb_fd)
            clean_rx_ofdm_sc_accum.append(clean_sc_ofdm_symb_fd)

        rx_sig_accum_arr = np.concatenate(rx_ofdm_sc_accum).ravel()
        clean_rx_sig_accum_arr = np.concatenate(clean_rx_ofdm_sc_accum).ravel()
        sc_ofdm_distortion_sig = rx_sig_accum_arr - my_rx.modem.alpha * clean_rx_sig_accum_arr

        # dist_ofdm_symb_freq_arr, dist_ofdm_symb_psd_arr = welch(sc_ofdm_distortion_sig, fs=psd_nfft, nfft=psd_nfft,
        #                                                         nperseg=n_samp_per_seg, return_onesided=False)
        # clean_ofdm_symb_freq_arr, clean_ofdm_symb_psd_arr = welch(clean_rx_sig_accum_arr, fs=psd_nfft, nfft=psd_nfft,
        #                                                           nperseg=n_samp_per_seg, return_onesided=False)
        # sdr_at_ibo[ibo_idx] = to_db(np.sum(clean_ofdm_symb_psd_arr)/np.sum(dist_ofdm_symb_psd_arr))

        sdr_at_ibo[ibo_idx] = to_db(
            np.sum(np.power(np.abs(clean_rx_sig_accum_arr), 2)) / np.sum(np.power(np.abs(sc_ofdm_distortion_sig), 2)))

    sdr_at_ibo_per_rayleigh.append(sdr_at_ibo)
    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
# plot signal to distortion ratio vs ibo
fig1, ax1 = plt.subplots(1, 1)
for chan_idx, chan_obj in enumerate(chan_lst):
    ax1.plot(ibo_arr, sdr_at_ibo_per_chan[chan_idx], label=str(chan_obj)+(", %d ant" % np.min(n_ant_arr)))
for n_ant_idx, n_ant_val in enumerate(n_ant_arr):
    ax1.plot(ibo_arr, sdr_at_ibo_per_rayleigh[n_ant_idx], label="Rayleigh, %d ant" % n_ant_val)

ax1.set_title("SDR vs IBO for given channel")
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel("SDR [dB]")
ax1.grid()
ax1.legend(title="Channel:")
plt.tight_layout()
plt.savefig(
    "../figs/sdr_vs_ibo_per_channel_ibo%dto%d_%dnant.png" % (
        min(ibo_arr), max(ibo_arr), np.max(n_ant_arr)),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
