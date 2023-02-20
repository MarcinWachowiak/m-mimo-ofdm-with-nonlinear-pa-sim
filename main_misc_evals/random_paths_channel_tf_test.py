# MISO OFDM simulation with nonlinearity
# Clipping noise cancellation eval
# %%
import os
import sys

sys.path.append(os.getcwd())

import copy

import matplotlib.pyplot as plt
import numpy as np

import channel
import distortion
import modulation
import transceiver
from plot_settings import set_latex_plot_style

set_latex_plot_style()

# %%
# parameters
n_ant_val = 8
ibo_arr = [0]
ebn0_step = [1]
cnc_n_iter_lst = [1, 2, 3, 4, 5, 6, 7, 8]
# include clean run is always True
# no distortion and standard RX always included
cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)

# print("Distortion IBO/TOI value:", ibo_db)
# print("Eb/n0 values: ", ebn0_arr)
# print("CNC iterations: ", cnc_n_iter_lst)

# modulation
constel_size = 64
n_fft = 4096
n_sub_carr = 2048
cp_len = 128

# accuracy
bits_sent_max = int(1e7)
n_err_min = int(1e5)

rx_loc_x, rx_loc_y = 212.0, 212.0
rx_loc_var = 10.0

my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len)

my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                center_freq=int(3.5e9), carrier_spacing=int(15e3))

my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                         cord_x=rx_loc_x, cord_y=rx_loc_y, cord_z=1.5,
                                         center_freq=int(3.5e9), carrier_spacing=int(15e3))

# %%
my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
# channel type
my_miso_los_chan = channel.MisoLosFd()
my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                  skip_attenuation=False)
my_miso_two_path_chan = channel.MisoTwoPathFd()
my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                       skip_attenuation=False)

my_miso_rayleigh_chan = channel.MisoRayleighFd(tx_transceivers=my_array.array_elements,
                                               rx_transceiver=my_standard_rx,
                                               seed=1234)

my_random_paths_miso_channel = channel.MisoRandomPathsFd(tx_transceivers=my_array.array_elements,
                                                         rx_transceiver=my_standard_rx, n_paths=3,
                                                         max_delay_spread=1000e-9)
chan_lst = [my_random_paths_miso_channel]

# %%
# plot the random path channel transfer function
freq_bin_vec = np.fft.fftfreq(my_mod.n_fft, 1 / my_mod.n_fft)
sorted_freq_bin_vec = sorted(freq_bin_vec)

chanels_tf_func_lst = []
for channel_idx, chan_obj in enumerate(chan_lst):
    channel_siso_tf = []
    channel_mat = chan_obj.get_channel_mat_fd()
    for ant_idx in range(channel_mat.shape[0]):
        siso_channel_tf_func = channel_mat[ant_idx, :]
        _, siso_channel_tf_func = zip(*sorted(zip(freq_bin_vec, siso_channel_tf_func)))
        channel_siso_tf.append(np.asarray(siso_channel_tf_func))
    chanels_tf_func_lst.append(channel_siso_tf)

fig1, ax1 = plt.subplots(1, 1)
for chan_idx, channel_obj in enumerate(chan_lst):
    for tx_idx in range(len(my_array.array_elements)):
        ax1.plot(10 * np.log10(np.abs(chanels_tf_func_lst[chan_idx][tx_idx])), label=tx_idx)

ax1.set_title("Random paths channel model transfer function")
ax1.set_ylabel("Transfer function |h(f)| [dB]")
ax1.set_xlabel("Subcarrier index [-]")
ax1.grid()
ax1.legend(title="TX-RX idx:")
plt.tight_layout()
plt.savefig(
    "../figs/random_paths_channel_tf_npaths%d_maxdelayspr%1.2e.png" % (
        my_random_paths_miso_channel.n_paths, my_random_paths_miso_channel.max_delay_spread),
    dpi=600, bbox_inches='tight')
plt.show()
