# antenna array evaluation
# %%
import os, sys
sys.path.append(os.getcwd())

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

n_ant_arr = 1
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

# %%
psd_nfft = 4096
n_samp_per_seg = 1024
n_snapshots = 30



my_miso_los_chan = channel.MisoLosFd()
my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                  skip_attenuation=False)
my_miso_two_path_chan = channel.MisoTwoPathFd()
my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                       skip_attenuation=False)

my_miso_rayleigh_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                               seed=1234)

chan_lst = [my_miso_two_path_chan, my_miso_los_chan, my_miso_rayleigh_chan]

chanel_tf_func_lst = []
for channel_idx, chan_obj in enumerate(chan_lst):
    channel_mat = chan_obj.get_channel_mat_fd()
    channel_tf_func = np.sum(channel_mat, axis=0)
    chanel_tf_func_lst.append(channel_tf_func)

freqs = np.fft.fftfreq(my_mod.n_fft, my_tx.carrier_spacing)


# plot signal to distortion ratio vs ibo
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(freqs, utilities.to_db(np.abs(chanel_tf_func_lst[2]), label="Rayleigh"))
ax1.plot(freqs, utilities.to_db(np.abs(chanel_tf_func_lst[1]), label="Two-Path"))
ax1.plot(freqs, utilities.to_db(np.abs(chanel_tf_func_lst[0]), label="LOS"))

ax1.set_title("Channel TF function")
ax1.set_xlabel("Frequency [subcarrier]")
ax1.set_ylabel("Mag [dB]")
ax1.grid()
# ax1.legend(title="Channel:")
plt.tight_layout()
plt.savefig(
    "../figs/channel_tf_mag_test.png",
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
