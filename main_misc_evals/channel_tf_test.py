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

#%%
freq_bin_vec = np.fft.fftfreq(my_mod.n_fft, 1/my_mod.n_fft)
chanel_tf_func_lst = []
for channel_idx, chan_obj in enumerate(chan_lst):
    channel_mat = chan_obj.get_channel_mat_fd()
    channel_tf_func = np.sum(channel_mat, axis=0)
    _, channel_tf_func = list(zip(*sorted(zip(freq_bin_vec, channel_tf_func))))
    chanel_tf_func_lst.append(channel_tf_func)
sorted_freq_bin_vec = sorted(freq_bin_vec)


#%%
# plot signal to distortion ratio vs ibo
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(sorted_freq_bin_vec, 10*np.log10(np.abs(chanel_tf_func_lst[2])), label="Rayleigh")
ax1.plot(sorted_freq_bin_vec, 10*np.log10(np.abs(chanel_tf_func_lst[1])), label="LOS")
ax1.plot(sorted_freq_bin_vec, 10*np.log10(np.abs(chanel_tf_func_lst[0])), label="Two-Path")

ax1.set_title("Channel TF function")
ax1.set_xlabel("Frequency [subcarrier]")
ax1.set_ylabel("Mag [dB]")
ax1.grid()
ax1.set_x_lim(-2048,2048)
ax1.legend(title="Channel:")
plt.tight_layout()
plt.savefig(
    "../figs/channel_tf_mag_test.png",
    dpi=600, bbox_inches='tight')
plt.show()

#%%
# plot signal to distortion ratio vs ibo
fig2, ax2 = plt.subplots(1, 1)
ax2.plot(sorted_freq_bin_vec, np.rad2deg(np.angle(chanel_tf_func_lst[2])), label="Rayleigh")
ax2.plot(sorted_freq_bin_vec, np.rad2deg(np.angle(chanel_tf_func_lst[1])), label="LOS")
ax2.plot(sorted_freq_bin_vec, np.rad2deg(np.angle(chanel_tf_func_lst[0])), label="Two-Path")

ax2.set_title("Channel TF function")
ax2.set_xlabel("Frequency [subcarrier]")
ax2.set_ylabel("Phase [deg]")
ax2.grid()
ax2.set_xlim(-128, 128)
ax2.set_ylim([-180, 180])

ax2.legend(title="Channel:")
plt.tight_layout()
plt.savefig(
    "../figs/channel_tf_phase_test.png",
    dpi=600, bbox_inches='tight')
plt.show()

#%%
channel_mat = my_miso_two_path_chan.get_channel_mat_fd()
my_array.set_precoding_matrix(channel_mat, mr_precoding=True)
precod_mat = my_array.array_elements[0].modem.precoding_mat

fig3, ax3 = plt.subplots(1, 1)
freq_bin_nsc = np.concatenate((freq_bin_vec[-(my_mod.n_sub_carr // 2):], freq_bin_vec[1:(my_mod.n_sub_carr // 2) + 1]), axis=0)
sorted_freq_bin_nsc = sorted(freq_bin_nsc)
ax3.plot(sorted_freq_bin_nsc, np.rad2deg(np.angle(precod_mat)), label="Two-path")

ax3.set_title("Precoding TF function")
ax3.set_xlabel("Frequency [subcarrier]")
ax3.set_ylabel("Phase [deg]")
ax3.grid()
ax3.set_xlim(-128, 128)
ax3.set_ylim([-180, 180])

ax3.legend(title="Channel:")
plt.tight_layout()
plt.savefig(
    "../figs/precoding_tf_phase_test.png",
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
