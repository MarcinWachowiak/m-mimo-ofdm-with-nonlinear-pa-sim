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
ibo_db = 3
print("IBO val [dB]: ", ibo_db)

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=ibo_db, avg_samp_pow=my_mod.avg_sample_power)
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

my_miso_rayleigh_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                               seed=1234)

chan_lst = [my_miso_los_chan, my_miso_two_path_chan, my_miso_rayleigh_chan]
my_chan = chan_lst[0]
channel_mat = my_chan.get_channel_mat_fd()
my_array.set_precoding_matrix(channel_mat, mr_precoding=True)
my_array.update_distortion(ibo_db=ibo_db, avg_sample_pow=my_mod.avg_sample_power)

# transmit single burst and evaluate the impact of nonlinearity on the symbol

tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
tx_ofdm_symbol, clean_ofdm_symbol = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

rescaled_tx_ofdm_symbol = tx_ofdm_symbol / my_mod.calc_alpha(ibo_db=ibo_db)

phase_after_nl = np.rad2deg(np.angle(rescaled_tx_ofdm_symbol))
phase_after_precod = np.rad2deg(np.angle(clean_ofdm_symbol))

# rx_ofdm_symbol = my_miso_two_path_chan.propagate(in_sig_mat=clean_ofdm_symbol)
# rx_ofdm_symbol = rx_ofdm_symbol / agc_corr_vec

freq_bin_vec = np.fft.fftfreq(my_mod.n_fft, 1/my_mod.n_fft)
# sorted_freq_bin_vec, phase_after_precod = list(zip(*sorted(zip(freq_bin_vec, phase_after_precod))))
# sorted_freq_bin_vec, phase_after_nl = list(zip(*sorted(zip(freq_bin_vec, phase_after_nl))))

phase_after_precod = np.array(phase_after_precod)
phase_after_nl = np.array(phase_after_nl)

phase_after_precod = np.concatenate((phase_after_precod[-(my_mod.n_sub_carr // 2):], phase_after_precod[1:(my_mod.n_sub_carr // 2) + 1]), axis=0)
phase_after_nl = np.concatenate((phase_after_nl[-(my_mod.n_sub_carr // 2):], phase_after_nl[1:(my_mod.n_sub_carr // 2) + 1]), axis=0)
sorted_freq_bin_nsc = np.concatenate((freq_bin_vec[-(my_mod.n_sub_carr // 2):], freq_bin_vec[1:(my_mod.n_sub_carr // 2) + 1]), axis=0)
phase_diff = phase_after_precod - phase_after_nl
#%%
props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(sorted_freq_bin_nsc, phase_after_precod, label="After precod")
ax1.plot(sorted_freq_bin_nsc, phase_after_nl, label="After nl")
ax1.plot(sorted_freq_bin_nsc, phase_diff, label="Difference")

ax1.text(0.75, 0.84,(r"$\bar{\Delta} = %1.2f\degree$" % np.mean(np.abs(phase_diff))), transform=ax1.transAxes, fontsize=8, verticalalignment='bottom', bbox=props)


ax1.set_title("Precoding after NL")
ax1.set_xlabel("Frequency [subcarrier]")
ax1.set_ylabel("Phase [deg]")
ax1.grid()
# ax1.set_xlim(-128, 128)
ax1.set_ylim([-180, 180])

ax1.legend(title="Channel:", loc="lower right")
plt.tight_layout()
plt.savefig(
    "../figs/precoding_after_nl_phase_ibo_%d_chan_%s.png" %(ibo_db, str(my_chan)),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
