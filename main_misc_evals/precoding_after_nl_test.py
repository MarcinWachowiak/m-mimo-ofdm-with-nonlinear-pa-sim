# antenna array evaluation
# %%
import os
import sys

import utilities

sys.path.append(os.getcwd())

import copy

import matplotlib.pyplot as plt
import numpy as np

import antenna_arrray
import channel
import distortion
import modulation
import transceiver
from plot_settings import set_latex_plot_style

from scipy import signal
# TODO: consider logger
set_latex_plot_style()
# %%
print("Multi antenna processing init!")
bit_rng = np.random.default_rng(4321)

n_ant_arr = 1
print("N ANT values:", n_ant_arr)
ibo_db = 0
print("IBO val [dB]: ", ibo_db)

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=1024, n_sub_carr=512, cp_len=128)
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
my_chan = chan_lst[2]

chan_mat = my_chan.get_channel_mat_fd()
freq_bin_vec = np.fft.fftfreq(my_mod.n_fft, 1 / my_mod.n_fft)
bit_rng2 = np.random.default_rng(4321)
# ordered_phase_vec = bit_rng2.normal(loc=0.0, scale=0.2, size=my_mod.n_fft)

phase = np.pi * 2 * freq_bin_vec / my_mod.n_sub_carr * 10

# # check the dependency on phase evolution
# phase_vec = np.zeros(my_mod.n_fft)
# rng_phase_vec = np.pi * bit_rng2.normal(loc=0.0, scale=0.001, size=my_mod.n_fft)
# phase_vec[0] = rng_phase_vec[0]
# for idx in range(len(phase_vec)):
#     if idx  == 0:
#         pass
#     else:
#         if phase_vec[idx-1] + rng_phase_vec[idx] > np.pi:
#             phase_vec[idx] = phase_vec[idx - 1] - rng_phase_vec[idx]
#         elif phase_vec[idx-1] + rng_phase_vec[idx] < -np.pi:
#             phase_vec[idx] = phase_vec[idx - 1] + rng_phase_vec[idx]
#         else:
#             phase_vec[idx] = phase_vec[idx - 1] + rng_phase_vec[idx]
#
# ordered_phase_vec = np.concatenate((phase_vec[-my_mod.n_fft // 2:], phase_vec[1:(my_mod.n_fft // 2) + 1]))
ordered_phase_vec = np.concatenate((phase[-my_mod.n_fft // 2:], phase[1:(my_mod.n_fft // 2) + 1]))
# new_chan_fd = np.expand_dims(np.exp(1j* np.pi * np.sin(ordered_phase_vec)), axis=0)
new_chan_fd = np.expand_dims(np.exp(1j* np.pi * signal.sawtooth(ordered_phase_vec)), axis=0)


# new_chan_fd = np.expand_dims(np.exp(1j * ordered_phase_vec), axis=0)
my_chan.channel_mat_fd = new_chan_fd


channel_mat = my_chan.get_channel_mat_fd()
my_array.set_precoding_matrix(channel_mat, mr_precoding=True)
my_array.update_distortion(ibo_db=ibo_db, avg_sample_pow=my_mod.avg_sample_power)

# transmit single burst and evaluate the impact of nonlinearity on the symbol

# tx_bits = bit_rng.choice((0, 1), my_mod.n_bits_per_ofdm_sym)
tx_bit_seq = [0,0,0,0,0,0]
tx_symbol = my_mod.modulate(tx_bit_seq, get_symbols_only=True)
symbols_init_phase = np.angle(tx_symbol)

# tx_bits = np.repeat(tx_bit_seq, my_mod.n_sub_carr)
# # tx_ofdm_symbol, clean_ofdm_symbol = my_array.transmit(in_bits=tx_bits, out_domain_fd=True, return_both=True)

modulated_symbols = np.repeat([7+0j], my_mod.n_sub_carr)
precoded_symbols = np.squeeze(my_array.array_elements[0].modem.precode_symbols(modulated_symbols, my_array.array_elements[0].modem.precoding_mat))
clean_ofdm_symbol = modulation._tx_ofdm_symbol(precoded_symbols, my_mod.n_fft, my_mod.n_sub_carr, my_mod.cp_len)
tx_ofdm_symbol = my_distortion.process(clean_ofdm_symbol)

# shifted_clean_ofdm_symbol = np.multiply(clean_ofdm_symbol,np.exp(2j * np.pi * -0.5))

# shifted_clean_ofdm_symbol = utilities.to_freq_domain(shifted_clean_ofdm_symbol, remove_cp=True, cp_len=my_mod.cp_len)
clean_ofdm_symbol = utilities.to_freq_domain(clean_ofdm_symbol, remove_cp=True, cp_len=my_mod.cp_len)
tx_ofdm_symbol = utilities.to_freq_domain(tx_ofdm_symbol, remove_cp=True, cp_len=my_mod.cp_len)

rescaled_tx_ofdm_symbol = tx_ofdm_symbol / my_mod.calc_alpha(ibo_db=ibo_db)

phase_after_nl = np.rad2deg(np.angle(rescaled_tx_ofdm_symbol))
phase_after_precod = np.rad2deg(np.angle(clean_ofdm_symbol))
# channel_phase = np.squeeze(np.rad2deg(np.angle(new_chan_fd)))

precoding_mat = my_array.array_elements[0].modem.precoding_mat
precod_phase = np.rad2deg(np.angle(precoding_mat))

# precoding_mat_td = utilities.to_time_domain(precoding_mat)
# shifted_precoding_mat_td = np.multiply(precoding_mat_td, np.exp(2j* np.pi * 0.25))
# shifted_precoding_mat = utilities.to_freq_domain(shifted_precoding_mat_td)
# shifted_precoding_mat = np.rad2deg(np.angle(shifted_precoding_mat))

freq_bin_vec = np.fft.fftfreq(my_mod.n_fft, 1 / my_mod.n_fft)
# sorted_freq_bin_vec, phase_after_precod = list(zip(*sorted(zip(freq_bin_vec, phase_after_precod))))
# sorted_freq_bin_vec, phase_after_nl = list(zip(*sorted(zip(freq_bin_vec, phase_after_nl))))

phase_after_precod = np.concatenate(
    (phase_after_precod[1:(my_mod.n_sub_carr // 2) + 1], phase_after_precod[-(my_mod.n_sub_carr // 2):]), axis=0)
phase_after_nl = np.concatenate(
    (phase_after_nl[1:(my_mod.n_sub_carr // 2) + 1], phase_after_nl[-(my_mod.n_sub_carr // 2):]), axis=0)
precod_phase = np.concatenate(
    (precod_phase[-(my_mod.n_sub_carr // 2):], precod_phase[0:(my_mod.n_sub_carr // 2)], ), axis=0)
sorted_freq_bin_nsc = np.concatenate(
    (freq_bin_vec[-(my_mod.n_sub_carr // 2):], freq_bin_vec[1:(my_mod.n_sub_carr // 2) + 1]), axis=0)
# shifted_precoding_mat = np.concatenate(
#     (shifted_precoding_mat[-(my_mod.n_sub_carr // 2):], shifted_precoding_mat[0:(my_mod.n_sub_carr // 2)]), axis=0)
# channel_phase_nsc = np.concatenate(
#     (channel_phase[1:(my_mod.n_sub_carr // 2) + 1], channel_phase[-(my_mod.n_sub_carr // 2):]), axis=0)

phase_diff = phase_after_precod - phase_after_nl
# %%
props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(sorted_freq_bin_nsc, phase_after_precod, label="After precod")
ax1.plot(sorted_freq_bin_nsc, phase_after_nl, label="After nl")
ax1.plot(sorted_freq_bin_nsc, phase_diff, label="Difference")
ax1.plot(sorted_freq_bin_nsc, precod_phase, label="Precod phase")
# ax1.plot(sorted_freq_bin_nsc, shifted_precoding_mat, label="Shifted precoding mat")
# ax1.plot(sorted_freq_bin_nsc, channel_phase_nsc, label="Channel phase")

ax1.text(0.75, 0.84, (r"$\bar{\Delta} = %1.2f\degree$" % np.mean(np.abs(phase_diff))), transform=ax1.transAxes,
         fontsize=8, verticalalignment='bottom', bbox=props)

ax1.set_title("Precoding after NL")
ax1.set_xlabel("Frequency [subcarrier]")
ax1.set_ylabel("Phase [deg]")
ax1.grid()
# ax1.set_xlim(-128, 128)
# ax1.set_ylim([-180, 180])

ax1.legend(loc="lower right")
plt.tight_layout()
plt.savefig(
    "../figs/precoding_after_nl_phase_ibo_%d_chan_%s.png" % (ibo_db, str(my_chan)),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
