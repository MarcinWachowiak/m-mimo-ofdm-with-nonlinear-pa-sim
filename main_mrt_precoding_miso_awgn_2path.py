# antenna array evaluation
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import time

import utilities
from utilities import count_mismatched_bits, snr_to_ebn0, ebn0_to_snr, to_db, pts_on_circum, signal_power
import channels
import modulation
import impairments
import transceiver
import antenna_arrray
import torch
from plot_settings import set_latex_plot_style
# TODO: consider logger
np.set_printoptions(threshold=np.inf)

set_latex_plot_style()
#%%
print("Multi antenna processing init!")
start_time = time.time()

radian_vals_lst = []
psd_at_angle_lst = []
bit_rng = np.random.default_rng(4321)

#for run_idx in range(1):
my_mod = modulation.OfdmQamModem(constel_size=4, n_fft=4096, n_sub_carr=1024, cp_len=128)
ibo_db = 5
my_distortion = impairments.SoftLimiter(ibo_db=ibo_db, avg_symb_pow=my_mod.ofdm_avg_sample_pow())
my_tx = transceiver.Transceiver(modem=my_mod, impairment=my_distortion, center_freq=int(3.5e9), carrier_spacing=int(150e3))
my_array = antenna_arrray.LinearArray(n_elements=8, transceiver=my_tx, center_freq=int(3.5e9), wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
my_rx = transceiver.Transceiver(modem=my_mod, impairment=None, cord_x=15, cord_y=15, cord_z=1.5)
my_rx.modem.correct_constellation(ibo_db=ibo_db)

my_miso_chan = channels.AwgnMisoTwoPath(n_inputs=my_array.n_elements, snr_db=10, is_complex=True, seed=1234)
my_array.set_precoding_single_point(rx_transceiver=my_rx, exact=True, two_path=True)
utilities.plot_spatial_config(ant_array=my_array,rx_transceiver=my_rx)
#if run_idx == 0:
#else:
# my_array.set_precoding_single_point(rx_transceiver=my_rx, exact=False)
# my_array.plot_configuration(plot_3d=True)
# utilities.plot_spatial_config(my_array, my_rx)

# bit_rng = np.random.default_rng(4321) tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym) arr_tx_sig
# my_array.transmit(in_bits=tx_bits) rx_sig = my_miso_chan.propagate(tx_transceivers=my_array.array_elements,
# rx_transceiver=my_rx, in_sig=arr_tx_sig, skip_noise=True)
#

#%%
# for a single carrier frequency plot the TX characteristics
n_points = 360
rx_points = pts_on_circum(r=22, n=n_points)
radian_vals = np.radians(np.linspace(0, 360, n_points+1))
psd_at_angle_desired = np.empty(radian_vals.shape)
psd_at_angle_dist = np.empty(radian_vals.shape)

psd_nfft = 4096
n_samp_per_seg = 1024
n_snapshots = 10

#%%
#chosen point idx
point_idx_psd = 45
precoding_point_idx = 45
rx_sig_at_point_clean = []
rx_sig_at_point_full = []


for pt_idx, point in enumerate(rx_points):
    # generate different channel for each point
    # precode only for single known point

    (x_cord, y_cord) = point
    my_rx.set_position(cord_x=x_cord, cord_y=y_cord, cord_z=1.5)
    rx_sig_accum = []
    clean_rx_sig_accum = []
    for snap_idx in range(n_snapshots):

        tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
        arr_tx_sig, clean_sig_mat = my_array.transmit(in_bits=tx_bits, return_both=True, )
        rx_sig = my_miso_chan.propagate(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                        in_sig_mat=arr_tx_sig, skip_noise=True, skip_attenuation=True)
        rx_sig = torch.fft.ifft(torch.from_numpy(rx_sig), norm="ortho").numpy()

        clean_rx_sig = my_miso_chan.propagate(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                        in_sig_mat=clean_sig_mat, skip_noise=True, skip_attenuation=True)
        clean_rx_sig = torch.fft.ifft(torch.from_numpy(clean_rx_sig), norm="ortho").numpy()

        if pt_idx == point_idx_psd:
            rx_sig_at_point_clean.append(clean_rx_sig)
            rx_sig_at_point_full.append(rx_sig)

        rx_sig_accum.append(rx_sig)
        clean_rx_sig_accum.append(clean_rx_sig)

    rx_sig_accum_arr = np.concatenate(rx_sig_accum).ravel()
    clean_rx_sig_accum_arr = np.concatenate(clean_rx_sig_accum).ravel()
    distortion_sig = rx_sig_accum_arr - my_rx.modem.alpha * clean_rx_sig_accum_arr


    dist_rx_sig_freq_arr, dist_rx_sig_psd_arr = welch(distortion_sig, fs=psd_nfft, nfft=psd_nfft,
                                          nperseg=n_samp_per_seg, return_onesided=False)
    clean_rx_sig_freq_arr, clean_rx_sig_psd_arr = welch(clean_rx_sig_accum_arr, fs=psd_nfft, nfft=psd_nfft,
                                          nperseg=n_samp_per_seg, return_onesided=False)

    psd_at_angle_desired[pt_idx] = to_db(np.sum(np.array(clean_rx_sig_psd_arr)))
    psd_at_angle_dist[pt_idx] = to_db(np.sum(np.array(dist_rx_sig_psd_arr)))


rx_sig_at_point_clean_arr = np.concatenate(rx_sig_at_point_clean).ravel()
rx_sig_at_point_full_arr = np.concatenate(rx_sig_at_point_full).ravel()
distortion_sig_at_point = rx_sig_at_point_full_arr - my_rx.modem.alpha * rx_sig_at_point_clean_arr

rx_clean_at_point_freq_arr, rx_clean_at_point_psd = welch(rx_sig_at_point_clean_arr, fs=psd_nfft, nfft=psd_nfft,
                                                    nperseg=n_samp_per_seg, return_onesided=False)
rx_dist_at_point_freq_arr, rx_dist_at_point_psd = welch(distortion_sig_at_point, fs=psd_nfft, nfft=psd_nfft,
                                                    nperseg=n_samp_per_seg, return_onesided=False)
psd_at_angle_lst.append(psd_at_angle_desired)
psd_at_angle_lst.append(psd_at_angle_dist)

#psd_at_angle_lst.append(psd_at_angle_dist)
print("--- Computation time: %f ---" % (time.time() - start_time))

#%%
fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
plt.tight_layout()
ax1.set_theta_zero_location("E")
#normalize psd at angle to get antenna gain
#ant_gain = to_db(psd_at_angle/np.max(psd_at_angle))
for res_idx in range(2):
    if res_idx==0:
        ax1.plot(radian_vals, psd_at_angle_lst[res_idx], label="Desired")
    else:
        ax1.plot(radian_vals, psd_at_angle_lst[res_idx], label="Distortion")
ax1.set_title("Power spectral density at angle [dB]")
ax1.set_thetalim(-np.pi, np.pi)
ax1.set_xticks(np.pi/180. * np.linspace(180, -180, 24, endpoint=False))
ax1.grid(True)
ax1.legend(title="IBO = 3dB, Signal:", loc='lower left')
plt.savefig("figs/desired_vs_dist_beamplot_ntx_%d_ibo_%d_scspac_%dkHz.png" % (my_array.n_elements, ibo_db, int(my_tx.carrier_spacing/1e3)), dpi=600, bbox_inches='tight')
plt.show()

#%%
fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
plt.tight_layout()
ax1.set_theta_zero_location("E")
#normalize psd at angle to get antenna gain
#ant_gain = to_db(psd_at_angle/np.max(psd_at_angle))
ax1.plot(radian_vals, psd_at_angle_lst[0] - psd_at_angle_lst[1], label="SDR")
ax1.set_title("Signal to distortion ratio at angle [dB]")
ax1.set_thetalim(-np.pi, np.pi)
ax1.set_xticks(np.pi/180. * np.linspace(180, -180, 24, endpoint=False))
ax1.grid(True)
ax1.legend(title="IBO = %d dB, Signal:" % ibo_db, loc='lower left')
plt.savefig("figs/signal_to_dist_beamplot_ntx_%d_ibo_%d_scspac_%dkHz.png" % (my_array.n_elements, ibo_db, int(my_tx.carrier_spacing/1e3)), dpi=600, bbox_inches='tight')
plt.show()

print("Finished processing!")

#TODO: better names for saved figures - more configuration details

#%%
fig1, ax1 = plt.subplots(1, 1)
sorted_clean_rx_at_point_freq_arr, sorted_clean_psd_at_point_arr = zip(*sorted(zip(rx_clean_at_point_freq_arr, rx_clean_at_point_psd)))
ax1.plot(np.array(sorted_clean_rx_at_point_freq_arr), to_db(np.array(sorted_clean_psd_at_point_arr)), label="Desired")
sorted_dist_rx_at_point_freq_arr, sorted_dist_psd_at_point_arr = zip(*sorted(zip(rx_dist_at_point_freq_arr, rx_dist_at_point_psd)))
ax1.plot(np.array(sorted_dist_rx_at_point_freq_arr), to_db(np.array(sorted_dist_psd_at_point_arr)), label="Distorted")

ax1.set_title("Power spectral density at angle %d$\degree$" % point_idx_psd)
ax1.set_xlabel("Subcarrier index [-]")
ax1.set_ylabel("Power [dB]")
ax1.legend(title="IBO = %d db dB" %ibo_db)
ax1.grid()
plt.tight_layout()
plt.savefig("figs/psd_at_angle_%d_deg_ibo_%d_scspac_%dkHz.png" % (point_idx_psd, ibo_db, int(my_tx.carrier_spacing/1e3)), dpi=600, bbox_inches='tight')
plt.show()




