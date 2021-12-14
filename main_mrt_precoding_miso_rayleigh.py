# antenna array evaluation
# %%
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch

import antenna_arrray
import channel
import impairment
import modulation
import transceiver
from plot_settings import set_latex_plot_style
from utilities import to_db, pts_on_circum, pts_on_semicircum

# TODO: consider logger
np.set_printoptions(threshold=np.inf)

set_latex_plot_style()
# %%
print("Multi antenna processing init!")
bit_rng = np.random.default_rng(4321)

my_mod = modulation.OfdmQamModem(constel_size=256, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = impairment.SoftLimiter(ibo_db=5, avg_symb_pow=my_mod.ofdm_avg_sample_pow())
my_tx = transceiver.Transceiver(modem=my_mod, impairment=my_distortion, center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))

my_rx = transceiver.Transceiver(modem=my_mod, impairment=None, cord_x=30, cord_y=30, cord_z=1.5, center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx.modem.correct_constellation(my_tx.impairment.ibo_db)
my_miso_chan = channel.AwgnMisoLosTdFd(n_inputs=1, snr_db=10, is_complex=True)

# %%
plot_full_circle = False

if plot_full_circle:
    n_points = 360
    rx_points = pts_on_circum(r=300, n=n_points)
else:
    n_points = 180
    rx_points = pts_on_semicircum(r=300, n=n_points)

radian_vals = np.radians(np.linspace(0, n_points, n_points + 1))

psd_nfft = 4096
n_samp_per_seg = 1024
n_snapshots = 10

# %%
# plot PSD for chosen point/angle
point_idx_psd = 50
n_ant_vec = [1, 2, 4, 8]
#precode for a single point
precoding_point_idx = 45

desired_psd_at_angle_lst = []
distortion_psd_at_angle_lst = []
rx_sig_at_point_clean = []
rx_sig_at_point_full = []

for n_ant in n_ant_vec:
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    my_array = antenna_arrray.LinearArray(n_elements=n_ant, transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5,
                                          cord_x=0, cord_y=0, cord_z=15)
    my_miso_chan = channel.RayleighMisoTdFd(n_inputs=my_array.n_elements, fd_samp_size=my_tx.modem.n_fft, snr_db=10,
                                        is_complex=True, seed=1234)
    my_rx.set_position(cord_x=212, cord_y=212, cord_z=1.5)
    chan_mat_at_point = my_miso_chan.get_channel_coeffs()
    my_array.set_precoding_single_point(rx_transceiver=my_rx, channel_fd_mat=chan_mat_at_point)

    psd_at_angle_desired = np.empty(radian_vals.shape)
    psd_at_angle_dist = np.empty(radian_vals.shape)
    for pt_idx, point in enumerate(rx_points):
        # generate different channel for each point
        # precode only for single known point
        if pt_idx == precoding_point_idx:
            my_miso_chan.set_channel_coeffs(chan_mat_at_point)
        else:
            my_miso_chan.reroll_channel_coeffs()

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

            # calculate PSD at point for the last value of N antennas
            if pt_idx == point_idx_psd and n_ant == n_ant_vec[-1]:
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

    desired_psd_at_angle_lst.append(psd_at_angle_desired)
    distortion_psd_at_angle_lst.append(psd_at_angle_dist)
    print("--- Computation time: %f ---" % (time.time() - start_time))

rx_sig_at_point_clean_arr = np.concatenate(rx_sig_at_point_clean).ravel()
rx_sig_at_point_full_arr = np.concatenate(rx_sig_at_point_full).ravel()
distortion_sig_at_point = rx_sig_at_point_full_arr - my_rx.modem.alpha * rx_sig_at_point_clean_arr

rx_clean_at_point_freq_arr, rx_clean_at_point_psd = welch(rx_sig_at_point_clean_arr, fs=psd_nfft, nfft=psd_nfft,
                                                          nperseg=n_samp_per_seg, return_onesided=False)
rx_dist_at_point_freq_arr, rx_dist_at_point_psd = welch(distortion_sig_at_point, fs=psd_nfft, nfft=psd_nfft,
                                                            nperseg=n_samp_per_seg, return_onesided=False)

# %%
# plot beampatterns of desired signal
fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
plt.tight_layout()
ax1.set_theta_zero_location("E")

if plot_full_circle:
    ax1.set_thetalim(-np.pi, np.pi)
    ax1.set_xticks(np.pi / 180. * np.linspace(180, -180, 24, endpoint=False))
else:
    ax1.set_thetalim(0, np.pi)
    ax1.set_xticks(np.pi / 180. * np.linspace(0, 180, 12, endpoint=False))

dist_lines_lst = []
for idx, n_ant in enumerate(n_ant_vec):
    ax1.plot(radian_vals, desired_psd_at_angle_lst[idx], label=n_ant, linewidth=1.5)
ax1.set_title("Desired signal PSD at angle [dB]")
ax1.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center')
ax1.grid(True)
plt.savefig("figs/desired_signal_beampattern_ibo%d_%dto%dant_sweep.png" %(my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
plt.show()

# %%
# plot beampatterns of distortion signal
fig2, ax2 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
plt.tight_layout()
ax2.set_theta_zero_location("E")
if plot_full_circle:
    ax2.set_thetalim(-np.pi, np.pi)
    ax2.set_xticks(np.pi / 180. * np.linspace(180, -180, 24, endpoint=False))
else:
    ax2.set_thetalim(0, np.pi)
    ax2.set_xticks(np.pi / 180. * np.linspace(0, 180, 12, endpoint=False))

dist_lines_lst = []
for idx, n_ant in enumerate(n_ant_vec):
    ax2.plot(radian_vals, distortion_psd_at_angle_lst[idx], label=n_ant, linewidth=1.5)
ax2.set_title("Distortion signal PSD at angle [dB]")
ax2.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center')
ax2.grid(True)
plt.savefig("figs/distortion_signal_beampattern_ibo%d_%dto%dant_sweep.png" %(my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
plt.show()

# %%
# Desired vs distortion PSD beampattern comparison for given number of antennas
fig3, ax3 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
plt.tight_layout()
ax3.set_theta_zero_location("E")
if plot_full_circle:
    ax3.set_thetalim(-np.pi, np.pi)
    ax3.set_xticks(np.pi / 180. * np.linspace(180, -180, 24, endpoint=False))
else:
    ax3.set_thetalim(0, np.pi)
    ax3.set_xticks(np.pi / 180. * np.linspace(0, 180, 12, endpoint=False))

dist_lines_lst = []
# select index to plot
sel_idx = 3
ax3.plot(radian_vals, desired_psd_at_angle_lst[sel_idx], label="Desired", linewidth=1.5)
ax3.plot(radian_vals, distortion_psd_at_angle_lst[sel_idx], label="Distortion", linewidth=1.5)
ax3.set_title("Power spectral density at angle [dB]")
ax3.legend(title="N antennas = %d, signals:" % n_ant_vec[sel_idx], ncol=2, loc='lower center')
ax3.grid(True)
plt.savefig("figs/desired_vs_distortion_beampattern_ibo%d_%dant.png" %(my_tx.impairment.ibo_db, np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
plt.show()

# %%
# plot signal to distortion ratio polar
fig4, ax4 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
plt.tight_layout()
ax4.set_theta_zero_location("E")
if plot_full_circle:
    ax4.set_thetalim(-np.pi, np.pi)
    ax4.set_xticks(np.pi / 180. * np.linspace(180, -180, 24, endpoint=False))
else:
    ax4.set_thetalim(0, np.pi)
    ax4.set_xticks(np.pi / 180. * np.linspace(0, 180, 12, endpoint=False))

for idx, n_ant in enumerate(n_ant_vec):
    ax4.plot(radian_vals, desired_psd_at_angle_lst[idx] - distortion_psd_at_angle_lst[idx], label=n_ant, linewidth=1.5)
ax4.set_title("Signal to distortion ratio at angle [dB]")
ax4.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center')
ax4.grid(True)
plt.savefig("figs/sdr_at_angle_polar_ibo%d_%dto%dant_sweep.png" %(my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
plt.show()

# %%
# plot signal to distortion ratio cartesian 
fig5, ax5 = plt.subplots(1, 1)
plt.tight_layout()
for idx, n_ant in enumerate(n_ant_vec):
    ax5.plot(np.degrees(radian_vals), desired_psd_at_angle_lst[idx] - distortion_psd_at_angle_lst[idx], label=n_ant, linewidth=1.5)

ax5.set_xlim([30, 60])
ax5.set_xlabel("Angle [$\degree$]")
ax5.set_ylabel("Signal to distortion ratio [dB]")

ax5.set_title("Signal to distortion ratio at angle")
ax5.legend(title="N antennas:")
ax5.grid(True)
plt.savefig("figs/sdr_at_angle_cartesian_ibo%d_%dto%dant_sweep.png" %(my_tx.impairment.ibo_db, np.min(n_ant_vec), np.max(n_ant_vec)), dpi=600, bbox_inches='tight')
plt.show()


# %%
# plot PSD at selected point/angle
fig6, ax6 = plt.subplots(1, 1)
sorted_clean_rx_at_point_freq_arr, sorted_clean_psd_at_point_arr = zip(
    *sorted(zip(rx_clean_at_point_freq_arr, rx_clean_at_point_psd)))
ax6.plot(np.array(sorted_clean_rx_at_point_freq_arr), to_db(np.array(sorted_clean_psd_at_point_arr)), label="Desired")
sorted_dist_rx_at_point_freq_arr, sorted_dist_psd_at_point_arr = zip(
    *sorted(zip(rx_dist_at_point_freq_arr, rx_dist_at_point_psd)))
ax6.plot(np.array(sorted_dist_rx_at_point_freq_arr), to_db(np.array(sorted_dist_psd_at_point_arr)), label="Distorted")

ax6.set_title("Power spectral density at angle %d$\degree$" % point_idx_psd)
ax6.set_xlabel("Subcarrier index [-]")
ax6.set_ylabel("Power [dB]")
ax6.legend(title="IBO = %d [dB]" % my_tx.impairment.ibo_db)
ax6.grid()
plt.tight_layout()
plt.savefig("figs/psd_at_angle_%ddeg_ibo%d_ant%d.png" % (point_idx_psd, my_tx.impairment.ibo_db, np.max(n_ant_vec)), dpi=600,
            bbox_inches='tight')
plt.show()

print("Finished processing!")
