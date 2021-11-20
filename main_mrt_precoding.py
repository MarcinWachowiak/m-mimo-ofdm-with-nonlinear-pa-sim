# antenna array evaluation
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import time

from utilities import count_mismatched_bits, snr_to_ebn0, ebn0_to_snr, to_db, points_on_circumference, signal_power
import channels
import modulation
import impairments
import transceiver
import antenna_arrray

from plot_settings import set_latex_plot_style

set_latex_plot_style()

print("Multi antenna processing init!")

my_mod = modulation.OfdmQamModem(constel_size=16, n_fft=4096, n_sub_carr=1024, cp_len=256)

my_distortion = impairments.SoftLimiter(ibo_db=3, avg_symb_pow=my_mod.ofdm_avg_sample_pow())
my_tx = transceiver.Transceiver(modem=my_mod, impairment=my_distortion)
my_array = antenna_arrray.LinearArray(n_elements=5, transceiver=my_tx, center_freq=0, wav_len_spacing=0.5)
bit_rng = np.random.default_rng(4321)

my_rx = transceiver.Transceiver(modem=my_mod, impairment=None, cord_x=10, cord_y=100)
my_array.set_precoding_single_point(rx_transceiver=my_rx, exact=True)
# my_array.plot_configuration()

my_miso_chan = channels.AwgnMisoChannel(n_inputs=5, snr_db=10, is_complex=True)

# bit_rng = np.random.default_rng(4321)
# tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
# arr_tx_sig = my_array.transmit(in_bits=tx_bits)
# rx_sig = my_miso_chan.propagate(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, in_sig=arr_tx_sig, skip_noise=True)
#

#%%
# for a single carrier frequency plot the TX characteristics
n_points = 360
rx_points = points_on_circumference(r=100, n=n_points)
radian_vals = np.radians(np.linspace(0, 359, n_points))
psd_at_angle = np.empty(radian_vals.shape)

psd_nfft = 4096
n_samp_per_seg = 512
n_snapshots = 10

#%%
start_time = time.time()

for pt_idx, point in enumerate(rx_points):
    (x_cord, y_cord) = point
    my_rx.set_position(cord_x=x_cord, cord_y=y_cord, cord_z=0)
    rx_sig_accum = []
    for snap_idx in range(n_snapshots):

        tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
        arr_tx_sig = my_array.transmit(in_bits=tx_bits)
        rx_sig = my_miso_chan.propagate(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, in_sig=arr_tx_sig, skip_noise=True)
        rx_sig_accum.append(rx_sig)

    rx_sig_freq_arr, rx_sig_psd_arr = welch(np.concatenate(rx_sig_accum).ravel(), fs=psd_nfft, nfft=psd_nfft,
                                          nperseg=n_samp_per_seg,
                                          return_onesided=False)
    rx_sig_freq = np.array(rx_sig_freq_arr[0:len(rx_sig_freq_arr) // 2])
    rx_sig_psd = to_db((np.array(rx_sig_psd_arr[0:len(rx_sig_psd_arr) // 2])))
    psd_at_angle[pt_idx] = rx_sig_psd[256]  # chose some frequency

print("--- Computation time: %f ---" % (time.time() - start_time))

#%%
fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
plt.tight_layout()
ax1.set_theta_zero_location("N")
ax1.plot(radian_vals, psd_at_angle, label="Test")
ax1.set_title("Power spectral density at angle")
ax1.grid(True)
#plt.savefig("figs/psd_toi.pdf", dpi=600, bbox_inches='tight')
plt.show()



# fig1, ax1 = plt.subplots(1, 1)
# ax1.plot(rx_sig_freq, rx_sig_psd, label="Test")
# ax1.set_title("Power spectral density")
# ax1.set_xlabel("Subcarrier index [-]")
# ax1.set_ylabel("Power [dB]")
# ax1.legend(title="IBO [dB]")
# ax1.grid()
# plt.tight_layout()
# #plt.savefig("figs/psd_toi.pdf", dpi=600, bbox_inches='tight')
# plt.show()




