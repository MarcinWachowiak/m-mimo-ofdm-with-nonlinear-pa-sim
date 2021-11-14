# Experimental OFDM simulation
# %%
import matplotlib.pyplot as plt

import impairments
import plot_settings

from utilities import gen_tx_bits, count_mismatched_bits, snr_to_ebn0, ebn0_to_snr, to_db
import channels
import modulation
import numpy as np
from numpy import log2
import time
from scipy.signal import welch
# %%
start_time = time.time()

constel_size = 16
n_sub_carr = 800
n_fft = 1024
cyclic_prefix_len = int(0.45 * n_fft)
n_bits_per_ofdm_sym = int(log2(constel_size) * n_sub_carr)

tot_bits = n_bits_per_ofdm_sym * 1000
tot_err = 0

my_mod = modulation.QamModem(constel_size)
# my_mod.plot_constellation()

my_limiter1 = impairments.SoftLimiter(-1, my_mod.avg_symbol_power)
# my_limiter1.plot_characteristics()

my_limiter2 = impairments.Rapp(-1, my_mod.avg_symbol_power, 5)
# my_limiter2.plot_characteristics()

my_limiter3 = impairments.ThirdOrderNonLin(15, my_mod.avg_symbol_power)
# my_limiter3.plot_characteristics()

my_chan = channels.AwgnChannel(0, True, 1234)
snrs = ebn0_to_snr(np.arange(0, 21, 2), n_fft, n_sub_carr, constel_size)
print("SNR array:", snrs)
bers = np.zeros([len(snrs)])
ofdm_symbols = int(tot_bits / n_bits_per_ofdm_sym)

plot_psd = True
n_colected_snapshots = 100
tx_sig_arr_for_psd = []
rx_sig_arr_for_psd = []

snapshot_counter = 0

# %%
for idx, snr in enumerate(snrs):
    n_err = 0
    for _ in range(ofdm_symbols):
        my_chan.set_snr(snr)
        tx_bits = gen_tx_bits(n_bits_per_ofdm_sym)
        tx_symb = my_mod.modulate(tx_bits)
        clean_ofdm_symbol = modulation.tx_ofdm_symbol(tx_symb, n_fft, n_sub_carr, cyclic_prefix_len)

        tx_ofdm_symbol = my_limiter1.process(clean_ofdm_symbol)
        rx_ofdm_symbol = my_chan.propagate(tx_ofdm_symbol, my_mod.avg_symbol_power, n_sub_carr, n_fft)

        if plot_psd and snapshot_counter < n_colected_snapshots:
            tx_sig_arr_for_psd.append(clean_ofdm_symbol)
            rx_sig_arr_for_psd.append(tx_ofdm_symbol)
            snapshot_counter += 1

        rx_symb = modulation.rx_ofdm_symbol(rx_ofdm_symbol, n_fft, n_sub_carr, cyclic_prefix_len)
        rx_bits = my_mod.demodulate(rx_symb)
        n_bit_err = count_mismatched_bits(tx_bits, rx_bits)

        n_err += n_bit_err

    bers[idx] = n_err / tot_bits

# %%
print("--- Computation time: %f ---" % (time.time() - start_time))
flat_tx_sig_for_psd = np.concatenate(tx_sig_arr_for_psd).ravel()
flat_rx_sig_for_psd = np.concatenate(rx_sig_arr_for_psd).ravel()

psd_nfft = 1024
n_samp_per_seg = 512

tx_freq_arr, tx_psd_arr = welch(flat_tx_sig_for_psd, fs=psd_nfft, nfft=psd_nfft, nperseg=n_samp_per_seg,
                                return_onesided=False)
tx_freqs, tx_sig_psd = zip(*sorted(zip(tx_freq_arr, tx_psd_arr)))

rx_freq_arr, rx_psd_arr = welch(flat_rx_sig_for_psd, fs=psd_nfft, nfft=psd_nfft, nperseg=n_samp_per_seg,
                                return_onesided=False)
rx_freqs, rx_sig_psd = zip(*sorted(zip(rx_freq_arr, rx_psd_arr)))

fig1, ax0 = plt.subplots(1, 1)
ax0.plot(tx_freqs, to_db(np.array(tx_sig_psd)))
ax0.set_title("Before propagation")
ax0.set_xlabel("Subcarrier index [-]")
ax0.set_ylabel("Power [dB]")

ax0.plot(rx_freqs, to_db(np.array(rx_sig_psd)))
ax0.set_title("After propagation")
ax0.set_xlabel("Subcarrier index [-]")
ax0.set_ylabel("Power [dB]")
ax0.grid()

plt.show()

print("BER arr:", bers)
eb_per_n0_arr = snr_to_ebn0(snrs, n_fft, n_sub_carr, constel_size)
print("EB/N0 arr:", eb_per_n0_arr)
fig2, ax = plt.subplots()
ax.set_yscale('log')
ax.scatter(eb_per_n0_arr, bers, label='QAM')
# fix log scaling
ax.set_ylim([1e-8, 1])
ax.set_xlabel('Eb/N0 [dB]')
ax.set_ylabel("BER")
for idx, ebn0_val in enumerate(snrs):
    ax.text(ebn0_val, bers[idx], "{:.5e}".format(bers[idx]))
ax.grid()
ax.legend()
plt.show()
print("Finished exectution!")
