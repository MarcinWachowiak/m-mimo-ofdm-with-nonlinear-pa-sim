# Experimental OFDM simulation
# %%
import matplotlib.pyplot as plt
import plot_settings

from utilities import gen_tx_bits, count_mismatched_bits, snr_to_ebn0, ebn0_to_snr
import channels
import modulation
import numpy as np
from numpy import log2
import time

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
my_chan = channels.AwgnChannel(0, True, 1234)
snrs = ebn0_to_snr(np.arange(0, 21, 2), n_fft, n_sub_carr, constel_size)
print("SNR array:", snrs)
bers = np.zeros([len(snrs)])
ofdm_symbols = int(tot_bits / n_bits_per_ofdm_sym)

plot_psd = False

for idx, snr in enumerate(snrs):
    n_err = 0
    for _ in range(ofdm_symbols):
        my_chan.set_snr(snr)
        tx_bits = gen_tx_bits(n_bits_per_ofdm_sym)
        tx_symb = my_mod.modulate(tx_bits)
        tx_ofdm_symbol = modulation.tx_ofdm_symbol(tx_symb, n_fft, n_sub_carr, cyclic_prefix_len)

        rx_ofdm_symbol = my_chan.propagate(tx_ofdm_symbol, my_mod.avg_symbol_power, n_sub_carr, n_fft)

        if plot_psd:
            fig, (ax0, ax1) = plt.subplots(2, 1)
            ax0.psd(tx_ofdm_symbol, 1024)
            ax0.set_title("Before propagation")
            ax1.psd(rx_ofdm_symbol, 1024)
            ax1.set_title("After propagation")
            plt.show()

            plot_psd = False

        rx_symb = modulation.rx_ofdm_symbol(rx_ofdm_symbol, n_fft, n_sub_carr, cyclic_prefix_len)
        rx_bits = my_mod.demodulate(rx_symb)
        n_bit_err = count_mismatched_bits(tx_bits, rx_bits)

        n_err += n_bit_err

    bers[idx] = n_err / tot_bits

# %%
print("--- Computation time: %f ---" % (time.time() - start_time))

print("BER arr:", bers)
eb_per_n0_arr = snr_to_ebn0(snrs, n_fft, n_sub_carr, constel_size)
print("EB/N0 arr:", eb_per_n0_arr)
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.scatter(eb_per_n0_arr, bers, label='QAM')
#fix log scaling
ax.set_ylim([1e-8, 1])
ax.set_xlabel('Eb/N0 [dB]')
ax.set_ylabel("BER")
for idx, ebn0_val in enumerate(snrs):
    ax.text(ebn0_val, bers[idx], "{:.5e}".format(bers[idx]))
ax.grid()
ax.legend()
plt.show()
print("Finished exectution!")

