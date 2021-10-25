# Experimental OFDM simulation
# %%
import matplotlib.pyplot as plt
from utilities import gen_tx_bits, count_mismatched_bits
import channels
# import plot_settings
import modulation
import numpy as np
from numpy import log2

# %%
# my_mod.plot_constellation()
constel_size = 16
n_sub_carr = 64
cyclic_prefix_len = int(64 * 0.25)
n_bits_per_ofdm_sym = int(log2(constel_size) * n_sub_carr)

tot_bits = n_bits_per_ofdm_sym * 1000
tot_err = 0

snrs = np.arange(0, 22, 2)
print("SNR array:", snrs)
bers = np.zeros([len(snrs)])

ofdm_symbols = int(tot_bits / n_bits_per_ofdm_sym)

my_mod = modulation.QamModem(constel_size)
my_chan = channels.AwgnChannel(0, True)

for idx, snr in enumerate(snrs):
    my_chan.set_snr(snr)
    n_err = 0
    for _ in range(ofdm_symbols):
        tx_bits = gen_tx_bits(n_bits_per_ofdm_sym)
        tx_symb = my_mod.modulate(tx_bits)
        tx_ofdm_symbol = modulation.tx_ofdm_symbol(tx_symb, n_sub_carr, cyclic_prefix_len)
        rx_ofdm_symbol = my_chan.propagate(tx_ofdm_symbol)
        rx_symb = modulation.rx_ofdm_symbol(rx_ofdm_symbol, n_sub_carr, cyclic_prefix_len)
        rx_bits = my_mod.demodulate(rx_symb)
        n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
        n_err += n_bit_err
    bers[idx] = n_err / tot_bits

# %%

print("BER arr:", bers)

fig, ax = plt.subplots()
ax.scatter(snrs, bers, label='QAM16')
ax.set_yscale('log')
ax.set_xlabel('SNR [dB]')
ax.set_ylabel("BER")
ax.grid()
ax.legend()
plt.show()

print("Finished exectution!")
