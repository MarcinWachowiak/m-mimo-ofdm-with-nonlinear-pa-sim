# Experimental OFDM simulation
# %%
import matplotlib.pyplot as plt
from utilities import gen_tx_bits, count_mismatched_bits
import channels
# import plot_settings
import modulation
import numpy as np


# %%
my_mod = modulation.QamModem(16)
my_chan = channels.AwgnChannel(0, True)
# my_mod.plot_constellation()

tx_bits = gen_tx_bits(64 * 4)
tx_symb = my_mod.modulate(tx_bits)

fig, ax = plt.subplots()
ax.scatter(tx_symb.real, tx_symb.imag)
plt.show()

# OFDM params
n_sub_carr = 64
cyclic_prefix_len = int(64 * 0.25)

tx_ofdm_symbol = modulation.tx_ofdm_symbol(tx_symb, n_sub_carr, cyclic_prefix_len)

rx_ofdm_symbol = my_chan.propagate(tx_ofdm_symbol)

rx_symb = modulation.rx_ofdm_symbol(rx_ofdm_symbol, n_sub_carr, cyclic_prefix_len)

fig, ax = plt.subplots()
ax.scatter(rx_symb.real, rx_symb.imag)
plt.show()

#fig, ax = plt.subplots()
#ax.plot(tx_ofdm_symbol)
#ax.plot(rx_ofdm_symbol)
#plt.show()

rx_bits = my_mod.demodulate(rx_symb)

n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
print("Number of bit errors: %d" % n_bit_err)

print("Finished exectution!")
