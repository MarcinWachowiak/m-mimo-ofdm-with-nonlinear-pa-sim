# Experimental OFDM simulation
# %%
# #import modules
import matplotlib.pyplot as plt
#import plot_settings
import modulation
import utilities
import numpy as np


# %%
# generate random sequences of 0,1 of given length
def gen_tx_bits(len):
    return np.random.randint(0, 2, len)


my_mod = modulation.QAMModem(16)
# my_mod.plot_constellation()

dat_seq = gen_tx_bits(128)
# fig1, ax1 = plt.subplots()
# ax1.stem(dat_seq)
# ax1.set_title("Binary sequence")
# ax1.set_xlabel("Index")
# plt.show()

sym_seq = my_mod.modulate(dat_seq)
# fig2, ax2 = plt.subplots()
# ax2.stem(sym_seq.real, label="I", linefmt='C0-', markerfmt='C0o')
# ax2.stem(sym_seq.imag, label="Q", linefmt='C1-', markerfmt='C1D')
# ax2.set_title("Symbol sequence")
# ax2.set_xlabel("Index")
# ax2.legend()
# plt.show()

fft_size = 32
tx_seq = modulation.ofdm_tx(sym_seq, fft_size, fft_size, np.floor(0.2*fft_size))

fig3, ax3 = plt.subplots()
ax3.plot(tx_seq)
ax3.set_title("OFDM signal")
plt.show()

print("Finished exectution!")
