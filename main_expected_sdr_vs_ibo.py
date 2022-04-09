# antenna array evaluation
# %%
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import modulation
from plot_settings import set_latex_plot_style

# TODO: consider logger

set_latex_plot_style()
# %%
my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
# %%
ibo_vec = np.linspace(start=0.1, stop=10, num=100)
# print("IBO values:", ibo_vec)

# get analytical value of alpha
alpha_vec = np.empty(len(ibo_vec))
for ibo_idx, ibo_val_db in enumerate(ibo_vec):
    alpha_vec[ibo_idx] = my_mod.calc_alpha(ibo_val_db)
# %%
# calculate expected SDR
sdr_vec = 20*np.log10(alpha_vec**2/(1-alpha_vec**2))

# %%
fig1, ax1 = plt.subplots(1, 1)

# plot analytical
ax1.plot(ibo_vec, sdr_vec, label="Analytical SDR")
ax1.set_title("Expected signal to distortion ratio (SDR)")
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel("SDR [dB]")
ax1.grid()

plt.tight_layout()
plt.savefig(
    "figs/expected_sdr_ibo%1.1fto%1.1f.pdf" % (min(ibo_vec), max(ibo_vec)),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")

#%%
