# antenna array evaluation
# %%
import os
import sys

sys.path.append(os.getcwd())

import copy

import matplotlib.pyplot as plt
import numpy as np

import distortion
import modulation
import transceiver
from plot_settings import set_latex_plot_style

set_latex_plot_style(use_tex=True, fig_width_in=5.89572, fig_height_in=3)

print("Multi antenna processing init!")
# remember to copy objects not to avoid shared properties modifications!
# check modifications before copy and what you copy!
my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=0, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=212,
                                cord_y=212, cord_z=1.5,
                                center_freq=int(3.5e9), carrier_spacing=int(15e3))
my_rx.correct_constellation()

# %%
n_ant_val = 8
# averaging length
n_ofdm_symb = 1e2
print("N antennas values:", n_ant_val)
ibo_arr = np.linspace(0.01, 10.0, 100)
print("IBO values:", ibo_arr)
lambda_per_ibo_analytical = []

# get analytical value of alpha
for ibo_idx, ibo_val_db in enumerate(ibo_arr):
    lambda_per_ibo_analytical.append(my_mod.calc_alpha(ibo_val_db))

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(ibo_arr, lambda_per_ibo_analytical, label="Analytical", linewidth=2)
ax1.set_xlim([-0.5, 10.5])
ax1.set_ylim([0.75, 1.05])
ax1.set_title(r"Alpha coefficient in regard to IBO")
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel(r"$\alpha$ [-]")
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax1.grid()

plt.tight_layout()
plt.savefig(
    "../figs/msc_figs/alpha_vs_ibo_analytical_ibo%1.1fto%1.1f.pdf" % (min(ibo_arr), max(ibo_arr)),
    dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
