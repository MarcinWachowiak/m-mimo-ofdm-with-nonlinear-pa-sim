# antenna array evaluation
# %%
import os
import sys

from matplotlib.ticker import MaxNLocator

sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import utilities
import antenna_arrray
import channel
import distortion
import modulation
import transceiver
from plot_settings import set_latex_plot_style


set_latex_plot_style(use_tex=True, fig_width_in=3.5)
# %%

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=2048, cp_len=128)
# %%
data_lst = utilities.read_from_csv(filename="alpha_vs_tx_power_per_ant64_ibo0.0")
ibo_per_tx = [data_lst[0], data_lst[2], data_lst[4]]
alpha_per_nant_per_ibo = [data_lst[1], data_lst[3], data_lst[5]]


# %%
fig1, ax1 = plt.subplots(1, 1)

ibo_range = np.linspace(np.min(ibo_per_tx), np.max(ibo_per_tx), 100)
alpha_analytical = my_mod.calc_alpha(ibo_db=ibo_range)
ax1.plot(ibo_per_tx[0], alpha_per_nant_per_ibo[0], '.', label="Rayleigh")
ax1.plot(ibo_per_tx[1], alpha_per_nant_per_ibo[1], '.', label="Two-path")
ax1.plot(ibo_per_tx[2], alpha_per_nant_per_ibo[2], '.', label="LOS")
ax1.plot(ibo_range, alpha_analytical, '--k', label="Analytical", alpha=0.7)

ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.xaxis.set_major_locator(MaxNLocator(6))
ax1.set_yticks([0.755, 0.765, 0.775, 0.785])
ax1.set_ylim([0.752, 0.788])
# ax1.set_title(r"$IBO_k$ and $\alpha_k$ spread, $K$ = 64, $\overline{IBO}$ = 0 [dB]")
ax1.set_xlabel(r"$\mathrm{IBO_k}$ [dB]")
ax1.set_ylabel(r"$\mathrm{\alpha_k}$ [-]")
ax1.grid()
ax1.legend(title="Channel:", loc="lower right")

plt.tight_layout()
plt.savefig("../figs/final_figs/alpha_vs_ibo_per_ant64_ibo0.0.pdf", dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
