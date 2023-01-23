import os
import sys

sys.path.append(os.getcwd())

import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

from plot_settings import set_latex_plot_style

set_latex_plot_style(use_tex=False, fig_width_in=5)

meng = matlab.engine.start_matlab()
meng.addpath(r"C:\Users\rkotrys\Documents\GitHub\mimo-simulation-py\main_quadriga_channel")
meng.rng(2137)

n_ant = 64
n_sub_carr = 4096
subcarr_spacing = 15e3
center_freq = 3.5e9
distance = 300
bandwidth = n_sub_carr * subcarr_spacing
channel_model_str = '3GPP_38.901_UMa_LOS'

meng.qd_channel_env_setup(meng.double(n_ant), meng.double(n_sub_carr), meng.double(subcarr_spacing), meng.double(center_freq), meng.double(distance), channel_model_str, nargout=0)

channel_mat_tmp = np.array(meng.qd_get_channel_mat(distance, 0, 1.5))

#%%
fig1, ax1 = plt.subplots(1, 1)
for idx in range(5):
    # despite no position change the channel is different
    channel_mat = np.array(meng.qd_get_channel_mat(distance, 0, 1.5))
    ax1.plot(10*np.log10(np.abs(channel_mat[idx, :])))

ax1.set_xlabel("Subcarrier index [-]")
ax1.set_ylabel("Channel attenuation [dB]")
ax1.set_xlim([0, n_sub_carr])

plt.grid()
plt.tight_layout()
plt.savefig("figs/quadriga_channel.png", dpi=600, bbox_inches='tight')
plt.show()