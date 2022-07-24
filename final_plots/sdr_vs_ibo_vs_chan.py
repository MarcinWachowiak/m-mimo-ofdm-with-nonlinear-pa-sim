# antenna array evaluation
# %%
import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

import utilities
from plot_settings import set_latex_plot_style

set_latex_plot_style(use_tex=True, fig_width_in=3.5)

data_lst = utilities.read_from_csv(filename="sdr_vs_ibo_per_channel_ibo0to8_32nant")
ibo_arr = data_lst[0]
sdr_at_ibo_per_n_ant = data_lst[1:]

# %%
# plot signal to distortion ratio vs ibo
fig1, ax1 = plt.subplots(1, 1)
p1, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[0], 'o-', markevery=2, fillstyle="none", color='#377eb8', label="1")
p2, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[3], 'o-', markevery=4, fillstyle="none", color='#ff7f00', label="4")
p3, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[6], 'o-', markevery=6, fillstyle="none", color='#4daf4a', label="32")

# leg1 = ax1.legend([p1,p2,p3], n_ant_arr, loc=1, title="LOS:")
# plt.gca().add_artist(leg1)

p4, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[1], 's-', markevery=2, fillstyle="none", color='#377eb8', label="1")
p5, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[4], 's-', markevery=4, fillstyle="none", color='#ff7f00', label="4")
p6, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[7], 's-', markevery=6, fillstyle="none", color='#4daf4a', label="32")

# leg2 = ax1.legend([p4,p5,p6], n_ant_arr, loc=2, title="Two-Path:")
# plt.gca().add_artist(leg2)

p7, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[2], '*-', markevery=4, fillstyle="none", color='#377eb8', label="1")
p8, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[5], '*-', markevery=4, fillstyle="none", color='#ff7f00', label="4")
p9, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[8], '*-', markevery=4, fillstyle="none", color='#4daf4a', label="32")

# leg3 = ax1.legend([p7,p8,p9], n_ant_arr, loc=3, title="Rayleigh:")
# plt.gca().add_artist(leg3)

import matplotlib.patches as mpatches

n_ant1 = mpatches.Patch(color='#377eb8', label='1')
n_ant4 = mpatches.Patch(color='#ff7f00', label='4')
n_ant32 = mpatches.Patch(color='#4daf4a', label='32')

leg1 = plt.legend(handles=[n_ant1, n_ant4, n_ant32], title="N antennas:", loc="lower right")
plt.gca().add_artist(leg1)

import matplotlib.lines as mlines

los = mlines.Line2D([0], [0], linestyle='none', marker="o", fillstyle="none", color='k', label='LOS')
twopath = mlines.Line2D([0], [0], linestyle='none', marker="s", fillstyle="none", color='k', label='Two-path')
rayleigh = mlines.Line2D([0], [0], linestyle='none', marker="*", fillstyle="none", color='k', label='Rayleigh')
leg2 = plt.legend(handles=[los, twopath, rayleigh], title="Channels:", loc="upper left")
plt.gca().add_artist(leg2)
#
# p10, = ax1.plot([0], marker='None',
#            linestyle='None', label='dummy-tophead')
# p11, = ax1.plot([0],  marker='None',
#            linestyle='None', label='dummy-empty')
# p12, = ax1.plot([0],  marker='None',
#            linestyle='None', label='dummy-empty')
#
# leg = ax1.legend([p10, p11, p12, p10, p11, p12, p10, p11, p12, p1, p2, p3, p4, p5, p6, p7, p8, p9],
#               ["LOS:", '', '', "Two-path:", '', '', "Rayleigh:", '', ''] + n_ant_arr + n_ant_arr + n_ant_arr,
#               ncol=2) # Two columns, horizontal group labels

# leg= ax1.legend([p11, p1, p2, p3, p11, p4, p5, p6, p11, p7, p8, p9],
#               ["LOS"] + n_ant_arr + ["Two-path"] + n_ant_arr + ["Rayleigh"] + n_ant_arr,
#               loc="lower right", ncol=3, title="Channel and N antennas:", columnspacing=0.2) # Two columns, horizontal group labels
#
# %%
ax1.set_ylim([5.75, 55.75])
ax1.set_xlim([0, 8])

# ax1.set_title("SDR in regard to IBO")
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel("SDR [dB]")
ax1.grid()
plt.tight_layout()
plt.savefig("../figs/final_figs/sdr_vs_ibo_per_channel_ibo0to8_1_4_32nant.pdf", dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
