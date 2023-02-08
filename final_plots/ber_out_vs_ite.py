# antenna array evaluation
# %%
import os
import sys

import plot_settings

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt, ticker as mticker

import utilities
from plot_settings import set_latex_plot_style

set_latex_plot_style(use_tex=True, fig_width_in=4)

cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7]
n_ant_val = 64
constel_size = 64

my_miso_chan = "los"
ibo_min = -9
ibo_max = 9
ibo_step = 0.5

cnc_bers_per_snr_lst = []
cnc_ibo_array_lst = []
mcnc_bers_per_snr_lst = []
mcnc_ibo_array_lst = []
no_dist_ber_limit = []
snr_lst = [12, 15, 18, 1000]
sel_snr_val_lst = [15, 1000]

ibo_val_lst = np.arange(ibo_min, ibo_max + 0.1, ibo_step)
ibo_sel_lst = [-3, 0, 1.5, 3]

for snr_idx, snr_val in enumerate(snr_lst):
    cnc_filename_str = "ber_vs_ibo_cnc_los_nant64_ebn0_%d_ibo_min%d_max%d_step%1.2f_niter1_2_3_4_5_6_7_8" % (
    snr_val, ibo_min, ibo_max, ibo_step)
    cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
    cnc_ibo_arr = cnc_data_lst[0]
    snr_ber_limit = np.average(cnc_data_lst[1])
    cnc_bers_per_ibo = cnc_data_lst[1:]

    cnc_bers_per_snr_lst.append(np.transpose(np.array((np.vstack(cnc_bers_per_ibo)))))
    cnc_ibo_array_lst.append(cnc_ibo_arr)
    no_dist_ber_limit.append(snr_ber_limit)

    mcnc_filename_str = "ber_vs_ibo_mcnc_los_nant64_ebn0_%d_ibo_min%d_max%d_step%1.2f_niter1_2_3_4_5_6_7_8" % (
    snr_val, ibo_min, ibo_max, ibo_step)
    mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
    mcnc_ibo_arr = mcnc_data_lst[0]
    # snr_ber_limit = np.average(cnc_data_lst[1])
    mcnc_bers_per_ibo = mcnc_data_lst[1:]
    mcnc_bers_per_snr_lst.append(np.transpose(np.array((np.vstack(mcnc_bers_per_ibo)))))
    mcnc_ibo_array_lst.append(mcnc_ibo_arr)
    # no_dist_ber_limit.append(snr_ber_limit)

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log', base=10)
# ax1.set_xscale('log', base=10)
# ax1.set_aspect('equal')

CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                  '#CFCFCF']

for snr_idx, snr_val in enumerate(snr_lst):
    color_idx = 2
    if snr_val in sel_snr_val_lst:
        for ibo_idx, ibo_val in enumerate(ibo_val_lst):
            if ibo_val in ibo_sel_lst:
                ax1.plot(cnc_n_iter_lst[1:], cnc_bers_per_snr_lst[snr_idx][ibo_idx][1:-1] / cnc_bers_per_snr_lst[snr_idx][ibo_idx][2:], "-",
                         color=CB_color_cycle[color_idx])
                ax1.plot(cnc_n_iter_lst[1:], mcnc_bers_per_snr_lst[snr_idx][ibo_idx][1:-1] / mcnc_bers_per_snr_lst[snr_idx][ibo_idx][2:], "--",
                         color=CB_color_cycle[color_idx])
                color_idx += 1
    plot_settings.reset_color_cycle()

import matplotlib.lines as mlines

n_ite_legend = []
# color_idx = 0
# for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
#     if ite_val in sel_cnc_iter_val:
#         n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[color_idx], label=ite_val))
#         color_idx += 1
# leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="upper right", ncol=1, framealpha=0.9)

import matplotlib.patches as mpatches

color_idx = 2
for ibo_idx, ibo_val in enumerate(ibo_sel_lst):
    n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ibo_val))
    color_idx += 1
leg1 = plt.legend(handles=n_ite_legend, title="IBO [dB]", loc="upper right", ncol=1, framealpha=0.9)
plt.gca().add_artist(leg1)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')

ax1.legend(handles=[cnc_leg, mcnc_leg], loc="upper center", framealpha=0.9, ncol=1)
# plt.gca().add_artist(leg2)

ax1.set_xlabel("I iterations [-]")
ax1.set_ylabel("BER out [-]")
ax1.set_xlim([1, 7])
# ax1.set_ylim([1e-5, 4e-1])
# ax1.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
# ax1.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

pt1 = ax1.get_ylim()
pt2 = ax1.get_xlim()
# ax1.plot([pt1[0], pt2[1]], [pt1[0], pt2[1]], color='k', linestyle=':', linewidth=1, label="No gain")
# ax1.text(0.2, 0.4, 'Eb/N0\n=15 [dB]', verticalalignment='center', horizontalalignment='center', transform=ax1.transAxes, fontsize=8)
# ax1.text(0.86, 0.1, 'Eb/N0\n=$\infty$ [dB]', verticalalignment='center', horizontalalignment='center', transform=ax1.transAxes, fontsize=8)


# for snr_idx, snr_val in enumerate(snr_lst):
#     if snr_val in sel_snr_val_lst:
#         ax1.scatter(no_dist_ber_limit[snr_idx], no_dist_ber_limit[snr_idx], color=CB_color_cycle[0], marker='o', zorder=3)

ax1.grid(which='major', linestyle='-')
# ax1.grid(which='minor', linestyle='--')

ax1.annotate('Greater Eb/No \n', xy=(4, 1e-4),  xycoords='data',
            xytext=(4, 1e-2), horizontalalignment="center", verticalalignment="center", textcoords='data',
            arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.tight_layout()

filename_str = "ber_gain_vs_ite_%s_nant%d_ebn0%s_ibo_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, '_'.join([str(val) for val in sel_snr_val_lst[:]]), ibo_min, ibo_max, ibo_step,
    '_'.join([str(val) for val in cnc_n_iter_lst[:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
