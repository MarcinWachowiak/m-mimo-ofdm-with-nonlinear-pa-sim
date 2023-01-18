# antenna array evaluation
# %%
import os
import sys

import plot_settings

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np

import utilities
from plot_settings import set_latex_plot_style

set_latex_plot_style(use_tex=False, fig_width_in=5)

cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sel_cnc_iter_val = [1, 2, 3, 4, 5, 6, 7, 8]

n_ant_val = 64
ebn0_db = 15
constel_size = 64

ibo_min = 0
ibo_max = 8
ibo_step = 0.5

my_miso_chan = "los"

cnc_filename_str = "ber_vs_ibo_cnc_%s_nant%d_ebn0_%d_ibo_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ebn0_db, ibo_min, ibo_max, ibo_step,
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
cnc_ibo_arr = cnc_data_lst[0]
cnc_bers_per_ibo = cnc_data_lst[1:]

# cnc_ber_per_ite = np.array(np.transpose(np.vstack(cnc_bers_per_ibo)))

mcnc_filename_str = "ber_vs_ibo_mcnc_%s_nant%d_ebn0_%d_ibo_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ebn0_db, ibo_min, ibo_max, ibo_step,
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
mcnc_ibo_arr = mcnc_data_lst[0]
mcnc_bers_per_ibo = mcnc_data_lst[1:]

# mcnc_ber_per_ite = np.array(np.transpose(np.vstack(mcnc_bers_per_ibo)))



# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log', base=10)
ax1.set_xscale('log', base=10)
ax1.set_aspect('equal')

CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                  '#CFCFCF']

color_idx = 1
for ite_idx, ite_val in enumerate(cnc_n_iter_lst[1:]):
    if ite_val in sel_cnc_iter_val:
        ax1.plot(cnc_bers_per_ibo[ite_idx], cnc_bers_per_ibo[ite_idx+1], "-", color=CB_color_cycle[color_idx])
        color_idx += 1
plot_settings.reset_color_cycle()

color_idx = 1
for ite_idx, ite_val in enumerate(cnc_n_iter_lst[1:]):
    if ite_val in sel_cnc_iter_val:
        ax1.plot(mcnc_bers_per_ibo[ite_idx], mcnc_bers_per_ibo[ite_idx+1], "--", color=CB_color_cycle[color_idx])  # , dashes=(5, 1 + ite_idx))
        color_idx += 1
import matplotlib.lines as mlines

n_ite_legend = []
# color_idx = 0
# for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
#     if ite_val in sel_cnc_iter_val:
#         n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[color_idx], label=ite_val))
#         color_idx += 1
# leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="upper right", ncol=1, framealpha=0.9)

import matplotlib.patches as mpatches

color_idx = 1
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val in sel_cnc_iter_val:
        n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
        color_idx += 1

leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="upper left", ncol=2, framealpha=0.9)
plt.gca().add_artist(leg1)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
ax1.legend(handles=[cnc_leg, mcnc_leg], loc="lower right", framealpha=0.9)
# plt.gca().add_artist(leg2)

ax1.set_xlabel("BER in [-]")
ax1.set_ylabel("BER out [-]")
ax1.set_xlim([6e-4, 1e-1])
ax1.set_ylim([6e-4, 1e-1])
ax1.grid(which='major', linestyle='-')
ax1.grid(which='minor', linestyle='--')


ax1.annotate('Greater IBO \n (lower distortion)', xy=(6e-4, 1e-2),  xycoords='data',
            xytext=(4e-3, 1e-2), horizontalalignment="center", verticalalignment="center", textcoords='data',
            arrowprops=dict(facecolor='black' , arrowstyle='->'))

plt.tight_layout()

filename_str = "berout_vs_berin_per_single_ite_%s_nant%d_ebn0%d_ibo_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ebn0_db, ibo_min, ibo_max, ibo_step,
    '_'.join([str(val) for val in sel_cnc_iter_val[:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
