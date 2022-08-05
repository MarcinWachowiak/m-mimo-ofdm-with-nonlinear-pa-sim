# antenna array evaluation
# %%
import os
import sys

import plot_settings

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

import utilities
from plot_settings import set_latex_plot_style

set_latex_plot_style(use_tex=False, fig_width_in=3.5)

cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sel_cnc_iter_val = [0, 2, 5, 8]

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

mcnc_filename_str = "ber_vs_ibo_mcnc_%s_nant%d_ebn0_%d_ibo_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ebn0_db, ibo_min, ibo_max, ibo_step,
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
mcnc_ibo_arr = mcnc_data_lst[0]
mcnc_bers_per_ibo = mcnc_data_lst[1:]

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log')
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val in sel_cnc_iter_val:
        ax1.plot(cnc_ibo_arr, cnc_bers_per_ibo[ite_idx], "-")

plot_settings.reset_color_cycle()

for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val in sel_cnc_iter_val:
        ax1.plot(mcnc_ibo_arr, mcnc_bers_per_ibo[ite_idx], "--")

import matplotlib.lines as mlines

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
n_ite_legend = []
# color_idx = 0
# for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
#     if ite_val in sel_cnc_iter_val:
#         n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[color_idx], label=ite_val))
#         color_idx += 1
# leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="upper right", ncol=1, framealpha=0.9)

import matplotlib.patches as mpatches

color_idx = 0
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val in sel_cnc_iter_val:
        n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
        color_idx += 1

leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="upper right", ncol=1, framealpha=0.9)
plt.gca().add_artist(leg1)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
ax1.legend(handles=[cnc_leg, mcnc_leg], loc="upper center", framealpha=0.9)
# plt.gca().add_artist(leg2)

ax1.set_xlim([0, 8])
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel("BER")
ax1.grid(which='major', linestyle='-')
ax1.grid(which='minor', linestyle='--')
plt.tight_layout()

filename_str = "ber_vs_ibo_%s_nant%d_ebn0_%d_ibo_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ebn0_db, min(cnc_ibo_arr), max(cnc_ibo_arr), cnc_ibo_arr[1] - cnc_ibo_arr[0],
    '_'.join([str(val) for val in sel_cnc_iter_val[1:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/final_figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
