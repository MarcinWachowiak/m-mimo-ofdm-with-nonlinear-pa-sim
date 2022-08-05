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
ibo_val_db = 0
constel_size = 64

ebn0_min = 5
ebn0_max = 20
ebn0_step = 0.5

my_miso_chan = "los"

cnc_filename_str = "ber_vs_ebn0_cnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
cnc_ebn0_arr = cnc_data_lst[0]
cnc_ber_per_dist = cnc_data_lst[1:]

# mcnc_filename_str = "ber_vs_ebn0_mcnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
#     my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
#     '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
# mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
# mcnc_ebn0_arr = mcnc_data_lst[0]
# mcnc_ber_per_dist = mcnc_data_lst[1:]

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log', base=10)

ax1.plot(cnc_ebn0_arr, cnc_ber_per_dist[0])
for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
    if cnc_iter_val in sel_cnc_iter_val:
        ax1.plot(cnc_ebn0_arr, cnc_ber_per_dist[idx + 1], "-")

plot_settings.reset_color_cycle()

# ax1.plot(mcnc_ebn0_arr, mcnc_ber_per_dist[0], "--")
# for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
#     if cnc_iter_val in sel_cnc_iter_val:
#         ax1.plot(mcnc_ebn0_arr, mcnc_ber_per_dist[idx + 1], "--")

import matplotlib.lines as mlines

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

n_ite_legend = []
# n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[0], label="No dist"))
# color_idx = 1
# for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
#     if ite_val in sel_cnc_iter_val:
#         n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[color_idx], label=ite_val))
#         color_idx += 1

import matplotlib.patches as mpatches

n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[0], label="No dist"))
color_idx = 1
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val in sel_cnc_iter_val:
        n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
        color_idx += 1

leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="lower left", ncol=1, framealpha=0.9)
plt.gca().add_artist(leg1)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
ax1.legend(handles=[cnc_leg, mcnc_leg], loc="lower center", framealpha=0.9)
# plt.gca().add_artist(leg2)

# ax1.set_title("BER vs Eb/N0, %s, CNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (
#     my_miso_chan, constel_size, n_ant_val, ibo_val_db))
ax1.set_xlim([10, 20])
ax1.set_ylim([1e-6, 3e-1])

ax1.grid(which='major', linestyle='-')
# ax1.grid(which='minor', linestyle='--')
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("BER")
plt.tight_layout()

filename_str = "ber_vs_ebn0_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ibo_val_db, min(cnc_ebn0_arr), max(cnc_ebn0_arr), cnc_ebn0_arr[1] - cnc_ebn0_arr[0],
    '_'.join([str(val) for val in sel_cnc_iter_val[1:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/final_figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
