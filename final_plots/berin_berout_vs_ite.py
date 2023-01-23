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
ibo_val_db = 3
constel_size = 64

ebn0_min = 5
ebn0_max = 20
ebn0_step = 1

my_miso_chan = "los"

cnc_filename_str = "ber_vs_ebn0_cnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
cnc_ebn0_arr = cnc_data_lst[0]
cnc_ber_per_dist = cnc_data_lst[1:]


mcnc_filename_str = "ber_vs_ebn0_mcnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
mcnc_ebn0_arr = mcnc_data_lst[0]
mcnc_ber_per_dist = mcnc_data_lst[1:]


# %%
# BER OUT vs BER IN PER SINGLE ITERATION
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log', base=10)
ax1.set_xscale('log', base=10)
ax1.set_aspect('equal')

CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                  '#CFCFCF']

# ber in as per standard detection corrected by alpha
color_idx = 1
for idx, cnc_iter_val in enumerate(cnc_n_iter_lst[1:]):
    if cnc_iter_val in sel_cnc_iter_val:
        # ber in vs ber out
        ax1.plot(cnc_ber_per_dist[idx+1][:], cnc_ber_per_dist[idx+2][:], "-", color=CB_color_cycle[color_idx])
        # ber in vs ber gain
        # ax1.plot(cnc_ber_per_dist[idx + 1][::2], np.array(cnc_ber_per_dist[idx + 1][::2]) / np.array(cnc_ber_per_dist[idx + 2][::2]), "-", color=CB_color_cycle[color_idx])

        color_idx += 1

color_idx = 1
for idx, mcnc_iter_val in enumerate(cnc_n_iter_lst[1:]):
    if mcnc_iter_val in sel_cnc_iter_val:
        # ber in vs ber out
        ax1.plot(mcnc_ber_per_dist[idx+1][:], mcnc_ber_per_dist[idx+2][:], "--", color=CB_color_cycle[color_idx])
        # ber in vs ber gain
        # ax1.plot(mcnc_ber_per_dist[idx+1][::2], np.array(mcnc_ber_per_dist[idx+1][::2]) / np.array(mcnc_ber_per_dist[idx+2][::2]), "--", color=CB_color_cycle[color_idx])

        color_idx += 1

import matplotlib.lines as mlines

n_ite_legend = []
# n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[0], label="No dist"))
# color_idx = 1
# for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
#     if ite_val in sel_cnc_iter_val:
#         n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[color_idx], label=ite_val))
#         color_idx += 1

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

# ax1.set_title("BER vs Eb/N0, %s, CNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (
#     my_miso_chan, constel_size, n_ant_val, ibo_val_db))
# ax1.set_xlim([10, 20])
# ax1.set_ylim([1e-5, 3e-1])

ax1.grid(which='major', linestyle='-')
# ax1.grid(which='minor', linestyle='--')
ax1.set_xlabel("BER in [-]")
ax1.set_ylabel("BER out [-]")
ax1.set_xlim([1e-4, 2e-1])
ax1.set_ylim([1e-4, 2e-1])

ax1.annotate('Greater SNR \n (lower noise)', xy=(1.1e-4, 5e-3),  xycoords='data',
            xytext=(1e-3, 5e-3), horizontalalignment="center", verticalalignment="center", textcoords='data',
            arrowprops=dict(facecolor='black' , arrowstyle='->'))

plt.tight_layout()

filename_str = "berout_vs_berin_per_single_ite_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ibo_val_db, min(cnc_ebn0_arr), max(cnc_ebn0_arr), cnc_ebn0_arr[1] - cnc_ebn0_arr[0],
    '_'.join([str(val) for val in sel_cnc_iter_val[:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()


# # %%
# # BER OUT VS BER IN PER TOTAL NUMBER OF ITERATIONS
# sel_cnc_iter_val = [1, 2, 3, 4, 5, 6, 7, 8]
#
# fig1, ax1 = plt.subplots(1, 1)
# ax1.set_yscale('log', base=10)
# # ax1.set_xscale('log', base=10)
#
# CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
#                   '#CFCFCF']
#
# # ber in as per standard detection corrected by alpha
# color_idx = 1
# for idx, cnc_iter_val in enumerate(cnc_n_iter_lst[1:]):
#     if cnc_iter_val in sel_cnc_iter_val:
#         # ber in vs ber out
#         ax1.plot(cnc_ber_per_dist[1][:], cnc_ber_per_dist[idx+2][:], "-", color=CB_color_cycle[color_idx])
#         # ber in vs ber gain
#         # ax1.plot(cnc_ber_per_dist[idx + 1][::2], np.array(cnc_ber_per_dist[idx + 1][::2]) / np.array(cnc_ber_per_dist[idx + 2][::2]), "-", color=CB_color_cycle[color_idx])
#
#         color_idx += 1
#
# color_idx = 1
# for idx, mcnc_iter_val in enumerate(cnc_n_iter_lst[1:]):
#     if mcnc_iter_val in sel_cnc_iter_val:
#         # ber in vs ber out
#         ax1.plot(mcnc_ber_per_dist[1][:], mcnc_ber_per_dist[idx+2][:], "--", color=CB_color_cycle[color_idx])
#         # ber in vs ber gain
#         # ax1.plot(mcnc_ber_per_dist[idx+1][::2], np.array(mcnc_ber_per_dist[idx+1][::2]) / np.array(mcnc_ber_per_dist[idx+2][::2]), "--", color=CB_color_cycle[color_idx])
#
#         color_idx += 1
#
# import matplotlib.lines as mlines
#
# n_ite_legend = []
# # n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[0], label="No dist"))
# # color_idx = 1
# # for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
# #     if ite_val in sel_cnc_iter_val:
# #         n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[color_idx], label=ite_val))
# #         color_idx += 1
#
# import matplotlib.patches as mpatches
#
# color_idx = 1
# for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
#     if ite_val in sel_cnc_iter_val:
#         n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
#         color_idx += 1
#
# leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="lower right", ncol=2, framealpha=0.9)
# plt.gca().add_artist(leg1)
#
# cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
# mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
# ax1.legend(handles=[cnc_leg, mcnc_leg], loc="center right", framealpha=0.9)
# # plt.gca().add_artist(leg2)
#
# # ax1.set_title("BER vs Eb/N0, %s, CNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (
# #     my_miso_chan, constel_size, n_ant_val, ibo_val_db))
# # ax1.set_xlim([10, 20])
# # ax1.set_ylim([1e-5, 3e-1])
#
# ax1.grid(which='major', linestyle='-')
# # ax1.grid(which='minor', linestyle='--')
# ax1.set_xlabel("BER in [-]")
# ax1.set_ylabel("BER out [-]")
# # ax1.set_xlim([8.3e-2, 1e-1])
# # ax1.set_ylim([1e-7, 5e-1])
#
# plt.tight_layout()
#
# filename_str = "berout_vs_berin_per_combined_ite_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
#     my_miso_chan, n_ant_val, ibo_val_db, min(cnc_ebn0_arr), max(cnc_ebn0_arr), cnc_ebn0_arr[1] - cnc_ebn0_arr[0],
#     '_'.join([str(val) for val in sel_cnc_iter_val[:]]))
# # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# # filename_str += "_" + timestamp
# plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
# plt.show()


print("Finished execution!")