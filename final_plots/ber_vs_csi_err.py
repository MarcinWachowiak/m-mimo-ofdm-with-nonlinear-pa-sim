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

set_latex_plot_style(use_tex=True, fig_width_in=3.5)

cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
n_ant_val = 64
constel_size = 64

my_miso_chan = "los"
ibo_val_db = 0

ebn0_min = 5
ebn0_max = 20
ebn0_step = 1

csi_eps_lst = [0.01, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
sel_eps_lst = [0.1, 0.2, 0.3]

sel_cnc_ite_lst = [2, 5, 8]

cnc_los_bers_per_eps = []
mcnc_los_bers_per_eps = []
ebno_lst = []


for eps_idx, eps_val in enumerate(csi_eps_lst):
    cnc_filename_str = "ber_vs_ebn0_cnc_%s_csi_eps%1.3f_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
        my_miso_chan, eps_val, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
    cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
    cnc_ebn0_arr = cnc_data_lst[0]
    cnc_ber_per_dist = cnc_data_lst[1:]

    cnc_los_bers_per_eps.append(cnc_ber_per_dist)
    ebno_lst.append(cnc_ebn0_arr)

    mcnc_filename_str = "ber_vs_ebn0_mcnc_%s_csi_eps%1.3f_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
        my_miso_chan, eps_val, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
    mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
    mcnc_ebn0_arr = mcnc_data_lst[0]
    mcnc_ber_per_dist = mcnc_data_lst[1:]

    mcnc_los_bers_per_eps.append(mcnc_ber_per_dist)
    ebno_lst.append(mcnc_ebn0_arr)

my_miso_chan_quadriga = "quadriga"
cnc_quadriga_bers_per_eps = []
mcnc_quadriga_bers_per_eps = []
ebno_quadriga_lst = []
quadriga_csi_eps_lst = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

for eps_idx, eps_val in enumerate(quadriga_csi_eps_lst):
    cnc_quadriga_filename_str = "ber_vs_ebn0_cnc_%s_csi_eps%1.3f_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
        my_miso_chan_quadriga, eps_val, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
    cnc_quadriga_data_lst = utilities.read_from_csv(filename=cnc_quadriga_filename_str)
    cnc_quadriga_ebn0_arr = cnc_quadriga_data_lst[0]
    cnc_quadriga_ber_per_dist = cnc_quadriga_data_lst[1:]

    cnc_quadriga_bers_per_eps.append(cnc_quadriga_ber_per_dist)
    ebno_quadriga_lst.append(cnc_quadriga_ebn0_arr)

    mcnc_quadriga_filename_str = "ber_vs_ebn0_mcnc_%s_csi_eps%1.3f_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
        my_miso_chan_quadriga, eps_val, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
    mcnc_quadriga_data_lst = utilities.read_from_csv(filename=mcnc_quadriga_filename_str)
    mcnc_quadriga_ebn0_arr = mcnc_quadriga_data_lst[0]
    mcnc_quadriga_ber_per_dist = mcnc_quadriga_data_lst[1:]

    mcnc_quadriga_bers_per_eps.append(mcnc_quadriga_ber_per_dist)
    ebno_quadriga_lst.append(mcnc_quadriga_ebn0_arr)

#%%
# BER VS Eb/N0
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log', base=10)

CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                  '#CFCFCF']

marker_lst = ['o', '^', '*', 'v', 'D']
marker_idx = 0
for eps_idx, eps_val in enumerate(csi_eps_lst):
    color_idx = 1
    ax1.plot(ebno_lst[0], cnc_los_bers_per_eps[eps_idx][0], "-", color=CB_color_cycle[0],
             marker=marker_lst[marker_idx], fillstyle='none')
    if eps_val in sel_eps_lst:
        for ite_idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
            if cnc_iter_val == 0:
                color_idx += 1
                continue
            if cnc_iter_val in sel_cnc_ite_lst:
                ax1.plot(ebno_lst[0], cnc_los_bers_per_eps[eps_idx][ite_idx + 1], "-", color=CB_color_cycle[color_idx], marker=marker_lst[marker_idx], fillstyle='none')
                ax1.plot(ebno_lst[0], mcnc_los_bers_per_eps[eps_idx][ite_idx + 1], "--", color=CB_color_cycle[color_idx],
                     marker=marker_lst[marker_idx], fillstyle='none')
            if cnc_iter_val in sel_cnc_ite_lst or cnc_iter_val == 1:
                color_idx += 1
        marker_idx += 1

import matplotlib.patches as mpatches
n_ite_legend = []
n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[0], label="No dist"))
color_idx = 2
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val in sel_cnc_ite_lst:
        n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
    if ite_val in sel_cnc_ite_lst or ite_val == 1:
        color_idx += 1

leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="lower left", ncol=1, framealpha=0.9)
plt.gca().add_artist(leg1)

import matplotlib.lines as mlines

marker_leg_lst = []
marker_idx = 0
for eps_idx, eps_val in enumerate(sel_eps_lst):
    if eps_val in sel_eps_lst:
        marker_leg_lst.append(mlines.Line2D([0], [0], linestyle='none', marker=marker_lst[marker_idx], fillstyle="none", color='k', label=eps_val))
        marker_idx += 1
leg2 = plt.legend(handles=marker_leg_lst, title=r"$\varepsilon$:", loc="lower center", framealpha=0.9)
plt.gca().add_artist(leg2)


cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
ax1.legend(handles=[cnc_leg, mcnc_leg], loc="lower right", framealpha=0.9)


ax1.grid(which='major', linestyle='-')
# ax1.grid(which='minor', linestyle='--')
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("BER")
plt.tight_layout()
# filename_str = "ber_vs_ebn0_%s_csi_eps%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
#     my_miso_chan, '_'.join([str(val) for val in sel_eps_lst]), n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
#     '_'.join([str(val) for val in sel_cnc_ite_lst])
# )
# plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
# plt.show()

#%%
# BER OUT vs BER IN
set_latex_plot_style(use_tex=True, fig_width_in=3.5)

sel_eps_lst = [0.1, 0.2, 0.3, 0.4]
sel_cnc_ite_lst = [5]

fig2, ax2 = plt.subplots(1, 1)
ax2.set_yscale('log', base=10)
ax2.set_xscale('log', base=10)

# LOS channel
marker_idx = 0
color_idx = 1
for eps_idx, eps_val in enumerate(csi_eps_lst):
    if eps_val in sel_eps_lst:
        for ite_idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
            if cnc_iter_val in sel_cnc_ite_lst:
                ax2.plot(cnc_los_bers_per_eps[eps_idx][1], cnc_los_bers_per_eps[eps_idx][ite_idx + 1], "-", color=CB_color_cycle[color_idx], marker=marker_lst[0], fillstyle='none', markersize=4)
                ax2.plot(mcnc_los_bers_per_eps[eps_idx][1], mcnc_los_bers_per_eps[eps_idx][ite_idx + 1], "--", color=CB_color_cycle[color_idx], marker=marker_lst[0], fillstyle='none', markersize=4)
        color_idx += 1
# Quadriga LOS channel
marker_idx = 0
color_idx = 1

for eps_idx, eps_val in enumerate(quadriga_csi_eps_lst):
    if eps_val in sel_eps_lst:
        for ite_idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
            if cnc_iter_val in sel_cnc_ite_lst:
                ax2.plot(cnc_quadriga_bers_per_eps[eps_idx][1], cnc_quadriga_bers_per_eps[eps_idx][ite_idx + 1], "-", color=CB_color_cycle[color_idx], marker=marker_lst[1], fillstyle='none', markersize=4)
                ax2.plot(mcnc_quadriga_bers_per_eps[eps_idx][1], mcnc_quadriga_bers_per_eps[eps_idx][ite_idx + 1], "--", color=CB_color_cycle[color_idx], marker=marker_lst[1], fillstyle='none', markersize=4)
        color_idx += 1

marker_leg_lst = []
marker_leg_lst.append(mlines.Line2D([0], [0], linestyle='none', marker=marker_lst[0], fillstyle="none", color='k', label="LOS"))
marker_leg_lst.append(mlines.Line2D([0], [0], linestyle='none', marker=marker_lst[1], fillstyle="none", color='k', label="38.901 LOS"))
leg2 = plt.legend(handles=marker_leg_lst, title="Channel:", loc="lower center", framealpha=0.9)
plt.gca().add_artist(leg2)

import matplotlib.patches as mpatches
color_idx = 1
eps_leg_lst = []
for eps_idx, eps_val in enumerate(csi_eps_lst):
    if eps_val in sel_eps_lst:
        eps_leg_lst.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=eps_val))
        color_idx += 1
leg1 = plt.legend(handles=eps_leg_lst, title=r"$\varepsilon$:", loc="lower right", ncol=1, framealpha=0.9)
plt.gca().add_artist(leg1)




cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
ax2.legend(handles=[cnc_leg, mcnc_leg], loc="lower center", framealpha=0.9)

ax2.grid(which='major', linestyle='-')

ax2.set_xlabel("BER in [-]")
ax2.set_ylabel("BER out [-]")
ax2.set_xlim([5e-2, 0.15])
ax2.set_ylim([1e-4, 0.5])

pt1 = ax2.get_ylim()
pt2 = ax2.get_xlim()
ax2.plot(np.linspace(pt1[0], pt2[1]), np.linspace(pt1[0], pt2[1]), color='k', linestyle=':', linewidth=1)

# leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="lower right", ncol=1, framealpha=0.9)
# plt.gca().add_artist(leg1)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
no_ber_gain = mlines.Line2D([0], [0], linestyle=':', color='k', label='No gain')

ax2.legend(handles=[cnc_leg, mcnc_leg, no_ber_gain], loc="lower left", framealpha=0.9)

plt.tight_layout()

filename_str = "berout_vs_berin_per_combined_ite_csi_eps%s_nant%d_ebn0_min%d_max%d_step%1.2f_niter%d" % (
    '_'.join([str(val) for val in sel_eps_lst[:]]), n_ant_val, ebn0_min, ebn0_max, ebn0_step, sel_cnc_ite_lst[0])
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
