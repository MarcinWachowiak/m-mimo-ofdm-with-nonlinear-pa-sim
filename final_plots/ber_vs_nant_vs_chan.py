# antenna array evaluation
# %%
import os
import sys

import plot_settings

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import utilities
from plot_settings import set_latex_plot_style
import channel
set_latex_plot_style(use_tex=True, fig_width_in=3.5, fig_height_in=3.5)

n_ant_arr = [1, 2, 4, 8, 16, 32, 64, 128]

cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sel_cnc_iter_val = [0, 2, 5, 8]

chan_lst = ['los', 'two_path', 'rayleigh']

ebn0_db = 15
ibo_val_db = 0
constel_size = 64

cnc_filename_str = "ber_vs_nant_cnc_nant%s_ebn0_%d_ibo%d_niter%s" % (
    '_'.join([str(val) for val in n_ant_arr]), ebn0_db, ibo_val_db,
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
cnc_n_ant_arr = cnc_data_lst[0]
cnc_bers_per_chan_per_nite_per_n_ant = cnc_data_lst[1:]

mcnc_filename_str = "ber_vs_nant_mcnc_nant%s_ebn0_%d_ibo%d_niter%s" % (
    '_'.join([str(val) for val in n_ant_arr]), ebn0_db, ibo_val_db,
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
mcnc_n_ant_arr = mcnc_data_lst[0]
mcnc_bers_per_chan_per_nite_per_n_ant = mcnc_data_lst[1:]

cnc_n_iter_lst = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
sel_cnc_iter_val = [-1, 0, 2, 5, 8]
# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=10)
ax1.set_xticks(n_ant_arr)
ax1.set_xticklabels(n_ant_arr)

CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                  '#CFCFCF']

cnc_chan_linestyles = ['o-', 's-', '*-']

for chan_idx, chan_obj in enumerate(chan_lst):
    color_idx = 1
    for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
        if ite_val == -1:
            ax1.plot(cnc_n_ant_arr, cnc_bers_per_chan_per_nite_per_n_ant[0 + chan_idx * len(cnc_n_iter_lst)],
                     cnc_chan_linestyles[chan_idx], fillstyle="none", label="No dist", color=CB_color_cycle[0])
            continue
        if ite_val == 1:
            color_idx += 1
        if ite_val in sel_cnc_iter_val:
            ax1.plot(cnc_n_ant_arr, cnc_bers_per_chan_per_nite_per_n_ant[ite_idx + chan_idx * len(cnc_n_iter_lst)],
                     cnc_chan_linestyles[chan_idx], fillstyle="none", label=ite_val, color=CB_color_cycle[color_idx])
            color_idx += 1
plot_settings.reset_color_cycle()

mcnc_chan_linestyles = ['o--', 's--', '*--']
for chan_idx, chan_obj in enumerate(chan_lst):
    color_idx = 1
    for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
        if ite_val == -1:
            # ax1.plot(mcnc_n_ant_arr, mcnc_bers_per_chan_per_nite_per_n_ant[0 + chan_idx * (len(cnc_n_iter_lst))],
            #          mcnc_chan_linestyles[chan_idx], fillstyle="none", label="No dist", color=CB_color_cycle[0])
            continue
        if ite_val == 1:
            color_idx += 1
        if ite_val in sel_cnc_iter_val:
            # if not (chan_idx == 2 and ite_val == 0):
            if isinstance(chan_obj, channel.MisoRayleighFd):
                ax1.plot(mcnc_n_ant_arr,
                         mcnc_bers_per_chan_per_nite_per_n_ant[ite_idx + chan_idx * (len(cnc_n_iter_lst))],
                         mcnc_chan_linestyles[chan_idx], fillstyle="none", label=ite_val,
                         color=CB_color_cycle[color_idx], dashes=(5, 1 + color_idx))
            else:
                ax1.plot(mcnc_n_ant_arr,
                         mcnc_bers_per_chan_per_nite_per_n_ant[ite_idx + chan_idx * (len(cnc_n_iter_lst))],
                         mcnc_chan_linestyles[chan_idx], fillstyle="none", label=ite_val,
                         color=CB_color_cycle[color_idx])
            color_idx += 1

plot_settings.reset_color_cycle()
# ax1.set_title("BER vs N ant, CNC, QAM %d, IBO = %d [dB], Eb/n0 = %d [dB], " % (constel_size, ibo_val_db, ebn0_db))
ax1.set_xlim([1, 128])
ax1.set_xlabel("K antennas [-]")
ax1.set_ylabel("BER")
ax1.grid(which='major', linestyle='-')
ax1.grid(which='minor', linestyle='--')

n_ite_legend = []
# color_idx = 0
# for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
#     if ite_val in sel_cnc_iter_val:
#         n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[color_idx], label=ite_val))
#         color_idx += 1

import matplotlib.patches as mpatches

cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sel_cnc_iter_val = [0, 2, 5, 8]
n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[0], label="No dist"))
color_idx = 1
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val == 1:
        color_idx += 1
    if ite_val in sel_cnc_iter_val:
        n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
        color_idx += 1

leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="upper center", ncol=1, framealpha=0.9,
                  bbox_to_anchor=(0.5 + 0.02, -0.15))
plt.gca().add_artist(leg1)

import matplotlib.lines as mlines

los = mlines.Line2D([0], [0], linestyle='none', marker="o", fillstyle="none", color='k', label='LOS')
twopath = mlines.Line2D([0], [0], linestyle='none', marker="s", fillstyle="none", color='k', label='Two-path')
rayleigh = mlines.Line2D([0], [0], linestyle='none', marker="*", fillstyle="none", color='k', label='Rayleigh')
leg2 = plt.legend(handles=[los, twopath, rayleigh], title="Channels:", loc="upper left", framealpha=0.9,
                  bbox_to_anchor=(-0.05 + 0.02, -0.15))
plt.gca().add_artist(leg2)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
ax1.legend(handles=[cnc_leg, mcnc_leg], loc="upper left", framealpha=0.9, bbox_to_anchor=(0.68 + 0.01, -0.15))
# plt.gca().add_artist(leg3)


plt.tight_layout()
# %%
filename_str = "ber_vs_nant_nant%s_ebn0_%d_ibo%d_niter%s" % (
    '_'.join([str(val) for val in n_ant_arr]), ebn0_db, ibo_val_db,
    '_'.join([str(val) for val in sel_cnc_iter_val[1:]]))

plt.savefig("../figs/final_figs/%s.pdf" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
