# antenna array evaluation
# %%
import os
import sys

import plot_settings

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

import utilities
from plot_settings import set_latex_plot_style

set_latex_plot_style(use_tex=True, fig_width_in=3.5)

cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sel_cnc_iter_val = [0, 1, 2, 5]

n_ant_val = 64
ibo_val_db = 0
constel_size = 64

ebn0_step = 0.25
my_miso_chan = "los"

code_rate_str_lst = ["1/3",  "2/3"]
ebn0_bounds_arr = [[-5.0, 5.1], [0.0, 15.1]]

ebn0_arr_lst = []
cnc_ber_per_ldpc_lst = []
mcnc_ber_per_ldpc_lst = []
for code_idx, code_rate_str in enumerate(code_rate_str_lst):
    cnc_filename_str = "ldpc_%s_ber_vs_ebn0_cnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
        code_rate_str.replace('/', '_'), my_miso_chan, n_ant_val, ibo_val_db, ebn0_bounds_arr[code_idx][0], ebn0_bounds_arr[code_idx][1], ebn0_step,
        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
    cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
    cnc_ebn0_arr = cnc_data_lst[0]
    cnc_ber_per_dist = cnc_data_lst[1:]

    ebn0_arr_lst.append(cnc_ebn0_arr)
    cnc_ber_per_ldpc_lst.append(cnc_ber_per_dist)

    mcnc_filename_str = "ldpc_%s_ber_vs_ebn0_mcnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
        code_rate_str.replace('/', '_'), my_miso_chan, n_ant_val, ibo_val_db, ebn0_bounds_arr[code_idx][0], ebn0_bounds_arr[code_idx][1], ebn0_step,
        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
    mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
    mcnc_ebn0_arr = mcnc_data_lst[0]
    mcnc_ber_per_dist = mcnc_data_lst[1:]
    mcnc_ber_per_ldpc_lst.append(mcnc_ber_per_dist)


# %%

fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log', base=10)

CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                  '#CFCFCF']
code_marker = ['*', '^']
for code_idx, code_rate_str in enumerate(code_rate_str_lst):
    color_idx = 1
    ax1.plot(ebn0_arr_lst[code_idx], cnc_ber_per_ldpc_lst[code_idx][0])
    for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
        if cnc_iter_val in sel_cnc_iter_val:
            if cnc_iter_val == 1:
                ax1.plot(ebn0_arr_lst[code_idx][::2], cnc_ber_per_ldpc_lst[code_idx][idx + 1][::2], "-", color=CB_color_cycle[color_idx])
            else:
                ax1.plot(ebn0_arr_lst[code_idx], cnc_ber_per_ldpc_lst[code_idx][idx + 1], "-", color=CB_color_cycle[color_idx])
        if cnc_iter_val in sel_cnc_iter_val or cnc_iter_val == 1:
            color_idx += 1
    plot_settings.reset_color_cycle()

    color_idx = 1
    for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
        if cnc_iter_val == 0:
            color_idx += 1
            continue

        if cnc_iter_val in sel_cnc_iter_val:
            if cnc_iter_val == 1:
                ax1.plot(ebn0_arr_lst[code_idx][::2], mcnc_ber_per_ldpc_lst[code_idx][idx + 1][::2], "--",
                         color=CB_color_cycle[color_idx])
            else:
                ax1.plot(ebn0_arr_lst[code_idx], mcnc_ber_per_ldpc_lst[code_idx][idx + 1], "--", color=CB_color_cycle[color_idx])
        if cnc_iter_val in sel_cnc_iter_val or cnc_iter_val == 1:
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

n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[0], label="No dist"))
color_idx = 1
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val in sel_cnc_iter_val:
        n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
    if ite_val in sel_cnc_iter_val or ite_val == 1:
        color_idx += 1

leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="lower left", ncol=1, framealpha=0.9)
plt.gca().add_artist(leg1)

ax1.text(0.54, 0.7, '1/3', verticalalignment='center', horizontalalignment='center', transform=ax1.transAxes, fontsize=11)
ax1.text(0.90, 0.7, '2/3', verticalalignment='center', horizontalalignment='center', transform=ax1.transAxes, fontsize=11)

# code_leg_lst = []
# for code_idx, code_rate_str in enumerate(code_rate_str_lst):
#     leg_obj = mlines.Line2D([0], [0], linestyle='none', marker=code_marker[code_idx], fillstyle="none", color='k', label=code_rate_str)
#     code_leg_lst.append(leg_obj)
#
# leg2 = plt.legend(handles=code_leg_lst, title="LDPC:", loc="upper left", framealpha=0.9, bbox_to_anchor=(0.004, 0.80))
# plt.gca().add_artist(leg2)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
ax1.legend(handles=[cnc_leg, mcnc_leg], loc="upper left", framealpha=0.9, bbox_to_anchor=(0.000, 0.92))


# cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
# mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
# ax1.legend(handles=[cnc_leg, mcnc_leg], loc="upper left", framealpha=0.9, bbox_to_anchor=(0.68, 0.69))
# plt.gca().add_artist(leg2)

# ax1.set_title("BER vs Eb/N0, %s, CNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (
#     my_miso_chan, constel_size, n_ant_val, ibo_val_db))
ax1.set_xlim([-10, 15])
ax1.set_ylim([1e-5, 5e-1])

ax1.grid(which='major', linestyle='-')
# ax1.grid(which='minor', linestyle='--')
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("BER")
plt.tight_layout()

filename_str = "ldpc_ber_vs_ebn0_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ibo_val_db, min(cnc_ebn0_arr), max(cnc_ebn0_arr), cnc_ebn0_arr[1] - cnc_ebn0_arr[0],
    '_'.join([str(val) for val in sel_cnc_iter_val[1:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/%s.pdf" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
