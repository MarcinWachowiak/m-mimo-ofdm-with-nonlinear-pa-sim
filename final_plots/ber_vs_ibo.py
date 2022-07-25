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

data_lst_cnc = utilities.read_from_csv(
    filename="ber_vs_ibo_cnc_two_path_nant1_ebn0_20_ibo_min0_max10_step1.00_niter1_2_3_5_8")
ibo_arr_cnc = data_lst_cnc[0]
bers_per_ibo_cnc = data_lst_cnc[1:]

data_lst_mcnc = utilities.read_from_csv(
    filename="ber_vs_ibo_mcnc_two_path_nant1_ebn0_20_ibo_min0_max10_step1.00_niter1_2_3_5_8")
ibo_arr_mcnc = data_lst_mcnc[0]
bers_per_ibo_mcnc = data_lst_mcnc[1:]

cnc_n_iter_lst = [0, 1, 2, 3, 5, 8]
n_ant_val = 1
ibo_val_db = 0
ebn0_db = 20
constel_size = 64
my_miso_chan = "Two-Path"

# %%
# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log')
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    ax1.plot(ibo_arr_cnc, bers_per_ibo_cnc[ite_idx], "-")

plot_settings.reset_color_cycle()

for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    ax1.plot(ibo_arr_mcnc, bers_per_ibo_mcnc[ite_idx], "--")

import matplotlib.lines as mlines

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
n_ite_legend = []
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[ite_idx], label=ite_val))
leg1 = plt.legend(handles=n_ite_legend, title="N iterations:", loc="upper right", ncol=3)
plt.gca().add_artist(leg1)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
leg2 = plt.legend(handles=[cnc_leg, mcnc_leg], loc="lower right")
plt.gca().add_artist(leg2)

ax1.set_title("BER vs IBO, %s, CNC, QAM %d, N ANT = %d, Eb/n0 = %d [dB], " % (
    my_miso_chan, constel_size, n_ant_val, ebn0_db))
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel("BER")
ax1.grid()
plt.tight_layout()

filename_str = "ber_vs_ibo_cnc_%s_nant%d_ebn0_%d_ibo_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ebn0_db, min(ibo_arr_cnc), max(ibo_arr_cnc), ibo_arr_cnc[1] - ibo_arr_cnc[0],
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/final_figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
