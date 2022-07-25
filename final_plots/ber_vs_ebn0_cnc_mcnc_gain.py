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

set_latex_plot_style(use_tex=False, fig_width_in=3.5)

cnc_n_iter_lst = [0, 1, 2, 3, 5, 8]
n_ant_val = 1
ibo_val_db = 0
constel_size = 64
my_miso_chan = "rayleigh"

data_lst_cnc = utilities.read_from_csv(
    filename="ber_vs_ebn0_cnc_%s_nant1_ibo0_ebn0_min0_max30_step1.00_niter2_3_5_8" % my_miso_chan)
ebn0_arr_cnc = data_lst_cnc[0]
ber_per_dist_cnc = data_lst_cnc[1:]

data_lst_mcnc = utilities.read_from_csv(
    filename="ber_vs_ebn0_mcnc_%s_nant1_ibo0_ebn0_min0_max30_step1.00_niter2_3_5_8" % my_miso_chan)
ebn0_arr_mcnc = data_lst_mcnc[0]
ber_per_dist_mcnc = data_lst_mcnc[1:]

# %%
fig1, ax1 = plt.subplots(1, 1)
# ax1.plot(ebn0_arr_cnc, ber_per_dist_cnc[0], label="No distortion")
for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
    ax1.plot(ebn0_arr_cnc, np.array(ber_per_dist_cnc[1]) / np.array(ber_per_dist_cnc[idx + 1]), "-")

plot_settings.reset_color_cycle()

for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
    ax1.plot(ebn0_arr_cnc, np.array(ber_per_dist_mcnc[1]) / np.array(ber_per_dist_mcnc[idx + 1]), "--")

import matplotlib.lines as mlines

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
n_ite_legend = []
# n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[0], label="Cln"))
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[ite_idx + 1], label=ite_val))

leg1 = plt.legend(handles=n_ite_legend, title="N iterations:", loc="lower left", ncol=3)
plt.gca().add_artist(leg1)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
leg2 = plt.legend(handles=[cnc_leg, mcnc_leg], loc="upper left")
plt.gca().add_artist(leg2)

# fix log scaling
ax1.set_yscale('log')
ax1.set_title("BER vs Eb/N0, %s, CNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (
    my_miso_chan, constel_size, n_ant_val, ibo_val_db))
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("BER")
ax1.grid()
plt.tight_layout()

filename_str = "ber_vs_ebn0_mcnc_cnc_gain_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
    my_miso_chan, n_ant_val, ibo_val_db, min(ebn0_arr_cnc), max(ebn0_arr_cnc), ebn0_arr_cnc[1] - ebn0_arr_cnc[0],
    '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/final_figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
