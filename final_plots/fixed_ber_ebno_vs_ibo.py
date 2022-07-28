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
target_ber_val = 1e-2
n_ant_val = 1
constel_size = 64
ebn0_db_arr = np.arange(10, 31, 1.0)
my_miso_chan = "los"

data_lst_cnc = utilities.read_from_csv(
    filename="fixed_ber1.0e-02_cnc_%s_nant1_ebn0_min14_max30_step0.50_ibo_min0_max7_step0.50_niter1_2_3_5_8" % my_miso_chan)
ibo_arr_cnc = data_lst_cnc[0]
req_ebn0_per_ibo_cnc = data_lst_cnc[1:]

data_lst_mcnc = utilities.read_from_csv(
    filename="fixed_ber1.0e-02_mcnc_%s_nant1_ebn0_min14_max30_step0.50_ibo_min0_max7_step0.50_niter1_2_3_5_8" % my_miso_chan)
ibo_arr_mcnc = data_lst_mcnc[0]
req_ebn0_per_ibo_mcnc = data_lst_mcnc[1:]

# %%
fig1, ax1 = plt.subplots(1, 1)
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    ax1.plot(ibo_arr_cnc, req_ebn0_per_ibo_cnc[ite_idx], "-")

plot_settings.reset_color_cycle()

for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    ax1.plot(ibo_arr_mcnc, req_ebn0_per_ibo_mcnc[ite_idx], "--")

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

ax1.set_title(
    "Fixed BER = %1.1e, %s, CNC, QAM %d, N ANT = %d" % (target_ber_val, my_miso_chan, constel_size, n_ant_val))
ax1.set_xlabel("IBO [dB]")
ax1.set_ylabel("Eb/n0 [dB]")
ax1.grid()
plt.tight_layout()

filename_str = "fixed_ber%1.1e_cnc_%s_nant%d_ebn0_min%d_max%d_step%1.2f_ibo_min%d_max%d_step%1.2f_niter%s" % \
               (target_ber_val, my_miso_chan, n_ant_val, min(ebn0_db_arr), max(ebn0_db_arr),
                ebn0_db_arr[1] - ebn0_db_arr[0], min(ibo_arr_cnc), max(ibo_arr_cnc), ibo_arr_cnc[1] - ibo_arr_cnc[0],
                '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig("../figs/final_figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
