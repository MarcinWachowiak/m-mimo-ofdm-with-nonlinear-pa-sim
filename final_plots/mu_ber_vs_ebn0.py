# %%
import matplotlib.pyplot as plt
import numpy as np

import plot_settings
import utilities

plot_settings.set_latex_plot_style(use_tex=True, fig_width_in=3.5)

n_ant_val = 64
ibo_val_db = 0
ebn0_step = 1
cnc_n_iter_lst = [1, 2, 3, 4, 5, 6, 7, 8]

usr_angles = [-30, 30]
usr_distances = [100, 316.3]
ebn0_min = 5
ebn0_max = 20

precoding_str = 'mr'
my_miso_chan = 'los'

cnc_filename_str = "ber_vs_ebn0_mu_%s_cnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s_angles%s_distances%s" % (
    precoding_str, my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
    '_'.join([str(val) for val in cnc_n_iter_lst]), '_'.join([str(val) for val in usr_angles]),
    '_'.join([str(val) for val in usr_distances]))

cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
cnc_ebn0_arr = cnc_data_lst[0]
cnc_ber_per_dist = cnc_data_lst[1:]
cnc_ber_per_dist_usr_1 = cnc_ber_per_dist[0:10]
cnc_ber_per_dist_usr_2 = cnc_ber_per_dist[10:20]
ber_per_dist_per_usr = [cnc_ber_per_dist_usr_1, cnc_ber_per_dist_usr_2]

# %%
mu_mcnc_combined_array = np.zeros((64, 10))
for proc_idx in range(12):
    mcnc_filename_str = "proc_%d_ber_vs_ebn0_mu_%s_mcnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s_angles%s_distances%s" % (
        proc_idx, precoding_str, my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
        '_'.join([str(val) for val in cnc_n_iter_lst]), '_'.join([str(val) for val in usr_angles]),
        '_'.join([str(val) for val in usr_distances]))
    mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
    mcnc_ebn0_arr = mcnc_data_lst[0]
    mu_mcnc_combined_array += np.array(mcnc_data_lst[1:])

mu_mcnc_bers = mu_mcnc_combined_array[0::2] / mu_mcnc_combined_array[1::2]
mu_mcnc_ber_usr_1 = np.transpose(mu_mcnc_bers[0::2])
mu_mcnc_ber_usr_2 = np.transpose(mu_mcnc_bers[1::2])
mu_mcnc_ber_per_usr = [mu_mcnc_ber_usr_1, mu_mcnc_ber_usr_2]
# %%
CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                  '#CFCFCF']

fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log')
usr_marker_lst = ['o', 's', '^', '*']

cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sel_cnc_iter_val = [0, 1, 2, 5]

for usr_idx, usr_ber in enumerate(ber_per_dist_per_usr):
    ax1.plot(cnc_ebn0_arr, usr_ber[0], color=CB_color_cycle[0], marker=usr_marker_lst[usr_idx], linestyle='-',
             fillstyle='none')
    color_idx = 1
    for ite_idx, iter_val in enumerate(cnc_n_iter_lst):
        if iter_val in sel_cnc_iter_val:
            ax1.plot(cnc_ebn0_arr, usr_ber[ite_idx + 1], color=CB_color_cycle[color_idx],
                     marker=usr_marker_lst[usr_idx], linestyle='-', fillstyle='none')
            ax1.plot(cnc_ebn0_arr, mu_mcnc_ber_per_usr[usr_idx][ite_idx + 1], color=CB_color_cycle[color_idx],
                     marker=usr_marker_lst[usr_idx], linestyle='--', fillstyle='none')
            color_idx += 1

plot_settings.reset_color_cycle()

import matplotlib.patches as mpatches

n_ite_legend = []
n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[0], label="No dist"))
color_idx = 1
for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
    if ite_val in sel_cnc_iter_val:
        n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
    if ite_val in sel_cnc_iter_val or ite_val == 1:
        color_idx += 1

leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="lower left", ncol=1, framealpha=0.9)
plt.gca().add_artist(leg1)

import matplotlib.lines as mlines

usr_1_leg = mlines.Line2D([0], [0], linestyle='none', marker=usr_marker_lst[0], fillstyle="none", color='k', label="1")
usr_2_leg = mlines.Line2D([0], [0], linestyle='none', marker=usr_marker_lst[1], fillstyle="none", color='k', label="2")
leg2 = plt.legend(handles=[usr_1_leg, usr_2_leg], title="User:", loc="lower left", framealpha=0.9,
                  bbox_to_anchor=(0.305, 0.0))
plt.gca().add_artist(leg2)

cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
ax1.legend(handles=[cnc_leg, mcnc_leg], loc="lower left", framealpha=0.9, bbox_to_anchor=(0.49, 0.0))

ax1.set_xlim([10, 20])
ax1.set_ylim([1e-6, 1e-0])
# fix log scaling
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("BER")
ax1.grid()
plt.tight_layout()

# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename_str += "_" + timestamp
plt.savefig(
    "../figs/%s.pdf" % "ber_vs_ebn0_mu_%s_cnc_mcnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s_angles%s_distances%s" % (
        precoding_str, my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
        '_'.join([str(val) for val in sel_cnc_iter_val[1:]]), '_'.join([str(val) for val in usr_angles]),
        '_'.join([str(val) for val in usr_distances])), dpi=600, bbox_inches='tight')
plt.show()
# plt.cla()
# plt.close()
