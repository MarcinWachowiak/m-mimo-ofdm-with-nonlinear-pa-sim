# %%
import ast
import os
import sys

sys.path.append(os.getcwd())

import numpy as np

from plot_settings import set_latex_plot_style
from utilities import to_db
import utilities

set_latex_plot_style(use_tex=True, fig_width_in=5.89572)

oint_idx_psd = 78
ibo_val_db = 3
n_snapshots = 100
n_points = 180
radial_distance = 300
precoding_angle = 45
sel_psd_angle = 78

channel_type_lst = ["los", "two_path", "rayleigh"]
n_ant_vec = [1, 2, 4, 8, 16, 32, 64, 128]
n_ant_vec_sel = [16, 32, 64, 128]
n_ant_val_max = 128

filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
    "los", ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val_max)
los_read_data = utilities.read_from_csv(filename=filename_str)

los_sdr_at_angle = []
for idx in range(len(n_ant_vec)):
    des_sig = ast.literal_eval(los_read_data[0][idx])
    dist_sig = ast.literal_eval(los_read_data[1][idx])
    los_sdr_at_angle.append(to_db(np.array(des_sig) / np.array(dist_sig)))
los_sdr_at_angle = los_sdr_at_angle[4:]

filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
    "two_path", ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val_max)
two_path_read_data = utilities.read_from_csv(filename=filename_str)

two_path_sdr_at_angle = []
for idx in range(len(n_ant_vec)):
    des_sig = ast.literal_eval(two_path_read_data[0][idx])
    dist_sig = ast.literal_eval(two_path_read_data[1][idx])
    two_path_sdr_at_angle.append(to_db(np.array(des_sig) / np.array(dist_sig)))
two_path_sdr_at_angle = two_path_sdr_at_angle[4:]

filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
    "rayleigh", ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val_max)
rayleigh_read_data = utilities.read_from_csv(filename=filename_str)

rayleigh_sdr_at_angle = []
for idx in range(len(n_ant_vec)):
    des_sig = ast.literal_eval(rayleigh_read_data[0][idx])
    dist_sig = ast.literal_eval(rayleigh_read_data[1][idx])
    rayleigh_sdr_at_angle.append(to_db(np.array(des_sig) / np.array(dist_sig)))
rayleigh_sdr_at_angle = rayleigh_sdr_at_angle[4:]

# # %%
# # plot beampatterns of distortion signal
# fig2, ax2 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(3.5, 3))
# ax2.set_theta_zero_location("E")
# ax2.set_thetalim(0, np.pi)
# ax2.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
# ax2.yaxis.set_major_locator(MaxNLocator(5))
#
# dist_lines_lst = []
# for idx, n_ant in enumerate(n_ant_vec):
#     ax2.plot(radian_vals, to_db(distortion_sig_power_per_nant[idx]), label=n_ant, linewidth=1.5)
# ax2.set_title("Distortion signal PSD at angle [dB]", pad=-15)
# ax2.legend(title="N antennas:", ncol=len(n_ant_vec), loc='lower center', borderaxespad=0)
# ax2.grid(True)
# plt.savefig("figs/beampatterns/%s_distortion_signal_beampattern_ibo%d_angle%d_npoints%d_nsnap%d_nant%s.png" % (
#     my_miso_chan, ibo_val_db, precoding_angle, n_points, n_snapshots, '_'.join([str(val) for val in n_ant_vec])),
#             dpi=600, bbox_inches='tight')
#
# # plt.show()
# plt.cla()
# plt.close()
