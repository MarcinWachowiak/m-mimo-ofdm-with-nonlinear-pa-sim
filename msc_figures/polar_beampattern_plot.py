# %%
import ast
import os
import sys

from matplotlib.ticker import MaxNLocator

sys.path.append(os.getcwd())

import numpy as np

from plot_settings import set_latex_plot_style
from utilities import to_db
import utilities
import matplotlib.pyplot as plt

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
n_ant_val_max = 128

rx_points = utilities.pts_on_semicircum(r=radial_distance, n=n_points)
radian_vals = np.radians(np.linspace(0, 180, n_points + 1))

filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
    "two_path", ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val_max)
two_path_read_data = utilities.read_from_csv(filename=filename_str)

two_path_des_sig, two_path_dist_sig = [], []
for idx in range(len(n_ant_vec)):
    two_path_des_sig.append(ast.literal_eval(two_path_read_data[0][idx]))
    two_path_dist_sig.append(ast.literal_eval(two_path_read_data[1][idx]))

filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
    "rayleigh", ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val_max)
rayleigh_read_data = utilities.read_from_csv(filename=filename_str)

rayleigh_des_sig, rayleigh_dist_sig = [], []
for idx in range(len(n_ant_vec)):
    rayleigh_des_sig.append(ast.literal_eval(rayleigh_read_data[0][idx]))
    rayleigh_dist_sig.append(ast.literal_eval(rayleigh_read_data[1][idx]))

# %%
# Two-path distortion signal radiation pattern
n_ant_val = 128
# plot beampatterns of desired vs distortion
fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(5.89572, 5.89572))
ax1.set_theta_zero_location("E")
ax1.set_thetalim(0, np.pi)
ax1.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
ax1.yaxis.set_major_locator(MaxNLocator(5))

two_path_max = np.max(two_path_dist_sig)
two_path_dist_norm = np.divide(np.array(two_path_dist_sig), two_path_max)

for idx, ant_val in enumerate(n_ant_vec):
    if ant_val >= 16:
        ax1.plot(radian_vals, to_db(two_path_dist_norm[idx]), label=ant_val, linewidth=1.5)

ax1.set_title("Normalized radiation pattern of distortion signal [dB]", pad=-50)
ax1.legend(title="Two-path channel, IBO=3 dB, Number of antennas:", ncol=4, loc='lower center', borderaxespad=4)
ax1.grid(True)

plt.savefig("../figs/msc_figs/%s_distortion_signal_beampattern_ibo%d_angle%d_npoints%d_nsnap%d_nant%s.pdf" % (
    "two_path", ibo_val_db, precoding_angle, n_points, n_snapshots, n_ant_val),
            dpi=600, bbox_inches='tight')

# plt.show()
plt.cla()
plt.close()

# %%
# %%
# Rayleigh channel distortion signal beampattern
n_ant_val = 128
# plot beampatterns of desired vs distortion
fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(5.89572, 5.89572))
ax1.set_theta_zero_location("E")
ax1.set_thetalim(0, np.pi)
ax1.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.set_ylim([-20, 10])
rayleigh_max = np.max(rayleigh_dist_sig)
rayleigh_dist_norm = np.divide(np.array(rayleigh_dist_sig), rayleigh_max)

for idx, ant_val in enumerate(n_ant_vec):
    if ant_val >= 16:
        ax1.plot(radian_vals, to_db(rayleigh_dist_norm[idx]), label=ant_val, linewidth=1.5)

ax1.set_title("Normalized radiation pattern of distortion signal [dB]", pad=-50)
ax1.legend(title="Rayleigh channel, IBO=3 dB, Number of antennas:", ncol=4, loc='lower center', borderaxespad=4)
ax1.grid(True)

plt.savefig("../figs/msc_figs/%s_distortion_signal_beampattern_ibo%d_angle%d_npoints%d_nsnap%d_nant%s.pdf" % (
    "rayleigh", ibo_val_db, precoding_angle, n_points, n_snapshots, n_ant_val),
            dpi=600, bbox_inches='tight')

# plt.show()
plt.cla()
plt.close()

# Two-path desired vs distortion signal radiation beampattern
# %%
fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(5.89572, 5.89572))
ax1.set_theta_zero_location("E")
ax1.set_thetalim(0, np.pi)
ax1.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
ax1.yaxis.set_major_locator(MaxNLocator(5))

norm_coeff = np.max(np.array([two_path_des_sig[7], two_path_dist_sig[7]]))

ax1.plot(radian_vals, to_db(np.divide(two_path_des_sig[7], norm_coeff)), label="Desired", linewidth=1.5)
ax1.plot(radian_vals, to_db(np.divide(two_path_dist_sig[7], norm_coeff)), label="Distortion", linewidth=1.5)

ax1.set_title("Normalized radiation pattern of signal components [dB]", pad=-50)
ax1.legend(title="Two-path channel, K=128, IBO=3 dB, Signals:", ncol=2, loc='lower center', borderaxespad=4)
ax1.grid(True)

plt.savefig(
    "../figs/msc_figs/%s_desired_vs_distortion_signal_beampattern_ibo%d_angle%d_npoints%d_nsnap%d_nant%s.pdf" % (
        "two_path", ibo_val_db, precoding_angle, n_points, n_snapshots, n_ant_val),
    dpi=600, bbox_inches='tight')

# plt.show()
plt.cla()
plt.close()
