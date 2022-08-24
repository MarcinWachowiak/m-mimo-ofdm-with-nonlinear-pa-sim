import ast
import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import utilities
from plot_settings import set_latex_plot_style
from utilities import to_db, pts_on_semicircum

set_latex_plot_style()
# %%
print("Multi antenna processing init!")

set_latex_plot_style(use_tex=True, fig_width_in=5.89572)
# %%
ibo_val_db = 3
n_snapshots = 10
n_points = 180 * 10
radial_distance = 300
sel_psd_angle = 78

# Multiple users data
usr_angles = [30, 60, 135]
usr_distances = [200, 250, 150]
n_users = len(usr_angles)
n_ant_vec = [128]
n_ant_val = 128

sel_ptx_idx = int(n_points / 180 * sel_psd_angle)

# PSD plotting params
psd_nfft = 4096
n_samp_per_seg = 2048

rx_points = pts_on_semicircum(r=radial_distance, n=n_points)
radian_vals = np.radians(np.linspace(0, 180, n_points + 1))

my_miso_chan = "rayleigh"

read_data = utilities.read_from_csv(
    filename="multiuser_mrt_sig_powers_vs_angle_two_path_chan_ibo3_npoints1800_nsnap10_angle78_nant128")
desired_sig_power = ast.literal_eval(read_data[0][0])
distortion_sig_power = ast.literal_eval(read_data[1][0])

# %%
fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(5.89572, 5.89572))
ax1.set_theta_zero_location("E")
plt.tight_layout()
ax1.set_thetalim(0, np.pi)
ax1.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.set_ylim([-90, 0])
dist_lines_lst = []

norm_coeff = np.max(np.array([desired_sig_power, distortion_sig_power]))
ax1.plot(radian_vals, to_db(np.divide(desired_sig_power, norm_coeff)), label="Desired", linewidth=1.5)
ax1.plot(radian_vals, to_db(np.divide(distortion_sig_power, norm_coeff)), label="Distortion", linewidth=1.5)
ax1.legend(title="Multi-user precoding, two-path channel, K = 128, IBO = 3 dB, Signals:", ncol=2, loc='lower center',
           borderaxespad=6)
# plot reference angles/directions
(y_min, y_max) = ax1.get_ylim()
ax1.vlines(np.deg2rad(usr_angles), y_min, y_max, colors='k', linestyles='--')  # label="Users")
ax1.margins(0.0, 0.0)

ax1.set_title("Normalized radiation pattern of signal components [dB]", pad=-60)
ax1.grid(True)
plt.savefig(
    "../figs/msc_figs/multiuser_%s_desired_vs_distortion_signal_beampattern_ibo%d_angles%s_distances%s_npoints%d_nsnap%d_nant%s.pdf" % (
        my_miso_chan, ibo_val_db, '_'.join([str(val) for val in usr_angles]),
        '_'.join([str(val) for val in usr_distances]), n_points, n_snapshots,
        '_'.join([str(val) for val in n_ant_vec])),
    dpi=600, bbox_inches='tight')
# plt.show()
plt.cla()
plt.close()

# %%
# plot signal to distortion ratio
fig3, ax3 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(5.89572, 5.89572))
ax3.set_theta_zero_location("E")
ax3.set_thetalim(0, np.pi)
ax3.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))

ax3.plot(radian_vals, to_db(np.array(desired_sig_power) / np.array(distortion_sig_power)), label=n_ant_val,
         linewidth=1.5)
# plot reference angles/directions
(y_min, y_max) = ax3.get_ylim()
ax3.vlines(np.deg2rad(usr_angles), y_min, y_max, colors='k', linestyles='--')  # label="Users")
ax3.margins(0.0, 0.0)

ax3.set_title("Signal to distortion ratio at angle [dB]", pad=-60)
ax3.legend(title="Multi-user precoding, two-path channel, K = 128, IBO = 3 dB, K antennas:", ncol=len(n_ant_vec),
           loc='lower center', borderaxespad=6)
ax3.grid(True)
plt.tight_layout()
plt.savefig(
    "../figs/msc_figs/multiuser_%s_sdr_beampattern_ibo%d_angles%s_distances%s_npoints%d_nsnap%d_nant%s.pdf" % (
        my_miso_chan, ibo_val_db, '_'.join([str(val) for val in usr_angles]),
        '_'.join([str(val) for val in usr_distances]), n_points, n_snapshots,
        '_'.join([str(val) for val in n_ant_vec])),
    dpi=600, bbox_inches='tight')

# plt.show()
plt.cla()
plt.close()

print("Finished processing!")
