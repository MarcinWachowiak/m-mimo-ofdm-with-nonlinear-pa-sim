"""
Master's thesis plotting script,
plot the effective radiation pattern of the desired and distortion signal of the antenna array with nonlinear
front-end amplifiers for selected number of antennas and channel models.
"""

# %%
import os
import sys

from matplotlib.ticker import MaxNLocator

sys.path.append(os.getcwd())

import numpy as np

from plot_settings import set_latex_plot_style
from utilities import to_db
import utilities
import matplotlib.pyplot as plt

if __name__ == '__main__':

    set_latex_plot_style(use_tex=True, fig_width_in=3.5)

    oint_idx_psd = 78
    ibo_val_db = 3
    n_snapshots = 10
    n_points = 180 * 10
    radial_distance = 300
    precoding_angle = 45
    sel_psd_angle = 78

    channel_type_lst = ["los", "two_path", "rayleigh"]
    n_ant_vec = [16, 32, 64, 128]
    n_ant_val_max = 128

    rx_points = utilities.pts_on_semicircum(radius=radial_distance, n_points=n_points)
    two_path_des_sig, two_path_dist_sig = [], []
    rayleigh_des_sig, rayleigh_dist_sig = [], []

    for n_ant_val in n_ant_vec:
        filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
            "two_path", ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val)
        two_path_read_data = utilities.read_from_csv(filename=filename_str)
        two_path_des_sig.append(two_path_read_data[0])
        two_path_dist_sig.append(two_path_read_data[1])

        filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
            "rayleigh", ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val)
        rayleigh_read_data = utilities.read_from_csv(filename=filename_str)
        rayleigh_des_sig.append(rayleigh_read_data[0])
        rayleigh_dist_sig.append(rayleigh_read_data[1])

    n_ant_val = 128

    fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(3.5, 3.0))
    ax1.set_theta_zero_location("E")
    ax1.set_thetalim(0, np.pi)
    ax1.set_xticks(np.pi / 180. * np.linspace(0, 180, 13, endpoint=True))
    ax1.yaxis.set_major_locator(MaxNLocator(5))

    norm_coeff = np.max(np.array([two_path_des_sig[-1], two_path_dist_sig[-1]]))
    radian_vals = np.radians(np.linspace(0, 180, 180 * 10 + 1))

    ax1.plot(radian_vals, to_db(np.divide(two_path_des_sig[-1], norm_coeff)), label="Desired", linewidth=1.0)
    ax1.plot(radian_vals, to_db(np.divide(two_path_dist_sig[-1], norm_coeff)), label="Distortion", linewidth=1.0)

    # ax1.set_title("Normalized radiation pattern of signal components [dB]", pad=-150)
    ax1.legend(title="Two-path channel, K = 128, IBO = 3 dB, Signals:", ncol=2, loc='lower center', borderaxespad=-0)
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(
        "../figs/wwrf_figs/%s_desired_vs_distortion_signal_beampattern_ibo%d_angle%d_npoints%d_nsnap%d_nant%s.pdf" % (
            "two_path", ibo_val_db, precoding_angle, n_points, n_snapshots, n_ant_val),
        dpi=600)

    # plt.show()
    plt.cla()
    plt.close()
