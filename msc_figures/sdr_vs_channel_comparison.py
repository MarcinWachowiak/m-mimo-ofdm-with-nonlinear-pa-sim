"""
Master's thesis plotting script,
signal-to-distortion (SDR) ratio as a function of the azimuth angle for selected number antennas and channel models.
"""

# %%
import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np

import utilities
from plot_settings import set_latex_plot_style
from utilities import to_db

if __name__ == '__main__':

    set_latex_plot_style(use_tex=True, )
    # %%
    print("Multi-antenna processing init!")
    bit_rng = np.random.default_rng(4321)

    # %%
    # plot PSD for chosen point/angle
    point_idx_psd = 78
    ibo_val_db = 3
    n_snapshots = 10
    n_points = 180 * 10
    radial_distance = 300
    precoding_angle = 45
    sel_psd_angle = 78

    rx_points = utilities.pts_on_semicircum(radius=radial_distance, n_points=n_points)
    radian_vals = np.radians(np.linspace(0, 180, n_points + 1))

    channel_type_lst = ["los", "two_path", "rayleigh"]
    n_ant_vec = [16, 32, 64, 128]
    n_ant_vec_sel = [16, 32, 64, 128]

    los_des_sig, los_dist_sig = [], []
    two_path_des_sig, two_path_dist_sig = [], []
    rayleigh_des_sig, rayleigh_dist_sig = [], []
    for n_ant_val in n_ant_vec:
        filename_str = "mrt_sig_powers_vs_angle_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
            "los", ibo_val_db, n_points, n_snapshots, precoding_angle, n_ant_val)
        los_read_data = utilities.read_from_csv(filename=filename_str)
        los_des_sig.append(los_read_data[0])
        los_dist_sig.append(los_read_data[1])

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

    los_sdr_at_angle, two_path_sdr_at_angle, rayleigh_sdr_at_angle = [], [], []
    for idx in range(len(n_ant_vec)):
        los_sdr_at_angle.append(to_db(np.divide(np.array(los_des_sig[idx]), np.array(los_dist_sig[idx]))))
        two_path_sdr_at_angle.append(to_db(np.divide(np.array(two_path_des_sig[idx]), np.array(two_path_dist_sig[idx]))))
        rayleigh_sdr_at_angle.append(to_db(np.divide(np.array(rayleigh_des_sig[idx]), np.array(rayleigh_dist_sig[idx]))))

    # %%
    # plot signal to distortion ratio
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5.89572, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1, 2]})
    deg_vals = np.rad2deg(radian_vals)
    # LOS
    for idx, n_ant in enumerate(n_ant_vec_sel):
        ax1.plot(deg_vals, los_sdr_at_angle[idx], label=n_ant, linewidth=1.5)
    ax1.grid(True)
    ax1.set_title("a) LOS channel", fontsize=8)
    ax1.set_ylabel("SDR [dB]", fontsize=8)
    ax1.set_ylim([2, 22])
    # ax1.set_yticks([14, 16, 18, 20, 22])
    # Two-path
    for idx, n_ant in enumerate(n_ant_vec_sel):
        ax2.plot(deg_vals, two_path_sdr_at_angle[idx], label=n_ant, linewidth=1.5)
    ax2.grid(True)
    ax2.set_title("b) Two-path channel", fontsize=8)
    ax2.set_ylabel("SDR [dB]", fontsize=8)
    ax2.set_ylim([2, 22])
    # ax2.set_yticks([14, 16, 18, 20, 22])

    # Rayleigh
    for idx, n_ant in enumerate(n_ant_vec_sel):
        ax3.plot(deg_vals, rayleigh_sdr_at_angle[idx], label=n_ant, linewidth=1.5)
    ax3.grid(True)
    ax3.set_title("c) Rayleigh channel", fontsize=8)
    ax3.legend(title="Number of antennas:", ncol=len(n_ant_vec_sel), loc="upper center", bbox_to_anchor=(0.5, -0.25),
               borderaxespad=0)
    ax3.set_xlabel("Angle [°]", fontsize=8)
    ax3.set_ylabel("SDR [dB]", fontsize=8)
    ax3.set_xlim([0, 180])
    ax3.set_ylim([18, 42])
    ax3.set_yticks([18, 22, 26, 30, 34, 38, 42])
    ax3.set_xticks(np.linspace(0, 180, 7, endpoint=True))
    fig.suptitle("Signal to distortion ratio in regard to channel and number of antennas")

    # zoom on Rayleigh SDR peak
    # inset axes....
    axins = ax3.inset_axes([0.4, 0.3, 0.55, 0.6])
    # sub region of the original image
    for idx, n_ant in enumerate(n_ant_vec_sel):
        axins.plot(deg_vals, rayleigh_sdr_at_angle[idx], label=n_ant, linewidth=1.5)

    axins.tick_params(axis='x', labelsize=8)
    axins.set_xlim(44.9, 45.1)
    axins.set_xticks([44.9, 44.95, 45, 45.05, 45.1])

    axins.tick_params(axis='y', labelsize=8)
    axins.set_ylim(28, 42)
    axins.set_yticks([31, 34, 37, 40])
    axins.grid()
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    ax3.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    plt.savefig("../figs/msc_figs/sdr_at_angle_ibo%d_%dto%dant_sweep.pdf" % (
        ibo_val_db, np.min(n_ant_vec_sel), np.max(n_ant_vec_sel)), dpi=600, bbox_inches='tight')
    plt.show()

    print("Finished processing!")
