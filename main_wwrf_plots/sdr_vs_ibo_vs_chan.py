"""
Master's thesis plotting script,
signal-to-distortion (SDR) ratio as a function of the IBO for selected number antennas and channel models.
"""

# %%
import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np

import utilities
from matplotlib import rcParams, cycler, rc
import matplotlib

if __name__ == '__main__':
    # set style similar to default Matlab
    CB_color_cycle = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.9290, 0.6940, 0.1250), (0.4940, 0.1840, 0.5560),
                      (0.4660, 0.6740, 0.1880), (0.3010, 0.7450, 0.9330), (0.6350, 0.0780, 0.1840)]
    rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)

    fig_width_in = 3.5
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    # fig_width_in = 3.5  # width in inches
    fig_height_in = fig_width_in * golden_mean  # height in inches
    fig_size = [fig_width_in, fig_height_in]

    params = {'backend': 'Qt5Agg',
              'axes.labelsize': 7,
              'font.size': 7,
              'legend.fontsize': 7,
              'xtick.labelsize': 7,
              'ytick.labelsize': 7,
              'figure.figsize': fig_size}
    rcParams['path.simplify'] = True

    rcParams.update(params)
    matplotlib.use("Qt5Agg")

    data_lst_1 = utilities.read_from_csv(filename="sdr_vs_ibo_per_channel_ibo0to8_1_4_16_32_64nant")
    ibo_arr_1 = data_lst_1[0]
    sdr_at_ibo_per_n_ant = 10 * np.log10(data_lst_1[1:])

    data_lst_2 = utilities.read_from_csv(filename="sdr_vs_ibo_per_channel_ibo0to8_1_4_16_32_64nant_quadriga_los")
    ibo_arr_2 = data_lst_2[0]
    sdr_at_ibo_per_n_ant_quadriga_los = 10 * np.log10(data_lst_2[1:])

    data_lst_3 = utilities.read_from_csv(filename="sdr_vs_ibo_per_channel_ibo0to8_1_4_16_32_64nant_quadriga_nlos")
    ibo_arr_3 = data_lst_3[0]
    sdr_at_ibo_per_n_ant_quadriga_nlos = 10 * np.log10(data_lst_3[1:])

    # %%
    # plot signal to distortion ratio vs ibo
    fig1, ax1 = plt.subplots(1, 1)
    # 1 antenna
    ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[0], 'o-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[0])
    # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[1], 's-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[0])
    ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[2], 's-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[0])
    ax1.plot(ibo_arr_2, sdr_at_ibo_per_n_ant_quadriga_los[0], 'x-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[0])
    ax1.plot(ibo_arr_3, sdr_at_ibo_per_n_ant_quadriga_nlos[0], '+-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[0])

    # 4 antennas
    ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[3], 'o-', markevery=4, linewidth=1.0, fillstyle="none", color=CB_color_cycle[1])
    # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[4], 's-', markevery=4, linewidth=1.0, fillstyle="none", color=CB_color_cycle[1])
    ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[5], 's-', markevery=4, linewidth=1.0, fillstyle="none", color=CB_color_cycle[1])
    ax1.plot(ibo_arr_2, sdr_at_ibo_per_n_ant_quadriga_los[1], 'x-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[1])
    ax1.plot(ibo_arr_3, sdr_at_ibo_per_n_ant_quadriga_nlos[1], '+-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[1])

    # # 16 antennas
    ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[6], 'o-', markevery=6, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[7], 's-', markevery=6, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[8], 's-', markevery=4, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    ax1.plot(ibo_arr_2, sdr_at_ibo_per_n_ant_quadriga_los[2], 'x-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    ax1.plot(ibo_arr_3, sdr_at_ibo_per_n_ant_quadriga_nlos[2], '+-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    #
    # 32 antennas
    # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[9], 'o-', markevery=6, linewidth=1.0, fillstyle="none", color=CB_color_cycle[3])
    # # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[10], 's-', markevery=6, linewidth=1.0, fillstyle="none", color=CB_color_cycle[3])
    # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[11], '*-', markevery=4, linewidth=1.0, fillstyle="none", color=CB_color_cycle[3])
    # ax1.plot(ibo_arr_2, sdr_at_ibo_per_n_ant_quadriga_los[3], 'x-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[3])
    # ax1.plot(ibo_arr_3, sdr_at_ibo_per_n_ant_quadriga_nlos[3], 's-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[3])
    #
    # # # 64 antennas
    # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[12], 'o-', markevery=6, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[13], 's-', markevery=6, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    # ax1.plot(ibo_arr_1, sdr_at_ibo_per_n_ant[14], '*-', markevery=4, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    # ax1.plot(ibo_arr_2, sdr_at_ibo_per_n_ant_quadriga_los[4], 'x-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])
    # ax1.plot(ibo_arr_3, sdr_at_ibo_per_n_ant_quadriga_nlos[4], 's-', markevery=2, linewidth=1.0, fillstyle="none", color=CB_color_cycle[2])

    import matplotlib.patches as mpatches

    n_ant1 = mpatches.Patch(color=CB_color_cycle[0], label='1')
    n_ant4 = mpatches.Patch(color=CB_color_cycle[1], label='4')
    n_ant16 = mpatches.Patch(color=CB_color_cycle[2], label='16')
    # n_ant32 = mpatches.Patch(color=CB_color_cycle[3], label='32')
    # n_ant64 = mpatches.Patch(color=CB_color_cycle[2], label='64')

    leg1 = plt.legend(handles=[n_ant1, n_ant4, n_ant16], title="K antennas:", loc="lower right", framealpha=0.9)
    plt.gca().add_artist(leg1)

    import matplotlib.lines as mlines

    los = mlines.Line2D([0], [0], linestyle='none', marker="o", fillstyle="none", color='k', label='LOS')
    # twopath = mlines.Line2D([0], [0], linestyle='none', marker="s", fillstyle="none", color='k', label='Two-path')
    rayleigh = mlines.Line2D([0], [0], linestyle='none', marker="s", fillstyle="none", color='k', label='Rayleigh')
    qd_los = mlines.Line2D([0], [0], linestyle='none', marker="x", fillstyle="none", color='k', label='38.901 LOS')
    qd_nlos = mlines.Line2D([0], [0], linestyle='none', marker="+", fillstyle="none", color='k', label='38.901 NLOS')

    ax1.legend(handles=[los, rayleigh, qd_los, qd_nlos], title="Channels:", loc="upper left", framealpha=0.9)
    # plt.gca().add_artist(leg2)
    #
    # p10, = ax1.plot([0], marker='None',
    #            linestyle='None', label='dummy-tophead')
    # p11, = ax1.plot([0],  marker='None',
    #            linestyle='None', label='dummy-empty')
    # p12, = ax1.plot([0],  marker='None',
    #            linestyle='None', label='dummy-empty')
    #
    # leg = ax1.legend([p10, p11, p12, p10, p11, p12, p10, p11, p12, p1, p2, p3, p4, p5, p6, p7, p8, p9],
    #               ["LOS:", '', '', "Two-path:", '', '', "Rayleigh:", '', ''] + n_ant_arr + n_ant_arr + n_ant_arr,
    #               ncol=2) # Two columns, horizontal group labels

    # leg= ax1.legend([p11, p1, p2, p3, p11, p4, p5, p6, p11, p7, p8, p9],
    #               ["LOS"] + n_ant_arr + ["Two-path"] + n_ant_arr + ["Rayleigh"] + n_ant_arr,
    #               loc="lower right", ncol=3, title="Channel and N antennas:", columnspacing=0.2) # Two columns, horizontal group labels
    #
    # %%
    ax1.set_ylim([10, 50])
    ax1.set_xlim([0, 7])

    # ax1.set_title("SDR in regard to IBO for selected channels and number of antennas ")
    ax1.set_xlabel("IBO [dB]")
    ax1.set_ylabel("SDR [dB]")
    ax1.grid()
    plt.tight_layout()
    plt.savefig("../figs/wwrf_figs/sdr_vs_ibo_per_channel_ibo0to8_1_4_16nant_sel_models.pdf", dpi=600, bbox_inches='tight')
    plt.show()

    print("Finished execution!")
