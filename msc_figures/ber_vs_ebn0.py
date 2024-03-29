"""
Master's thesis plotting script,
bit error rate (BER) as a function of Eb/N0 for a selected number of CNC/MCNC iterations,
"""

# %%
import os
import sys

import plot_settings

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

import utilities
from plot_settings import set_latex_plot_style

if __name__ == '__main__':

    set_latex_plot_style(use_tex=True, fig_width_in=5.89572)

    cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    n_ant_val = 64
    ibo_val_db = 0
    constel_size = 64

    ebn0_min = 5
    ebn0_max = 20
    ebn0_step = 0.5

    my_miso_chan_lst = ["los", "two_path", "rayleigh"]
    miso_chan_str = ["LOS", "two-path", "Rayleigh"]
    for chan_idx, my_miso_chan in enumerate(my_miso_chan_lst):
        if my_miso_chan == "rayleigh":
            sel_cnc_iter_val = [0, 1, 2, 5]
        else:
            sel_cnc_iter_val = [0, 2, 5, 8]

        cnc_filename_str = "ber_vs_ebn0_cnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
            my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
            '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
        cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
        cnc_ebn0_arr = cnc_data_lst[0]
        cnc_ber_per_dist = cnc_data_lst[1:]

        mcnc_filename_str = "ber_vs_ebn0_mcnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
            my_miso_chan, n_ant_val, ibo_val_db, ebn0_min, ebn0_max, ebn0_step,
            '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
        mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
        mcnc_ebn0_arr = mcnc_data_lst[0]
        mcnc_ber_per_dist = mcnc_data_lst[1:]

        # %%
        fig1, ax1 = plt.subplots(1, 1)
        ax1.set_yscale('log', base=10)

        CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                          '#CFCFCF']

        color_idx = 1
        ax1.plot(cnc_ebn0_arr, cnc_ber_per_dist[0])
        for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
            if cnc_iter_val in sel_cnc_iter_val:
                ax1.plot(cnc_ebn0_arr[::2], cnc_ber_per_dist[idx + 1][::2], "-", color=CB_color_cycle[color_idx])
            if cnc_iter_val in sel_cnc_iter_val or cnc_iter_val == 1:
                color_idx += 1
        plot_settings.reset_color_cycle()

        color_idx = 1
        ax1.plot(mcnc_ebn0_arr, mcnc_ber_per_dist[0], "--")
        for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
            if cnc_iter_val in sel_cnc_iter_val:
                if my_miso_chan == "rayleigh":
                    ax1.plot(mcnc_ebn0_arr[::2], mcnc_ber_per_dist[idx + 1][::2], "--", color=CB_color_cycle[color_idx],
                             dashes=(5, 1 + idx))
                else:
                    ax1.plot(mcnc_ebn0_arr[::2], mcnc_ber_per_dist[idx + 1][::2], "--", color=CB_color_cycle[color_idx])
            if cnc_iter_val in sel_cnc_iter_val or cnc_iter_val == 1:
                color_idx += 1

        import matplotlib.lines as mlines

        n_ite_legend = []
        # n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[0], label="No dist"))
        # color_idx = 1
        # for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
        #     if ite_val in sel_cnc_iter_val:
        #         n_ite_legend.append(mlines.Line2D([0], [0], color=CB_color_cycle[color_idx], label=ite_val))
        #         color_idx += 1

        import matplotlib.patches as mpatches

        n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[0], label="No dist"))
        color_idx = 1
        for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
            if ite_val in sel_cnc_iter_val:
                n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
            if ite_val in sel_cnc_iter_val or ite_val == 1:
                color_idx += 1

        leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="lower left", ncol=1, framealpha=0.9)
        plt.gca().add_artist(leg1)

        cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
        mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
        ax1.legend(handles=[cnc_leg, mcnc_leg], loc="lower left", framealpha=0.9, bbox_to_anchor=(0.17, 0.0))
        # plt.gca().add_artist(leg2)

        ax1.set_title("BER in regard to Eb/N0, %s channel, QAM %d, K = %d, IBO = %d dB" % (
            miso_chan_str[chan_idx], constel_size, n_ant_val, ibo_val_db))
        ax1.set_xlim([10, 20])
        ax1.set_ylim([1e-5, 3e-1])

        ax1.grid(which='major', linestyle='-')
        # ax1.grid(which='minor', linestyle='--')
        ax1.set_xlabel("Eb/N0 [dB]")
        ax1.set_ylabel("BER")
        plt.tight_layout()

        filename_str = "ber_vs_ebn0_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
            my_miso_chan, n_ant_val, ibo_val_db, min(cnc_ebn0_arr), max(cnc_ebn0_arr), cnc_ebn0_arr[1] - cnc_ebn0_arr[0],
            '_'.join([str(val) for val in sel_cnc_iter_val[1:]]))
        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # filename_str += "_" + timestamp
        plt.savefig("../figs/msc_figs/%s.pdf" % filename_str, dpi=600, bbox_inches='tight')
        plt.show()

        print("Finished execution!")
