"""
Master's thesis plotting script.
Eb/N0 required for a constant bit error rate (BER) as a function of the IBO,
for selected number of the CNC/MCNC iterations.
"""

# %%
import os
import sys

import plot_settings

sys.path.append(os.getcwd())

import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
import utilities

from plot_settings import set_latex_plot_style

if __name__ == '__main__':

    set_latex_plot_style(use_tex=True, fig_width_in=5.89572)

    cnc_n_iter_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    sel_cnc_iter_val = [0, 1, 2, 5, 8]
    target_ber_val = 1e-2
    n_ant_val = 64
    constel_size = 64

    ebn0_min = 10
    ebn0_max = 22.1
    ebn0_step = 0.5

    ibo_min = 0
    ibo_max = 7
    ibo_step = 0.5

    my_miso_chan_lst = ["los", "two_path", "rayleigh"]
    miso_chan_str = ["LOS", "two-path", "Rayleigh"]
    for chan_idx, my_miso_chan in enumerate(my_miso_chan_lst):

        ebn0_db_arr = np.arange(ebn0_min, ebn0_max, ebn0_step)
        cnc_filename_str = "fixed_ber%1.1e_cnc_%s_nant%d_ebn0_min%d_max%d_step%1.2f_ibo_min%d_max%d_step%1.2f_niter%s" % \
                           (target_ber_val, my_miso_chan, n_ant_val, ebn0_min, ebn0_max,
                            ebn0_step, ibo_min, ibo_max, ibo_step, '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
        cnc_data_lst = utilities.read_from_csv(filename=cnc_filename_str)
        cnc_ibo_arr = cnc_data_lst[0]
        cnc_tmp_data = cnc_data_lst[1:]

        # parse the data to interpolate
        cnc_ber_per_ibo_snr_iter = np.zeros((len(cnc_ibo_arr), len(ebn0_db_arr), len(cnc_n_iter_lst)))
        for ibo_idx, ibo_val_db in enumerate(cnc_ibo_arr):
            for snr_idx, snr_db_val in enumerate(ebn0_db_arr):
                for ite_idx in range(len(cnc_n_iter_lst)):
                    cnc_ber_per_ibo_snr_iter[ibo_idx, snr_idx, ite_idx] = \
                        cnc_tmp_data[ibo_idx * len(ebn0_db_arr) + snr_idx][
                            ite_idx]

        CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                          '#CFCFCF']
        # %%
        # extract SNR value providing given BER from collected data
        cnc_req_ebn0_per_ibo = np.zeros((len(cnc_n_iter_lst), len(cnc_ibo_arr)))
        plot_once = False
        for iter_idx, iter_val in enumerate(cnc_n_iter_lst):
            for ibo_idx, ibo_val in enumerate(cnc_ibo_arr):
                ber_per_ebn0 = cnc_ber_per_ibo_snr_iter[ibo_idx, :, iter_idx]
                # investigate better interpolation options
                interpol_func = interpolate.interp1d(ber_per_ebn0, ebn0_db_arr, kind="linear")
                if ibo_val == 2 and iter_val == 3:
                    # ber vector
                    if plot_once:
                        fig1, ax1 = plt.subplots(1, 1)
                        ax1.set_yscale('log')
                        ax1.plot(ebn0_db_arr, ber_per_ebn0, label=iter_val)
                        ax1.plot(interpol_func(ber_per_ebn0), ber_per_ebn0, label=iter_val)
                        ax1.set_xlabel("Eb/n0 [dB]")
                        ax1.set_ylabel("BER [-]")
                        ax1.grid()
                        plt.tight_layout()
                        plt.show()
                        print("Required Eb/No:", interpol_func(target_ber_val))

                        fig2, ax2 = plt.subplots(1, 1)
                        ax2.set_yscale('log')
                        ax2.plot(ebn0_db_arr, ber_per_ebn0)
                        plot_once = False
                try:
                    cnc_req_ebn0_per_ibo[iter_idx, ibo_idx] = interpol_func(target_ber_val)
                except:
                    # value not found in interpolation, replace with inf
                    cnc_req_ebn0_per_ibo[iter_idx, ibo_idx] = np.inf

        # %%
        color_idx = 1
        fig1, ax1 = plt.subplots(1, 1)
        for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
            if ite_val in sel_cnc_iter_val:
                ax1.plot(cnc_ibo_arr, cnc_req_ebn0_per_ibo[ite_idx], "-", color=CB_color_cycle[color_idx])
                color_idx += 1

        plot_settings.reset_color_cycle()

        mcnc_filename_str = "fixed_ber%1.1e_mcnc_%s_nant%d_ebn0_min%d_max%d_step%1.2f_ibo_min%d_max%d_step%1.2f_niter%s" % \
                            (target_ber_val, my_miso_chan, n_ant_val, ebn0_min, ebn0_max,
                             ebn0_step, ibo_min, ibo_max, ibo_step, '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
        mcnc_data_lst = utilities.read_from_csv(filename=mcnc_filename_str)
        mcnc_ibo_arr = mcnc_data_lst[0]
        mcnc_tmp_data = mcnc_data_lst[1:]

        # parse the data to interpolate
        mcnc_ber_per_ibo_snr_iter = np.zeros((len(cnc_ibo_arr), len(ebn0_db_arr), len(cnc_n_iter_lst)))
        for ibo_idx, ibo_val_db in enumerate(cnc_ibo_arr):
            for snr_idx, snr_db_val in enumerate(ebn0_db_arr):
                for ite_idx in range(len(cnc_n_iter_lst)):
                    mcnc_ber_per_ibo_snr_iter[ibo_idx, snr_idx, ite_idx] = \
                        mcnc_tmp_data[ibo_idx * len(ebn0_db_arr) + snr_idx][
                            ite_idx]

        # %%
        # extract SNR value providing given BER from collected data
        mcnc_req_ebn0_per_ibo = np.zeros((len(cnc_n_iter_lst), len(cnc_ibo_arr)))
        plot_once = False
        for iter_idx, iter_val in enumerate(cnc_n_iter_lst):
            for ibo_idx, ibo_val in enumerate(cnc_ibo_arr):
                ber_per_ebn0 = mcnc_ber_per_ibo_snr_iter[ibo_idx, :, iter_idx]
                # investigate better interpolation options
                interpol_func = interpolate.interp1d(ber_per_ebn0, ebn0_db_arr)
                if ibo_val == 2 and iter_val == 3:
                    # ber vector
                    if plot_once:
                        fig1, ax1 = plt.subplots(1, 1)
                        ax1.set_yscale('log')
                        ax1.plot(ebn0_db_arr, ber_per_ebn0, label=iter_val)
                        ax1.plot(interpol_func(ber_per_ebn0), ber_per_ebn0, label=iter_val)
                        ax1.set_xlabel("Eb/n0 [dB]")
                        ax1.set_ylabel("BER [-]")
                        ax1.grid()
                        plt.tight_layout()
                        plt.show()
                        print("Required Eb/No:", interpol_func(target_ber_val))

                        fig2, ax2 = plt.subplots(1, 1)
                        ax2.set_yscale('log')
                        ax2.plot(ebn0_db_arr, ber_per_ebn0)
                        plot_once = False
                try:
                    mcnc_req_ebn0_per_ibo[iter_idx, ibo_idx] = interpol_func(target_ber_val)
                except:
                    # value not found in interpolation, replace with inf
                    mcnc_req_ebn0_per_ibo[iter_idx, ibo_idx] = np.inf

        color_idx = 1
        for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
            if ite_val in sel_cnc_iter_val:
                ax1.plot(mcnc_ibo_arr, mcnc_req_ebn0_per_ibo[ite_idx], "--", color=CB_color_cycle[color_idx])
                # dashes=(5, 1 + color_idx))
                color_idx += 1

        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        n_ite_legend = []
        color_idx = 1
        for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
            if ite_val in sel_cnc_iter_val:
                n_ite_legend.append(mpatches.Patch(color=CB_color_cycle[color_idx], label=ite_val))
                color_idx += 1
        if my_miso_chan == "rayleigh":
            leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="upper left", ncol=1, framealpha=0.9)
        else:
            leg1 = plt.legend(handles=n_ite_legend, title="I iterations:", loc="upper right", ncol=1, framealpha=0.9)

        plt.gca().add_artist(leg1)

        cnc_leg = mlines.Line2D([0], [0], linestyle='-', color='k', label='CNC')
        mcnc_leg = mlines.Line2D([0], [0], linestyle='--', color='k', label='MCNC')
        if my_miso_chan == "rayleigh":
            ax1.legend(handles=[cnc_leg, mcnc_leg], loc="upper center", framealpha=0.9, bbox_to_anchor=(0.24, 1.0))
        else:
            ax1.legend(handles=[cnc_leg, mcnc_leg], loc="upper center", framealpha=0.9, bbox_to_anchor=(0.76, 1.0))
        # plt.gca().add_artist(leg2)
        ax1.set_title("Eb/N0 in regard to IBO for fixed BER = $10^{-2}$, %s channel, QAM %d, K = %d" % (
            miso_chan_str[chan_idx], constel_size, n_ant_val))
        ax1.set_xlabel("IBO [dB]")
        ax1.set_ylabel("Eb/N0 [dB]")
        ax1.set_xlim([0, 7])
        if my_miso_chan == "rayleigh":
            ax1.set_ylim([11.5, 16])
        else:
            ax1.set_ylim([11.5, 20])

        ax1.grid()
        plt.tight_layout()

        filename_str = "fixed_ber%1.1e_%s_nant%d_ebn0_min%d_max%d_step%1.2f_ibo_min%d_max%d_step%1.2f_niter%s" % \
                       (target_ber_val, my_miso_chan, n_ant_val, ebn0_min, ebn0_max,
                        ebn0_step, min(cnc_ibo_arr), max(cnc_ibo_arr), cnc_ibo_arr[1] - cnc_ibo_arr[0],
                        '_'.join([str(val) for val in sel_cnc_iter_val[1:]]))
        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # filename_str += "_" + timestamp
        plt.savefig("../figs/msc_figs/%s.pdf" % filename_str, dpi=600, bbox_inches='tight')
        plt.show()

        print("Finished execution!")
