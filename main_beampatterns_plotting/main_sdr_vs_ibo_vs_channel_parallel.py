"""
Measure the signal-to-distortion ratio as a function of input back-off for selected number of antennas and channels.
"""
import ctypes
import multiprocessing as mp
import os
import sys
from multiprocessing import Array, Value

from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime

import numpy as np

import channel
import distortion
import modulation
import transceiver
import utilities
import antenna_array
from plot_settings import set_latex_plot_style

from sdr_mp import Sdr_vs_ibo_vs_chan_link

if __name__ == '__main__':
    # %%

    reroll_chan = True

    set_latex_plot_style()
    print("Multi-antenna processing init!")

    ibo_arr = np.arange(0, 8.01, 0.25)
    print("IBO values:", ibo_arr)

    n_ant_arr = [1, 4, 16, 32, 64]
    print("N ANT values:", n_ant_arr)

    n_snapshots = 500
    rx_loc_x, rx_loc_y = 212.0, 212.0
    rx_loc_var = 10.0

    my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=2048, cp_len=128)
    my_distortion = distortion.SoftLimiter(ibo_db=3.0, avg_samp_pow=my_mod.avg_sample_power)
    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(3.5e9),
                                    carrier_spacing=int(15e3))

    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=rx_loc_x,
                                             cord_y=rx_loc_y, cord_z=1.5, center_freq=int(3.5e9),
                                             carrier_spacing=int(15e3))

    my_array = antenna_array.LinearArray(n_elements=np.min(n_ant_arr), base_transceiver=my_tx, center_freq=int(3.5e9),
                                         wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
    my_standard_rx.set_position(cord_x=rx_loc_x, cord_y=rx_loc_y, cord_z=1.5)

    # %%
    seed_rng = np.random.default_rng(2137)

    # plot PSD for chosen point/angle
    sdr_at_ibo_per_n_ant = []
    # direct visibility channels - increasing antennas does not increase SDR - run for lowest value of n_ant_arr
    for n_ant_idx, n_ant_val in enumerate(n_ant_arr):
        start_time = time.time()
        print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sdr_at_ibo_per_chan = []
        my_array = antenna_array.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                             wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)

        my_miso_los_chan = channel.MisoLosFd()
        my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                          skip_attenuation=False)
        my_miso_two_path_chan = channel.MisoTwoPathFd()
        my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                               skip_attenuation=False)

        my_miso_rayleigh_chan = channel.MisoRayleighFd(tx_transceivers=my_array.array_elements,
                                                       rx_transceiver=my_standard_rx,
                                                       seed=1234)
        channel_model_str_los = '3GPP_38.901_UMa_LOS'
        my_miso_quadriga_chan_los = channel.MisoQuadrigaFd(tx_transceivers=my_array.array_elements,
                                                           rx_transceiver=my_standard_rx,
                                                           channel_model_str=channel_model_str_los,
                                                           start_matlab_eng=False)
        channel_model_str_nlos = '3GPP_38.901_UMa_NLOS'
        my_miso_quadriga_chan_nlos = channel.MisoQuadrigaFd(tx_transceivers=my_array.array_elements,
                                                            rx_transceiver=my_standard_rx,
                                                            channel_model_str=channel_model_str_nlos,
                                                            start_matlab_eng=False)

        chan_lst = [my_miso_quadriga_chan_los, my_miso_quadriga_chan_nlos]
        num_cores_lst = [8, 8]

        for chan_idx, my_miso_chan in enumerate(chan_lst):
            sdr_at_ibo = np.zeros(len(ibo_arr))
            num_cores = num_cores_lst[chan_idx]

            mp_link_obj = Sdr_vs_ibo_vs_chan_link(mod_obj=my_mod, array_obj=my_array, std_rx_obj=my_standard_rx,
                                                  chan_obj=my_miso_chan, rx_loc_var=rx_loc_var)

            for ibo_idx, ibo_val_db in enumerate(ibo_arr):
                utilities.print_progress_bar(ibo_idx + 1, len(ibo_arr), prefix='IBO loop progress:')

                mp_link_obj.update_distortion(ibo_val_db=ibo_val_db)

                n_snap_idx = Value('i', 0, lock=True)
                sdr_at_ibo_per_symb = Array(ctypes.c_double, n_snapshots, lock=True)

                proc_seed_lst = seed_rng.integers(0, high=sys.maxsize, size=(num_cores, 2))
                processes = []
                for idx in range(num_cores):
                    p = mp.Process(target=mp_link_obj.simulate,
                                   args=(reroll_chan, proc_seed_lst[idx],
                                         n_snap_idx, sdr_at_ibo_per_symb))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

                sdr_at_ibo[ibo_idx] = np.average(sdr_at_ibo_per_symb)
                # sdr_at_ibo[ibo_idx] = to_db(
                #     np.sum_signals(np.power(np.abs(clean_rx_sig_accum_arr), 2)) / np.sum_signals(
                #         np.power(np.abs(sc_ofdm_distortion_sig), 2)))

            sdr_at_ibo_per_chan.append(sdr_at_ibo)

        sdr_at_ibo_per_n_ant.append(sdr_at_ibo_per_chan)
        print("--- Computation time: %f ---" % (time.time() - start_time))

    # %%
    # plot signal to distortion ratio vs ibo
    # from utilities import to_db
    #
    # fig1, ax1 = plt.subplots(1, 1)
    # p1, = ax1.plot(ibo_arr, to_db(sdr_at_ibo_per_n_ant[0][0]), '-', color='#377eb8', label="1")
    # p2, = ax1.plot(ibo_arr, to_db(sdr_at_ibo_per_n_ant[1][0]), '--', color='#377eb8', label="4")
    # p3, = ax1.plot(ibo_arr, to_db(sdr_at_ibo_per_n_ant[2][0]), ':', color='#377eb8', label="32")

    # leg1 = ax1.legend([p1,p2,p3], n_ant_arr, loc=1, title="LOS:")
    # plt.gca().add_artist(leg1)
    #
    # p4, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[0][1], '-', color='#ff7f00', label="1")
    # p5, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[1][1], '--', color='#ff7f00', label="4")
    # p6, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[2][1], ':', color='#ff7f00', label="32")
    #
    # # leg2 = ax1.legend([p4,p5,p6], n_ant_arr, loc=2, title="Two-Path:")
    # # plt.gca().add_artist(leg2)
    #
    # p7, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[0][2], '-', color='#4daf4a', label="1")
    # p8, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[1][2], '--', color='#4daf4a', label="4")
    # p9, = ax1.plot(ibo_arr, sdr_at_ibo_per_n_ant[2][2], ':', color='#4daf4a', label="32")

    # leg3 = ax1.legend([p7,p8,p9], n_ant_arr, loc=3, title="Rayleigh:")
    # plt.gca().add_artist(leg3)
    #
    # import matplotlib.patches as mpatches
    #
    # los = mpatches.Patch(color='#377eb8', label='LOS')
    # twopath = mpatches.Patch(color='#ff7f00', label='Two-path')
    # rayleigh = mpatches.Patch(color='#4daf4a', label='Rayleigh')
    #
    # leg1 = plt.legend(handles=[los, twopath, rayleigh], title="Channel:", loc="upper left")
    # plt.gca().add_artist(leg1)
    #
    # import matplotlib.lines as mlines
    #
    # n_ant1 = mlines.Line2D([0], [0], linestyle='-', color='k', label='1')
    # n_ant4 = mlines.Line2D([0], [0], linestyle='--', color='k', label='4')
    # n_ant32 = mlines.Line2D([0], [0], linestyle=':', color='k', label='32')
    # leg2 = plt.legend(handles=[n_ant1, n_ant4, n_ant32], title="N antennas:", loc="lower right")
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
    #
    # # # %%
    # ax1.set_title("SDR vs IBO for given channel")
    # ax1.set_xlabel("IBO [dB]")
    # ax1.set_ylabel("SDR [dB]")
    # ax1.grid()
    # # # ax1.legend(title="Channel:")
    # # plt.tight_layout()
    # # plt.savefig("figs/sdr_vs_ibo_per_channel_ibo%dto%d_%dnant.png" % (
    # #     min(ibo_arr), max(ibo_arr), np.max(n_ant_arr)),
    # #             dpi=600, bbox_inches='tight')
    # #
    # plt.show()

    # %%
    # save data to csv file
    data_lst = []
    data_lst.append(ibo_arr)
    for arr1 in sdr_at_ibo_per_n_ant:
        for arr2 in arr1:
            data_lst.append(arr2.tolist())

    utilities.save_to_csv(data_lst=data_lst, filename="sdr_vs_ibo_per_channel_ibo%dto%d_%dnant_quadriga" % (
        min(ibo_arr), max(ibo_arr), np.max(n_ant_arr)), )
    print("Finished execution!")
