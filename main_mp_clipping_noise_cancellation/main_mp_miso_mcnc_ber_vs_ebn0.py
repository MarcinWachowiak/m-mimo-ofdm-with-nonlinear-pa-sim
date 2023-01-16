# MISO OFDM simulation with nonlinearity
# Multiple Antenna Clipping noise cancellation eval
# %%
import os
import sys

sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import antenna_arrray
import channel
import distortion
import modulation
import noise
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import ebn0_to_snr

import mp_model
import multiprocessing as mp
import ctypes

if __name__ == '__main__':

    set_latex_plot_style()
    num_cores = mp.cpu_count()

    # %%
    # parameters
    n_ant_arr = [64]
    ibo_arr = [1, 3]
    ebn0_step = [1]
    cnc_n_iter_lst = [1, 2, 3, 4, 5, 6, 7, 8]
    # include clean run is always True
    # no distortion and standard RX always included
    incl_clean_run = True
    reroll_chan = True
    cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)

    # modulation
    constel_size = 64
    n_fft = 4096
    n_sub_carr = 2048
    cp_len = 128

    # accuracy
    bits_sent_max = int(1e7)
    n_err_min = int(1e6)

    rx_loc_x, rx_loc_y = 212.0, 212.0
    rx_loc_var = 10.0

    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len)

    my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
    # my_distortion = distortion.Rapp(ibo_db=0, p_hardness=4.0, avg_samp_pow=my_mod.avg_sample_power)

    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(3.5e9),
                                    carrier_spacing=int(15e3))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=rx_loc_x, cord_y=rx_loc_y, cord_z=1.5,
                                             center_freq=int(3.5e9), carrier_spacing=int(15e3))

    seed_rng = np.random.default_rng(2137)
    for n_ant_val in n_ant_arr:
        my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                              wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
        # channel type
        my_miso_los_chan = channel.MisoLosFd()
        my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                          skip_attenuation=False)
        my_miso_two_path_chan = channel.MisoTwoPathFd()
        my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                               skip_attenuation=False)

        my_miso_rayleigh_chan = channel.MisoRayleighFd(tx_transceivers=my_array.array_elements,
                                                       rx_transceiver=my_standard_rx,
                                                       seed=1234)

        my_random_paths_miso_channel = channel.MisoRandomPathsFd(tx_transceivers=my_array.array_elements,
                                                                 rx_transceiver=my_standard_rx, n_paths=8,
                                                                 max_delay_spread=1e-6)

        chan_lst = [my_miso_two_path_chan, my_miso_rayleigh_chan]
        my_noise = noise.Awgn(snr_db=10, seed=1234)

        for my_miso_chan in chan_lst:

            mp_link_obj = mp_model.Link(mod_obj=my_mod, array_obj=my_array, std_rx_obj=my_standard_rx,
                                        chan_obj=my_miso_chan, noise_obj=my_noise,
                                        rx_loc_var=rx_loc_var, n_err_min=n_err_min,
                                        bits_sent_max=bits_sent_max, is_mcnc=True)

            for ibo_val_db in ibo_arr:
                mp_link_obj.update_distortion(ibo_val_db=ibo_val_db)

                for ebn0_step_val in ebn0_step:
                    start_time = time.time()
                    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                    ebn0_arr = np.arange(5, 20.1, ebn0_step_val)
                    snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_sub_carr, my_mod.n_sub_carr, my_mod.constel_size)
                    ber_per_dist = []

                    for snr_idx, snr_db_val in enumerate(snr_arr):
                        mp_link_obj.set_snr(snr_db_val=snr_db_val)

                        bers = np.zeros([len(cnc_n_iter_lst) + 1])
                        n_err_shared_arr = mp.Array(ctypes.c_double, len(cnc_n_iter_lst) + 1, lock=True)
                        n_bits_sent_shared_arr = mp.Array(ctypes.c_double, len(cnc_n_iter_lst) + 1, lock=True)

                        proc_seed_lst = seed_rng.integers(0, high=sys.maxsize, size=(num_cores, 3))
                        processes = []
                        for idx in range(num_cores):
                            p = mp.Process(target=mp_link_obj.simulate,
                                           args=(incl_clean_run, reroll_chan, cnc_n_iter_lst, proc_seed_lst[idx],
                                                 n_err_shared_arr, n_bits_sent_shared_arr))
                            processes.append(p)
                            p.start()

                        for p in processes:
                            p.join()

                        for ite_idx in range(len(bers)):
                            if n_bits_sent_shared_arr[ite_idx] == 0:
                                bers[ite_idx] = np.nan
                            else:
                                bers[ite_idx] = n_err_shared_arr[ite_idx] / n_bits_sent_shared_arr[ite_idx]
                        ber_per_dist.append(bers)
                    ber_per_dist = np.column_stack(ber_per_dist)
                    print("--- Computation time: %f ---" % (time.time() - start_time))

                    # %%
                    fig1, ax1 = plt.subplots(1, 1)
                    ax1.set_yscale('log')
                    ax1.plot(ebn0_arr, ber_per_dist[0], label="No distortion")
                    for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
                        if idx == 0:
                            ax1.plot(ebn0_arr, ber_per_dist[idx + 1], label="Standard RX")
                        else:
                            ax1.plot(ebn0_arr, ber_per_dist[idx + 1], label="MCNC NI = %d" % (cnc_iter_val))

                    # fix log scaling
                    ax1.set_title("BER vs Eb/N0, %s, MCNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (
                        my_miso_chan, my_mod.constellation_size, n_ant_val, ibo_val_db))
                    ax1.set_xlabel("Eb/N0 [dB]")
                    ax1.set_ylabel("BER")
                    ax1.grid()
                    ax1.legend()
                    plt.tight_layout()

                    filename_str = "ber_vs_ebn0_mcnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
                        my_miso_chan, n_ant_val, ibo_val_db, min(ebn0_arr), max(ebn0_arr), ebn0_arr[1] - ebn0_arr[0],
                        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
                    # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    # filename_str += "_" + timestamp
                    plt.savefig("figs/vm_worker_results/%s.png" % filename_str, dpi=600, bbox_inches='tight')
                    # plt.show()
                    plt.cla()
                    plt.close()

                    # %%
                    data_lst = []
                    data_lst.append(ebn0_arr)
                    for arr1 in ber_per_dist:
                        data_lst.append(arr1)
                    utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

                    # read_data = utilities.read_from_csv(filename=filename_str)

    print("Finished execution!")
