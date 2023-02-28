"""
Multiprocessing/parallel version of:
Simulate the clipping noise cancellation (CNC) receiver with LDPC channel coding in a multi-antenna scenario,
measure the BER as a function of Eb/N0 for selected number of iterations and channels.
"""


# %%
import ctypes
import os
import sys

sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

import channel
import distortion
import modulation
import noise
import transceiver
import antenna_array
import utilities
from plot_settings import set_latex_plot_style
from utilities import ebn0_to_snr
import mp_ldpc_model

if __name__ == '__main__':

    set_latex_plot_style()
    num_cores = 8

    # parameters
    n_ant_arr = [64]
    ibo_arr = [0]
    ebn0_step = [0.25]
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

    center_freq = 3.5e9
    subcarr_spacing = 15e3

    distance = 300
    # bandwidth = n_sub_carr * subcarr_spacing
    # channel_model_str = '3GPP_38.901_UMa_LOS'

    # main code tuning variable
    code_rate_str_lst = ["1/3", "2/3"]
    ebn0_bounds_arr = [[-5.0, 5.1], [0.0, 15.1]]
    max_ldpc_ite = 12

    # accuracy
    bits_sent_max = int(1e7)
    n_err_min = int(1e6)

    rx_loc_x, rx_loc_y = 212.0, 212.0
    rx_loc_var = 10.0

    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len)

    my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
    # my_distortion = distortion.Rapp(ibo_db=0, p_hardness=4.0, avg_samp_pow=my_mod.avg_sample_power)

    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(center_freq),
                                    carrier_spacing=int(subcarr_spacing))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=rx_loc_x, cord_y=rx_loc_y, cord_z=1.5,
                                             center_freq=int(center_freq), carrier_spacing=int(subcarr_spacing))
    seed_rng = np.random.default_rng(2137)

    for code_idx, code_rate_str in enumerate(code_rate_str_lst):
        ebn0_min = ebn0_bounds_arr[code_idx][0]
        ebn0_max = ebn0_bounds_arr[code_idx][1]

        num, den = code_rate_str.split('/')
        code_rate = float(num) / float(den)

        for n_ant_val in n_ant_arr:

            my_array = antenna_array.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx,
                                                  center_freq=int(center_freq),
                                                  wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
            # channel type

            my_miso_los_chan = channel.MisoLosFd()
            my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                              skip_attenuation=False)

            chan_lst = [my_miso_los_chan]
            my_noise = noise.Awgn(snr_db=10, seed=1234)

            for my_miso_chan in chan_lst:

                mp_link_obj = mp_ldpc_model.LinkLdpc(mod_obj=my_mod, array_obj=my_array, std_rx_obj=my_standard_rx,
                                                     chan_obj=my_miso_chan, noise_obj=my_noise,
                                                     rx_loc_var=rx_loc_var, n_err_min=n_err_min,
                                                     bits_sent_max=bits_sent_max, is_mcnc=False, code_rate=code_rate,
                                                     max_ldpc_ite=12)

                for ibo_val_db in ibo_arr:
                    mp_link_obj.update_distortion(ibo_val_db=ibo_val_db)

                    for ebn0_step_val in ebn0_step:

                        start_time = time.time()
                        print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                        ebn0_arr = np.arange(ebn0_min, ebn0_max, ebn0_step_val)
                        snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_sub_carr, my_mod.n_sub_carr, my_mod.constel_size)
                        ber_per_dist = []

                        for snr_idx, snr_db_val in enumerate(snr_arr):
                            utilities.print_progress_bar(snr_idx + 1, len(snr_arr), prefix='SNR loop progress:')

                            mp_link_obj.set_snr(snr_db_val=snr_db_val)
                            bers = np.zeros([len(cnc_n_iter_lst) + 1])
                            n_err_shared_arr = mp.Array(ctypes.c_double, len(cnc_n_iter_lst) + 1, lock=True)
                            n_bits_sent_shared_arr = mp.Array(ctypes.c_double, len(cnc_n_iter_lst) + 1, lock=True)

                            proc_seed_lst = seed_rng.integers(0, high=sys.maxsize, size=(num_cores, 6))
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

                        ax1.plot(ebn0_arr, ber_per_dist[0, :], label="No distortion")
                        for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
                            if idx == 0:
                                ax1.plot(ebn0_arr, ber_per_dist[idx + 1, :], label="Standard RX")
                            else:
                                ax1.plot(ebn0_arr, ber_per_dist[idx + 1, :], label="CNC NI = %d" % cnc_iter_val)

                        # fix log scaling
                        ax1.set_title("BER vs Eb/N0, LDPC %s, %s, CNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (
                            code_rate_str, my_miso_chan, my_mod.constellation_size, n_ant_val, ibo_val_db))
                        ax1.set_xlabel("Eb/N0 [dB]")
                        ax1.set_ylabel("BER")
                        ax1.grid()
                        ax1.legend()
                        plt.tight_layout()

                        filename_str = "ldpc_%s_ber_vs_ebn0_cnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
                        code_rate_str.replace('/', '_'),
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
