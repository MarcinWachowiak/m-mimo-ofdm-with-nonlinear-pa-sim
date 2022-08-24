# antenna array evaluation
# %%
import os
import sys

sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

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
    n_ant_arr = [64]
    target_ber_arr = [1e-2]
    ebn0_step_arr = [0.5]
    ibo_step_arr = [0.5]
    cnc_n_iter_lst = [1, 2, 3, 4, 5, 6, 7, 8]
    cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)
    incl_clean_run = False
    reroll_chan = True

    # modulation
    constel_size = 64
    n_fft = 4096
    n_sub_carr = 2048
    cp_len = 128

    # BER accuracy settings
    bits_sent_max = int(5e6)
    n_err_min = int(1e5)

    rx_loc_x, rx_loc_y = 212.0, 212.0
    rx_loc_var = 10.0

    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len)
    my_distortion = distortion.SoftLimiter(ibo_db=5, avg_samp_pow=my_mod.avg_sample_power)
    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(3.5e9),
                                    carrier_spacing=int(15e3))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=rx_loc_x,
                                             cord_y=rx_loc_y, cord_z=1.5,
                                             center_freq=int(3.5e9), carrier_spacing=int(15e3))

    seed_rng = np.random.default_rng(2137)
    for n_ant_val in n_ant_arr:
        my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                              wav_len_spacing=0.5,
                                              cord_x=0, cord_y=0, cord_z=15)
        # channel type
        my_miso_los_chan = channel.MisoLosFd()
        my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                          skip_attenuation=False)
        my_miso_two_path_chan = channel.MisoTwoPathFd()
        my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                               skip_attenuation=False)

        my_miso_rayleigh_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements,
                                                       rx_transceiver=my_standard_rx,
                                                       seed=1234)
        chan_lst = [my_miso_los_chan, my_miso_two_path_chan, my_miso_rayleigh_chan]
        my_noise = noise.Awgn(snr_db=10, seed=1234)

        for my_miso_chan in chan_lst:

            mp_link_obj = mp_model.Link(mod_obj=my_mod, array_obj=my_array, std_rx_obj=my_standard_rx,
                                        chan_obj=my_miso_chan, noise_obj=my_noise,
                                        rx_loc_var=rx_loc_var, n_err_min=n_err_min,
                                        bits_sent_max=bits_sent_max, is_mcnc=False)

            for target_ber_val in target_ber_arr:

                for ibo_step_val in ibo_step_arr:
                    ibo_arr = np.arange(0, 8, ibo_step_val)

                    for ebn0_step_val in ebn0_step_arr:
                        start_time = time.time()
                        print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                        ebn0_db_arr = np.arange(10.0, 23.1, ebn0_step_val)
                        snr_db_vals = ebn0_to_snr(ebn0_db_arr, my_mod.n_sub_carr, my_mod.n_sub_carr,
                                                  my_mod.constel_size)
                        ber_per_ibo_snr_iter = np.zeros((len(ibo_arr), len(snr_db_vals), len(cnc_n_iter_lst)))

                        # %%
                        # BER vs IBO eval
                        for ibo_idx, ibo_val_db in enumerate(ibo_arr):
                            mp_link_obj.update_distortion(ibo_val_db=ibo_val_db)

                            for snr_idx, snr_db_val in enumerate(snr_db_vals):
                                mp_link_obj.set_snr(snr_db_val=snr_db_val)

                                n_err_shared_arr = mp.Array(ctypes.c_double, len(cnc_n_iter_lst), lock=True)
                                n_bits_sent_shared_arr = mp.Array(ctypes.c_double, len(cnc_n_iter_lst), lock=True)
                                bers_per_ite = np.zeros(len(cnc_n_iter_lst))

                                proc_seed_lst = seed_rng.integers(0, high=sys.maxsize, size=(num_cores, 3))
                                processes = []
                                for idx in range(num_cores):
                                    p = mp.Process(target=mp_link_obj.simulate,
                                                   args=(
                                                       incl_clean_run, reroll_chan, cnc_n_iter_lst, proc_seed_lst[idx],
                                                       n_err_shared_arr, n_bits_sent_shared_arr))
                                    processes.append(p)
                                    p.start()

                                for p in processes:
                                    p.join()

                                for ite_idx in range(len(cnc_n_iter_lst)):
                                    if n_bits_sent_shared_arr[ite_idx] == 0:
                                        ber_per_ibo_snr_iter[ibo_idx, snr_idx, ite_idx] = np.nan
                                    else:
                                        ber_per_ibo_snr_iter[ibo_idx, snr_idx, ite_idx] = n_err_shared_arr[ite_idx] / \
                                                                                          n_bits_sent_shared_arr[
                                                                                              ite_idx]

                        print("--- Computation time: %f ---" % (time.time() - start_time))

                        # %%
                        # extract SNR value providing given BER from collected data
                        req_ebn0_per_ibo = np.zeros((len(cnc_n_iter_lst), len(ibo_arr)))
                        plot_once = False
                        for iter_idx, iter_val in enumerate(cnc_n_iter_lst):
                            for ibo_idx, ibo_val in enumerate(ibo_arr):
                                ber_per_ebn0 = ber_per_ibo_snr_iter[ibo_idx, :, iter_idx]
                                # investigate better interpolation options
                                interpol_func = interpolate.interp1d(ber_per_ebn0, ebn0_db_arr)
                                if ibo_val == 0 and iter_val == 0:
                                    # ber vector
                                    if plot_once:
                                        fig1, ax1 = plt.subplots(1, 1)
                                        ax1.set_yscale('log')
                                        ax1.plot(ber_per_ebn0, ebn0_db_arr, label=iter_val)
                                        ax1.plot(ber_per_ebn0, interpol_func(ber_per_ebn0), label=iter_val)

                                        ax1.grid()
                                        plt.tight_layout()
                                        plt.show()
                                        print("Required Eb/No:", interpol_func(target_ber_val))

                                        fig2, ax2 = plt.subplots(1, 1)
                                        ax2.set_yscale('log')
                                        ax2.plot(ebn0_db_arr, ber_per_ebn0)

                                        plot_once = False
                                try:
                                    req_ebn0_per_ibo[iter_idx, ibo_idx] = interpol_func(target_ber_val)
                                except:
                                    # value not found in interpolation, replace with inf
                                    req_ebn0_per_ibo[iter_idx, ibo_idx] = np.inf

                        # %%
                        fig1, ax1 = plt.subplots(1, 1)
                        for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
                            # read by columns
                            if ite_idx == 0:
                                ite_val = "0 - standard"
                            ax1.plot(ibo_arr, req_ebn0_per_ibo[ite_idx, :], label=ite_val)

                        ax1.set_title("Fixed BER = %1.1e, %s, CNC, QAM %d, N ANT = %d" % (
                            target_ber_val, my_miso_chan, my_mod.constellation_size, n_ant_val))
                        ax1.set_xlabel("IBO [dB]")
                        ax1.set_ylabel("Eb/n0 [dB]")
                        ax1.grid()
                        ax1.legend(title="CNC N ite:")
                        plt.tight_layout()

                        filename_str = "fixed_ber%1.1e_cnc_%s_nant%d_ebn0_min%d_max%d_step%1.2f_ibo_min%d_max%d_step%1.2f_niter%s" % \
                                       (target_ber_val, my_miso_chan, n_ant_val, min(ebn0_db_arr), max(ebn0_db_arr),
                                        ebn0_db_arr[1] - ebn0_db_arr[0], min(ibo_arr), max(ibo_arr),
                                        ibo_arr[1] - ibo_arr[0], '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
                        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                        # filename_str += "_" + timestamp
                        plt.savefig("figs/vm_worker_results/%s.png" % filename_str, dpi=600, bbox_inches='tight')
                        # plt.show()
                        plt.cla()
                        plt.close()

                        # %%
                        data_lst = []
                        data_lst.append(ibo_arr)
                        for arr1 in ber_per_ibo_snr_iter:
                            for arr2 in arr1:
                                data_lst.append(arr2)
                        utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

                        read_data = utilities.read_from_csv(filename=filename_str)

    print("Finished execution!")
