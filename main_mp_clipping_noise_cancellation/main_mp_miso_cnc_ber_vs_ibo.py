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

import antenna_arrray
import channel
import distortion
import modulation
import noise
import transceiver
from plot_settings import set_latex_plot_style
from utilities import ebn0_to_snr
import utilities

import mp_model
import multiprocessing as mp
import ctypes

if __name__ == '__main__':

    set_latex_plot_style()
    num_cores = mp.cpu_count()

    # %%
    n_ant_arr = [64]
    ebn0_db_arr = [15]
    ibo_step_arr = [0.5]
    cnc_n_iter_lst = [1, 2, 3, 4, 5, 6, 7, 8]
    # standard RX
    cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)
    incl_clean_run = False
    reroll_chan = True

    # modulation
    constel_size = 64
    n_fft = 4096
    n_sub_carr = 2048
    cp_len = 128

    # BER accuracy settings
    bits_sent_max = int(1e7)
    n_err_min = int(1e5)

    rx_loc_x, rx_loc_y = 212.0, 212.0
    rx_loc_var = 10.0

    # remember to copy objects not to avoid shared properties modifications!
    # check modifications before copy and what you copy!
    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len)

    my_distortion = distortion.SoftLimiter(ibo_db=0, avg_samp_pow=my_mod.avg_sample_power)
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

        my_miso_rayleigh_chan = channel.MisoRayleighFd(tx_transceivers=my_array.array_elements,
                                                       rx_transceiver=my_standard_rx,
                                                       seed=1234)
        chan_lst = [my_miso_los_chan, my_miso_two_path_chan, my_miso_rayleigh_chan]
        my_noise = noise.Awgn(snr_db=10, seed=1234)

        for my_miso_chan in chan_lst:

            mp_link_obj = mp_model.Link(mod_obj=my_mod, array_obj=my_array, std_rx_obj=my_standard_rx,
                                        chan_obj=my_miso_chan, noise_obj=my_noise,
                                        rx_loc_var=rx_loc_var, n_err_min=n_err_min,
                                        bits_sent_max=bits_sent_max, is_mcnc=False)

            for ibo_step_val in ibo_step_arr:
                ibo_arr = np.arange(0, 9.1, ibo_step_val)

                for ebn0_db in ebn0_db_arr:
                    start_time = time.time()
                    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    snr_db_val = ebn0_to_snr(ebn0_db, my_mod.n_sub_carr, my_mod.n_sub_carr, my_mod.constel_size)
                    bers_per_ibo = np.zeros((len(cnc_n_iter_lst), len(ibo_arr)))
                    mp_link_obj.set_snr(snr_db_val=snr_db_val)

                    # BER vs IBO eval
                    for ibo_idx, ibo_val_db in enumerate(ibo_arr):
                        mp_link_obj.update_distortion(ibo_val_db=ibo_val_db)

                        bers = np.zeros([len(cnc_n_iter_lst) + 1])
                        n_err_shared_arr = mp.Array(ctypes.c_double, len(cnc_n_iter_lst), lock=True)
                        n_bits_sent_shared_arr = mp.Array(ctypes.c_double, len(cnc_n_iter_lst), lock=True)

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

                        for ite_idx in range(len(cnc_n_iter_lst)):
                            if n_bits_sent_shared_arr[ite_idx] == 0:
                                bers_per_ibo[ite_idx][ibo_idx] = np.nan
                            else:
                                bers_per_ibo[ite_idx][ibo_idx] = n_err_shared_arr[ite_idx] / n_bits_sent_shared_arr[
                                    ite_idx]

                    print("--- Computation time: %f ---" % (time.time() - start_time))

                    # %%
                    fig1, ax1 = plt.subplots(1, 1)
                    ax1.set_yscale('log')
                    for ite_idx, ite_val in enumerate(cnc_n_iter_lst):
                        # read by columns
                        if ite_idx == 0:
                            ite_val = "0 - standard RX"
                        ax1.plot(ibo_arr, bers_per_ibo[ite_idx, :], label=ite_val)

                    ax1.set_title("BER vs IBO, %s, CNC, QAM %d, N ANT = %d, Eb/n0 = %d [dB], " % (
                        my_miso_chan, my_mod.constellation_size, n_ant_val, ebn0_db))
                    ax1.set_xlabel("IBO [dB]")
                    ax1.set_ylabel("BER")
                    ax1.grid()
                    ax1.legend(title="CNC N ite:")
                    plt.tight_layout()

                    filename_str = "ber_vs_ibo_cnc_%s_nant%d_ebn0_%d_ibo_min%d_max%d_step%1.2f_niter%s" % (
                        my_miso_chan, n_ant_val, ebn0_db, min(ibo_arr), max(ibo_arr), ibo_arr[1] - ibo_arr[0],
                        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
                    # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    # filename_str += "_" + timestamp
                    plt.savefig("figs/vm_worker_results/%s.png" % filename_str, dpi=600, bbox_inches='tight')
                    # plt.show()
                    plt.cla()
                    plt.close()

                    # %%
                    data_lst = []
                    data_lst.append(ibo_arr)
                    for arr1 in bers_per_ibo:
                        data_lst.append(arr1)
                    utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

                    read_data = utilities.read_from_csv(filename=filename_str)

    print("Finished execution!")
