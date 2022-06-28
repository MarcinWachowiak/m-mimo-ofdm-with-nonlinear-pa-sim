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
import corrector
import distortion
import modulation
import noise
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits

set_latex_plot_style()

# %%
# parameters
n_ant_arr = [1, 8]
ibo_arr = [0, 2, 3, 5, 7]
ebn0_step = [0.5, 1, 2]
cnc_n_iter_lst = [1, 2, 3, 5, 8]
# include clean run is always True
# no distortion and standard RX always included
cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)
cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)

# print("Distortion IBO/TOI value:", ibo_db)
# print("Eb/n0 values: ", ebn0_arr)
# print("CNC iterations: ", cnc_n_iter_lst)

#modulation
constel_size = 64
n_fft = 4096
n_sub_carr = 2048
cp_len = 128

# accuracy
bits_sent_max = int(1e6)
n_err_min = 1000

my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len)


my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                         cord_x=212, cord_y=212, cord_z=1.5,
                                         center_freq=int(3.5e9), carrier_spacing=int(15e3))

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
    
    my_miso_rayleigh_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                                   seed=1234)
    chan_lst = [my_miso_two_path_chan, my_miso_los_chan, my_miso_rayleigh_chan]
    
    for my_miso_chan in chan_lst:
        my_mcnc_rx = corrector.McncReceiver(copy.deepcopy(my_array), copy.deepcopy(my_miso_chan))
        cnc_n_upsamp = int(my_mod.n_fft / my_mod.n_sub_carr)
    
        for ibo_val in ibo_arr:
            ibo_db = ibo_val
    
            for ebn0_step_val in ebn0_step:
                ebn0_arr = np.arange(0, 31, ebn0_step_val)
    
                my_noise = noise.Awgn(snr_db=10, seed=1234)
                bit_rng = np.random.default_rng(4321)
                snr_arr = ebn0_arr
    
                chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
                my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)
                agc_corr_vec = np.sqrt(np.sum(np.power(np.abs(chan_mat_at_point), 2), axis=0))
                agc_corr_nsc = np.concatenate((agc_corr_vec[-my_mod.n_sub_carr // 2:], agc_corr_vec[1:(my_mod.n_sub_carr // 2) + 1]))
    
                # %%
                my_array.update_distortion(ibo_db=ibo_db, avg_sample_pow=my_mod.avg_sample_power)
                my_mcnc_rx.update_distortion(ibo_db=ibo_db)
                abs_lambda = my_mod.calc_alpha(ibo_db=ibo_db)
                ber_per_dist = []
    
                for run_idx, cnc_n_iter_val in enumerate(cnc_n_iter_lst):
                    start_time = time.time()
                    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
                    bers = np.zeros([len(snr_arr)])
                    for idx, snr in enumerate(snr_arr):
                        my_noise.snr_db = snr
                        n_err = 0
                        bits_sent = 0
    
                        while bits_sent < bits_sent_max and n_err < n_err_min:
                            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
                            tx_ofdm_symbol, clean_ofdm_symbol = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)
                            if run_idx == 0:
                                rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol)
                                rx_ofdm_symbol = my_noise.process(rx_ofdm_symbol, avg_sample_pow=my_mod.avg_symbol_power * np.mean(
                                    np.power(agc_corr_nsc, 2)), disp_data=False)
                            else:
                                rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol)
                                rx_ofdm_symbol = my_noise.process(rx_ofdm_symbol, avg_sample_pow=my_mod.avg_symbol_power * np.mean(
                                    np.power(agc_corr_nsc, 2)) * abs_lambda ** 2)
                            # apply AGC
                            rx_ofdm_symbol = rx_ofdm_symbol / agc_corr_vec
    
                            if run_idx == 0:
                                # standard reception
                                rx_ofdm_symbol = utilities.to_time_domain(rx_ofdm_symbol)
                                rx_ofdm_symbol = np.concatenate((rx_ofdm_symbol[-my_mod.cp_len:], rx_ofdm_symbol))
                                rx_bits = my_standard_rx.receive(rx_ofdm_symbol)
                            else:
                                # enchanced CNC reception
                                rx_bits = my_mcnc_rx.receive(n_iters_lst=cnc_n_iter_val, in_sig_fd=rx_ofdm_symbol)
    
                            n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
                            bits_sent += my_mod.n_bits_per_ofdm_sym
                            n_err += n_bit_err
                        bers[idx] = n_err / bits_sent
                    ber_per_dist.append(bers)
                    print("--- Computation time: %f ---" % (time.time() - start_time))
    
                # %%
                fig1, ax1 = plt.subplots(1, 1)
                ax1.set_yscale('log')
                for idx, cnc_iter_val in enumerate(cnc_n_iter_lst):
                    if idx == 0:
                        ax1.plot(ebn0_arr, ber_per_dist[idx], label="No distortion")
                    elif idx == 1:
                        ax1.plot(ebn0_arr, ber_per_dist[idx], label="Standard RX")
                    else:
                        ax1.plot(ebn0_arr, ber_per_dist[idx], label="MCNC NI = %d" % (cnc_iter_val))
    
                # fix log scaling
                ax1.set_title("BER vs Eb/N0, %s, MCNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (my_miso_chan, my_mod.constellation_size, n_ant_val, ibo_db))
                ax1.set_xlabel("Eb/N0 [dB]")
                ax1.set_ylabel("BER")
                ax1.grid()
                ax1.legend()
                plt.tight_layout()
    
                filename_str = "ber_vs_ebn0_mcnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (my_miso_chan, n_ant_val, ibo_db, min(ebn0_arr), max(ebn0_arr), ebn0_arr[1]-ebn0_arr[0], '_'.join([str(val) for val in cnc_n_iter_lst[2:]]))
                # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                # filename_str += "_" + timestamp
                plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
                plt.show()
    
                #%%
                data_lst = []
                data_lst.append(ebn0_arr)
                for arr1 in ber_per_dist:
                    data_lst.append(arr1)
                utilities.save_to_csv(data_lst=data_lst, filename=filename_str)
    
                # read_data = utilities.read_from_csv(filename=filename_str)
    
print("Finished execution!")
