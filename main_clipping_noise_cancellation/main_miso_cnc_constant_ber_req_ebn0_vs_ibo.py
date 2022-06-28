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
import corrector
import distortion
import modulation
import noise
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, ebn0_to_snr

# TODO: consider logger

set_latex_plot_style()
# %%
# %%
n_ant_arr = [1, 8]
target_ber_arr = [1e-2]
ebn0_step_arr = [2]
ibo_step_arr = [1]
cnc_n_iter_lst = [1, 2, 3, 5, 8]
cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)

# print("Eb/n0 values:", ebn0_db_arr)
# print("IBO values:", ibo_arr)
# print("CNC N iterations:", cnc_n_iter_lst)

#modulation
constel_size = 64
n_fft = 4096
n_sub_carr = 2048
cp_len = 128

# BER accuracy settings
bits_sent_max = int(1e6)
n_err_min = 1e3

my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len)
my_distortion = distortion.SoftLimiter(ibo_db=5, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))
my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=212,
                                cord_y=212, cord_z=1.5,
                                center_freq=int(3.5e9), carrier_spacing=int(15e3))


for n_ant_val in n_ant_arr:
    my_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5,
                                          cord_x=0, cord_y=0, cord_z=15)
    # channel type
    my_miso_los_chan = channel.MisoLosFd()
    my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                      skip_attenuation=False)
    my_miso_two_path_chan = channel.MisoTwoPathFd()
    my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                           skip_attenuation=False)

    my_miso_rayleigh_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                                   seed=1234)
    chan_lst = [my_miso_two_path_chan, my_miso_los_chan, my_miso_rayleigh_chan]

    for my_miso_chan in chan_lst:
        my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))
        cnc_n_upsamp = int(my_mod.n_fft / my_mod.n_sub_carr)

        chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
        my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)
        agc_corr_vec = np.sqrt(np.sum(np.power(np.abs(chan_mat_at_point), 2), axis=0))
        for target_ber_val in target_ber_arr:

            for ibo_step_val in ibo_step_arr:
                ibo_arr = np.arange(0, 8, ibo_step_val)

                for ebn0_step_val in ebn0_step_arr:
                    ebn0_db_arr = np.arange(10, 31, ebn0_step_val)

                    snr_db_vals = ebn0_to_snr(ebn0_db_arr, my_mod.n_fft, my_mod.n_sub_carr, my_mod.constel_size)
                    print("SNR value:", snr_db_vals)

                    ber_per_ibo_snr_iter = np.zeros((len(ibo_arr), len(snr_db_vals), len(cnc_n_iter_lst)))

                    estimate_lambda = False
                    # lambda estimation phase
                    if estimate_lambda:
                        abs_alpha_per_ibo = np.zeros(len(ibo_arr))
                        for ibo_idx, ibo_val_db in enumerate(ibo_arr):
                            start_time = time.time()
                            print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
                            # correct avg sample power in nonlinearity after precoding

                            # estimate lambda correcting coefficient
                            # same seed is required
                            bit_rng = np.random.default_rng(4321)
                            dist_rx_pow_coeff = (np.abs(my_mod.calc_alpha(ibo_val_db))) ** 2

                            n_ofdm_symb = 1e3
                            ofdm_symb_idx = 0
                            lambda_numerator_vecs = []
                            lambda_denominator_vecs = []
                            while ofdm_symb_idx < n_ofdm_symb:
                                tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
                                tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

                                rx_sig_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
                                rx_sig_clean_fd = my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol_fd)

                                clean_nsc_ofdm_symb_fd = np.concatenate(
                                    (rx_sig_clean_fd[-my_mod.n_sub_carr // 2:], rx_sig_clean_fd[1:(my_mod.n_sub_carr // 2) + 1]))
                                rx_nsc_ofdm_symb_fd = np.concatenate(
                                    (rx_sig_fd[-my_mod.n_sub_carr // 2:], rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))

                                lambda_numerator_vecs.append(np.multiply(rx_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))
                                lambda_denominator_vecs.append(np.multiply(clean_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))

                                ofdm_symb_idx += 1

                                # calculate lambda estimate
                            lambda_num = np.average(np.vstack(lambda_numerator_vecs), axis=0)
                            lambda_denum = np.average(np.vstack(lambda_denominator_vecs), axis=0)
                            abs_alpha_per_ibo[ibo_idx] = np.abs(np.average(lambda_num / lambda_denum))

                            print("--- Computation time: %f ---" % (time.time() - start_time))
                    else:
                        abs_alpha_per_ibo = my_mod.calc_alpha(ibo_arr)

                    # %%
                    # BER vs IBO eval
                    for ibo_idx, ibo_val_db in enumerate(ibo_arr):
                        start_time = time.time()
                        print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
                        my_cnc_rx.update_distortion(ibo_db=ibo_val_db)

                        for snr_idx, snr_val_db in enumerate(snr_db_vals):

                            for iter_idx, cnc_iter_val in enumerate(cnc_n_iter_lst):

                                my_noise = noise.Awgn(snr_db=20, noise_p_dbm=-90, seed=1234)
                                my_noise.snr_db = snr_val_db
                                # same seed is required
                                bit_rng = np.random.default_rng(4321)
                                bers = np.zeros([len(cnc_n_iter_lst)])
                                n_err = 0
                                bits_sent = 0

                                while bits_sent < bits_sent_max and n_err < n_err_min:
                                    tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)

                                    tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True,
                                                                                                return_both=True)
                                    rx_ofdm_symbol_fd = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
                                    rx_ofdm_symbol_fd = my_noise.process(rx_ofdm_symbol_fd, avg_sample_pow=my_mod.avg_symbol_power * (
                                        np.average(agc_corr_vec ** 2)) * abs_alpha_per_ibo[ibo_idx] ** 2, disp_data=False)
                                    rx_ofdm_symbol_fd = np.divide(rx_ofdm_symbol_fd, agc_corr_vec)

                                    # CNC reception
                                    rx_bits = my_cnc_rx.receive(n_iters=cnc_iter_val, in_sig_fd=rx_ofdm_symbol_fd)

                                    n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
                                    bits_sent += my_mod.n_bits_per_ofdm_sym
                                    n_err += n_bit_err

                                ber_per_ibo_snr_iter[ibo_idx, snr_idx, iter_idx] = n_err / bits_sent

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
                            if ibo_val == 2 and iter_val == 0:
                                # ber vector
                                if plot_once:
                                    fig1, ax1 = plt.subplots(1, 1)
                                    ax1.set_yscale('log')
                                    ax1.plot(ber_per_ebn0, ebn0_db_arr, label=iter_val)
                                    ax1.plot(ber_per_ebn0, interpol_func(ber_per_ebn0), label=iter_val)

                                    ax1.grid()
                                    plt.tight_layout()
                                    # plt.show()
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

                    ax1.set_title("Fixed BER = %1.1e, %s, CNC, QAM %d, N ANT = %d" % (target_ber_val, my_miso_chan, my_mod.constellation_size, n_ant_val))
                    ax1.set_xlabel("IBO [dB]")
                    ax1.set_ylabel("Eb/n0 [dB]")
                    ax1.grid()
                    ax1.legend(title="CNC N ite:")
                    plt.tight_layout()

                    filename_str = "fixed_ber%1.1e_cnc_%s_nant%d_ebn0_min%d_max%d_step%1.2f_ibo_min%d_max%d_step%1.2f_niter%s" % \
                                   (target_ber_val, my_miso_chan, n_ant_val, min(ebn0_db_arr), max(ebn0_db_arr), ebn0_db_arr[1]-ebn0_db_arr[0], min(ibo_arr), max(ibo_arr), ibo_arr[1]-ibo_arr[0], '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
                    # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    # filename_str += "_" + timestamp
                    plt.savefig("figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
                    # plt.show()
                    plt.cla()
                    plt.close()
                    
                    #%%
                    data_lst = []
                    data_lst.append(ibo_arr)
                    for arr1 in req_ebn0_per_ibo:
                        data_lst.append(arr1)
                    utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

                    read_data = utilities.read_from_csv(filename=filename_str)


print("Finished execution!")