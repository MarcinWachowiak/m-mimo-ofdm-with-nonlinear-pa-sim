"""
Simulate the clipping noise cancellation (CNC) receiver in a multi-antenna scenario with the Quadriga channel model,
measure the BER as a function of Eb/N0 for selected number of iterations.
"""

# %%
import os
import sys

import corrector

sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime

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

if __name__ == '__main__':

    set_latex_plot_style(use_tex=False, fig_width_in=5)

    # parameters
    n_ant_arr = [64]
    ibo_arr = [0]
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

    center_freq = 3.5e9
    subcarr_spacing = 15e3

    distance = 300
    bandwidth = n_sub_carr * subcarr_spacing
    channel_model_str = '3GPP_38.901_UMa_LOS'

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
    for n_ant_val in n_ant_arr:
        my_array = antenna_array.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx,
                                              center_freq=int(center_freq),
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
        my_miso_quadriga_chan = channel.MisoQuadrigaFd(tx_transceivers=my_array.array_elements,
                                                       rx_transceiver=my_standard_rx,
                                                       channel_model_str=channel_model_str)
        chan_lst = [my_miso_quadriga_chan]

        for my_miso_chan in chan_lst:
            loc_rng = np.random.default_rng(2137)
            my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))

            for ibo_val_db in ibo_arr:
                my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
                my_cnc_rx.update_distortion(ibo_db=ibo_val_db)

                for ebn0_step_val in ebn0_step:
                    ebn0_arr = np.arange(5, 21, ebn0_step_val)

                    my_noise = noise.Awgn(snr_db=10, seed=1234)
                    bit_rng = np.random.default_rng(4321)
                    snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_sub_carr, my_mod.n_sub_carr, my_mod.constel_size)

                    ber_per_dist = []
                    start_time = time.time()
                    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    for snr_idx, snr_db_val in enumerate(snr_arr):
                        my_noise.snr_db = snr_db_val
                        utilities.print_progress_bar(snr_idx + 1, len(snr_arr), prefix='SNR loop progress:')

                        bers = np.zeros([len(cnc_n_iter_lst) + 1])
                        n_err = np.zeros([len(cnc_n_iter_lst) + 1])
                        bits_sent = np.zeros([len(cnc_n_iter_lst) + 1])
                        # clean RX run
                        snap_cnt = 0
                        while True:
                            # for direct visibility channel and CNC algorithm channel impact must be averaged
                            if not isinstance(my_miso_chan, channel.MisoRayleighFd):
                                # reroll location
                                my_standard_rx.set_position(
                                    cord_x=rx_loc_x + loc_rng.uniform(low=-rx_loc_var / 2.0, high=rx_loc_var / 2.0),
                                    cord_y=rx_loc_y + loc_rng.uniform(low=-rx_loc_var / 2.0, high=rx_loc_var / 2.0),
                                    cord_z=my_standard_rx.cord_z)
                                my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                              rx_transceiver=my_standard_rx)
                            else:
                                my_miso_rayleigh_chan.reroll_channel_coeffs()

                            chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
                            my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)
                            my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                            hk_mat = np.concatenate((chan_mat_at_point[:, -my_mod.n_sub_carr // 2:],
                                                     chan_mat_at_point[:, 1:(my_mod.n_sub_carr // 2) + 1]), axis=1)
                            vk_mat = my_array.get_precoding_mat()
                            vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)
                            hk_vk_agc = np.multiply(hk_mat, vk_mat)
                            hk_vk_agc_avg_vec = np.sum(hk_vk_agc, axis=0)
                            hk_vk_noise_scaler = np.mean(np.power(np.abs(hk_vk_agc_avg_vec), 2))

                            hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                            hk_vk_agc_nfft[-(n_sub_carr // 2):] = hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                            hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = hk_vk_agc_avg_vec[n_sub_carr // 2:]

                            ibo_vec = 10 * np.log10(
                                10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                            ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
                            ak_vect = np.expand_dims(ak_vect, axis=1)

                            ak_hk_vk_agc = ak_vect * hk_vk_agc
                            ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)
                            ak_hk_vk_noise_scaler = np.mean(np.power(np.abs(ak_hk_vk_agc_avg_vec), 2))

                            ak_hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                            ak_hk_vk_agc_nfft[-(n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                            ak_hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[n_sub_carr // 2:]

                            if np.logical_and((n_err[0] < n_err_min), (bits_sent[0] < bits_sent_max)):
                                tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
                                clean_ofdm_symbol = my_array.transmit(tx_bits, out_domain_fd=True, return_both=False,
                                                                      skip_dist=True)
                                clean_rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol)
                                clean_rx_ofdm_symbol = my_noise.process(clean_rx_ofdm_symbol,
                                                                        avg_sample_pow=my_mod.avg_symbol_power * hk_vk_noise_scaler,
                                                                        disp_data=False)
                                clean_rx_ofdm_symbol = np.divide(clean_rx_ofdm_symbol, hk_vk_agc_nfft)
                                clean_rx_ofdm_symbol = utilities.to_time_domain(clean_rx_ofdm_symbol)
                                clean_rx_ofdm_symbol = np.concatenate(
                                    (clean_rx_ofdm_symbol[-my_mod.cp_len:], clean_rx_ofdm_symbol))
                                rx_bits = my_standard_rx.receive(clean_rx_ofdm_symbol)

                                n_bit_err = utilities.count_mismatched_bits(tx_bits, rx_bits)
                                n_err[0] += n_bit_err
                                bits_sent[0] += my_mod.n_bits_per_ofdm_sym
                            else:
                                break
                            snap_cnt += 1
                        # print("Eb/N0: %1.1f, chan_rerolls: %d" %(utilities.snr_to_ebn0(snr=snr_db_val, n_fft=n_sub_carr, n_sub_carr=n_sub_carr, constel_size=constel_size), snap_cnt))

                        # distorted RX run
                        snap_cnt = 0
                        while True:
                            ite_use_flags = np.logical_and((n_err[1:] < n_err_min), (bits_sent[1:] < bits_sent_max))

                            if ite_use_flags.any() == True:
                                curr_ite_lst = cnc_n_iter_lst[ite_use_flags]
                            else:
                                break

                            # for direct visibility channel and CNC algorithm channel impact must be averaged
                            snap_cnt += 1
                            if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan,
                                                                                         channel.MisoTwoPathFd) \
                                    or isinstance(my_miso_chan, channel.MisoQuadrigaFd):
                                # reroll location
                                my_standard_rx.set_position(
                                    cord_x=rx_loc_x + loc_rng.uniform(low=-rx_loc_var / 2.0, high=rx_loc_var / 2.0),
                                    cord_y=rx_loc_y + loc_rng.uniform(low=-rx_loc_var / 2.0, high=rx_loc_var / 2.0),
                                    cord_z=my_standard_rx.cord_z)
                                my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                              rx_transceiver=my_standard_rx)
                            else:
                                my_miso_rayleigh_chan.reroll_channel_coeffs()

                            chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
                            my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)
                            my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                            hk_mat = np.concatenate((chan_mat_at_point[:, -my_mod.n_sub_carr // 2:],
                                                     chan_mat_at_point[:, 1:(my_mod.n_sub_carr // 2) + 1]), axis=1)
                            vk_mat = my_array.get_precoding_mat()
                            vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)
                            hk_vk_agc = np.multiply(hk_mat, vk_mat)
                            hk_vk_agc_avg_vec = np.sum(hk_vk_agc, axis=0)
                            hk_vk_noise_scaler = np.mean(np.power(np.abs(hk_vk_agc_avg_vec), 2))

                            hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                            hk_vk_agc_nfft[-(n_sub_carr // 2):] = hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                            hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = hk_vk_agc_avg_vec[n_sub_carr // 2:]

                            ibo_vec = 10 * np.log10(
                                10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                            ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
                            ak_vect = np.expand_dims(ak_vect, axis=1)

                            ak_hk_vk_agc = ak_vect * hk_vk_agc
                            ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)
                            ak_hk_vk_noise_scaler = np.mean(np.power(np.abs(ak_hk_vk_agc_avg_vec), 2))

                            ak_hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                            ak_hk_vk_agc_nfft[-(n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                            ak_hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[n_sub_carr // 2:]

                            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
                            tx_ofdm_symbol = my_array.transmit(tx_bits, out_domain_fd=True, skip_dist=False)
                            rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol)
                            rx_ofdm_symbol = my_noise.process(rx_ofdm_symbol,
                                                              avg_sample_pow=my_mod.avg_symbol_power * ak_hk_vk_noise_scaler)
                            # apply AGC

                            # enchanced CNC reception
                            rx_ofdm_symbol = np.divide(rx_ofdm_symbol, ak_hk_vk_agc_nfft)
                            rx_bits_per_iter_lst = my_cnc_rx.receive(n_iters_lst=curr_ite_lst, in_sig_fd=rx_ofdm_symbol)

                            ber_idx = np.array(list(range(len(cnc_n_iter_lst))))
                            act_ber_idx = ber_idx[ite_use_flags] + 1
                            for idx in range(len(rx_bits_per_iter_lst)):
                                n_bit_err = utilities.count_mismatched_bits(tx_bits, rx_bits_per_iter_lst[idx])
                                n_err[act_ber_idx[idx]] += n_bit_err
                                bits_sent[act_ber_idx[idx]] += my_mod.n_bits_per_ofdm_sym
                            snap_cnt += 1

                        # print("Eb/N0: %1.1f, chan_rerolls: %d" %(utilities.snr_to_ebn0(snr=snr_db_val, n_fft=n_sub_carr, n_sub_carr=n_sub_carr, constel_size=constel_size), snap_cnt))
                        for ite_idx in range(len(bers)):
                            bers[ite_idx] = n_err[ite_idx] / bits_sent[ite_idx]
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
                    ax1.set_title("BER vs Eb/N0, %s, CNC, QAM %d, N ANT = %d, IBO = %d [dB]" % (
                        channel_model_str, my_mod.constellation_size, n_ant_val, ibo_val_db))
                    ax1.set_xlabel("Eb/N0 [dB]")
                    ax1.set_ylabel("BER")
                    ax1.grid()
                    ax1.legend()
                    plt.tight_layout()

                    filename_str = "ber_vs_ebn0_cnc_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s" % (
                        channel_model_str, n_ant_val, ibo_val_db, min(ebn0_arr), max(ebn0_arr),
                        ebn0_arr[1] - ebn0_arr[0],
                        '_'.join([str(val) for val in cnc_n_iter_lst[1:]]))
                    # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    # filename_str += "_" + timestamp
                    plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')
                    plt.show()
                    # plt.cla()
                    # plt.close()

                    # %%
                    data_lst = []
                    data_lst.append(ebn0_arr)
                    for arr1 in ber_per_dist:
                        data_lst.append(arr1)
                    utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

                    # read_data = utilities.read_from_csv(filename=filename_str)

    print("Finished execution!")
