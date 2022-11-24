# antenna array evaluation
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
from matplotlib.ticker import MaxNLocator

import antenna_arrray
import channel
import distortion
import modulation
import noise
import transceiver
import utilities
from plot_settings import set_latex_plot_style

if __name__ == '__main__':
    set_latex_plot_style()
    CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                      '#CFCFCF']
    # Multiple users data
    usr_angles = np.array([-30, 60])
    usr_distances = [100, 300]
    usr_pos_tup = []
    for usr_idx, usr_angle in enumerate(usr_angles + 90):
        usr_pos_x = np.cos(np.deg2rad(usr_angle)) * usr_distances[usr_idx]
        usr_pos_y = np.sin(np.deg2rad(usr_angle)) * usr_distances[usr_idx]
        usr_pos_tup.append((usr_pos_x, usr_pos_y))
    # custom x, y coordinates
    # usr_pos_tup = [(45, 45), (120, 120), (150, 150)]
    n_users = len(usr_pos_tup)

    n_ant_arr = [16]
    ibo_arr = [0]
    ebn0_step = [1]
    mcnc_n_iter_lst = [1, 2, 3, 4]  # 5, 6, 7, 8]
    # include clean run is always True
    # no distortion and standard RX always included
    mcnc_n_iter_lst = np.insert(mcnc_n_iter_lst, 0, 0)

    # print("Distortion IBO/TOI value:", ibo_db)
    # print("Eb/n0 values: ", ebn0_arr)
    # print("mmcnc iterations: ", mcnc_n_iter_lst)

    # modulation
    constel_size = 64
    n_fft = 4096
    n_sub_carr = 2048
    cp_len = 1

    # BER analysis
    bits_sent_max = int(1e5)
    n_err_min = int(1e4)
    ber_reroll_pos = False

    rx_loc_x, rx_loc_y = 212.0, 212.0
    rx_loc_var = 10.0

    # SDR
    meas_usr_sdr = True
    sdr_n_snapshots = 10
    sdr_reroll_pos = False

    # Beampatterns
    plot_precoding_beampatterns = False
    beampattern_n_snapshots = 10
    n_points = 180 * 1
    radial_distance = usr_distances[0]
    rx_points = utilities.pts_on_semicircum(r=radial_distance, n=n_points)
    radian_vals = np.radians(np.linspace(-90, 90, n_points + 1))

    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len,
                                     n_users=len(usr_angles))
    my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=rx_loc_x, cord_y=rx_loc_y, cord_z=1.5,
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

        my_miso_rayleigh_chan = channel.MisoRayleighFd(tx_transceivers=my_array.array_elements,
                                                       rx_transceiver=my_standard_rx,
                                                       seed=1234)
        chan_lst = [my_miso_los_chan]

        for my_miso_chan in chan_lst:

            loc_rng = np.random.default_rng(2137)

            for ibo_val_db in ibo_arr:
                usr_chan_mat_lst = []
                my_mcnc_rx_lst = []
                for usr_idx, usr_pos_tuple_val in enumerate(usr_pos_tup):
                    usr_pos_x, usr_pos_y = usr_pos_tuple_val
                    my_standard_rx.set_position(cord_x=usr_pos_x, cord_y=usr_pos_y, cord_z=1.5)

                    if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan, channel.MisoTwoPathFd):
                        my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                      rx_transceiver=my_standard_rx,
                                                      skip_attenuation=False)
                    else:
                        my_miso_chan.reroll_channel_coeffs()
                    usr_chan_mat_lst.append(my_miso_chan.get_channel_mat_fd())

                    # array parameters automatically updated in the MCNC rx due to shared object - not copied
                    my_mcnc_rx = corrector.McncMuReceiver(my_array, copy.deepcopy(my_miso_chan), usr_idx=usr_idx)
                    my_mcnc_rx_lst.append(my_mcnc_rx)

                # set precoding and calculate AGC
                my_array.set_precoding_matrix(channel_mat_fd=usr_chan_mat_lst, mr_precoding=True)
                my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
                for my_mcnc_rx_obj in my_mcnc_rx_lst:
                    my_mcnc_rx_obj.update_agc()

                estimate_alpha = False
                # lambda estimation phase
                if estimate_alpha:
                    print("Alpha estimation init!")
                    start_time = time.time()
                    bit_rng = np.random.default_rng(4321)
                    n_ofdm_symb = 1e2
                    ofdm_symb_idx = 0
                    alpha_val_vec = []

                    while ofdm_symb_idx < n_ofdm_symb:
                        tx_bits = np.squeeze(bit_rng.choice((0, 1), (n_users, my_tx.modem.n_bits_per_ofdm_sym)))
                        tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True,
                                                                                    return_both=True)
                        clean_nsc_ofdm_symb_fd = np.concatenate(
                            (clean_ofdm_symbol_fd[:, -my_mod.n_sub_carr // 2:],
                             clean_ofdm_symbol_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]), axis=1)
                        rx_nsc_ofdm_symb_fd = np.concatenate(
                            (tx_ofdm_symbol_fd[:, -my_mod.n_sub_carr // 2:],
                             tx_ofdm_symbol_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]), axis=1)

                        alpha_numerator_vec = np.multiply(rx_nsc_ofdm_symb_fd,
                                                          np.conjugate(clean_nsc_ofdm_symb_fd))
                        alpha_denominator_vec = np.multiply(clean_nsc_ofdm_symb_fd,
                                                            np.conjugate(clean_nsc_ofdm_symb_fd))

                        ofdm_symb_idx += 1
                        alpha_val_vec.append(np.abs(np.average(alpha_numerator_vec / alpha_denominator_vec, axis=1)))

                    # calculate alpha average
                    alpha_vec_est = np.average(alpha_val_vec, axis=0)
                    print("Alpha coeff estimate:", alpha_vec_est)
                    print("--- Computation time: %f ---" % (time.time() - start_time))

                ak_hk_vk_noise_scaler_lst = []
                hk_vk_noise_scaler_lst = []
                hk_vk_agc_nfft_lst = []
                ak_hk_vk_agc_nfft_lst = []

                for usr_idx, usr_pos in enumerate(usr_pos_tup):
                    chan_mat_at_point = usr_chan_mat_lst[usr_idx]
                    hk_mat = np.concatenate((chan_mat_at_point[:, -my_mod.n_sub_carr // 2:],
                                             chan_mat_at_point[:, 1:(my_mod.n_sub_carr // 2) + 1]), axis=1)
                    vk_mat = my_array.get_precoding_mat()
                    vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                    hk_vk_agc = np.multiply(hk_mat, vk_mat[:, usr_idx, :])
                    hk_vk_agc_avg_vec = np.sum(hk_vk_agc, axis=0)
                    hk_vk_noise_scaler = np.mean(np.power(np.abs(hk_vk_agc_avg_vec), 2))
                    hk_vk_noise_scaler_lst.append(hk_vk_noise_scaler)

                    hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                    hk_vk_agc_nfft[-(n_sub_carr // 2):] = hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                    hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = hk_vk_agc_avg_vec[n_sub_carr // 2:]
                    hk_vk_agc_nfft_lst.append(hk_vk_agc_nfft)

                    ibo_vec = 10 * np.log10(10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                    ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
                    # ak_vect = alpha_vec_est
                    ak_vect = np.expand_dims(ak_vect, axis=1)

                    ak_hk_vk_agc = ak_vect * hk_vk_agc
                    ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)
                    ak_hk_vk_noise_scaler = np.mean(np.power(np.abs(ak_hk_vk_agc_avg_vec), 2))
                    ak_hk_vk_noise_scaler_lst.append(ak_hk_vk_noise_scaler)

                    ak_hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                    ak_hk_vk_agc_nfft[-(n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                    ak_hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[n_sub_carr // 2:]
                    ak_hk_vk_agc_nfft_lst.append(ak_hk_vk_agc_nfft)

                # measure SDR at selected RX points
                loc_rng = np.random.default_rng(2137)
                if meas_usr_sdr:
                    print("Per user SDR")
                    print("User, \t\t X[-], \t Y[-], \t Dist[m], \t SDR[dB]")
                    bit_rng = np.random.default_rng(4321)

                    desired_sig_pow_arr = np.zeros((n_users, sdr_n_snapshots))
                    distortion_sig_pow_arr = np.zeros((n_users, sdr_n_snapshots))
                    for snap_idx in range(sdr_n_snapshots):
                        tx_bits = np.squeeze(bit_rng.choice((0, 1), (n_users, my_tx.modem.n_bits_per_ofdm_sym)))

                        if sdr_reroll_pos:
                            usr_chan_mat_lst = []
                            for usr_idx, user_pos_tup in enumerate(usr_pos_tup):
                                usr_pos_x, usr_pos_y = user_pos_tup
                                # for direct visibility channel and mcnc algorithm channel impact must be averaged
                                if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan,
                                                                                             channel.MisoTwoPathFd):
                                    # reroll location
                                    my_standard_rx.set_position(
                                        cord_x=usr_pos_x + loc_rng.uniform(low=-rx_loc_var / 2.0,
                                                                           high=rx_loc_var / 2.0),
                                        cord_y=usr_pos_y + loc_rng.uniform(low=-rx_loc_var / 2.0,
                                                                           high=rx_loc_var / 2.0),
                                        cord_z=my_standard_rx.cord_z)
                                    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                                  rx_transceiver=my_standard_rx,
                                                                  skip_attenuation=False)
                                else:
                                    my_miso_rayleigh_chan.reroll_channel_coeffs()

                                usr_chan_mat_lst.append(my_miso_chan.get_channel_mat_fd())

                            my_array.set_precoding_matrix(channel_mat_fd=usr_chan_mat_lst, mr_precoding=True)
                            my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                            vk_mat = my_array.get_precoding_mat()
                            vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                            ibo_vec = 10 * np.log10(
                                10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                            ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
                            ak_vect = np.expand_dims(ak_vect, axis=1)

                        arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True,
                                                                            return_both=True)
                        for usr_idx, user_pos_tup in enumerate(usr_pos_tup):
                            rx_sig_fd = np.multiply(arr_tx_sig_fd, usr_chan_mat_lst[usr_idx])
                            rx_sc_ofdm_symb_fd = np.concatenate(
                                (rx_sig_fd[:, -my_mod.n_sub_carr // 2:], rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                axis=1)
                            # rx_sc_ofdm_symb_td = utilities.to_time_domain(rx_sc_ofdm_symb_fd)

                            clean_rx_sig_fd = np.multiply(clean_sig_mat_fd, usr_chan_mat_lst[usr_idx])

                            clean_sc_ofdm_symb_fd = np.concatenate(
                                (clean_rx_sig_fd[:, -my_mod.n_sub_carr // 2:],
                                 clean_rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                axis=1)

                            sc_ofdm_distortion_sig = np.subtract(rx_sc_ofdm_symb_fd, (ak_vect * clean_sc_ofdm_symb_fd))

                            desired_sig_pow_arr[usr_idx, snap_idx] = np.sum(
                                np.power(np.abs(np.sum(ak_vect * clean_sc_ofdm_symb_fd, axis=0)), 2))
                            distortion_sig_pow_arr[usr_idx, snap_idx] = np.sum(
                                np.power(np.abs(np.sum(sc_ofdm_distortion_sig, axis=0)), 2))
                            # calculate SDR on OFDM symbol basis
                    sdr_per_usr = utilities.to_db(
                        np.sum(desired_sig_pow_arr, axis=1) / np.sum(distortion_sig_pow_arr, axis=1))
                    # %%
                    for usr_idx, point_tup in enumerate(usr_pos_tup):
                        print("%d, \t\t  %1.1f, \t %1.1f, \t %1.1f, \t %1.2f" % (
                            usr_idx, point_tup[0], point_tup[1], np.sqrt(point_tup[0] ** 2 + point_tup[1] ** 2),
                            sdr_per_usr[usr_idx]))

                if plot_precoding_beampatterns:
                    bit_rng = np.random.default_rng(4321)
                    desired_sig_pow_per_pt = []
                    distorted_sig_pow_per_pt = []
                    for pt_idx, point in enumerate(rx_points):
                        (x_cord, y_cord) = point
                        my_standard_rx.set_position(cord_x=x_cord, cord_y=y_cord, cord_z=1.5)
                        # update channel matrix constant for a given point
                        if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan,
                                                                                     channel.MisoTwoPathFd):
                            my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                          rx_transceiver=my_standard_rx,
                                                          skip_attenuation=False)
                        else:
                            pass
                            # if pt_idx == precoding_point_idx:
                            #     my_miso_chan.set_channel_mat_fd(channel_mat_at_point_fd)
                            # else:
                            #     my_miso_chan.reroll_channel_coeffs()

                        desired_sig_pow_arr = np.zeros(beampattern_n_snapshots)
                        distortion_sig_pow_arr = np.zeros(beampattern_n_snapshots)
                        for snap_idx in range(beampattern_n_snapshots):
                            tx_bits = np.squeeze(bit_rng.choice((0, 1), (n_users, my_tx.modem.n_bits_per_ofdm_sym)))
                            arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True,
                                                                                return_both=True)

                            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd, sum=False)
                            rx_sc_ofdm_symb_fd = np.concatenate(
                                (rx_sig_fd[:, -my_mod.n_sub_carr // 2:], rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                axis=1)
                            # rx_sc_ofdm_symb_td = utilities.to_time_domain(rx_sc_ofdm_symb_fd)

                            clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd, sum=False)

                            clean_sc_ofdm_symb_fd = np.concatenate(
                                (clean_rx_sig_fd[:, -my_mod.n_sub_carr // 2:],
                                 clean_rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                axis=1)

                            sc_ofdm_distortion_sig = np.subtract(rx_sc_ofdm_symb_fd, (ak_vect * clean_sc_ofdm_symb_fd))
                            desired_sig_pow_arr[snap_idx] = np.sum(
                                np.power(np.abs(np.sum(ak_vect * clean_sc_ofdm_symb_fd, axis=0)), 2))
                            distortion_sig_pow_arr[snap_idx] = np.sum(
                                np.power(np.abs(np.sum(sc_ofdm_distortion_sig, axis=0)), 2))
                            # calculate SDR on symbol basis
                        desired_sig_pow_per_pt.append(np.sum(desired_sig_pow_arr))
                        distorted_sig_pow_per_pt.append(np.sum(distortion_sig_pow_arr))
                    # %%
                    # plot beampatterns of desired and distortion components
                    fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(3.5, 3))
                    ax1.set_theta_zero_location("N")
                    plt.tight_layout()
                    ax1.set_thetalim(-np.pi / 2, np.pi / 2)
                    ax1.set_xticks(np.pi / 180. * np.linspace(-90, 90, 13, endpoint=True))
                    ax1.yaxis.set_major_locator(MaxNLocator(5))

                    dist_lines_lst = []

                    ax1.plot(radian_vals, utilities.to_db(desired_sig_pow_per_pt), label="Desired", linewidth=1.5)
                    ax1.plot(radian_vals, utilities.to_db(distorted_sig_pow_per_pt), label="Distorted", linewidth=1.5)

                    # plot reference angles/directions
                    (y_min, y_max) = ax1.get_ylim()
                    ax1.vlines(np.deg2rad(usr_angles), y_min, y_max, colors='k', linestyles='--')  # label="Users")
                    ax1.margins(0.0, 0.0)
                    ax1.set_title("Signal power at angle [dB]", pad=-15)
                    ax1.legend(title="Signal:", ncol=2, loc='lower center', borderaxespad=0)
                    ax1.grid(True)

                    plt.savefig(
                        "../figs/multiuser/distortion_directions_eval/zf_multiuser_%s_desired_and_distortion_signal_beampattern_ibo%d_angles%s_distances%s_npoints%d_nsnap%d_nant%s.png" % (
                            my_miso_chan, ibo_val_db, '_'.join([str(val) for val in usr_angles]),
                            '_'.join([str(val) for val in usr_distances]), n_points, beampattern_n_snapshots,
                            '_'.join([str(val) for val in [n_ant_val]])),
                        dpi=600, bbox_inches='tight')
                    # plt.show()
                    plt.cla()
                    plt.close()

                # BER measurement
                for ebn0_step_val in ebn0_step:
                    ebn0_arr = np.arange(5, 21, ebn0_step_val)

                    my_noise = noise.Awgn(snr_db=10, seed=1234)
                    bit_rng = np.random.default_rng(4321)
                    snr_arr = utilities.ebn0_to_snr(ebn0_arr, my_mod.n_sub_carr, my_mod.n_sub_carr, my_mod.constel_size)

                    bers_per_usr = [[] for x in range(n_users)]
                    start_time = time.time()
                    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    for snr_idx, snr_db_val in enumerate(snr_arr):
                        my_noise.snr_db = snr_db_val

                        bers = np.zeros((n_users, len(mcnc_n_iter_lst) + 1))
                        n_err = np.zeros((n_users, len(mcnc_n_iter_lst) + 1))
                        bits_sent = np.zeros((n_users, len(mcnc_n_iter_lst) + 1))

                        # clean RX run
                        snap_cnt = 0
                        while True:
                            # for each frame reroll position and recalculate AGC
                            if ber_reroll_pos:
                                usr_chan_mat_lst = []
                                my_mcnc_rx_lst = []
                                for usr_idx, user_pos_tup in enumerate(usr_pos_tup):
                                    usr_pos_x, usr_pos_y = user_pos_tup
                                    # for direct visibility channel and mcnc algorithm channel impact must be averaged
                                    if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan,
                                                                                                 channel.MisoTwoPathFd):
                                        # reroll location
                                        my_standard_rx.set_position(
                                            cord_x=usr_pos_x + loc_rng.uniform(low=-rx_loc_var / 2.0,
                                                                               high=rx_loc_var / 2.0),
                                            cord_y=usr_pos_y + loc_rng.uniform(low=-rx_loc_var / 2.0,
                                                                               high=rx_loc_var / 2.0),
                                            cord_z=my_standard_rx.cord_z)
                                        my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                                      rx_transceiver=my_standard_rx,
                                                                      skip_attenuation=False)
                                    else:
                                        my_miso_rayleigh_chan.reroll_channel_coeffs()
                                    usr_chan_mat_lst.append(my_miso_chan.get_channel_mat_fd())

                                    my_mcnc_rx = corrector.McncMuReceiver(my_array, copy.deepcopy(my_miso_chan),
                                                                          usr_idx=usr_idx)
                                    my_mcnc_rx_lst.append(my_mcnc_rx)

                                my_array.set_precoding_matrix(channel_mat_fd=usr_chan_mat_lst, mr_precoding=True)
                                my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                                for my_mcnc_rx_obj in my_mcnc_rx_lst:
                                    my_mcnc_rx_obj.update_agc()

                                vk_mat = my_array.get_precoding_mat()
                                vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                                ak_hk_vk_noise_scaler_lst = []
                                hk_vk_noise_scaler_lst = []
                                hk_vk_agc_nfft_lst = []
                                ak_hk_vk_agc_nfft_lst = []

                                for usr_idx, usr_pos in enumerate(usr_pos_tup):
                                    chan_mat_at_point = usr_chan_mat_lst[usr_idx]
                                    hk_mat = np.concatenate((chan_mat_at_point[:, -my_mod.n_sub_carr // 2:],
                                                             chan_mat_at_point[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                                            axis=1)
                                    vk_mat = my_array.get_precoding_mat()
                                    vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                                    hk_vk_agc = np.multiply(hk_mat, vk_mat[:, usr_idx, :])
                                    hk_vk_agc_avg_vec = np.sum(hk_vk_agc, axis=0)
                                    hk_vk_noise_scaler = np.mean(np.power(np.abs(hk_vk_agc_avg_vec), 2))
                                    hk_vk_noise_scaler_lst.append(hk_vk_noise_scaler)

                                    hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                                    hk_vk_agc_nfft[-(n_sub_carr // 2):] = hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                                    hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = hk_vk_agc_avg_vec[n_sub_carr // 2:]
                                    hk_vk_agc_nfft_lst.append(hk_vk_agc_nfft)

                                    ibo_vec = 10 * np.log10(
                                        10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                                    ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
                                    ak_vect = np.expand_dims(ak_vect, axis=1)

                                    ak_hk_vk_agc = ak_vect * hk_vk_agc
                                    ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)
                                    ak_hk_vk_noise_scaler = np.mean(np.power(np.abs(ak_hk_vk_agc_avg_vec), 2))
                                    ak_hk_vk_noise_scaler_lst.append(ak_hk_vk_noise_scaler)

                                    ak_hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                                    ak_hk_vk_agc_nfft[-(n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                                    ak_hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[n_sub_carr // 2:]
                                    ak_hk_vk_agc_nfft_lst.append(ak_hk_vk_agc_nfft)

                            # analyze single user receiver performance in multi-user MRT
                            if any(np.logical_and((n_err[:, 0] < n_err_min), (bits_sent[:, 0] < bits_sent_max))):
                                tx_bits = np.squeeze(bit_rng.choice((0, 1), (n_users, my_tx.modem.n_bits_per_ofdm_sym)))
                                clean_ofdm_symbol = my_array.transmit(tx_bits, out_domain_fd=True, return_both=False,
                                                                      skip_dist=True)
                                for usr_idx in range(n_users):
                                    # clean_rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol)
                                    clean_rx_ofdm_symbol = np.sum(
                                        np.multiply(clean_ofdm_symbol, usr_chan_mat_lst[usr_idx]), axis=0)

                                    clean_rx_ofdm_symbol = my_noise.process(clean_rx_ofdm_symbol,
                                                                            avg_sample_pow=my_mod.avg_symbol_power *
                                                                                           hk_vk_noise_scaler_lst[
                                                                                               usr_idx],
                                                                            disp_data=False)
                                    clean_rx_ofdm_symbol = np.divide(clean_rx_ofdm_symbol, hk_vk_agc_nfft_lst[usr_idx])
                                    clean_rx_ofdm_symbol = utilities.to_time_domain(clean_rx_ofdm_symbol)
                                    clean_rx_ofdm_symbol = np.concatenate(
                                        (clean_rx_ofdm_symbol[-my_mod.cp_len:], clean_rx_ofdm_symbol))
                                    rx_bits = my_standard_rx.receive(clean_rx_ofdm_symbol)

                                    n_bit_err = utilities.count_mismatched_bits(tx_bits[usr_idx], rx_bits)
                                    n_err[usr_idx, 0] += n_bit_err
                                    bits_sent[usr_idx, 0] += my_mod.n_bits_per_ofdm_sym
                            else:
                                break
                            snap_cnt += 1
                        # print("Eb/N0: %1.1f, chan_rerolls: %d" %(utilities.snr_to_ebn0(snr=snr_db_val, n_fft=n_sub_carr, n_sub_carr=n_sub_carr, constel_size=constel_size), snap_cnt))

                        # distorted RX run
                        snap_cnt = 0
                        while True:
                            ite_use_flags_per_usr = []
                            for usr_idx in range(n_users):
                                ite_use_flags = np.logical_and((n_err[usr_idx, 1:] < n_err_min),
                                                               (bits_sent[usr_idx, 1:] < bits_sent_max))
                                ite_use_flags_per_usr.append(ite_use_flags)

                            curr_ite_lst_per_usr = []
                            for usr_idx in range(n_users):
                                if ite_use_flags_per_usr[usr_idx].any() == True:
                                    curr_ite_lst = mcnc_n_iter_lst[ite_use_flags_per_usr[usr_idx]]
                                    curr_ite_lst_per_usr.append(curr_ite_lst)
                                else:
                                    curr_ite_lst_per_usr.append([])

                            # check if there is any mcnc iteration to run
                            if not all(len(curr_ite_lst) for curr_ite_lst in curr_ite_lst_per_usr):
                                break

                            # for direct visibility channel and mcnc algorithm channel impact must be averaged
                            snap_cnt += 1
                            if ber_reroll_pos:
                                usr_chan_mat_lst = []
                                my_mcnc_rx_lst = []
                                for usr_idx, usr_pos_tuple_val in enumerate(usr_pos_tup):
                                    usr_pos_x, usr_pos_y = usr_pos_tuple_val
                                    my_standard_rx.set_position(cord_x=usr_pos_x, cord_y=usr_pos_y, cord_z=1.5)

                                    if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan,
                                                                                                 channel.MisoTwoPathFd):
                                        my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                                      rx_transceiver=my_standard_rx,
                                                                      skip_attenuation=False)
                                    else:
                                        my_miso_chan.reroll_channel_coeffs()

                                    usr_chan_mat_lst.append(my_miso_chan.get_channel_mat_fd())
                                    my_mcnc_rx = corrector.McncMuReceiver(my_array, copy.deepcopy(my_miso_chan),
                                                                          usr_idx=usr_idx)
                                    my_mcnc_rx_lst.append(my_mcnc_rx)

                                # set precoding and calculate AGC
                                my_array.set_precoding_matrix(channel_mat_fd=usr_chan_mat_lst, mr_precoding=True)
                                my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
                                # array parameters automatically updated in the MCNC rx due to shared object - not copied
                                for my_mcnc_rx_obj in my_mcnc_rx_lst:
                                    my_mcnc_rx_obj.update_agc()

                                vk_mat = my_array.get_precoding_mat()
                                vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                                ak_hk_vk_noise_scaler_lst = []
                                hk_vk_noise_scaler_lst = []
                                hk_vk_agc_nfft_lst = []
                                ak_hk_vk_agc_nfft_lst = []

                                for usr_idx, usr_pos in enumerate(usr_pos_tup):
                                    chan_mat_at_point = usr_chan_mat_lst[usr_idx]
                                    hk_mat = np.concatenate((chan_mat_at_point[:, -my_mod.n_sub_carr // 2:],
                                                             chan_mat_at_point[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                                            axis=1)
                                    vk_mat = my_array.get_precoding_mat()
                                    vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                                    hk_vk_agc = np.multiply(hk_mat, vk_mat[:, usr_idx, :])
                                    hk_vk_agc_avg_vec = np.sum(hk_vk_agc, axis=0)
                                    hk_vk_noise_scaler = np.mean(np.power(np.abs(hk_vk_agc_avg_vec), 2))
                                    hk_vk_noise_scaler_lst.append(hk_vk_noise_scaler)

                                    hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                                    hk_vk_agc_nfft[-(n_sub_carr // 2):] = hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                                    hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = hk_vk_agc_avg_vec[n_sub_carr // 2:]
                                    hk_vk_agc_nfft_lst.append(hk_vk_agc_nfft)

                                    ibo_vec = 10 * np.log10(
                                        10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                                    ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
                                    ak_vect = np.expand_dims(ak_vect, axis=1)

                                    ak_hk_vk_agc = ak_vect * hk_vk_agc
                                    ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)
                                    ak_hk_vk_noise_scaler = np.mean(np.power(np.abs(ak_hk_vk_agc_avg_vec), 2))
                                    ak_hk_vk_noise_scaler_lst.append(ak_hk_vk_noise_scaler)

                                    ak_hk_vk_agc_nfft = np.ones(my_mod.n_fft, dtype=np.complex128)
                                    ak_hk_vk_agc_nfft[-(n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:n_sub_carr // 2]
                                    ak_hk_vk_agc_nfft[1:(n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[n_sub_carr // 2:]
                                    ak_hk_vk_agc_nfft_lst.append(ak_hk_vk_agc_nfft)

                            tx_bits = np.squeeze(bit_rng.choice((0, 1), (n_users, my_tx.modem.n_bits_per_ofdm_sym)))
                            tx_ofdm_symbol, clean_ofdm_symbol = my_array.transmit(tx_bits, out_domain_fd=True,
                                                                                  return_both=True, skip_dist=False)

                            for usr_idx in range(n_users):
                                if not any(curr_ite_lst_per_usr[usr_idx]):
                                    continue
                                # rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol)
                                rx_ofdm_symbol = np.sum(np.multiply(tx_ofdm_symbol, usr_chan_mat_lst[usr_idx]), axis=0)

                                rx_ofdm_symbol = my_noise.process(rx_ofdm_symbol,
                                                                  avg_sample_pow=my_mod.avg_symbol_power *
                                                                                 ak_hk_vk_noise_scaler_lst[usr_idx])

                                # apply AGC
                                rx_ofdm_symbol = np.divide(rx_ofdm_symbol, ak_hk_vk_agc_nfft_lst[usr_idx])

                                rx_bits_per_iter_lst = my_mcnc_rx_lst[usr_idx].receive(
                                    n_iters_lst=curr_ite_lst_per_usr[usr_idx], in_sig_fd=rx_ofdm_symbol,
                                    other_usr_bits=tx_bits[1 - usr_idx, :])

                                ber_idx = np.array(list(range(len(mcnc_n_iter_lst))))
                                act_ber_idx = ber_idx[ite_use_flags_per_usr[usr_idx]] + 1

                                for idx in range(len(rx_bits_per_iter_lst)):
                                    n_bit_err = utilities.count_mismatched_bits(tx_bits[usr_idx, :],
                                                                                rx_bits_per_iter_lst[idx])
                                    n_err[usr_idx, act_ber_idx[idx]] += n_bit_err
                                    bits_sent[usr_idx, act_ber_idx[idx]] += my_mod.n_bits_per_ofdm_sym
                                snap_cnt += 1

                        # print("Eb/N0: %1.1f, chan_rerolls: %d" %(utilities.snr_to_ebn0(snr=snr_db_val, n_fft=n_sub_carr, n_sub_carr=n_sub_carr, constel_size=constel_size), snap_cnt))
                        for usr_idx in range(n_users):

                            for ite_idx in range(len(mcnc_n_iter_lst) + 1):
                                bers[usr_idx, ite_idx] = n_err[usr_idx, ite_idx] / bits_sent[usr_idx, ite_idx]
                            bers_per_usr[usr_idx].append(bers[usr_idx, :])

                    for usr_idx in range(n_users):
                        bers_per_usr[usr_idx] = np.column_stack(bers_per_usr[usr_idx])

                    print("--- Computation time: %f ---" % (time.time() - start_time))

                    # %%
                    fig1, ax1 = plt.subplots(1, 1)
                    ax1.set_yscale('log')
                    usr_marker_lst = ['o', 's', '^', '*']

                    for usr_idx in range(n_users):
                        ax1.plot(ebn0_arr, bers_per_usr[usr_idx][0, :], label="No distortion", color=CB_color_cycle[0],
                                 marker=usr_marker_lst[usr_idx], fillstyle='none')
                        for ite_idx, mcnc_iter_val in enumerate(mcnc_n_iter_lst):
                            if ite_idx == 0:
                                ax1.plot(ebn0_arr, bers_per_usr[usr_idx][ite_idx + 1, :], label="Standard RX",
                                         color=CB_color_cycle[ite_idx + 1], marker=usr_marker_lst[usr_idx],
                                         fillstyle='none')
                            else:
                                ax1.plot(ebn0_arr, bers_per_usr[usr_idx][ite_idx + 1, :],
                                         label="MCNC NI = %d" % mcnc_iter_val, color=CB_color_cycle[ite_idx + 1],
                                         marker=usr_marker_lst[usr_idx], fillstyle='none')

                    # fix log scaling
                    ax1.set_title("BER vs Eb/N0, %s, MCNCWI, QAM %d, N ANT = %d, IBO = %d [dB]" % (
                        my_miso_chan, my_mod.constellation_size, n_ant_val, ibo_val_db))
                    ax1.set_xlabel("Eb/N0 [dB]")
                    ax1.set_ylabel("BER")
                    ax1.grid()
                    ax1.legend(loc="lower left")
                    plt.tight_layout()

                    filename_str = "ber_vs_ebn0_mu_mcncwi_%s_nant%d_ibo%d_ebn0_min%d_max%d_step%1.2f_niter%s_angles%s_distances%s" % (
                        my_miso_chan, n_ant_val, ibo_val_db, min(ebn0_arr), max(ebn0_arr), ebn0_arr[1] - ebn0_arr[0],
                        '_'.join([str(val) for val in mcnc_n_iter_lst[1:]]), '_'.join([str(val) for val in usr_angles]),
                        '_'.join([str(val) for val in usr_distances]))

                    # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    # filename_str += "_" + timestamp
                    plt.savefig("../figs/multiuser/%s.png" % filename_str, dpi=600, bbox_inches='tight')
                    plt.show()
                    # plt.cla()
                    # plt.close()

                    # # %%
                    # data_lst = []
                    # data_lst.append(ebn0_arr)
                    # for arr1 in ber_per_dist:
                    #     data_lst.append(arr1)
                    # utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

                    # read_data = utilities.read_from_csv(filename=filename_str)

print("Finished processing!")
