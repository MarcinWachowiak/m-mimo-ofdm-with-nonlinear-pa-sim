"""
Measure the effective radiation pattern of uniform circular array with nonlinear front-end amplifiers
in a multi-user scenario. Evaluate the distortion beamforming directions.
"""

# %%
import os
import sys

import corrector

sys.path.append(os.getcwd())

import copy
import time
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.signal import welch

import channel
import distortion
import modulation
import transceiver
import antenna_array

import utilities
from plot_settings import set_latex_plot_style

if __name__ == '__main__':
    set_latex_plot_style()
    # Multiple users data
    usr_angles_deg = np.array([-10, 10])
    usr_angles_rad = np.deg2rad(usr_angles_deg)
    usr_distances = [300, 300]
    usr_pos_tup = []
    for usr_idx, usr_angle in enumerate(usr_angles_deg + 90):
        usr_pos_x = np.cos(np.deg2rad(usr_angle)) * usr_distances[usr_idx]
        usr_pos_y = np.sin(np.deg2rad(usr_angle)) * usr_distances[usr_idx]
        usr_pos_tup.append((usr_pos_x, usr_pos_y))
    # custom x, y coordinates
    # usr_pos_tup = [(45, 45), (120, 120), (150, 150)]
    n_users = len(usr_pos_tup)

    n_ant_arr = [64]
    ibo_arr = [7]
    ebn0_step = [1]
    cnc_n_iter_lst = [1, 2, 3, 4]  # 5, 6, 7, 8]
    # include clean run is always True
    # no distortion and standard RX always included
    cnc_n_iter_lst = np.insert(cnc_n_iter_lst, 0, 0)

    # print("Distortion IBO/TOI value:", ibo_db)
    # print("Eb/n0 values: ", ebn0_arr)
    # print("CNC iterations: ", cnc_n_iter_lst)

    # modulation
    constel_size = 64
    n_fft = 128
    n_sub_carr = 64
    cp_len = 1

    # BER analysis
    bits_sent_max = int(1e5)
    n_err_min = int(1e5)
    ber_reroll_pos = False

    rx_loc_x, rx_loc_y = 212.0, 212.0
    rx_loc_var = 10.0

    # SDR
    meas_usr_sdr = False
    sdr_n_snapshots = 10
    sdr_reroll_pos = False

    # Beampatterns
    plot_precoding_beampatterns = True
    beampattern_n_snapshots = 10
    n_points = 360
    radial_distance = 3000
    rx_points = utilities.pts_on_circum(radius=radial_distance, n_points=n_points)
    radian_vals = np.radians(np.linspace(-90, 270, n_points + 1))

    # PSD at angle
    plot_psd = False
    sel_psd_angle = 51
    sel_ptx_idx = int(n_points / 180 * (sel_psd_angle + 90))
    # PSD plotting params
    psd_nfft = 1024
    n_samp_per_seg = 256

    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len,
                                     n_users=len(usr_angles_deg))

    my_distortion = distortion.ThirdOrderNonLin(toi_db=ibo_arr[0], avg_samp_pow=my_mod.avg_sample_power)
    # my_distortion = distortion.SoftLimiter(ibo_db=ibo_arr[0], avg_samp_pow=my_mod.avg_sample_power)

    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(3.5e9), carrier_spacing=int(15e3))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=rx_loc_x, cord_y=rx_loc_y, cord_z=1.5,
                                             center_freq=int(3.5e9), carrier_spacing=int(15e3))

    for n_ant_val in n_ant_arr:
        my_array = antenna_array.CircularArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
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
            my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))

            for ibo_val_db in ibo_arr:
                rx_sig_at_sel_point_des = []
                rx_sig_at_sel_point_dist = []
                rx_sig_at_sel_point_cln = []

                estimate_alpha = True
                # lambda estimation phase
                if estimate_alpha:
                    my_tmp_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr,
                                                         cp_len=cp_len,
                                                         n_users=1)
                    my_tmp_tx = transceiver.Transceiver(modem=copy.deepcopy(my_tmp_mod),
                                                        impairment=copy.deepcopy(my_distortion),
                                                        center_freq=int(3.5e9), carrier_spacing=int(15e3))

                    my_tmp_array = antenna_array.LinearArray(n_elements=n_ant_val, base_transceiver=my_tmp_tx,
                                                              center_freq=int(3.5e9),
                                                              wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
                    my_tmp_miso_los_chan = channel.MisoLosFd()
                    my_tmp_miso_los_chan.calc_channel_mat(tx_transceivers=my_tmp_array.array_elements,
                                                          rx_transceiver=my_standard_rx,
                                                          skip_attenuation=False)
                    start_time = time.time()
                    bit_rng = np.random.default_rng(4321)
                    n_ofdm_symb = 1e3
                    ofdm_symb_idx = 0
                    alpha_val_vec = []

                    my_tmp_array.set_precoding_matrix(channel_mat_fd=my_tmp_miso_los_chan.get_channel_mat_fd(),
                                                      mr_precoding=True)
                    my_tmp_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_tmp_mod.avg_sample_power)

                    while ofdm_symb_idx < n_ofdm_symb:
                        tx_bits = bit_rng.choice((0, 1), my_tmp_tx.modem.n_bits_per_ofdm_sym)
                        tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_tmp_array.transmit(tx_bits, out_domain_fd=True,
                                                                                        return_both=True)

                        rx_sig_fd = my_tmp_miso_los_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd, sum_signals=False)
                        rx_sig_clean_fd = my_tmp_miso_los_chan.propagate(in_sig_mat=clean_ofdm_symbol_fd, sum_signals=False)

                        clean_nsc_ofdm_symb_fd = np.concatenate((rx_sig_clean_fd[:, -my_mod.n_sub_carr // 2:],
                                                                 rx_sig_clean_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                                                axis=1)
                        rx_nsc_ofdm_symb_fd = np.concatenate(
                            (rx_sig_fd[:, -my_mod.n_sub_carr // 2:], rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                            axis=1)

                        alpha_numerator_vec = np.multiply(rx_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd))
                        alpha_denominator_vec = np.multiply(clean_nsc_ofdm_symb_fd,
                                                            np.conjugate(clean_nsc_ofdm_symb_fd))

                        ofdm_symb_idx += 1
                        alpha_val_vec.append(np.abs(np.average(alpha_numerator_vec / alpha_denominator_vec, axis=1)))

                    # calculate alpha average
                    alpha_vec_est = np.average(alpha_val_vec, axis=0)
                    print("Alpha coeff estimate:", alpha_vec_est)
                    print("--- Computation time: %f ---" % (time.time() - start_time))
                else:
                    alpha_vec_est = 1.0
                    # alpha_estimate = my_mod.calc_alpha(dist_val_arr)

                my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
                my_cnc_rx.update_distortion(ibo_db=ibo_val_db)

                usr_chan_mat_lst = []

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

                my_array.set_precoding_matrix(channel_mat_fd=usr_chan_mat_lst, mr_precoding=True)
                my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                vk_mat = my_array.get_precoding_mat()
                vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                if estimate_alpha:
                    ak_vect = alpha_vec_est
                else:
                    if isinstance(my_distortion, distortion.ThirdOrderNonLin):
                        ak_vect = np.repeat(1.0, n_ant_val)
                    else:
                        ibo_vec = 10 * np.log10(10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                        ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)

                ak_vect = np.expand_dims(ak_vect, axis=1)

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
                                # for direct visibility channel and CNC algorithm channel impact must be averaged
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
                    distorted_sig_inband_pow_per_pt = []
                    distorted_sig_oob_pow_per_pt = []
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
                        distortion_inband_sig_pow_arr = np.zeros(beampattern_n_snapshots)
                        distortion_oob_sig_pow_arr = np.zeros(beampattern_n_snapshots)
                        for snap_idx in range(beampattern_n_snapshots):
                            tx_bits = np.squeeze(bit_rng.choice((0, 1), (n_users, my_tx.modem.n_bits_per_ofdm_sym)))
                            arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits,
                                                                                out_domain_fd=True,
                                                                                return_both=True)

                            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd, sum_signals=False)
                            clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd, sum_signals=False)
                            distortion_sig_fd = np.subtract(rx_sig_fd, (ak_vect * clean_rx_sig_fd))

                            clean_sc_ofdm_symb_fd = np.concatenate(
                                (clean_rx_sig_fd[:, -my_mod.n_sub_carr // 2:],
                                 clean_rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                axis=1)

                            sc_inband_ofdm_distortion_sig = np.concatenate(
                                (distortion_sig_fd[:, -my_mod.n_sub_carr // 2:],
                                 distortion_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                axis=1)

                            sc_oob_ofdm_distortion_sig = np.concatenate(
                                (np.expand_dims(distortion_sig_fd[:, 0], axis=1),
                                 distortion_sig_fd[:, (my_mod.n_sub_carr // 2) + 1: -my_mod.n_sub_carr // 2]),
                                axis=1)

                            desired_sig_pow_arr[snap_idx] = np.sum(
                                np.power(np.abs(np.sum(ak_vect * clean_sc_ofdm_symb_fd, axis=0)), 2))
                            distortion_inband_sig_pow_arr[snap_idx] = np.sum(
                                np.power(np.abs(np.sum(sc_inband_ofdm_distortion_sig, axis=0)), 2))
                            distortion_oob_sig_pow_arr[snap_idx] = np.sum(
                                np.power(np.abs(np.sum(sc_oob_ofdm_distortion_sig, axis=0)), 2))

                            if plot_psd and pt_idx == sel_ptx_idx:
                                # for PSD plotting take into consideration full BW not only SC
                                desired_sig = np.sum(ak_vect * clean_rx_sig_fd, axis=0)
                                distortion_sig = np.sum(np.subtract(rx_sig_fd, (ak_vect * clean_rx_sig_fd)), axis=0)
                                rx_sig_at_sel_point_cln.append(utilities.to_time_domain(clean_rx_sig_fd))
                                rx_sig_at_sel_point_des.append(utilities.to_time_domain(desired_sig))
                                rx_sig_at_sel_point_dist.append(utilities.to_time_domain(distortion_sig))

                            # calculate SDR on symbol basis
                        desired_sig_pow_per_pt.append(np.sum(desired_sig_pow_arr))
                        distorted_sig_inband_pow_per_pt.append(np.sum(distortion_inband_sig_pow_arr))
                        distorted_sig_oob_pow_per_pt.append(np.sum(distortion_oob_sig_pow_arr))

                    if plot_psd:
                        rx_sig_at_sel_point_des_arr = np.concatenate(rx_sig_at_sel_point_des).ravel()
                        rx_sig_at_sel_point_dist_arr = np.concatenate(rx_sig_at_sel_point_dist).ravel()
                        rx_sig_at_sel_point_cln_arr = np.concatenate(rx_sig_at_sel_point_cln).ravel()

                        rx_des_at_sel_point_freq_arr, rx_des_at_sel_point_psd = welch(rx_sig_at_sel_point_des_arr,
                                                                                      fs=psd_nfft,
                                                                                      nfft=psd_nfft,
                                                                                      nperseg=n_samp_per_seg,
                                                                                      return_onesided=False)
                        rx_dist_at_sel_point_freq_arr, rx_dist_at_sel_point_psd = welch(rx_sig_at_sel_point_dist_arr,
                                                                                        fs=psd_nfft,
                                                                                        nfft=psd_nfft,
                                                                                        nperseg=n_samp_per_seg,
                                                                                        return_onesided=False)
                        rx_cln_at_sel_point_freq_arr, rx_cln_at_sel_point_psd = welch(rx_sig_at_sel_point_cln_arr,
                                                                                      fs=psd_nfft,
                                                                                      nfft=psd_nfft,
                                                                                      nperseg=n_samp_per_seg,
                                                                                      return_onesided=False)

                        psd_sel_filename_str = "multiuser_uca_shared_sc_psd_%s_%s_chan_ibo%d_npoints%d_nsnap%d_angle%d_nant%d" % (
                            my_distortion, my_miso_chan, ibo_val_db, n_points, beampattern_n_snapshots, sel_psd_angle,
                            n_ant_val)

                        # data_lst_sel = []
                        # tmp_lst_sel = [rx_des_at_sel_point_freq_arr, rx_des_at_sel_point_psd, rx_dist_at_sel_point_freq_arr,
                        #                rx_dist_at_sel_point_psd, rx_cln_at_sel_point_freq_arr, rx_cln_at_sel_point_psd]
                        # for arr1 in tmp_lst_sel:
                        #     data_lst_sel.append(arr1)
                        # utilities.save_to_csv(data_lst=data_lst_sel, filename=psd_sel_filename_str)
                        # %%
                        fig5, ax5 = plt.subplots(1, 1)
                        sorted_des_rx_at_sel_freq_arr, sorted_des_psd_at_sel_arr = zip(
                            *sorted(zip(rx_des_at_sel_point_freq_arr, rx_des_at_sel_point_psd)))
                        ax5.plot(np.array(sorted_des_rx_at_sel_freq_arr),
                                 utilities.to_db(np.array(sorted_des_psd_at_sel_arr)),
                                 label="Desired")
                        sorted_dist_rx_at_sel_freq_arr, sorted_dist_psd_at_sel_arr = zip(
                            *sorted(zip(rx_dist_at_sel_point_freq_arr, rx_dist_at_sel_point_psd)))
                        ax5.plot(np.array(sorted_dist_rx_at_sel_freq_arr),
                                 utilities.to_db(np.array(sorted_dist_psd_at_sel_arr)),
                                 label="Distorted")

                        ax5.set_title("Power spectral density at angle %d$\degree$" % sel_psd_angle)
                        ax5.set_xlabel("Subcarrier index [-]")
                        ax5.set_ylabel("Power [dB]")
                        ax5.legend(title="IBO = %d [dB]" % ibo_val_db)
                        ax5.grid()
                        plt.tight_layout()
                        plt.savefig("../figs/multiuser/psd/%s.png" % psd_sel_filename_str, dpi=600, bbox_inches='tight')
                        plt.show()
                        # plt.cla()
                        # plt.close()

                    # %%
                    # utilities.plot_array_config(my_array)
                    # # my_standard_rx.set_position(0.0, 0.0, 0.0)
                    # utilities.plot_spatial_config(my_array, rx_transceiver=my_standard_rx, plot_3d=True)

                    # %%
                    # plot beampatterns of desired and distortion components
                    fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize=(3.5, 3))
                    ax1.set_theta_zero_location("N")
                    plt.tight_layout()
                    ax1.set_thetalim(-np.pi, np.pi)
                    ax1.set_xticks(np.pi / 180. * np.linspace(-180, 180, 25, endpoint=True))
                    ax1.yaxis.set_major_locator(MaxNLocator(5))

                    dist_lines_lst = []

                    ax1.plot(radian_vals, utilities.to_db(desired_sig_pow_per_pt), label="Desired",
                             linewidth=1.5)
                    ax1.plot(radian_vals, utilities.to_db(distorted_sig_inband_pow_per_pt), label="Distorted IB",
                             linewidth=1.5)
                    ax1.plot(radian_vals, utilities.to_db(distorted_sig_oob_pow_per_pt), label="Distorted OOB",
                             linewidth=1.5)

                    # plot reference angles/directions
                    (y_min, y_max) = ax1.get_ylim()
                    ax1.vlines(np.deg2rad(usr_angles_deg), y_min, y_max, colors='k', linestyles='--',
                               zorder=10)  # label="Users")

                    dist_angles = []
                    arcsin_arg_periodize = lambda val_a: val_a - 2.0 if val_a > 1.0 else (
                        val_a + 2.0 if val_a < -1.0 else val_a)

                    print(len(list(itertools.product(range(n_users), repeat=3))))
                    for usr_ang_idx_1, usr_ang_idx_2, usr_ang_idx_3 in itertools.product(range(n_users), repeat=3):
                        phase_val = np.sin(usr_angles_rad[usr_ang_idx_1]) + np.sin(
                            usr_angles_rad[usr_ang_idx_2]) - np.sin(usr_angles_rad[usr_ang_idx_3])
                        arccos_val = arcsin_arg_periodize(phase_val)
                        dist_angles.append(np.arcsin(arccos_val))
                        dist_angles.append(-np.arcsin(arccos_val))


                    # generate usr angle idx tuples to calculate the distortion angle
                    def dist_get_usr_angle_idx():
                        for idx_1 in range(n_users):
                            for idx_2 in range(n_users):
                                for idx_3 in range(n_users):
                                    pass


                    ax1.vlines(np.array(dist_angles), y_min, y_max, colors='r', linestyles=':',
                               zorder=10)  # label="Expected distortion")
                    ax1.margins(0.0, 0.0)
                    ax1.set_title("Signal power at angle [dB]", pad=-10)
                    ax1.legend(title="Signal:", ncol=2, loc='lower center', borderaxespad=-8)
                    ax1.grid(True)
                    beampattern_filename_str = "multiuser_uca_shared_sc_no_precod_beampatterns_%s_%s_nfft%d_nsc%d_ibo%d_angles%s_distances%s_npoints%d_nsnap%d_nant%s" % (
                        my_distortion, my_miso_chan, n_fft, n_sub_carr, ibo_val_db,
                        '_'.join([str(val) for val in usr_angles_deg]),
                        '_'.join([str(val) for val in usr_distances]), n_points, beampattern_n_snapshots,
                        '_'.join([str(val) for val in [n_ant_val]]))
                    plt.savefig("../figs/multiuser/distortion_directions_eval/%s.png" % (beampattern_filename_str),
                                dpi=600, bbox_inches='tight')
                    plt.show()
                    # plt.cla()
                    # plt.close()

                    # #%%
                    # fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 3))
                    # plt.tight_layout()
                    # ax1.plot(radian_vals, utilities.to_db(np.array(desired_sig_pow_per_pt) / (np.array(distorted_sig_inband_pow_per_pt) + np.array(distorted_sig_oob_pow_per_pt))), label="SDR",
                    #          linewidth=1.5)
                    # ax1.set_title("Signal to distortion ratio at angle [dB]")
                    # ax1.set_xlabel("Angle [°]")
                    # ax1.set_ylabel("SDR [dB]")
                    # ax1.grid()
                    # plt.show()
