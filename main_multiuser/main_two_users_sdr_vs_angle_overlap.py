# antenna array evaluation
# %%
import os
import sys

sys.path.append(os.getcwd())

import copy
import time

import matplotlib.pyplot as plt
import numpy as np

import antenna_arrray
import channel
import distortion
import modulation
import transceiver
import utilities
from plot_settings import set_latex_plot_style

if __name__ == '__main__':
    set_latex_plot_style()
    CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                      '#CFCFCF']
    # Multiple users data
    main_usr_angle = 60
    main_user_dist = 300
    usr_pos_tup = []
    n_users = 2

    main_usr_pos_x = np.cos(np.deg2rad(main_usr_angle)) * main_user_dist
    main_usr_pos_y = np.sin(np.deg2rad(main_usr_angle)) * main_user_dist

    n_ant_arr = [16] # 256]
    ibo_val_db = 0
    # modulation
    constel_size = 64
    n_fft = 4096
    n_sub_carr = 2048
    cp_len = 128
    rx_loc_x, rx_loc_y = 212.0, 212.0

    # Beampattern sweeping params
    sdr_n_snapshots = 2
    n_points = 180 * 1
    radial_distance = main_user_dist
    rx_points = utilities.pts_on_semicircum(r=radial_distance, n=n_points)
    radian_vals = np.radians(np.linspace(0, 180, n_points + 1))

    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len,
                                     n_users=n_users)

    my_distortion = distortion.SoftLimiter(ibo_val_db, my_mod.avg_sample_power)
    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=rx_loc_x, cord_y=rx_loc_y, cord_z=1.5,
                                             center_freq=int(3.5e9), carrier_spacing=int(15e3))
    corr_per_n_ant, sdr_per_n_ant = [], []
    for ant_idx, n_ant_val in enumerate(n_ant_arr):

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

        corr_per_chan, sdr_per_chan = [], []
        for chan_idx, my_miso_chan in enumerate(chan_lst):
            bit_rng = np.random.default_rng(4321)
            start_time = time.time()
            # get reference user channel mat
            my_standard_rx.set_position(cord_x=main_usr_pos_x, cord_y=main_usr_pos_y, cord_z=1.5)

            if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan, channel.MisoTwoPathFd):
                my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                              rx_transceiver=my_standard_rx,
                                              skip_attenuation=False)
            else:
                my_miso_chan.reroll_channel_coeffs()

            main_usr_channel_mat = my_miso_chan.get_channel_mat_fd()
            # my_array.set_precoding_matrix(channel_mat_fd=main_usr_channel_mat, mr_precoding=True)
            # main_usr_channel_mat = my_array.get_precoding_mat()
            # sweep channels and correlate
            corr_vect = np.zeros(n_points + 1)
            sdr_vec = np.zeros((n_users, n_points+1))
            for point_idx, rx_point in enumerate(rx_points):
                usr_pos_x, usr_pos_y = rx_point
                my_standard_rx.set_position(cord_x=usr_pos_x, cord_y=usr_pos_y, cord_z=1.5)

                if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan, channel.MisoTwoPathFd):
                    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                  rx_transceiver=my_standard_rx,
                                                  skip_attenuation=False)
                else:
                    if np.all(np.isclose([usr_pos_x, usr_pos_y], [main_usr_pos_x, main_usr_pos_y])):
                        my_miso_chan.channel_mat_fd = main_usr_channel_mat
                    else:
                        my_miso_chan.reroll_channel_coeffs()

                sec_usr_channel_mat = my_miso_chan.get_channel_mat_fd()

                main_usr_sc_channel_mat = np.concatenate((main_usr_channel_mat[:, -my_mod.n_sub_carr // 2:],
                                                          main_usr_channel_mat[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                                         axis=1)
                sec_usr_sc_channel_mat = np.concatenate((sec_usr_channel_mat[:, -my_mod.n_sub_carr // 2:],
                                                         sec_usr_channel_mat[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                                        axis=1)
                nomin = np.trace(
                    np.abs(np.matmul(np.transpose(main_usr_sc_channel_mat), np.conjugate(sec_usr_sc_channel_mat))))
                denomin = np.sqrt(np.sum(np.power(np.abs(main_usr_sc_channel_mat), 2)) * np.sum(
                    np.power(np.abs(sec_usr_sc_channel_mat), 2)))
                corr_coeff = nomin / denomin
                corr_vect[point_idx] = corr_coeff

                usr_chan_mat_lst = [main_usr_channel_mat, sec_usr_channel_mat]
                my_array.set_precoding_matrix(channel_mat_fd=usr_chan_mat_lst, mr_precoding=True)
                my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                vk_mat = my_array.get_precoding_mat()
                vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                ibo_vec = 10 * np.log10(
                    10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
                ak_vect = np.expand_dims(ak_vect, axis=1)

                desired_sig_pow_arr = np.zeros((n_users, sdr_n_snapshots))
                distortion_sig_pow_arr = np.zeros((n_users, sdr_n_snapshots))
                for snap_idx in range(sdr_n_snapshots):
                    tx_bits = np.squeeze(bit_rng.choice((0, 1), (n_users, my_tx.modem.n_bits_per_ofdm_sym)))

                    arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True,
                                                                        return_both=True)
                    for usr_idx in range(n_users):
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
                sdr_vec[:, point_idx] = sdr_per_usr

            corr_per_chan.append(corr_vect)
            sdr_per_chan.append(sdr_vec)
        corr_per_n_ant.append(corr_per_chan)
        sdr_per_n_ant.append(sdr_per_chan)

        print("--- Computation time: %f ---" % (time.time() - start_time))

    # %%
    for chan_idx, chan_obj in enumerate(chan_lst):
        fig1, ax1 = plt.subplots(1, 1)
        ax2 = ax1.twinx()

        for ant_idx, n_ant_val in enumerate(n_ant_arr):
            ax1.plot(np.rad2deg(radian_vals), corr_per_n_ant[ant_idx][chan_idx], label=n_ant_val, linestyle='--', color=CB_color_cycle[-1])
        ax1.legend(title="K antennas:", loc="center left")
        ax1.set_ylabel("Correlation coefficient")

        for ant_idx, n_ant_val in enumerate(n_ant_arr):
             ax2.plot(np.rad2deg(radian_vals), sdr_per_n_ant[ant_idx][chan_idx][0, :], label="Main", linestyle='-')
             ax2.plot(np.rad2deg(radian_vals), sdr_per_n_ant[ant_idx][chan_idx][1, :], label="Secondary", linestyle='-')
        ax2.set_ylabel("SDR [dB]")
        ax2.legend(title="User:", loc="center right")


        # plot ref user
        (y_min, y_max) = ax1.get_ylim()
        ax1.margins(0.0, 0.0)
        ax1.vlines(main_usr_angle, y_min, y_max, colors='k', linestyles='--', label='Precoding angle',
                   zorder=10)  # label="Users")
        # fix log scaling
        ax1.set_title("Correlation coefficient vs angle and K antennas, %s" % (chan_obj))
        ax1.set_xlabel("Angle [°]")

        ax1.grid()


        plt.tight_layout()

        filename_str = "sdr_and_channel_mat_corr_coeff_%s_distance%d_angle%d_nant%s" % (
        chan_obj, main_user_dist, main_usr_angle, '_'.join([str(val) for val in n_ant_arr]))
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
