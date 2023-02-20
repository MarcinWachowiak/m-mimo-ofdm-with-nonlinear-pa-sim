# antenna array evaluation
# %%
import os
import sys

sys.path.append(os.getcwd())

import copy
import time

import matplotlib.pyplot as plt
import numpy as np

import channel
import distortion
import modulation
import transceiver
import utilities
from plot_settings import set_latex_plot_style

if __name__ == '__main__':
    # %%
    set_latex_plot_style()
    CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                      '#CFCFCF']
    # Multiple users data
    main_usr_angle = 45
    main_user_dist = 300
    usr_pos_tup = []

    main_usr_pos_x = np.cos(np.deg2rad(main_usr_angle)) * main_user_dist
    main_usr_pos_y = np.sin(np.deg2rad(main_usr_angle)) * main_user_dist

    ibo_val_db = 0
    n_ant_arr = [2, 4, 8, 16, 32, 64]
    beampattern_n_snapshots = 1

    # modulation
    constel_size = 64
    n_fft = 4096
    n_sub_carr = 2048
    cp_len = 128
    rx_loc_x, rx_loc_y = 212.0, 212.0

    # Beampattern sweeping params
    n_points = 36 * 1
    radial_distance = main_user_dist
    rx_points = utilities.pts_on_semicircum(r=radial_distance, n=n_points)
    radian_vals = np.radians(np.linspace(0, 180, n_points + 1))
    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len,
                                     n_users=1)

    my_distortion = distortion.SoftLimiter(ibo_val_db, my_mod.avg_sample_power)
    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=rx_loc_x, cord_y=rx_loc_y, cord_z=1.5,
                                             center_freq=int(3.5e9), carrier_spacing=int(15e3))
    corr_per_n_ant_cln = []
    corr_per_n_ant_full = []
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

        chan_lst = [my_miso_los_chan, my_miso_two_path_chan, my_miso_rayleigh_chan]

        corr_per_chan_cln = []
        corr_per_chan_full = []
        for chan_idx, my_miso_chan in enumerate(chan_lst):
            start_time = time.time()
            corr_vect_cln = np.zeros(n_points + 1)
            corr_vect_full = np.zeros(n_points + 1)

            # get reference user channel mat
            my_standard_rx.set_position(cord_x=main_usr_pos_x, cord_y=main_usr_pos_y, cord_z=1.5)

            if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan, channel.MisoTwoPathFd):
                my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                              rx_transceiver=my_standard_rx,
                                              skip_attenuation=False)
            else:
                my_miso_chan.reroll_channel_coeffs()

            main_usr_chan_mat = my_miso_chan.get_channel_mat_fd()
            my_array.set_precoding_matrix(channel_mat_fd=main_usr_chan_mat, mr_precoding=True)
            my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

            bit_rng = np.random.default_rng(4321)
            main_usr_clean_beampattern = np.zeros(len(rx_points))
            main_usr_full_beampattern = np.zeros(len(rx_points))
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
                    if np.all(np.isclose([x_cord, y_cord], [main_usr_pos_x, main_usr_pos_y])):
                        my_miso_chan.channel_mat_fd = main_usr_chan_mat
                    else:
                        my_miso_chan.reroll_channel_coeffs()

                clean_sig_pow_arr = np.zeros(beampattern_n_snapshots)
                full_rx_sig_pow_arr = np.zeros(beampattern_n_snapshots)
                for snap_idx in range(beampattern_n_snapshots):
                    tx_bits = np.squeeze(bit_rng.choice((0, 1), (1, my_tx.modem.n_bits_per_ofdm_sym)))
                    arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True,
                                                                        return_both=True)

                    rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd, sum=True)
                    rx_sc_ofdm_symb_fd = np.concatenate(
                        (rx_sig_fd[-my_mod.n_sub_carr // 2:], rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))

                    clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd, sum=True)
                    clean_sc_ofdm_symb_fd = np.concatenate(
                        (clean_rx_sig_fd[-my_mod.n_sub_carr // 2:],
                         clean_rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))

                    clean_sig_pow_arr[snap_idx] = np.sum(
                        np.power(np.abs(clean_sc_ofdm_symb_fd), 2))
                    full_rx_sig_pow_arr[snap_idx] = np.sum(
                        np.power(np.abs(rx_sc_ofdm_symb_fd), 2))

                main_usr_clean_beampattern[pt_idx] = np.average(clean_sig_pow_arr)
                main_usr_full_beampattern[pt_idx] = np.average(full_rx_sig_pow_arr)

            test_usr_clean_beampattern_lst = []
            test_usr_full_beampattern_lst = []
            for idx, rx_point in enumerate(rx_points):
                test_usr_pos_x, test_usr_pos_y = rx_point
                my_standard_rx.set_position(cord_x=test_usr_pos_x, cord_y=test_usr_pos_y, cord_z=1.5)

                if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan, channel.MisoTwoPathFd):
                    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                  rx_transceiver=my_standard_rx,
                                                  skip_attenuation=False)
                else:
                    my_miso_chan.reroll_channel_coeffs()

                test_usr_chan_mat = my_miso_chan.get_channel_mat_fd()
                my_array.set_precoding_matrix(channel_mat_fd=test_usr_chan_mat, mr_precoding=True)
                my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                bit_rng = np.random.default_rng(4321)
                test_usr_clean_beampattern = np.zeros(len(rx_points))
                test_usr_full_beampattern = np.zeros(len(rx_points))
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
                        if np.all(np.isclose([x_cord, y_cord], [test_usr_pos_x, test_usr_pos_y])):
                            my_miso_chan.channel_mat_fd = test_usr_chan_mat
                        else:
                            my_miso_chan.reroll_channel_coeffs()

                    clean_sig_pow_arr = np.zeros(beampattern_n_snapshots)
                    full_rx_sig_pow_arr = np.zeros(beampattern_n_snapshots)
                    for snap_idx in range(beampattern_n_snapshots):
                        tx_bits = np.squeeze(bit_rng.choice((0, 1), (1, my_tx.modem.n_bits_per_ofdm_sym)))
                        arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True,
                                                                            return_both=True)

                        rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd, sum=True)
                        rx_sc_ofdm_symb_fd = np.concatenate(
                            (rx_sig_fd[-my_mod.n_sub_carr // 2:],
                             rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))

                        clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd, sum=True)
                        clean_sc_ofdm_symb_fd = np.concatenate(
                            (clean_rx_sig_fd[-my_mod.n_sub_carr // 2:],
                             clean_rx_sig_fd[1:(my_mod.n_sub_carr // 2) + 1]))

                        clean_sig_pow_arr[snap_idx] = np.sum(
                            np.power(np.abs(clean_sc_ofdm_symb_fd), 2))
                        full_rx_sig_pow_arr[snap_idx] = np.sum(
                            np.power(np.abs(rx_sc_ofdm_symb_fd), 2))

                    test_usr_clean_beampattern[pt_idx] = np.average(clean_sig_pow_arr)
                    test_usr_full_beampattern[pt_idx] = np.average(full_rx_sig_pow_arr)

                test_usr_clean_beampattern_lst.append(test_usr_clean_beampattern)
                test_usr_full_beampattern_lst.append(test_usr_full_beampattern)

            # %%
            for corr_pt_idx, point in enumerate(rx_points):
                corr_vect_cln[corr_pt_idx] = np.abs(
                    np.matmul(main_usr_clean_beampattern, np.transpose(test_usr_clean_beampattern_lst[corr_pt_idx]))) \
                                             / np.sqrt(np.sum(np.power(np.abs(main_usr_clean_beampattern), 2)) * np.sum(
                    np.power(np.abs(test_usr_clean_beampattern_lst[corr_pt_idx]), 2)))
                # corr_vect_full[corr_pt_idx] = np.abs( np.matmul(main_usr_full_beampattern, np.transpose(
                # test_usr_full_beampattern_lst[corr_pt_idx]))) \ / np.sqrt(np.sum(np.power(np.abs(
                # main_usr_full_beampattern), 2)) * np.sum( np.power(np.abs(test_usr_full_beampattern_lst[
                # corr_pt_idx]), 2)))

            corr_per_chan_cln.append(corr_vect_cln)
            # corr_per_chan_full.append(corr_vect_full)
        corr_per_n_ant_cln.append(corr_per_chan_cln)
        # corr_per_n_ant_full.append(corr_per_chan_full)

    print("--- Computation time: %f ---" % (time.time() - start_time))

    # %%
    for chan_idx, chan_obj in enumerate(chan_lst):
        fig1, ax1 = plt.subplots(1, 1)
        for ant_idx, n_ant_val in enumerate(n_ant_arr):
            ax1.plot(np.rad2deg(radian_vals), corr_per_n_ant_cln[ant_idx][chan_idx], label=n_ant_val,
                     color=CB_color_cycle[ant_idx], linestyle='-')
            # ax1.plot(np.rad2deg(radian_vals), corr_per_n_ant_full[ant_idx][chan_idx], label=n_ant_val,
            #          color=CB_color_cycle[ant_idx], linestyle='--')
        # plot ref user
        (y_min, y_max) = ax1.get_ylim()
        ax1.margins(0.0, 0.0)
        ax1.vlines(main_usr_angle, y_min, y_max, colors='k', linestyles='--', label='Precoding angle',
                   zorder=5)  # label="Users")
        # fix log scaling
        ax1.set_title("Spatial correlation coefficient vs angle and K antennas, %s" % chan_obj)
        ax1.set_xlabel("Angle [°]")
        ax1.set_ylabel("Correlation coefficient")
        ax1.legend(title="K antennas:")
        ax1.grid()
        plt.tight_layout()

        filename_str = "channel_spatial_corr_coeff_%s_distance%d_angle%d_nant%s" % (
            chan_obj, main_user_dist, main_usr_angle, '_'.join([str(val) for val in n_ant_arr]))
        plt.savefig("../figs/multiuser/channel_spatial_correlation/%s.png" % filename_str, dpi=600, bbox_inches='tight')
        plt.show()
        plt.cla()
        plt.close()

# # %%
# data_lst = []
# data_lst.append(ebn0_arr)
# for arr1 in ber_per_dist:
#     data_lst.append(arr1)
# utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

# read_data = utilities.read_from_csv(filename=filename_str)

print("Finished processing!")
