# antenna array evaluation
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

import antenna_arrray
import channel
import distortion
import modulation
import transceiver
import utilities
from plot_settings import set_latex_plot_style

if __name__ == '__main__':
    set_latex_plot_style()
    # Multiple users data
    # usr angles: (elevation [-90, +90],  azimuth [-90, +90]) tuples
    usr_angles_deg = np.array([(15, 15), (-15, -15)])
    usr_angles_rad = np.deg2rad(usr_angles_deg)
    usr_distances = [300, 300]
    usr_pos_tup = []
    # user positions in relation to array center
    arr_center_x = 0
    arr_center_y = 0
    arr_center_z = 15
    for usr_idx, (usr_azimuth, usr_elevation) in enumerate(usr_angles_deg + 90):
        usr_pos_x = -usr_distances[usr_idx] * np.sin(np.deg2rad(usr_elevation)) * np.cos(np.deg2rad(usr_azimuth)) + arr_center_x
        usr_pos_y = -usr_distances[usr_idx] * np.sin(np.deg2rad(usr_elevation)) * np.sin(np.deg2rad(usr_azimuth)) + arr_center_y
        usr_pos_z = -usr_distances[usr_idx] * np.cos(np.deg2rad(usr_elevation)) + arr_center_z
        usr_pos_tup.append((usr_pos_x, usr_pos_y, usr_pos_z))
    n_users = len(usr_pos_tup)

    n_ant_arr = [256]
    ibo_arr = [10]
    ebn0_step = [1]

    # modulation
    constel_size = 64
    n_fft = 128
    n_sub_carr = 64
    cp_len = 1

    # BER analysis
    bits_sent_max = int(1e5)
    n_err_min = int(1e5)
    ber_reroll_pos = False

    sdr_n_snapshots = 10

    # Beampatterns
    plot_precoding_beampatterns = True
    beampattern_n_snapshots = 5
    n_points = 90**2 * 1
    radial_distance = 300
    rx_points = utilities.pts_on_semisphere(r=radial_distance, n=n_points, center_x=arr_center_x, center_y=arr_center_y, center_z=arr_center_z)

    radian_vals = np.radians(np.linspace(-90, 90, n_points + 1))

    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len,
                                     n_users=len(usr_angles_deg))

    my_distortion = distortion.ThirdOrderNonLin(toi_db=ibo_arr[0], avg_samp_pow=my_mod.avg_sample_power)
    # my_distortion = distortion.SoftLimiter(ibo_db=ibo_arr[0], avg_samp_pow=my_mod.avg_sample_power)

    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=212, cord_y=212, cord_z=1.5,
                                             center_freq=int(3.5e9), carrier_spacing=int(15e3))

    for n_ant_val in n_ant_arr:
        my_array = antenna_arrray.PlanarRectangularArray(n_elements_per_row=int(np.sqrt(n_ant_val)), n_elements_per_col=int(np.sqrt(n_ant_val)), base_transceiver=my_tx, center_freq=int(3.5e9),
                                              wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)

        # verify coordinate and angular relations
        # utilities.plot_spatial_config(ant_array=my_array, rx_points_lst=rx_points, plot_3d=True)
        # my_standard_rx.set_position(cord_x=usr_pos_tup[0][0], cord_y=usr_pos_tup[0][1], cord_z=usr_pos_tup[0][2])
        # utilities.plot_spatial_config(ant_array=my_array, rx_transceiver=my_standard_rx, plot_3d=True)

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
                                                        impairment=copy.deepcopy(my_distortion))

                    my_tmp_array = antenna_arrray.LinearArray(n_elements=n_ant_val, base_transceiver=my_tmp_tx,
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

                        rx_sig_fd = my_tmp_miso_los_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd, sum=False)
                        rx_sig_clean_fd = my_tmp_miso_los_chan.propagate(in_sig_mat=clean_ofdm_symbol_fd, sum=False)

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
                    usr_pos_x, usr_pos_y, usr_pos_z = usr_pos_tuple_val
                    my_standard_rx.set_position(cord_x=usr_pos_x, cord_y=usr_pos_y, cord_z=usr_pos_z)

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

                if plot_precoding_beampatterns:
                    bit_rng = np.random.default_rng(4321)
                    desired_sig_pow_per_pt = []
                    distorted_sig_inband_pow_per_pt = []
                    distorted_sig_oob_pow_per_pt = []
                    for pt_idx, point in enumerate(rx_points):

                        (x_cord, y_cord, z_cord) = point
                        my_standard_rx.set_position(cord_x=x_cord, cord_y=y_cord, cord_z=z_cord)
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

                            rx_sig_fd = my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd, sum=False)
                            clean_rx_sig_fd = my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd, sum=False)
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

                            # calculate SDR on symbol basis
                        desired_sig_pow_per_pt.append(np.sum(desired_sig_pow_arr))
                        distorted_sig_inband_pow_per_pt.append(np.sum(distortion_inband_sig_pow_arr))
                        distorted_sig_oob_pow_per_pt.append(np.sum(distortion_oob_sig_pow_arr))
                    # %%
                    # plot beampatterns of desired and distortion components
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989',
                                      '#A2C8EC', '#FFBC79',
                                      '#CFCFCF']

                    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
                    # fig.set_tight_layout(True)
                    # arrange into matrix
                    n_points_per_row_col = int(np.sqrt(n_points))
                    desired_sig_pow_mat = 10*np.log10(np.reshape(desired_sig_pow_per_pt, (n_points_per_row_col, n_points_per_row_col)))
                    distorted_sig_inband_pow_mat = 10*np.log10(np.reshape(distorted_sig_inband_pow_per_pt, (n_points_per_row_col, n_points_per_row_col)))
                    distorted_sig_oob_pow_mat = 10*np.log10(np.reshape(distorted_sig_oob_pow_per_pt, (n_points_per_row_col, n_points_per_row_col)))

                    # mark predicted intermodulation/distortion directions
                    print("Number of intermodulation combinations: %d" %(len(list(itertools.product(range(n_users), repeat=3)))))
                    dist_marker_azimuths = []
                    dist_marker_elevations = []
                    arcsin_arg_periodize = lambda val_a: val_a - 1.0 if val_a > 1.0 else (
                        val_a + 1.0 if val_a < -1.0 else val_a)

                    for usr_idx_1, usr_idx_2, usr_idx_3 in itertools.product(range(n_users), repeat=3):
                        usr_1_u = np.cos(usr_angles_rad[usr_idx_1][1]) * np.sin(usr_angles_rad[usr_idx_1][0])
                        usr_1_v = np.sin(usr_angles_rad[usr_idx_1][1])
                        usr_2_u = np.cos(usr_angles_rad[usr_idx_2][1]) * np.sin(usr_angles_rad[usr_idx_2][0])
                        usr_2_v = np.sin(usr_angles_rad[usr_idx_2][1])
                        usr_3_u = np.cos(usr_angles_rad[usr_idx_3][1]) * np.sin(usr_angles_rad[usr_idx_3][0])
                        usr_3_v = np.sin(usr_angles_rad[usr_idx_3][1])

                        imd_u = usr_1_u + usr_2_u - usr_3_u
                        imd_v = usr_1_v + usr_2_v - usr_3_v
                        dist_marker_azimuths.append(np.rad2deg(np.arcsin(imd_v)))
                        dist_marker_elevations.append(np.rad2deg(np.arctan(imd_u / np.sqrt(np.abs(1 - imd_u**2 - imd_v**2)))))

                    for ax in axs[1::]:
                        ax.scatter(dist_marker_azimuths, dist_marker_elevations, color=CB_color_cycle[5], marker="*", label="Distortion", zorder=2)

                    # mark user directions
                    usr_marker_azimuths = []
                    usr_marker_elevations = []
                    for usr_angle_tuple in usr_angles_deg:
                        usr_marker_azimuths.append(usr_angle_tuple[0])
                        usr_marker_elevations.append(usr_angle_tuple[1])

                    for ax in axs:
                        ax.scatter(usr_marker_elevations, usr_marker_azimuths, color="black", marker="x", label="User", zorder=2)


                    im1 = axs[0].contourf(desired_sig_pow_mat, extent=[-90, 90, -90, 90])
                    axs[0].set_aspect("equal")
                    axs[0].set_title("Desired signal power [dB]")
                    axs[0].set_xlabel("Azimuth angle [°]")
                    axs[0].set_ylabel("Elevation angle [°]")
                    x_and_y_ticks = np.linspace(-90, 90, 7, endpoint=True)
                    axs[0].set_xticks(x_and_y_ticks)
                    axs[0].set_yticks(x_and_y_ticks)
                    axs[0].grid(True)

                    im2 = axs[1].contourf(distorted_sig_inband_pow_mat, extent=[-90, 90, -90, 90])
                    axs[1].set_aspect("equal")
                    axs[1].set_title("Distortion IB power [dB]")
                    axs[1].set_xlabel("Azimuth angle [°]")
                    axs[1].set_xticks(x_and_y_ticks)
                    axs[1].grid(True)
                    axs[1].legend(ncol=2, loc="lower center", borderaxespad=-5.5)

                    im3 = axs[2].contourf(distorted_sig_oob_pow_mat, extent=[-90, 90, -90, 90])
                    axs[2].set_aspect("equal")
                    axs[2].set_title("Distortion OOB power [dB]")
                    axs[2].set_xlabel("Azimuth angle [°]")
                    axs[2].set_xticks(x_and_y_ticks)
                    axs[2].grid(True)
                    fig.colorbar(im3, ax=axs.ravel().tolist(), location='right', shrink=0.8)

                    beampattern_filename_str = "multiuser_shared_sc_mrt_beampatterns_%s_%s_nfft%d_nsc%d_ibo%d_angles%s_distances%s_npoints%d_nsnap%d_nant%s" % (
                        my_distortion, my_miso_chan, n_fft, n_sub_carr, ibo_val_db,
                        '_'.join([str(val) for val in usr_angles_deg]),
                        '_'.join([str(val) for val in usr_distances]), n_points, beampattern_n_snapshots,
                        '_'.join([str(val) for val in [n_ant_val]]))
                    plt.savefig("../figs/multiuser/planar_arrays/%s.png" % (beampattern_filename_str),
                                dpi=600, bbox_inches='tight')
                    plt.show()
                    # plt.cla()
                    # plt.close()
