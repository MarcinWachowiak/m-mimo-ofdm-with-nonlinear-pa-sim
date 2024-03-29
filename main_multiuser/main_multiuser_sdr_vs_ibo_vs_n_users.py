"""
Measure the signal-to-distortion (SDR) as a function of the IBO in a multi-user scenario
for MRT precoding and selected number of users allocated simultaneously in the system.
"""

# %%
import os
import sys

sys.path.append(os.getcwd())

import copy

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

import channel
import distortion
import modulation
import transceiver
import antenna_array
import utilities
from plot_settings import set_latex_plot_style

if __name__ == '__main__':
    set_latex_plot_style()
    CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                      '#CFCFCF']
    # Multiple users data

    ibo_arr = np.arange(0, 7.01, 0.25)
    n_ant_arr = [32]
    n_users_lst = [1, 2, 3, 4, 5]
    radial_distance = 300
    angular_margin = 10
    # SDR
    sdr_n_snapshots = 100
    sdr_reroll_pos = False

    # modulation
    constel_size = 64
    n_fft = 4096
    n_sub_carr = 2048
    cp_len = 128

    my_mod = modulation.OfdmQamModem(constel_size=constel_size, n_fft=n_fft, n_sub_carr=n_sub_carr, cp_len=cp_len,
                                     n_users=1)
    my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
    my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                             cord_x=212.0, cord_y=212.0, cord_z=1.5,
                                             center_freq=int(3.5e9), carrier_spacing=int(15e3))
    sdr_per_scenario = []
    for scenario_idx, usr_count in enumerate(n_users_lst):
        my_mod.n_users = usr_count
        my_tx.modem.n_users = usr_count

        for n_ant_val in n_ant_arr:
            my_array = antenna_array.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                                  wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)
            # channel type
            my_miso_los_chan = channel.MisoLosFd()
            my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                              skip_attenuation=False)
            my_miso_two_path_chan = channel.MisoTwoPathFd()
            my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                   rx_transceiver=my_standard_rx,
                                                   skip_attenuation=False)

            my_miso_rayleigh_chan = channel.MisoRayleighFd(tx_transceivers=my_array.array_elements,
                                                           rx_transceiver=my_standard_rx,
                                                           seed=1234)
            chan_lst = [my_miso_los_chan]

            for chan_idx, my_miso_chan in enumerate(chan_lst):
                loc_rng = np.random.default_rng(2137)
                bit_rng = np.random.default_rng(4321)
                sdr_per_ibo = []
                for ibo_idx, ibo_val_db in enumerate(ibo_arr):

                    sdr_per_snap = np.zeros((usr_count, sdr_n_snapshots))
                    for snap_idx in range(sdr_n_snapshots):
                        # generate usr positions
                        if usr_count == 1:
                            first_usr_ang = loc_rng.uniform(low=angular_margin, high=180 - angular_margin)
                            usr_angles = [first_usr_ang]
                        else:
                            # favourable case TOI distortion not beamformed to user angles
                            usr_ang_spacing = (180.0 - 2 * angular_margin) / usr_count
                            if usr_ang_spacing <= 180.0 / n_ant_val * 2:
                                raise Warning(
                                    "User angular separation is too small - increase antenna count or reduce the number of users!")

                            usr_angles = []
                            for usr_idx in range(usr_count):
                                if usr_idx == 0:
                                    new_usr_ang = loc_rng.uniform(low=angular_margin,
                                                                  high=angular_margin + usr_ang_spacing)
                                else:
                                    new_usr_ang = loc_rng.uniform(low=usr_angles[-1] + usr_ang_spacing,
                                                                  high=angular_margin + usr_ang_spacing * (usr_idx + 1))
                                usr_angles.append(new_usr_ang)

                        usr_pos_tup_lst = []
                        for usr_idx, usr_angle in enumerate(usr_angles):
                            usr_pos_x = np.cos(np.deg2rad(usr_angle)) * radial_distance
                            usr_pos_y = np.sin(np.deg2rad(usr_angle)) * radial_distance
                            usr_pos_tup_lst.append((usr_pos_x, usr_pos_y), )

                        usr_chan_mat_lst = []
                        for usr_idx, usr_pos_tup in enumerate(usr_pos_tup_lst):
                            if isinstance(my_miso_chan, channel.MisoLosFd) or isinstance(my_miso_chan,
                                                                                         channel.MisoTwoPathFd):
                                my_standard_rx.set_position(cord_x=usr_pos_tup[0], cord_y=usr_pos_tup[1],
                                                            cord_z=my_standard_rx.cord_z)
                                my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements,
                                                              rx_transceiver=my_standard_rx,
                                                              skip_attenuation=False)

                            else:
                                my_miso_rayleigh_chan.reroll_channel_coeffs()
                            usr_chan_mat_lst.append(my_miso_chan.get_channel_mat_fd())

                        # set precoding and calculate AGC
                        if usr_count == 1:
                            usr_precod_chan_mat_lst = usr_chan_mat_lst[0]
                        else:
                            usr_precod_chan_mat_lst = usr_chan_mat_lst

                        my_array.set_precoding_matrix(channel_mat_fd=usr_precod_chan_mat_lst, mr_precoding=True)
                        my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                        vk_mat = my_array.get_precoding_mat()
                        if usr_count == 1:
                            vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)
                        else:
                            vk_pow_vec = np.sum(np.sum(np.power(np.abs(vk_mat), 2), axis=2), axis=1)

                        ibo_vec = 10 * np.log10(
                            10 ** (ibo_val_db / 10) * my_mod.n_sub_carr / (vk_pow_vec * n_ant_val))
                        ak_vect = my_mod.calc_alpha(ibo_db=ibo_vec)
                        ak_vect = np.expand_dims(ak_vect, axis=1)

                        tx_bits = np.squeeze(bit_rng.choice((0, 1), (usr_count, my_tx.modem.n_bits_per_ofdm_sym)))

                        arr_tx_sig_fd, clean_sig_mat_fd = my_array.transmit(in_bits=tx_bits, out_domain_fd=True,
                                                                            return_both=True, sum_usr_signals=True)
                        if usr_count > 1:
                            per_usr_clean_sig_fd_lst = my_array.transmit(in_bits=tx_bits, out_domain_fd=True,
                                                                         return_both=False, skip_dist=True,
                                                                         sum_usr_signals=False)

                        for usr_idx, user_pos_tup in enumerate(usr_pos_tup_lst):
                            rx_sig_fd = np.multiply(arr_tx_sig_fd, usr_chan_mat_lst[usr_idx])
                            clean_rx_sig_fd = np.multiply(clean_sig_mat_fd, usr_chan_mat_lst[usr_idx])
                            rx_sc_ofdm_symb_fd = np.concatenate(
                                (rx_sig_fd[:, -my_mod.n_sub_carr // 2:], rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                axis=1)
                            clean_sc_ofdm_symb_fd = np.concatenate(
                                (clean_rx_sig_fd[:, -my_mod.n_sub_carr // 2:],
                                 clean_rx_sig_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                                axis=1)

                            sc_ofdm_distortion_sig = np.subtract(rx_sc_ofdm_symb_fd, (ak_vect * clean_sc_ofdm_symb_fd))

                            if usr_count == 1:
                                desired_usr_rx_signal_sc = clean_sc_ofdm_symb_fd
                            else:
                                desired_usr_rx_signal_fb = np.multiply(per_usr_clean_sig_fd_lst[usr_idx],
                                                                       usr_chan_mat_lst[usr_idx])
                                desired_usr_rx_signal_sc = np.concatenate((desired_usr_rx_signal_fb[:,
                                                                           -my_mod.n_sub_carr // 2:],
                                                                           desired_usr_rx_signal_fb[:,
                                                                           1:(my_mod.n_sub_carr // 2) + 1]), axis=1)

                            sdr_per_snap[usr_idx, snap_idx] = np.sum(
                                np.power(np.abs(np.sum(ak_vect * desired_usr_rx_signal_sc, axis=0)), 2)) / np.sum(
                                np.power(np.abs(np.sum(sc_ofdm_distortion_sig, axis=0)), 2))

                    sdr_per_usr = utilities.to_db(np.average(sdr_per_snap, axis=1))
                    sdr_per_ibo.append(sdr_per_usr)
                sdr_per_scenario.append(sdr_per_ibo)

    sdr_per_scen_res = []
    for sdr_lst in sdr_per_scenario:
        sdr_per_scen_res.append(np.transpose(np.vstack(sdr_lst)))

    # %%
    usr_marker_lst = ['o', 's', '^', '*', 'v', '<', '>']
    fig1, ax1 = plt.subplots(1, 1)
    legend_handles = []
    for n_usrs_scen_idx, n_users_scen in enumerate(n_users_lst):
        for usr_idx in range(n_users_scen):
            ax1.plot(ibo_arr, sdr_per_scen_res[n_usrs_scen_idx][usr_idx], linestyle='-',
                     color=CB_color_cycle[n_usrs_scen_idx], marker=usr_marker_lst[usr_idx], fillstyle='none')
        legend_handles.append(
            mlines.Line2D([0], [0], linestyle='-', color=CB_color_cycle[n_usrs_scen_idx], label=n_users_scen))

    ax1.set_title("Multiuser SDR vs IBO")
    ax1.set_xlabel("IBO [dB]")
    ax1.set_ylabel("SDR [dB]")
    ax1.legend(handles=legend_handles, title="N users:")
    ax1.grid()

    plt.tight_layout()
    plt.savefig(
        "../figs/multiuser/sdr_vs_ibo_vs_nusr/multiuser_sdr_per_usr_vs_ibo_ibo%dto%d_%dnant_nsnap%d_nusrs%s.png" % (
            min(ibo_arr), max(ibo_arr), n_ant_arr[0], sdr_n_snapshots, '_'.join([str(val) for val in n_users_lst])),
        dpi=600, bbox_inches='tight')

    plt.show()

    # %%
    # average per user SDR
    fig1, ax1 = plt.subplots(1, 1)
    legend_handles = []
    for n_usrs_scen_idx, n_users_scen in enumerate(n_users_lst):
        ax1.plot(ibo_arr, np.average(sdr_per_scen_res[n_usrs_scen_idx], axis=0), linestyle='-',
                 color=CB_color_cycle[n_usrs_scen_idx], label=n_users_scen)

    ax1.set_title("Multiuser SDR vs IBO")
    ax1.set_xlabel("IBO [dB]")
    ax1.set_ylabel("SDR [dB]")
    ax1.legend(title="N users:")
    ax1.grid()

    plt.tight_layout()
    plt.savefig(
        "../figs/multiuser/sdr_vs_ibo_vs_nusr/multiuser_sdr_averaged_vs_ibo_ibo%dto%d_%dnant_nsnap%d_nusrs%s.png" % (
            min(ibo_arr), max(ibo_arr), n_ant_arr[0], sdr_n_snapshots, '_'.join([str(val) for val in n_users_lst])),
        dpi=600, bbox_inches='tight')

    plt.show()

    print("Finished processing!")
