"""
Estimate the alpha coefficient for each singe antenna under precoding for a selected channel models
and plot the spread of it as a function of the set IBO per antenna.
"""

# %%
import os
import sys

sys.path.append(os.getcwd())

import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import channel
import distortion
import modulation
import transceiver
import antenna_array
from plot_settings import set_latex_plot_style


if __name__ == '__main__':

    set_latex_plot_style()
    # %%
    print("Multi-antenna processing init!")

    my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=128)
    my_distortion = distortion.SoftLimiter(ibo_db=0, avg_samp_pow=my_mod.avg_sample_power)
    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(3.5e9),
                                    carrier_spacing=int(15e3))
    my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=212,
                                    cord_y=212, cord_z=1.5,
                                    center_freq=int(3.5e9), carrier_spacing=int(15e3))
    my_rx.correct_constellation()

    # %%
    n_ant_val = 8
    # averaging length
    n_ofdm_symb = 1e2
    print("N antennas values:", n_ant_val)
    ibo_arr = np.linspace(0.01, 10.0, 10)
    print("IBO values:", ibo_arr)
    alpha_per_ibo_analytical = []

    # get analytical value of alpha
    for ibo_idx, ibo_val_db in enumerate(ibo_arr):
        alpha_per_ibo_analytical.append(my_mod.calc_alpha(ibo_val_db))
    # %%
    my_array = antenna_array.LinearArray(n_elements=n_ant_val, base_transceiver=my_tx, center_freq=int(3.5e9),
                                          wav_len_spacing=0.5,
                                          cord_x=0, cord_y=0, cord_z=15)
    my_miso_rayleigh_chan = channel.MisoRayleighFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, seed=1234)
    my_miso_los_chan = channel.MisoLosFd()
    my_miso_two_path_chan = channel.MisoTwoPathFd()

    my_miso_los_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx, skip_attenuation=False)
    my_miso_two_path_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_rx,
                                           skip_attenuation=False)

    # list of channel objects
    chan_lst = [my_miso_rayleigh_chan, my_miso_two_path_chan, my_miso_los_chan]

    alpha_per_nant_per_ibo = np.zeros((len(chan_lst), len(ibo_arr), n_ant_val))

    for chan_idx, chan_obj in enumerate(chan_lst):
        start_time = time.time()
        print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        tmp_chan_mat = chan_obj.get_channel_mat_fd()
        my_array.set_precoding_matrix(channel_mat_fd=tmp_chan_mat, mr_precoding=True)

        for ibo_idx, ibo_val_db in enumerate(ibo_arr):
            my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

            # estimate alpha correcting coefficient
            # same seed is required
            bit_rng = np.random.default_rng(4321)
            ofdm_symb_idx = 0
            alpha_numerator_vecs = []
            alpha_denominator_vecs = []
            while ofdm_symb_idx < n_ofdm_symb:
                # reroll coeffs for each symbol for rayleigh chan
                # if chan_idx == 0:
                #     chan_obj.reroll_channel_coeffs()
                #     tmp_chan_mat = chan_obj.get_channel_mat_fd()
                #     my_array.set_precoding_matrix(channel_mat_fd=tmp_chan_mat, mr_precoding=True)
                #     my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)

                tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
                tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

                clean_nsc_ofdm_symb_fd = np.concatenate(
                    (clean_ofdm_symbol_fd[:, -my_mod.n_sub_carr // 2:],
                     clean_ofdm_symbol_fd[:, 1:(my_mod.n_sub_carr // 2) + 1, ]), axis=1)
                rx_nsc_ofdm_symb_fd = np.concatenate(
                    (tx_ofdm_symbol_fd[:, -my_mod.n_sub_carr // 2:], tx_ofdm_symbol_fd[:, 1:(my_mod.n_sub_carr // 2) + 1]),
                    axis=1)
                # estimate alpha parameters for each antenna and compare in regard to the average
                alpha_numerator_vecs.append(np.multiply(rx_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))
                alpha_denominator_vecs.append(np.multiply(clean_nsc_ofdm_symb_fd, np.conjugate(clean_nsc_ofdm_symb_fd)))

                ofdm_symb_idx += 1

                # calculate alpha estimate
            alpha_num = np.average(np.hstack(alpha_numerator_vecs), axis=1)
            alpha_denum = np.average(np.hstack(alpha_denominator_vecs), axis=1)
            alpha_per_nant_per_ibo[chan_idx, ibo_idx, :] = np.abs(alpha_num / alpha_denum)
        print("--- Computation time: %f ---" % (time.time() - start_time))

    # %%
    fig1, ax1 = plt.subplots(1, 1)
    ax1.fill_between(ibo_arr, np.amin(alpha_per_nant_per_ibo[0, :, :], axis=1),
                     np.amax(alpha_per_nant_per_ibo[0, :, :], axis=1), alpha=0.9, label="Rayleigh")
    ax1.fill_between(ibo_arr, np.amin(alpha_per_nant_per_ibo[1, :, :], axis=1),
                     np.amax(alpha_per_nant_per_ibo[1, :, :], axis=1), alpha=0.9, label="Two-path")
    ax1.fill_between(ibo_arr, np.amin(alpha_per_nant_per_ibo[2, :, :], axis=1),
                     np.amax(alpha_per_nant_per_ibo[2, :, :], axis=1), alpha=0.9, label="LOS")
    ax1.plot(ibo_arr, alpha_per_ibo_analytical, 'k--', label="Analytical", linewidth=0.5, )

    ax1.set_title(r"Coefficient $\alpha$ [-]")
    ax1.set_xlabel("IBO [dB]")
    ax1.set_ylabel(r"$\alpha$ [-]")
    ax1.grid()
    ax1.legend(title="Channel:")

    plt.tight_layout()
    plt.savefig(
        "../figs/alpha_per_antenna_n_ant%d_ibo%1.1fto%1.1f.png" % (n_ant_val, min(ibo_arr), max(ibo_arr)),
        dpi=600, bbox_inches='tight')
    plt.show()

    print("Finished execution!")
