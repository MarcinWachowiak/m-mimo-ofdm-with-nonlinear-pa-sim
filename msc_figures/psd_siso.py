"""
Master's thesis plotting script,
plot the power spectral density (PSD) of the transmitted signal decomposed into desired and distortion parts
for selected values of IBO.
"""

# %%
import os
import sys

sys.path.append(os.getcwd())

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import distortion
import modulation
import noise
import transceiver
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, ebn0_to_snr, to_db

if __name__ == '__main__':

    set_latex_plot_style(use_tex=True, fig_width_in=5.89572)

    # %%

    my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=2048, cp_len=256)

    my_distortion = distortion.SoftLimiter(5, my_mod.avg_sample_power)
    my_limiter2 = distortion.Rapp(0, my_mod.avg_sample_power, p_hardness=5)
    my_limiter3 = distortion.ThirdOrderNonLin(25, my_mod.avg_sample_power)

    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(3.5e9), carrier_spacing=int(15e3))
    # my_tx.impairment.plot_characteristics()
    my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(3.5e9), carrier_spacing=int(15e3))

    my_noise = noise.Awgn(0, True, 1234)
    bit_rng = np.random.default_rng(4321)

    ebn0_arr = np.arange(0, 21, 2)
    print("Eb/n0 values:", ebn0_arr)
    snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_fft, my_mod.n_sub_carr, my_mod.constel_size)

    plot_psd = True
    n_collected_snapshots = 100
    psd_nfft = 4096
    n_samp_per_seg = 1024

    bits_sent_max = int(1e6)
    n_err_min = 1000
    convergence_epsilon = 0.001  # e.g. 0.1%
    conv_ite_th = 10  # number of iterations after the convergence threshold is activated

    # %%
    dist_vals_db = [0, 3, 5, 7]
    include_clean_run = True
    if include_clean_run:
        dist_vals_db = np.insert(dist_vals_db, 0, 0)

    print("Distortion IBO/TOI values:", dist_vals_db)
    ber_per_dist, freq_arr, clean_ofdm_psd, distortion_psd, desired_psd, tx_ofdm_psd = ([] for i in range(6))

    start_time = time.time()
    for dist_idx, dist_val_db in enumerate(dist_vals_db):

        if not (include_clean_run and dist_idx == 0):
            my_rx.modem.correct_constellation(dist_val_db)
            my_tx.impairment.set_ibo(dist_val_db)

        snapshot_counter = 0
        clean_ofdm_for_psd = []
        distortion_for_psd = []
        tx_ofdm_for_psd = []
        desired_for_psd = []
        bers = np.zeros([len(snr_arr)])

        for idx, snr in enumerate(snr_arr):
            my_noise.snr_db = snr
            n_err = 0
            bits_sent = 0
            ite_cnt = 0
            while bits_sent < bits_sent_max and n_err < n_err_min:
                tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
                tx_ofdm_symbol, clean_ofdm_symbol = my_tx.transmit(tx_bits, out_domain_fd=False, return_both=True)

                if include_clean_run and dist_idx == 0:
                    rx_ofdm_symbol = my_noise.process(clean_ofdm_symbol, my_mod.avg_sample_power)
                else:
                    rx_ofdm_symbol = my_noise.process(tx_ofdm_symbol, my_mod.avg_sample_power)

                if plot_psd and snapshot_counter < n_collected_snapshots:
                    tx_ofdm_for_psd.append(tx_ofdm_symbol)
                    clean_ofdm_for_psd.append(clean_ofdm_symbol)
                    desired_for_psd.append(np.multiply(clean_ofdm_symbol, my_rx.modem.alpha))
                    distortion_for_psd.append(tx_ofdm_symbol - my_rx.modem.alpha * clean_ofdm_symbol)
                    snapshot_counter += 1
                else:
                    break

                rx_bits = my_rx.receive(rx_ofdm_symbol)
                n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
            if bits_sent != 0:
                bers[idx] = n_err / bits_sent
            else:
                bers[idx] = np.nan

        ber_per_dist.append(bers)

        if plot_psd:
            # flatten psd data

            tx_odfm_freq_arr, tx_odfm_psd_arr = welch(np.concatenate(tx_ofdm_for_psd).ravel(), fs=psd_nfft,
                                                      nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                      return_onesided=False)
            clean_odfm_freq_arr, clean_odfm_psd_arr = welch(np.concatenate(clean_ofdm_for_psd).ravel(), fs=psd_nfft,
                                                            nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                            return_onesided=False)
            desired_odfm_freq_arr, desired_odfm_psd_arr = welch(np.concatenate(desired_for_psd).ravel(), fs=psd_nfft,
                                                                nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                                return_onesided=False)
            distortion_freq_arr, distortion_odfm_psd_arr = welch(np.concatenate(distortion_for_psd).ravel(), fs=psd_nfft,
                                                                 nfft=psd_nfft,
                                                                 nperseg=n_samp_per_seg,
                                                                 return_onesided=False)

            sorted_freq_arr, sorted_clean_odfm_psd_arr = zip(
                *sorted(zip(clean_odfm_freq_arr, clean_odfm_psd_arr)))
            sorted_freq_arr, sorted_tx_odfm_psd_arr = zip(
                *sorted(zip(tx_odfm_freq_arr, tx_odfm_psd_arr)))
            sorted_freq_arr, sorted_desired_psd_arr = zip(
                *sorted(zip(desired_odfm_freq_arr, desired_odfm_psd_arr)))
            sorted_freq_arr, sorted_distortion_odfm_psd_arr = zip(
                *sorted(zip(distortion_freq_arr, distortion_odfm_psd_arr)))

            freq_arr.append(np.array(sorted_freq_arr))
            clean_ofdm_psd.append(to_db(np.array(sorted_clean_odfm_psd_arr)))
            tx_ofdm_psd.append(to_db(np.array(sorted_tx_odfm_psd_arr)))
            distortion_psd.append(to_db(np.array(sorted_distortion_odfm_psd_arr)))
            desired_psd.append(to_db(np.array(sorted_desired_psd_arr)))

    print("--- Computation time: %f ---" % (time.time() - start_time))

    # %%
    if plot_psd:
        # normalize
        # Plot PSD at the receiver
        fig1, ax1 = plt.subplots(1, 1)
        for idx, dist_val in enumerate(dist_vals_db):
            if include_clean_run:
                if idx == 0:
                    ax1.plot(freq_arr[0], clean_ofdm_psd[0], label="No dist")
                else:
                    ax1.plot(freq_arr[0], tx_ofdm_psd[idx], label=dist_val)
            else:
                ax1.plot(freq_arr[0], tx_ofdm_psd[idx], label=dist_val)

        ax1.set_title("Relative power spectral density of transmitted OFDM signal in regard to IBO")
        ax1.set_xlabel("Subcarrier index [-]")
        ax1.set_ylabel("Relative power spectral density [dB]")
        ax1.legend(title="IBO [dB]:")
        ax1.grid()

        plt.tight_layout()
        plt.savefig(
            "../figs/msc_figs/ofdm_psd_soft_lim_combined_ibo%s.pdf" % ('_'.join([str(val) for val in dist_vals_db[1:]])),
            dpi=600, bbox_inches='tight')
        plt.show()

        # %%
        # Plot decomposed PSD of desired signal and distortion separately
        CB_color_cycle = ['#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                          '#CFCFCF']
        color_idx = 0
        fig2, ax2 = plt.subplots(1, 1)
        for idx, dist_val in enumerate(dist_vals_db):
            if idx == 0:
                continue
            ax2.plot(freq_arr[0], distortion_psd[idx], label=dist_val, color=CB_color_cycle[color_idx], alpha=0.5)
            color_idx += 1

        color_idx = 0
        for idx, dist_val in enumerate(dist_vals_db):
            if idx == 0:
                continue
            ax2.plot(freq_arr[0], desired_psd[idx], label=dist_val, color=CB_color_cycle[color_idx])
            color_idx += 1

        ax2.set_title("Relative power spectral density of signal components in regard to IBO")
        ax2.set_xlabel("Subcarrier index [-]")
        ax2.set_ylabel("Relative power spectral density [dB]")
        ax2.grid()

        import matplotlib.lines as mlines

        desired_ibo_0 = mlines.Line2D([0], [0], color='#FF800E', label=0)
        desired_ibo_3 = mlines.Line2D([0], [0], color='#ABABAB', label=3)
        desired_ibo_5 = mlines.Line2D([0], [0], color='#595959', label=5)
        desired_ibo_7 = mlines.Line2D([0], [0], color='#5F9ED1', label=7)
        leg1 = plt.legend(handles=[desired_ibo_0, desired_ibo_3, desired_ibo_5, desired_ibo_7],
                          title="Desired signal\n for IBO [dB]:", loc="upper left",
                          framealpha=0.9)
        plt.gca().add_artist(leg1)

        distorted_ibo_0 = mlines.Line2D([0], [0], color='#FF800E', label=0, alpha=0.5)
        distorted_ibo_3 = mlines.Line2D([0], [0], color='#ABABAB', label=3, alpha=0.5)
        distorted_ibo_5 = mlines.Line2D([0], [0], color='#595959', label=5, alpha=0.5)
        distorted_ibo_7 = mlines.Line2D([0], [0], color='#5F9ED1', label=7, alpha=0.5)
        ax2.legend(handles=[distorted_ibo_0, distorted_ibo_3, distorted_ibo_5, distorted_ibo_7],
                   title="Distortion signal\n for IBO [dB]:", loc="upper right",
                   framealpha=0.9)

        plt.tight_layout()
        plt.savefig(
            "../figs/msc_figs/ofdm_psd_soft_lim_decomposed_ibo%s.pdf" % ('_'.join([str(val) for val in dist_vals_db[1:]])),
            dpi=600, bbox_inches='tight')
        plt.show()

    # # %%
    # fig3, ax3 = plt.subplots(1, 1)
    # ax3.set_yscale('log')
    # for idx, dist_val in enumerate(dist_vals_db):
    #     if include_clean_run:
    #         if idx == 0:
    #             ax3.plot(ebn0_arr, ber_per_dist[idx], label="No distortion")
    #         else:
    #             ax3.plot(ebn0_arr, ber_per_dist[idx], label=dist_val)
    #     else:
    #         ax3.plot(ebn0_arr, ber_per_dist[idx], label=dist_val)
    # # fix log scaling
    # ax3.set_title("Bit error rate, QAM" + str(my_mod.constellation_size))
    # ax3.set_xlabel("Eb/N0 [dB]")
    # ax3.set_ylabel("BER")
    # ax3.grid()
    # ax3.legend(title="IBO [dB]")
    #
    # plt.tight_layout()
    # plt.savefig("../figs/ber_soft_lim_siso.png", dpi=600, bbox_inches='tight')
    # plt.show()

    print("Finished execution!")
    # %%
