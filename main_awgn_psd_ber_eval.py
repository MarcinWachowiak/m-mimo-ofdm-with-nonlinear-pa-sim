# SISO OFDM simulation with nonlinearity
# Desired vs distorted signal PSD and BER comparison
# %%
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

set_latex_plot_style()

# %%

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=256)

my_distortion = distortion.SoftLimiter(5, my_mod.avg_sample_power)
my_limiter2 = distortion.Rapp(0, my_mod.avg_sample_power, p_hardness=5)
my_limiter3 = distortion.ThirdOrderNonLin(25, my_mod.avg_sample_power)

my_tx = transceiver.Transceiver(modem=my_mod, impairment=my_distortion)
my_tx.impairment.plot_characteristics()

my_demod = copy.deepcopy(my_mod)
my_rx = transceiver.Transceiver(my_demod)

my_noise = noise.Awgn(0, True, 1234)
bit_rng = np.random.default_rng(4321)

ebn0_arr = np.arange(0, 21, 2)
print("Eb/n0 values:", ebn0_arr)
snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_fft, my_mod.n_sub_carr, my_mod.constel_size)

plot_psd = True
n_collected_snapshots = 100
psd_nfft = 4096
n_samp_per_seg = 512

bits_sent_max = int(1e6)
n_err_min = 1000

# %%
dist_vals_db = [3, 5, 7]
include_clean_run = True
if include_clean_run:
    dist_vals_db = np.insert(dist_vals_db, 0, 0)

print("Distortion IBO/TOI values:", dist_vals_db)
ber_per_dist, freq_arr, clean_ofdm_psd, distortion_psd, tx_ofdm_psd = ([] for i in range(5))

start_time = time.time()
for dist_idx, dist_val_db in enumerate(dist_vals_db):

    if not (include_clean_run and dist_idx == 0):
        my_rx.modem.correct_constellation(dist_val_db)
        my_tx.impairment.set_ibo(dist_val_db)

    snapshot_counter = 0
    clean_ofdm_for_psd = []
    distortion_for_psd = []
    tx_ofdm_for_psd = []

    bers = np.zeros([len(snr_arr)])

    for idx, snr in enumerate(snr_arr):
        my_noise.snr_db = snr
        n_err = 0
        bits_sent = 0
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
                distortion_for_psd.append(tx_ofdm_symbol - my_rx.modem.alpha * clean_ofdm_symbol)
                snapshot_counter += 1

            rx_bits = my_rx.receive(rx_ofdm_symbol)
            n_bit_err = count_mismatched_bits(tx_bits, rx_bits)

            bits_sent += my_mod.n_bits_per_ofdm_sym
            n_err += n_bit_err
        bers[idx] = n_err / bits_sent
    ber_per_dist.append(bers)

    if plot_psd:
        # flatten psd data

        tx_odfm_freq_arr, tx_odfm_psd_arr = welch(np.concatenate(tx_ofdm_for_psd).ravel(), fs=psd_nfft,
                                                  nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                  return_onesided=False)
        clean_odfm_freq_arr, clean_odfm_psd_arr = welch(np.concatenate(clean_ofdm_for_psd).ravel(), fs=psd_nfft,
                                                        nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                        return_onesided=False)
        distortion_freq_arr, distortion_odfm_psd_arr = welch(np.concatenate(distortion_for_psd).ravel(), fs=psd_nfft,
                                                             nfft=psd_nfft,
                                                             nperseg=n_samp_per_seg,
                                                             return_onesided=False)
        freq_arr.append(np.array(clean_odfm_freq_arr[0:len(clean_odfm_freq_arr) // 2]))
        tx_ofdm_psd.append(to_db(np.array(tx_odfm_psd_arr[0:len(tx_odfm_psd_arr) // 2])))
        clean_ofdm_psd.append(to_db(np.array(clean_odfm_psd_arr[0:len(clean_odfm_psd_arr) // 2])))
        distortion_psd.append(to_db(np.array(distortion_odfm_psd_arr[0:len(distortion_odfm_psd_arr) // 2])))

print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
if plot_psd:
    # Plot PSD at the receiver
    fig1, ax1 = plt.subplots(1, 1)
    for idx, dist_val in enumerate(dist_vals_db):
        if include_clean_run:
            if idx == 0:
                ax1.plot(freq_arr[0], clean_ofdm_psd[0], label="No distortion")
            else:
                ax1.plot(freq_arr[0], tx_ofdm_psd[idx], label=dist_val)
        else:
            ax1.plot(freq_arr[0], tx_ofdm_psd[idx], label=dist_val)

    ax1.set_title("Power spectral density at TX")
    ax1.set_xlabel("Subcarrier index [-]")
    ax1.set_ylabel("Power [dB]")
    ax1.legend(title="IBO [dB]")
    ax1.grid()

    plt.tight_layout()
    plt.savefig("figs/psd_soft_lim_combined.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Plot decomposed PSD of desired signal and distortion separately
    fig2, ax2 = plt.subplots(1, 1)
    for idx, dist_val in enumerate(dist_vals_db):
        if include_clean_run:
            if idx == 0:
                ax2.plot(freq_arr[0], clean_ofdm_psd[0], label="Desired")
            else:
                ax2.plot(freq_arr[0], distortion_psd[idx], label="Distortion IBO = %d [dB]" % dist_val)
        else:
            ax2.plot(freq_arr[0], distortion_psd[idx], label="Distortion IBO = %d [dB]" % dist_val)

    ax2.set_title("Power spectral density at TX decomposed")
    ax2.set_xlabel("Subcarrier index [-]")
    ax2.set_ylabel("Power [dB]")
    ax2.legend(title="Signals:")
    ax2.grid()

    plt.tight_layout()
    plt.savefig("figs/psd_soft_lim_decomposed.png", dpi=600, bbox_inches='tight')
    plt.show()

# %%
fig3, ax3 = plt.subplots(1, 1)
ax3.set_yscale('log')
for idx, dist_val in enumerate(dist_vals_db):
    if include_clean_run:
        if idx == 0:
            ax3.plot(ebn0_arr, ber_per_dist[idx], label="No distortion")
        else:
            ax3.plot(ebn0_arr, ber_per_dist[idx], label=dist_val)
    else:
        ax3.plot(ebn0_arr, ber_per_dist[idx], label=dist_val)
# fix log scaling
ax3.set_title("Bit error rate, QAM" + str(my_mod.constellation_size))
ax3.set_xlabel("Eb/N0 [dB]")
ax3.set_ylabel("BER")
ax3.grid()
ax3.legend(title="IBO [dB]")

plt.tight_layout()
plt.savefig("figs/ber_soft_lim_siso.png", dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
# %%