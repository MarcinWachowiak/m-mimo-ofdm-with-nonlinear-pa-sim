# Experimental OFDM simulation
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import time

from utilities import count_mismatched_bits, snr_to_ebn0, ebn0_to_snr, to_db
import channels
import modulation
import impairments
import transceiver

from plot_settings import set_latex_plot_style

set_latex_plot_style()

# %%

my_mod = modulation.OfdmQamModem(constel_size=16, n_fft=4096, n_sub_carr=1024, cp_len=256)
my_distortion = impairments.SoftLimiter(3, my_mod.ofdm_avg_sample_pow())
my_tx= transceiver.Transceiver(my_mod, my_distortion)
my_tx.impairment.plot_characteristics()

my_demod = modulation.OfdmQamModem(constel_size=16, n_fft=4096, n_sub_carr=1024, cp_len=256)
my_rx = transceiver.Transceiver(my_demod)

my_chan = channels.AwgnChannel(0, True, 1234)
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
dist_vals_db = [5]  # np.arange(8, -1, -1)

include_clean_run = False
clean_run_flag = include_clean_run
if include_clean_run:
    dist_vals_db = np.insert(dist_vals_db, 0, 0)

print("Distortion IBO/TOI values:", dist_vals_db)
ber_per_dist, freq_arr, clean_ofdm_psd, distortion_psd = ([] for i in range(4))

sample_constellation = True
constel_snapshot = []

start_time = time.time()
for dist_idx, dist_val_db in enumerate(dist_vals_db):

    # if dist_idx == 1:
    my_rx.modem.correct_constellation(dist_val_db)
    my_tx.impairment.set_ibo(dist_val_db)

    snapshot_counter = 0
    clean_ofdm_for_psd = []
    distortion_for_psd = []
    bers = np.zeros([len(snr_arr)])

    for idx, snr in enumerate(snr_arr):
        my_chan.set_snr(snr)
        n_err = 0
        bits_sent = 0
        while bits_sent < bits_sent_max and n_err < n_err_min:
            tx_bits = bit_rng.choice((0, 1), my_mod.n_bits_per_ofdm_sym)
            tx_ofdm_symbol, clean_ofdm_symbol = my_tx.transmit(tx_bits, return_both=True)

            rx_ofdm_symbol = my_chan.propagate(tx_ofdm_symbol, my_mod.ofdm_avg_sample_pow())

            if plot_psd and snapshot_counter < n_collected_snapshots:
                clean_ofdm_for_psd.append(clean_ofdm_symbol)
                distortion_for_psd.append(tx_ofdm_symbol - my_demod.alpha * clean_ofdm_symbol)
                snapshot_counter += 1

            rx_bits = my_rx.receive(rx_ofdm_symbol)
            n_bit_err = count_mismatched_bits(tx_bits, rx_bits)

            # if sample_constellation:
            #     constel_snapshot.append(tx_symb)
            #     constel_snapshot.append(rx_symb)
            #     sample_constellation = False

            bits_sent += my_mod.n_bits_per_ofdm_sym
            n_err += n_bit_err
        bers[idx] = n_err / bits_sent
    ber_per_dist.append(bers)
    clean_run_flag = False

    if plot_psd:
        # flatten psd data
        clean_odfm_freq_arr, clean_odfm_psd_arr = welch(np.concatenate(clean_ofdm_for_psd).ravel(), fs=psd_nfft,
                                                        nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                        return_onesided=False)
        distortion_freq_arr, distortion_odfm_psd_arr = welch(np.concatenate(distortion_for_psd).ravel(), fs=psd_nfft,
                                                             nfft=psd_nfft,
                                                             nperseg=n_samp_per_seg,
                                                             return_onesided=False)
        freq_arr.append(np.array(clean_odfm_freq_arr[0:len(clean_odfm_freq_arr) // 2]))
        clean_ofdm_psd.append(to_db(np.array(clean_odfm_psd_arr[0:len(clean_odfm_psd_arr) // 2])))
        distortion_psd.append(to_db(np.array(distortion_odfm_psd_arr[0:len(distortion_odfm_psd_arr) // 2])))

print("--- Computation time: %f ---" % (time.time() - start_time))
# %%
# #plot sampled constellation - check shrinking
# fig, ax = plt.subplots()
# ax.grid()
# ax.scatter(constel_snapshot[1].real, constel_snapshot[1].imag, label="Soft lim, IBO = -8dB]", marker="*")
# ax.scatter(constel_snapshot[0].real, constel_snapshot[0].imag, label="Clean sig")
# ax.legend()
# ax.set_title('Constellation comparison')
# ax.set_xlabel("In-phase")
# ax.set_ylabel("Quadrature")
# plt.show()

# %%
fig1, ax1 = plt.subplots(1, 1)
if not include_clean_run:
    ax1.plot(freq_arr[0], clean_ofdm_psd[0], label="Desired")

for idx, dist_val in enumerate(dist_vals_db):
    if include_clean_run:
        if idx == 0:
            ax1.plot(freq_arr[0], clean_ofdm_psd[0], label="Desired")
        else:
            ax1.plot(freq_arr[0], distortion_psd[idx], label="Desired")
    else:
        ax1.plot(freq_arr[0], distortion_psd[idx], label="Distortion")

ax1.set_title("Power spectral density")
ax1.set_xlabel("Subcarrier index [-]")
ax1.set_ylabel("Power [dB]")
ax1.legend(title="Signals, IBO: %d [dB]" % dist_vals_db[0])
ax1.grid()

plt.tight_layout()
plt.savefig("figs/psd_soft_lim.pdf", dpi=600, bbox_inches='tight')
plt.show()

# %%
fig2, ax2 = plt.subplots(1, 1)
ax2.set_yscale('log')
for idx, dist_val in enumerate(dist_vals_db):
    if include_clean_run:
        if idx == 0:
            ax2.plot(ebn0_arr, ber_per_dist[idx], label="No dist")
        else:
            ax2.plot(ebn0_arr, ber_per_dist[idx], label=dist_val)
    else:
        ax2.plot(ebn0_arr, ber_per_dist[idx], label=dist_val)
# fix log scaling
ax2.set_title("Bit error rate, QAM" + str(my_mod.constellation_size))
ax2.set_xlabel("Eb/N0 [dB]")
ax2.set_ylabel("BER")
ax2.grid()
ax2.legend(title="IBO [dB]")

plt.tight_layout()
plt.savefig("figs/ber_soft_lim.pdf", dpi=600, bbox_inches='tight')
plt.show()

print("Finished exectution!")
# %%
