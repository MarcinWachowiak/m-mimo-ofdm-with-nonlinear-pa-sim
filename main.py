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

from plot_settings import set_latex_plot_style

set_latex_plot_style()

# %%
constel_size = 16
n_sub_carr = 1024
n_fft = 4096
cyclic_prefix_len = int(0.15 * n_fft)
n_bits_per_ofdm_sym = int(np.log2(constel_size) * n_sub_carr)

my_mod = modulation.QamModem(constel_size)
print("Avg symbol power:", my_mod.avg_symbol_power)
# my_mod.plot_constellation()
my_chan = channels.AwgnChannel(0, True, 1234)
bit_rng = np.random.default_rng(4321)

my_limiter1 = impairments.SoftLimiter(0, my_mod.avg_symbol_power)
my_distortion = impairments.Rapp(0, my_mod.avg_symbol_power, 5)

print("Saturation power:", my_distortion.sat_pow)
tx_ofdm_symbol = my_distortion.process(np.array([5+5j, 5+7j, 7+5j, 7+7j]))
print(tx_ofdm_symbol)
my_limiter3 = impairments.ThirdOrderNonLin(15, my_mod.avg_symbol_power)
my_distortion.plot_characteristics(-18, 18, 0.1)

ebn0_arr = np.arange(0, 21, 2)
print("Eb/n0 values:", ebn0_arr)
snr_arr = ebn0_to_snr(ebn0_arr, n_fft, n_sub_carr, constel_size)

plot_psd = True
n_collected_snapshots = 100
psd_nfft = 4096
n_samp_per_seg = 512

bits_sent_max = int(1e6)
n_err_min = 1000

# %%
dist_vals_db = np.arange(5, -1, -1)

include_clean_run = True
clean_run_flag = include_clean_run
if include_clean_run:
    dist_vals_db = np.insert(dist_vals_db, 0, 0)

print("Distortion IBO/TOI values:", dist_vals_db)
ber_per_dist, freq_arr, clean_ofdm_psd, tx_ofdm_psd = ([] for i in range(4))

sample_constellation = True
constel_snapshot = []
# %%
start_time = time.time()
for dist_val_db in dist_vals_db:
    snapshot_counter = 0
    my_distortion.set_ibo(dist_val_db)
    clean_ofdm_for_psd = []
    tx_ofm_for_psd = []
    bers = np.zeros([len(snr_arr)])

    for idx, snr in enumerate(snr_arr):
        my_chan.set_snr(snr)
        n_err = 0
        bits_sent = 0
        while bits_sent < bits_sent_max and n_err < n_err_min:
            tx_bits = bit_rng.choice((0, 1), n_bits_per_ofdm_sym)
            tx_symb = my_mod.modulate(tx_bits)
            clean_ofdm_symbol = modulation.tx_ofdm_symbol(tx_symb, n_fft, n_sub_carr, cyclic_prefix_len)

            if not clean_run_flag:
                tx_ofdm_symbol = my_distortion.process(clean_ofdm_symbol)
            else:
                tx_ofdm_symbol = clean_ofdm_symbol

            rx_ofdm_symbol = my_chan.propagate(tx_ofdm_symbol, my_mod.avg_symbol_power, n_sub_carr, n_fft)

            if plot_psd and snapshot_counter < n_collected_snapshots:
                clean_ofdm_for_psd.append(clean_ofdm_symbol)
                tx_ofm_for_psd.append(tx_ofdm_symbol)
                snapshot_counter += 1

            rx_symb = modulation.rx_ofdm_symbol(rx_ofdm_symbol, n_fft, n_sub_carr, cyclic_prefix_len)
            rx_bits = my_mod.demodulate(rx_symb)
            n_bit_err = count_mismatched_bits(tx_bits, rx_bits)

            # if sample_constellation:
            #     constel_snapshot.append(tx_symb)
            #     constel_snapshot.append(rx_symb)
            #     sample_constellation = False

            bits_sent += n_bits_per_ofdm_sym
            n_err += n_bit_err
        bers[idx] = n_err / bits_sent
    ber_per_dist.append(bers)
    clean_run_flag = False

    if plot_psd:
        # flatten psd data
        clean_odfm_freq_arr, clean_odfm_psd_arr = welch(np.concatenate(clean_ofdm_for_psd).ravel(), fs=psd_nfft,
                                                        nfft=psd_nfft, nperseg=n_samp_per_seg,
                                                        return_onesided=False)
        tx_odfm_freq_arr, tx_odfm_psd_arr = welch(np.concatenate(tx_ofm_for_psd).ravel(), fs=psd_nfft, nfft=psd_nfft,
                                                  nperseg=n_samp_per_seg,
                                                  return_onesided=False)
        freq_arr.append(np.array(clean_odfm_freq_arr[0:len(clean_odfm_freq_arr) // 2]))
        clean_ofdm_psd.append(to_db(np.array(clean_odfm_psd_arr[0:len(clean_odfm_psd_arr) // 2])))
        tx_ofdm_psd.append(to_db(np.array(tx_odfm_psd_arr[0:len(tx_odfm_psd_arr) // 2])))

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
for idx, dist_val in enumerate(dist_vals_db):
    if idx == 0:
        ax1.plot(freq_arr[0], tx_ofdm_psd[idx], label="No dist")
    else:
        ax1.plot(freq_arr[0], tx_ofdm_psd[idx], label=dist_val)
ax1.set_title("Power spectral density")
ax1.set_xlabel("Subcarrier index [-]")
ax1.set_ylabel("Power [dB]")
ax1.legend(title="IBO [dB]")
ax1.grid()

plt.tight_layout()
#plt.savefig("figs/psd_toi.pdf", dpi=600, bbox_inches='tight')
plt.show()

# %%
fig2, ax2 = plt.subplots(1, 1)
ax2.set_yscale('log')
for idx, dist_val in enumerate(dist_vals_db):
    if idx == 0:
        ax2.plot(ebn0_arr, ber_per_dist[idx], label="No dist")
    else:
        ax2.plot(ebn0_arr, ber_per_dist[idx], label=dist_val)
# fix log scaling
ax2.set_title("Bit error rate, QAM" + str(my_mod.constellation_size))
ax2.set_xlabel("Eb/N0 [dB]")
ax2.set_ylabel("BER")
ax2.grid()
ax2.legend(title="IBO [dB]")

plt.tight_layout()
#plt.savefig("figs/ber_toi.pdf", dpi=600, bbox_inches='tight')
plt.show()

print("Finished exectution!")
