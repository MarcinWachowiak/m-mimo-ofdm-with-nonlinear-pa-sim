# SISO OFDM simulation with nonlinearity
# Clipping noise cancellation eval
# %%
import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import antenna_arrray
import channel
import corrector
import distortion
import modulation
import noise
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, ebn0_to_snr

set_latex_plot_style()

# %%

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=2048, cp_len=128)
my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
# my_mod.plot_constellation()
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
my_array = antenna_arrray.LinearArray(n_elements=1, base_transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5,
                                      cord_x=0, cord_y=0, cord_z=15)
my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                         cord_x=212, cord_y=212, cord_z=1.5,
                                         center_freq=int(3.5e9), carrier_spacing=int(15e3))
my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))

# my_miso_chan = channel.MisoTwoPathFd()
my_miso_chan = channel.RayleighMisoFd(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx, seed=1234)
my_noise = noise.Awgn(snr_db=10, seed=1234)
bit_rng = np.random.default_rng(4321)

snr_arr = np.arange(15, 41, 2)
print("SNR values:", snr_arr)
n_symb_sent_max = int(1e6)
n_symb_err_min = 10e3

if not isinstance(my_miso_chan, channel.RayleighMisoFd):
    my_miso_chan.calc_channel_mat(tx_transceivers=my_array.array_elements, rx_transceiver=my_standard_rx,
                                  skip_attenuation=False)

chan_mat_at_point = my_miso_chan.get_channel_mat_fd()
# my_array.set_precoding_matrix(channel_mat_fd=chan_mat_at_point, mr_precoding=True)

agc_corr_vec = np.sqrt(np.sum(np.power(np.abs(chan_mat_at_point), 2), axis=0))
agc_corr_nsc = np.concatenate((agc_corr_vec[-my_mod.n_sub_carr // 2:], agc_corr_vec[1:(my_mod.n_sub_carr // 2) + 1]))

plot_psd = False
n_collected_snapshots = 100
psd_nfft = 128
n_samp_per_seg = 64

# %%
# Number of CNC iterations eval, upsample ratio fixed
ibo_val_db = 0
my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=my_mod.avg_sample_power)
my_cnc_rx.impairment.set_ibo(ibo_val_db)

print("Distortion IBO/TOI value:", ibo_val_db)
cnc_n_iters_lst = [1, 2, 3, 5, 12]
print("CNC number of iteration list:", cnc_n_iters_lst)
cnc_n_upsamp = 2
# Single CNC iteration is equal to standard reception without distortion compensation
cnc_n_iters_lst = np.insert(cnc_n_iters_lst, 0, 0)

include_clean_run = True
if include_clean_run:
    cnc_n_iters_lst = np.insert(cnc_n_iters_lst, 0, 0)

ser_qam_per_ncnc, freq_arr, clean_ofdm_psd, distortion_psd, tx_ofdm_psd = ([] for i in range(5))

estimate_eta = True
if estimate_eta:
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    my_tx.impairment.set_ibo(ibo_val_db)
    # estimate lambda correcting coefficient
    # same seed is required
    bit_rng = np.random.default_rng(4321)
    n_ofdm_symb = 1e3
    ofdm_symb_idx = 0
    nsc_sig_after_dist_fd = []
    lambda_denominator_vecs = []
    while ofdm_symb_idx < n_ofdm_symb:
        tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
        tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_tx.transmit(tx_bits, out_domain_fd=True, return_both=True)

        rx_nsc_ofdm_symb_fd = np.concatenate(
            (tx_ofdm_symbol_fd[-my_mod.n_sub_carr // 2:], tx_ofdm_symbol_fd[1:(my_mod.n_sub_carr // 2) + 1]))
        nsc_sig_after_dist_fd.append(rx_nsc_ofdm_symb_fd)

        ofdm_symb_idx += 1

    eta_pwr_ratio = (np.sum(np.power(np.abs(np.hstack(nsc_sig_after_dist_fd)), 2)) / n_ofdm_symb) / (
            my_mod.n_sub_carr * my_mod.avg_symbol_power)

    print("--- Computation time: %f ---" % (time.time() - start_time))
else:
    eta_pwr_ratio = my_mod.calc_alpha(ibo_db=ibo_val_db) ** 2

print("Eta ratio: ", eta_pwr_ratio)

for run_idx, cnc_n_iter_val in enumerate(cnc_n_iters_lst):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if not (include_clean_run and run_idx == 0):
        my_tx.impairment.set_ibo(ibo_val_db)
        my_cnc_rx.impairment.set_ibo(ibo_val_db)

    tmp_ser = np.zeros([len(snr_arr)])
    for idx, snr in enumerate(snr_arr):
        my_noise.snr_db = snr
        n_symb_err = 0
        n_symb_sent = 0
        while n_symb_sent < n_symb_sent_max and n_symb_err < n_symb_err_min:
            tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
            tx_symbols = np.split(tx_bits, my_mod.n_sub_carr)

            tx_ofdm_symbol_fd, clean_ofdm_symbol_fd = my_array.transmit(tx_bits, out_domain_fd=True, return_both=True)

            if include_clean_run and run_idx == 0:
                rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol_fd)
                rx_ofdm_symbol = my_noise.process(rx_ofdm_symbol, avg_sample_pow=my_mod.avg_sample_power * np.average(
                    np.power(chan_mat_at_point, 2)), disp_data=False)
            else:
                rx_ofdm_symbol = my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol_fd)
                rx_ofdm_symbol = my_noise.process(rx_ofdm_symbol, avg_sample_pow=my_mod.avg_sample_power *
                                                                                 np.average(np.power(chan_mat_at_point,
                                                                                                     2)) * eta_pwr_ratio)
            # apply AGC
            rx_ofdm_symbol = np.squeeze(np.divide(rx_ofdm_symbol, chan_mat_at_point))

            if include_clean_run and run_idx == 0:
                # standard reception - no distortion
                rx_bits = my_cnc_rx.receive(n_iters=0, upsample_factor=1,
                                            in_sig_fd=rx_ofdm_symbol, lambda_estimation=1.0)
            else:
                # enchanced CNC reception
                rx_bits = my_cnc_rx.receive(n_iters=cnc_n_iter_val, upsample_factor=cnc_n_upsamp,
                                            in_sig_fd=rx_ofdm_symbol)
            rx_symbols = np.split(rx_bits, my_mod.n_sub_carr)
            # compare symbols
            for arr_idx in range(len(tx_symbols)):
                if count_mismatched_bits(tx_symbols[arr_idx], rx_symbols[arr_idx]):
                    n_symb_err += 1
            n_symb_sent += len(tx_symbols)

        tmp_ser[idx] = n_symb_err / n_symb_sent
    ser_qam_per_ncnc.append(tmp_ser)
    print("--- Computation time: %f ---" % (time.time() - start_time))

    # %%
tmp = np.array(ser_qam_per_ncnc)
ser_pam_per_ncn = 1 - np.sqrt(1 - np.array(ser_qam_per_ncnc))
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
ax1.set_yscale('log')
for idx, cnc_iter_val in enumerate(cnc_n_iters_lst):
    if include_clean_run:
        if idx == 0:
            ax1.plot(snr_arr, ser_pam_per_ncn[idx], '-o', label="No distortion")
        elif idx == 1:
            ax1.plot(snr_arr, ser_pam_per_ncn[idx], '-o', label="Standard RX")
        else:
            ax1.plot(snr_arr, ser_pam_per_ncn[idx], '-o', label="CNC NI = %d, J = %d" % (cnc_iter_val, cnc_n_upsamp))
    else:
        if idx == 0:
            ax1.plot(snr_arr, ser_pam_per_ncn[idx], '-o', label="Standard RX")
        else:
            ax1.plot(snr_arr, ser_pam_per_ncn[idx],
                     label="CNC NI = %d, J = %d" % (cnc_iter_val, cnc_n_upsamp))

ax1.plot(snr_arr + 0.17, ser_pam_per_ncn[0], 'k--', label="FD Lower Bound")
# fix log scaling
ax1.set_title("Symbol error rate, PAM (QAM) %d, NSC = %d, IBO = %d [dB]" % (
    np.sqrt(my_mod.constellation_size), my_mod.n_sub_carr, ibo_val_db))
ax1.set_xlabel("SNR [dB]")
ax1.set_ylabel("SER")
ax1.grid()
ax1.legend()
ax1.set_xlim([15, 40])
plt.tight_layout()
plt.savefig("../figs/ser_soft_lim_rayleigh_cnc_ibo%d_niter%d_sweep_nupsamp%d_nsc%d.png" % (
    my_tx.impairment.ibo_db, np.max(cnc_n_iters_lst), cnc_n_upsamp, my_mod.n_sub_carr), dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
