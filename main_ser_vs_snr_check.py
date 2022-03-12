# SISO OFDM simulation with nonlinearity
# Clipping noise cancellation eval
# %%
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import corrector
import distortion
import modulation
import noise
import transceiver
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, snr_to_ebn0
from datetime import datetime

set_latex_plot_style()

# %%

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=1024, n_sub_carr=256, cp_len=32)
my_distortion = distortion.SoftLimiter(0, my_mod.avg_sample_power)
# my_mod.plot_constellation()
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
# my_tx.impairment.plot_characteristics()

my_standard_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=None)
my_cnc_rx = corrector.CncReceiver(copy.deepcopy(my_mod), copy.deepcopy(my_distortion))

my_noise = noise.Awgn(snr_db=10, seed=1234)
bit_rng = np.random.default_rng(4321)

snr_arr = np.arange(15, 31, 1)
print("SNR values:", snr_arr)
ebn0_arr = snr_to_ebn0(snr_arr, my_mod.n_fft, my_mod.n_sub_carr, my_mod.constel_size)
print("SNR values:", snr_arr)

n_symb_sent_max = int(1e5)
n_symb_err_min = 1000

# %%
# Number of CNC iterationss eval, upsample ratio fixed
ibo_val_db = 0
print("Distortion IBO/TOI value:", ibo_val_db)
cnc_n_iters_lst = [1, 2, 3, 5, 12]
print("CNC number of iteration list:", cnc_n_iters_lst)
cnc_n_upsamp = 2
# Single CNC iteration is equal to standard reception without distortion compensation
cnc_n_iters_lst = np.insert(cnc_n_iters_lst, 0, 0)

include_clean_run = True
if include_clean_run:
    cnc_n_iters_lst = np.insert(cnc_n_iters_lst, 0, 0)

ser_per_ncnc, freq_arr, clean_ofdm_psd, distortion_psd, tx_ofdm_psd = ([] for i in range(5))

start_time = time.time()
for run_idx, cnc_n_iter_val in enumerate(cnc_n_iters_lst):
    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if not (include_clean_run and run_idx == 0):
        my_standard_rx.modem.correct_constellation(ibo_val_db)
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

            tx_ofdm_symbol, clean_ofdm_symbol = my_tx.transmit(tx_bits, out_domain_fd=False, return_both=True)

            if include_clean_run and run_idx == 0:
                rx_ofdm_symbol = my_noise.process(clean_ofdm_symbol, my_mod.avg_sample_power)
            else:
                rx_ofdm_symbol = my_noise.process(tx_ofdm_symbol, my_mod.avg_sample_power)

            if include_clean_run and run_idx == 0:
                # standard reception
                rx_bits = my_standard_rx.receive(rx_ofdm_symbol)
            else:
                # enchanced CNC reception
                # Change domain TD of RX signal to FD
                no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(rx_ofdm_symbol[my_cnc_rx.modem.cp_len:]),
                                                 norm="ortho").numpy()
                rx_bits = my_cnc_rx.receive(n_iters=cnc_n_iter_val, upsample_factor=cnc_n_upsamp,
                                            in_sig_fd=no_cp_fd_sig_mat)
            rx_symbols = np.split(rx_bits, my_mod.n_sub_carr)
            # compare symbols
            for arr_idx in range(len(tx_symbols)):
                if count_mismatched_bits(tx_symbols[arr_idx], rx_symbols[arr_idx]):
                    n_symb_err += 1
            n_symb_sent += len(tx_symbols)

        tmp_ser[idx] = n_symb_err / n_symb_sent
    ser_per_ncnc.append(tmp_ser)
    print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log')
for idx, cnc_iter_val in enumerate(cnc_n_iters_lst):
    if include_clean_run:
        if idx == 0:
            ax1.plot(snr_arr, ser_per_ncnc[idx], label="No distortion")
        elif idx == 1:
            ax1.plot(snr_arr, ser_per_ncnc[idx], label="Standard RX")
        else:
            ax1.plot(snr_arr, ser_per_ncnc[idx], label="CNC NI = %d, J = %d" % (cnc_iter_val, cnc_n_upsamp))
    else:
        if idx == 0:
            ax1.plot(snr_arr, ser_per_ncnc[idx], label="Standard RX")
        else:
            ax1.plot(snr_arr, ser_per_ncnc[idx],
                     label="CNC NI = %d, J = %d" % (cnc_iter_val, cnc_n_upsamp))
# fix log scaling
ax1.set_title("Symbol error rate, QAM %d, IBO = %d [dB]" % (my_mod.constellation_size, ibo_val_db))
ax1.set_xlabel("SNR [dB]")
ax1.set_ylabel("SER")
ax1.grid()
ax1.legend()
ax1.set_xlim([15,30])
plt.tight_layout()
plt.savefig("figs/ser_soft_lim_siso_cnc_ibo%d_niter%d_sweep_nupsamp%d.png" % (
    my_tx.impairment.ibo_db, np.max(cnc_n_iters_lst), cnc_n_upsamp), dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
