# SISO OFDM simulation with nonlinearity
# Desired vs distorted signal PSD and BER comparison
# %%
import os
import sys

import utilities

sys.path.append(os.getcwd())

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import matlab.engine

import distortion
import modulation
import noise
import transceiver
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, ebn0_to_snr, to_db

set_latex_plot_style(use_tex=False, fig_width_in=5)

matlab = matlab.engine.start_matlab()
matlab.rng(2137)
# %%
# main code tuning variable
target_code_rate = 1 / 3  # % Target code rate, a real number between 0 and 1
max_ldpc_ite = 12
cbs_info_dict = matlab.nrDLSCHInfo(4096, target_code_rate)

base_graph_number = int(cbs_info_dict['BGN'])
n_info_bits_per_block = int(cbs_info_dict['K'])
total_bits_per_block = int(cbs_info_dict['N'])
code_rate = n_info_bits_per_block / total_bits_per_block

n_sub_carr = int(total_bits_per_block / 6)
print("Resultant code rate: %f,  N_SC: %d" %(code_rate, n_sub_carr))

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=n_sub_carr, cp_len=8)
my_distortion = distortion.SoftLimiter(5, my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))

my_noise = noise.Awgn(0, True, 1234)
bit_rng = np.random.default_rng(4321)

ebn0_arr = np.arange(-5, 5, 1)
print("Eb/n0 values:", ebn0_arr)
snr_arr = ebn0_to_snr(ebn0_arr, my_mod.n_sub_carr, my_mod.n_sub_carr, my_mod.constel_size)

bits_sent_max = int(1e7)
n_err_min = int(1e6)


#%%
bers_lst = []
start_time = time.time()

# # CLEAN RUN - NO DIST, AWGN ONLY
ber_arr = np.zeros([len(snr_arr)])
for idx, snr in enumerate(snr_arr):
    my_noise.snr_db = snr
    n_err = 0
    bits_sent = 0
    ite_cnt = 0
    while bits_sent < bits_sent_max and n_err < n_err_min:
        tx_bits = bit_rng.choice((0, 1), my_tx.modem.n_bits_per_ofdm_sym)
        tx_ofdm_symbol, clean_ofdm_symbol = my_tx.transmit(tx_bits, out_domain_fd=False, return_both=True)
        rx_ofdm_symbol = my_noise.process(clean_ofdm_symbol, my_mod.avg_sample_power)
        rx_bits = my_rx.receive(rx_ofdm_symbol)
        n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
        n_err += n_bit_err
        bits_sent += my_mod.n_bits_per_ofdm_sym
    if bits_sent != 0:
        ber_arr[idx] = n_err / bits_sent
    else:
        ber_arr[idx] = np.nan
bers_lst.append(ber_arr)

# # CLEAN RUN + LDPC - NO DIST, AWGN ONLY

ber_arr = np.zeros([len(snr_arr)])
for idx, snr in enumerate(snr_arr):
    my_noise.snr_db = snr
    noise_var = np.complex128(2 * my_mod.avg_symbol_power / (10 ** (my_noise.snr_db / 10)))

    n_err = 0
    bits_sent = 0
    ite_cnt = 0

    while bits_sent < bits_sent_max and n_err < n_err_min:
        matlab.rng(2137)
        bit_rng = np.random.default_rng(2137)

        tx_bits = bit_rng.choice((0, 1), n_info_bits_per_block)
        ldpc_encoded_bits = np.squeeze(np.array(matlab.nrLDPCEncode(matlab.int8(matlab.transpose(tx_bits)), base_graph_number)))

        tx_ofdm_symbol, clean_ofdm_symbol = my_tx.transmit(ldpc_encoded_bits, out_domain_fd=False, return_both=True)
        rx_ofdm_symbol = my_noise.process(clean_ofdm_symbol, my_mod.avg_symbol_power)

        rx_symbols = my_mod.demodulate(clean_ofdm_symbol, get_symbols_only=True)
        rx_llr_soft_bits = -my_mod.soft_detection_llr(rx_symbols, noise_var=noise_var)

        # checking soft-detection
        # soft_bits_tmp = np.copy(rx_llr_soft_bits)
        # soft_bits_tmp[soft_bits_tmp == -np.inf] = 0
        # soft_bits_tmp[soft_bits_tmp == np.inf] = 1
        # tmp = count_mismatched_bits(ldpc_encoded_bits, np.asarray(soft_bits_tmp, dtype=np.int8))

        rx_bits = np.squeeze(np.array(matlab.transpose(matlab.nrLDPCDecode(matlab.transpose(matlab.double(rx_llr_soft_bits)), base_graph_number, max_ldpc_ite))))

        n_bit_err = count_mismatched_bits(tx_bits, rx_bits)
        n_err += n_bit_err
        bits_sent += n_info_bits_per_block
    if bits_sent != 0:
        ber_arr[idx] = n_err / bits_sent
    else:
        ber_arr[idx] = np.nan

    utilities.print_progress_bar(idx + 1, len(snr_arr), prefix='LDPC SNR loop progress:')

bers_lst.append(ber_arr)

print("--- Computation time: %f ---" % (time.time() - start_time))

# %%
fig1, ax1 = plt.subplots(1, 1)
ax1.set_yscale('log')

ax1.plot(ebn0_arr, bers_lst[0], label="No dist")
ax1.plot(ebn0_arr, bers_lst[1], label="No dist + LDPC 1/3")

# fix log scaling
ax1.set_title("Bit error rate, QAM" + str(my_mod.constellation_size))
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("BER")
ax1.grid()
ax1.legend(title="IBO [dB]")

plt.tight_layout()
plt.savefig("../figs/tmp.png", dpi=600, bbox_inches='tight')
plt.show()

print("Finished execution!")
# %%
