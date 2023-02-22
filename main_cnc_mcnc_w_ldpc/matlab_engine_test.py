import matlab.engine
import numpy as np

import modulation
from utilities import count_mismatched_bits

matlab = matlab.engine.start_matlab()

matlab.rng(2137)  # Set RNG state for repeatability
bit_rng = np.random.default_rng(2137)
my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=2048, cp_len=128)

# main code tuning variable
code_rate = 1 / 3  # % Target code rate, a real number between 0 and 1
max_ldpc_ite = 12

rv = matlab.double(0)
modulation_format_str = "64QAM"
bits_per_symbol = 6
n_layers = matlab.double(1)

transport_block_size = matlab.int64(my_mod.n_bits_per_ofdm_sym * code_rate)
out_len = matlab.double(np.ceil(transport_block_size / code_rate))
if out_len != my_mod.n_bits_per_ofdm_sym:
    raise ValueError('Code output length does not match modulator input length!')

cbs_info_dict = matlab.nrDLSCHInfo(transport_block_size, code_rate)
print("DL-SCH coding parameters", cbs_info_dict)

# %%
# input bits vector
bits_in = bit_rng.choice((0, 1), transport_block_size)

crc_encoded_bits = matlab.nrCRCEncode(matlab.int8(matlab.transpose(bits_in)), cbs_info_dict['CRC'])
# CRC encoding
code_block_segment_in = matlab.nrCodeBlockSegmentLDPC(crc_encoded_bits, cbs_info_dict['BGN'])
ldpc_encoded_bits = matlab.nrLDPCEncode(code_block_segment_in, cbs_info_dict['BGN'])
channel_in = matlab.nrRateMatchLDPC(ldpc_encoded_bits, out_len, rv, modulation_format_str, n_layers)

symbols_in = matlab.nrSymbolModulate(channel_in, modulation_format_str)
# # AWGN channel
snrdB = 9990.0
rx_sig, noise_var = matlab.awgn(symbols_in, snrdB, nargout=2)
# Symbol demapping
soft_bits_llr = matlab.nrSymbolDemodulate(rx_sig, modulation_format_str, noise_var)
rate_recovered_bits = matlab.nrRateRecoverLDPC(soft_bits_llr, transport_block_size, code_rate, rv,
                                               modulation_format_str, n_layers)
decBits = matlab.nrLDPCDecode(rate_recovered_bits, cbs_info_dict['BGN'], max_ldpc_ite)
blocks_out = matlab.nrCodeBlockDesegmentLDPC(decBits, cbs_info_dict['BGN'], transport_block_size + cbs_info_dict['L'])
crc_decoded_bits = matlab.nrCRCDecode(blocks_out, cbs_info_dict['CRC'])
bits_out = np.array(matlab.transpose(crc_decoded_bits))

n_bit_err = count_mismatched_bits(bits_in, bits_out)
print("N err: ", n_bit_err)

# # %%
# # input bits vector
# # bits_in = bit_rng.choice((0, 1), n_data_bits)
# # matlab_bits_in = matlab.int8(matlab.transpose(bits_in))
# base_graph_number = int(cbs_info_dict['BGN'])
# n_bits_per_codeblock = int(cbs_info_dict['K'])
# n_bits_per_output_block = int(cbs_info_dict['N'])
# code_rate = n_bits_per_codeblock / n_bits_per_output_block
# print("Code rate: ", code_rate)
#
# bits_in = np.ones(n_bits_per_codeblock, dtype=np.int8)
# print("N data bits: ", len(bits_in))
# # fillers = -np.squeeze(np.full((n_fillers, n_codeblocks), -1, dtype=np.int8))
# # in_bits_w_fillers = np.concatenate((bits_in, fillers))
#
# coded_bits = np.array(matlab.nrLDPCEncode(matlab.transpose(matlab.int8(bits_in)), base_graph_number))
# print("N coded bits ", len(coded_bits))
# soft_detected_bits = 1 - 2 * np.array(coded_bits)
# # filler_indices = np.where(coded_bits == -1)
# # make soft decoded fillers to 0
# # soft_detected_bits[filler_indices] = 0
# matlab_bits_out = matlab.nrLDPCDecode(matlab.double(soft_detected_bits), base_graph_number, max_ldpc_ite)
#
# bits_out = np.squeeze(np.array(matlab.transpose(matlab_bits_out)))
# #
# n_bit_err = count_mismatched_bits(bits_in, bits_out)
# print("N err: ", n_bit_err)
