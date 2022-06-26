import numpy as np
import torch

from antenna_arrray import LinearArray
from distortion import SoftLimiter
from modulation import OfdmQamModem


# Nonlinear distortion recovery technique based on Ochiai, Clipping noise cancellation
class CncReceiver():
    def __init__(self, modem: OfdmQamModem, impairment: SoftLimiter):
        self.modem = modem
        self.impairment = impairment
        self.modem.alpha = self.modem.calc_alpha(self.impairment.ibo_db)
        self.upsample_factor = self.modem.n_fft / self.modem.n_sub_carr
        self.impairment.set_avg_sample_power(avg_samp_pow=self.modem.avg_symbol_power * (1 / self.upsample_factor))

    def update_distortion(self, ibo_db):
        self.impairment.set_ibo(ibo_db)
        self.modem.alpha = self.modem.calc_alpha(ibo_db)

    def receive(self, n_iters: int, in_sig_fd, alpha_estimation=None):
        # strip input fd signal of the OOB - include only the symbol data
        n_sub_carr = self.modem.n_sub_carr
        rx_ofdm_nsc_fd = np.concatenate((in_sig_fd[-n_sub_carr // 2:], in_sig_fd[1:(n_sub_carr // 2) + 1]))

        if n_iters == 0:
            if alpha_estimation is not None:
                rx_symbols = self.modem.symbol_detection(rx_ofdm_nsc_fd / alpha_estimation)
            else:
                rx_symbols = self.modem.symbol_detection(rx_ofdm_nsc_fd / self.modem.alpha)
            return self.modem.symbols_to_bits(rx_symbols)

        # allow a fixed number of iterations
        for iter_idx in range(n_iters + 1):
            # skip estimate subtraction for first iteration
            if iter_idx != 0:
                corr_in_sig_fd = rx_ofdm_nsc_fd - distortion_estimate_fd
            else:
                corr_in_sig_fd = rx_ofdm_nsc_fd

            # perform detection with corrected RX constellation - get symbols
            if alpha_estimation is not None:
                rx_symbols = self.modem.symbol_detection(corr_in_sig_fd / alpha_estimation)
            else:
                rx_symbols = self.modem.symbol_detection(corr_in_sig_fd / self.modem.alpha)

            # perform upsampled modulation
            ofdm_sym_fd_upsampled = np.zeros(int(self.modem.n_sub_carr * self.upsample_factor), dtype=np.complex128)

            ofdm_sym_fd_upsampled[-(n_sub_carr // 2):] = rx_symbols[0:n_sub_carr // 2]
            ofdm_sym_fd_upsampled[1:(n_sub_carr // 2) + 1] = rx_symbols[n_sub_carr // 2:]

            # simulate OFDM transmit
            ofdm_sym_td = torch.fft.ifft(torch.from_numpy(ofdm_sym_fd_upsampled), norm="ortho").numpy()
            # perform clipping
            clipped_ofdm_sym_td = self.impairment.process(ofdm_sym_td)

            # simulate OFDM receive
            clipped_ofdm_sym_fd = torch.fft.fft(torch.from_numpy(clipped_ofdm_sym_td), norm="ortho").numpy()

            rx_symbols_estimate = np.concatenate(
                (clipped_ofdm_sym_fd[-n_sub_carr // 2:], clipped_ofdm_sym_fd[1:(n_sub_carr // 2) + 1]))

            # calculate distortion estimate
            if alpha_estimation is not None:
                distortion_estimate_fd = rx_symbols_estimate - (rx_symbols * alpha_estimation)
            else:
                distortion_estimate_fd = rx_symbols_estimate - (rx_symbols * self.modem.alpha)

        return self.modem.symbols_to_bits(rx_symbols)


class McncReceiver():
    def __init__(self, antenna_array: LinearArray, channel):
        self.antenna_array = antenna_array
        self.channel = channel
        channel_mat_at_point = self.channel.get_channel_mat_fd()
        self.antenna_array.set_precoding_matrix(channel_mat_fd=channel_mat_at_point, mr_precoding=True)
        self.agc_corr_vec = np.sqrt(np.sum(np.power(np.abs(channel_mat_at_point), 2), axis=0))

    def update_distortion(self, ibo_db):
        self.antenna_array.update_distortion(ibo_db=ibo_db,
                                             avg_sample_pow=self.antenna_array.base_transceiver.modem.avg_sample_power)

    def receive(self, n_iters: int, in_sig_fd, alpha_estimation=None):
        # strip input fd signal of the OOB - include only the symbol data
        n_sub_carr = self.antenna_array.array_elements[0].modem.n_sub_carr
        rx_ofdm_nsc_fd = np.concatenate((in_sig_fd[-n_sub_carr // 2:], in_sig_fd[1:(n_sub_carr // 2) + 1]))

        if n_iters == 0:
            if alpha_estimation is not None:
                rx_symbols = self.antenna_array.array_elements[0].modem.symbol_detection(
                    rx_ofdm_nsc_fd / alpha_estimation)
            else:
                rx_symbols = self.antenna_array.array_elements[0].modem.symbol_detection(
                    rx_ofdm_nsc_fd / self.antenna_array.array_elements[0].modem.alpha)
            return self.antenna_array.array_elements[0].modem.symbols_to_bits(rx_symbols)

        # allow a fixed number of iterations
        for iter_idx in range(n_iters + 1):
            # skip estimate subtraction for first iteration
            if iter_idx != 0:
                corr_in_sig_fd = rx_ofdm_nsc_fd - distortion_estimate_fd
            else:
                corr_in_sig_fd = rx_ofdm_nsc_fd

            # perform detection with corrected RX constellation - get symbols
            if alpha_estimation is not None:
                rx_symbols = self.antenna_array.array_elements[0].modem.symbol_detection(
                    corr_in_sig_fd / alpha_estimation)
            else:
                rx_symbols = self.antenna_array.array_elements[0].modem.symbol_detection(
                    corr_in_sig_fd / self.antenna_array.array_elements[0].modem.alpha)

            rx_bits = self.antenna_array.array_elements[0].modem.symbols_to_bits(rx_symbols)

            tx_ofdm_symbol = self.antenna_array.transmit(rx_bits, out_domain_fd=True, return_both=False)
            rx_ofdm_symbol = self.channel.propagate(in_sig_mat=tx_ofdm_symbol)
            rx_ofdm_symbol = np.divide(rx_ofdm_symbol, self.agc_corr_vec)

            rx_symbols_estimate = np.concatenate(
                (rx_ofdm_symbol[-n_sub_carr // 2:], rx_ofdm_symbol[1:(n_sub_carr // 2) + 1]))

            # calculate distortion estimate
            if alpha_estimation is not None:
                distortion_estimate_fd = rx_symbols_estimate - (rx_symbols * alpha_estimation)
            else:
                distortion_estimate_fd = rx_symbols_estimate - (
                        rx_symbols * self.antenna_array.array_elements[0].modem.alpha)

        return self.antenna_array.array_elements[0].modem.symbols_to_bits(rx_symbols)
