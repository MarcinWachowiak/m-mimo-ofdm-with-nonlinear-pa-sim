import numpy as np
import torch

from distortion import SoftLimiter
from modulation import OfdmQamModem


# Nonlinear distortion recovery technique based on Ochiai, Clipping noise cancellation
class CncReceiver():
    def __init__(self, modem: OfdmQamModem, impairment: SoftLimiter):
        self.modem = modem
        self.impairment = impairment
        self.modem.alpha = self.modem.calc_alpha(self.impairment.ibo_db)

    def update_distortion(self, ibo_db):
        self.impairment.set_ibo(ibo_db)
        self.modem.alpha = self.modem.calc_alpha(ibo_db)

    def receive(self, n_iters: int, upsample_factor: int, in_sig_fd, lambda_estimation=None):
        # strip input fd signal of the OOB - include only the symbol data
        n_sub_carr = self.modem.n_sub_carr
        rx_ofdm_nsc_fd = np.concatenate((in_sig_fd[-n_sub_carr // 2:], in_sig_fd[1:(n_sub_carr // 2) + 1]))

        # allow a fixed number of iterations
        for iter_idx in range(n_iters + 1):
            # skip estimate subtraction for first iteration
            if iter_idx != 0:
                corr_in_sig_fd = rx_ofdm_nsc_fd - distortion_estimate_fd
            else:
                corr_in_sig_fd = rx_ofdm_nsc_fd

            # perform detection with corrected RX constellation - get symbols
            if lambda_estimation is not None:
                rx_symbols = self.modem.symbol_detection(corr_in_sig_fd / lambda_estimation)
            else:
                rx_symbols = self.modem.symbol_detection(corr_in_sig_fd / self.modem.alpha)

            # perform upsampled modulation
            ofdm_sym_fd_upsampled = np.zeros(self.modem.n_fft * upsample_factor, dtype=np.complex128)
            # keep subcarrier mapping?
            ofdm_sym_fd_upsampled[-(n_sub_carr // 2):] = rx_symbols[0:n_sub_carr // 2]
            ofdm_sym_fd_upsampled[1:(n_sub_carr // 2) + 1] = rx_symbols[n_sub_carr // 2:]

            # simulate OFDM transmit
            ofdm_sym_td = np.sqrt(upsample_factor) * torch.fft.ifft(torch.from_numpy(ofdm_sym_fd_upsampled),
                                                                    norm="ortho").numpy()
            # perform clipping
            clipped_ofdm_sym_td = self.impairment.process(ofdm_sym_td)

            # simulate OFDM receive
            clipped_ofdm_sym_fd = torch.fft.fft(torch.from_numpy(clipped_ofdm_sym_td), norm="ortho").numpy() / np.sqrt(
                upsample_factor)

            rx_symbols_estimate = np.concatenate(
                (clipped_ofdm_sym_fd[-n_sub_carr // 2:], clipped_ofdm_sym_fd[1:(n_sub_carr // 2) + 1]))

            # calculate distortion estimate
            if lambda_estimation is not None:
                distortion_estimate_fd = rx_symbols_estimate - (rx_symbols * lambda_estimation)
            else:
                distortion_estimate_fd = rx_symbols_estimate - (rx_symbols * self.modem.alpha)

        return self.modem.symbols_to_bits(rx_symbols)
