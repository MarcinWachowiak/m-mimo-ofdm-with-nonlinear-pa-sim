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

    def receive(self, n_iters_lst, in_sig_fd, alpha_estimation=None):
        # strip input fd signal of the OOB - include only the symbol data
        n_sub_carr = self.modem.n_sub_carr
        rx_ofdm_nsc_fd = np.concatenate((in_sig_fd[-n_sub_carr // 2:], in_sig_fd[1:(n_sub_carr // 2) + 1]))

        bits_per_iter_lst = []

        # allow a fixed number of iterations
        for iter_idx in range(np.max(n_iters_lst) + 1):
            # skip estimate subtraction for first iteration
            if iter_idx != 0:
                corr_in_sig_fd = rx_ofdm_nsc_fd - distortion_estimate_fd
            else:
                corr_in_sig_fd = rx_ofdm_nsc_fd

            rx_symbols = self.modem.symbol_detection(corr_in_sig_fd)

            if iter_idx in n_iters_lst:
                bits_per_iter_lst.append(self.modem.symbols_to_bits(rx_symbols))

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

            if alpha_estimation is not None:
                rx_symbols_estimate = np.divide(rx_symbols_estimate, alpha_estimation)
            else:
                rx_symbols_estimate = np.divide(rx_symbols_estimate, self.modem.alpha)

            # calculate distortion estimate
            distortion_estimate_fd = rx_symbols_estimate - rx_symbols

        return bits_per_iter_lst


class McncReceiver():
    def __init__(self, antenna_array: LinearArray, channel):
        self.antenna_array = antenna_array
        self.channel = channel
        channel_mat_at_point = self.channel.get_channel_mat_fd()
        self.antenna_array.set_precoding_matrix(channel_mat_fd=channel_mat_at_point, mr_precoding=True)
        self.n_sub_carr = self.antenna_array.array_elements[0].modem.n_sub_carr

        chan_mat_at_point = self.channel.get_channel_mat_fd()
        hk_mat = np.concatenate((chan_mat_at_point[:, -self.antenna_array.array_elements[0].modem.n_sub_carr // 2:],
                                 chan_mat_at_point[:,
                                 1:(self.antenna_array.array_elements[0].modem.n_sub_carr // 2) + 1]), axis=1)
        vk_mat = self.antenna_array.get_precoding_mat()
        vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)
        hk_vk_agc = np.multiply(hk_mat, vk_mat)

        ibo_vec = 10 * np.log10(10 ** (self.antenna_array.array_elements[0].impairment.ibo_db / 10) * self.n_sub_carr / (vk_pow_vec * len(self.antenna_array.array_elements)))
        ak_vect = self.antenna_array.array_elements[0].modem.calc_alpha(ibo_db=ibo_vec)
        ak_vect = np.expand_dims(ak_vect, axis=1)

        ak_hk_vk_agc = ak_vect * hk_vk_agc
        ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)

        ak_hk_vk_agc_nfft = np.ones(self.antenna_array.array_elements[0].modem.n_fft, dtype=np.complex128)
        ak_hk_vk_agc_nfft[-(self.n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:self.n_sub_carr // 2]
        ak_hk_vk_agc_nfft[1:(self.n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[self.n_sub_carr // 2:]

        self.agc_corr_vec = ak_hk_vk_agc_nfft

    def receive(self, n_iters_lst, in_sig_fd, alpha_estimation=None):
        # strip input fd signal of the OOB - include only the symbol data
        rx_ofdm_nsc_fd = np.concatenate((in_sig_fd[-self.n_sub_carr // 2:], in_sig_fd[1:(self.n_sub_carr // 2) + 1]))

        bits_per_iter_lst = []

        # allow a fixed number of iterations
        for iter_idx in range(np.max(n_iters_lst) + 1):
            # skip estimate subtraction for first iteration
            if iter_idx != 0:
                corr_in_sig_fd = rx_ofdm_nsc_fd - distortion_estimate_fd
            else:
                corr_in_sig_fd = rx_ofdm_nsc_fd

            # perform detection - get symbols
            rx_symbols = self.antenna_array.array_elements[0].modem.symbol_detection(corr_in_sig_fd)

            rx_bits = self.antenna_array.array_elements[0].modem.symbols_to_bits(rx_symbols)

            if iter_idx in n_iters_lst:
                bits_per_iter_lst.append(rx_bits)

            tx_ofdm_symbol = self.antenna_array.transmit(rx_bits, out_domain_fd=True, return_both=False)
            rx_ofdm_symbol = self.channel.propagate(in_sig_mat=tx_ofdm_symbol)
            rx_ofdm_symbol = np.divide(rx_ofdm_symbol, self.agc_corr_vec)

            rx_symbols_estimate = np.concatenate(
                (rx_ofdm_symbol[-self.n_sub_carr // 2:], rx_ofdm_symbol[1:(self.n_sub_carr // 2) + 1]))

            # calculate distortion estimate
            if alpha_estimation is not None:
                distortion_estimate_fd = rx_symbols_estimate - rx_symbols
            else:
                distortion_estimate_fd = rx_symbols_estimate - rx_symbols

        return bits_per_iter_lst

    def update_agc(self):
        # channel_mat_at_point = self.channel.get_channel_mat_fd()
        # self.antenna_array.set_precoding_matrix(channel_mat_fd=channel_mat_at_point, mr_precoding=True)
        # self.n_sub_carr = self.antenna_array.array_elements[0].modem.n_sub_carr

        chan_mat_at_point = self.channel.get_channel_mat_fd()
        hk_mat = np.concatenate((chan_mat_at_point[:, -self.antenna_array.array_elements[0].modem.n_sub_carr // 2:],
                                 chan_mat_at_point[:,
                                 1:(self.antenna_array.array_elements[0].modem.n_sub_carr // 2) + 1]), axis=1)
        vk_mat = self.antenna_array.get_precoding_mat()
        vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)
        hk_vk_agc = np.multiply(hk_mat, vk_mat)

        ibo_vec = 10 * np.log10(
            10 ** (self.antenna_array.array_elements[0].impairment.ibo_db / 10) * self.n_sub_carr / (
                    vk_pow_vec * len(self.antenna_array.array_elements)))
        ak_vect = self.antenna_array.array_elements[0].modem.calc_alpha(ibo_db=ibo_vec)
        ak_vect = np.expand_dims(ak_vect, axis=1)

        ak_hk_vk_agc = ak_vect * hk_vk_agc
        ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)

        ak_hk_vk_agc_nfft = np.ones(self.antenna_array.array_elements[0].modem.n_fft, dtype=np.complex128)
        ak_hk_vk_agc_nfft[-(self.n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:self.n_sub_carr // 2]
        ak_hk_vk_agc_nfft[1:(self.n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[self.n_sub_carr // 2:]

        self.agc_corr_vec = ak_hk_vk_agc_nfft
