import copy

import numpy as np
import scipy as scp
import torch


class LinearArray:
    def __init__(self, n_elements, transceiver, center_freq, wav_len_spacing, cord_x=0, cord_y=0, cord_z=0):
        self.n_elements = n_elements
        self.transceiver = transceiver
        self.center_freq = center_freq
        self.wav_len_spacing = wav_len_spacing
        self.array_elements = []
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z

        if isinstance(self.transceiver, list) and len(self.transceiver) == self.n_elements and len(
                self.transceiver) == self.n_elements:
            # extend for custom list of transceivers
            for idx, base_modem in enumerate(self.transceiver):
                pass
        else:
            # antenna position vector centered around 0
            wavelength_at_freq = scp.constants.c / self.center_freq
            ant_vec = np.linspace(-(self.n_elements - 1) * self.wav_len_spacing * wavelength_at_freq / 2,
                                  (self.n_elements - 1) * self.wav_len_spacing * wavelength_at_freq / 2,
                                  self.n_elements)
            for idx in range(self.n_elements):
                tmp_transceiver = copy.deepcopy(self.transceiver)
                tmp_transceiver.cord_x = ant_vec[idx]
                tmp_transceiver.cord_y = 0
                tmp_transceiver.cord_z = self.cord_z
                self.array_elements.append(tmp_transceiver)

    def transmit(self, in_bits, return_both=False):
        out_sig_mat = np.empty([self.n_elements, self.transceiver.modem.n_fft + self.transceiver.modem.cp_len],
                               dtype=np.complex128)

        if return_both:
            clean_sig_mat = np.empty([self.n_elements, self.transceiver.modem.n_fft + self.transceiver.modem.cp_len],
                                     dtype=np.complex128)
            for idx, transceiver in enumerate(self.array_elements):
                out_sig_mat[idx, :], clean_sig_mat[idx, :] = transceiver.transmit(in_bits, return_both=True)

            return out_sig_mat, clean_sig_mat
        else:
            for idx, transceiver in enumerate(self.array_elements):
                out_sig_mat[idx, :] = transceiver.transmit(in_bits, return_both=return_both)
            return out_sig_mat

    def set_precoding_single_point(self, rx_transceiver, channel_fd_mat=None, exact=False, two_path=False):
        precoding_mat_fd = np.empty([self.n_elements, rx_transceiver.modem.n_fft], dtype=np.complex128)

        if channel_fd_mat is None:
            for idx, tx_transceiver in enumerate(self.array_elements):
                # get frequency of each subcarrier
                sig_freq_vals = (torch.fft.fftfreq(self.array_elements[idx].modem.n_fft,
                                                   d=1 / self.array_elements[idx].modem.n_fft).numpy() *
                                 self.array_elements[idx].carrier_spacing + self.array_elements[idx].center_freq)

                los_distance = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                    tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                    tx_transceiver.cord_z - rx_transceiver.cord_z, 2))

                los_fd_shift_mat = np.exp(2j * np.pi * los_distance * (sig_freq_vals / scp.constants.c))

                if two_path:
                    dim_ratio = (tx_transceiver.cord_z + rx_transceiver.cord_z) / (
                        np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                            tx_transceiver.cord_y - rx_transceiver.cord_y, 2)))
                    # angle of elevation (angle in relation to the ground plane) = 90 deg - angle of incidence
                    angle_of_elev_rad = np.arctan(dim_ratio)

                    second_path_len = rx_transceiver.cord_z / np.sin(angle_of_elev_rad)
                    first_path_len = tx_transceiver.cord_z / np.sin(angle_of_elev_rad)

                    reflection_coeff = -1.0
                    sec_fd_shift_mat = reflection_coeff * np.exp(2j * np.pi * (first_path_len + second_path_len) *
                                              (sig_freq_vals / scp.constants.c))
                    two_path_fd_chan = np.add(los_fd_shift_mat, sec_fd_shift_mat)
                    # normalize to exclude amplification/attenuation
                    two_path_fd_chan_normalized = np.exp(1j * np.angle(two_path_fd_chan))

                    precoding_vec_fd = np.conjugate(two_path_fd_chan_normalized)
                else:
                    if exact:
                        precoding_vec_fd = np.conjugate(los_fd_shift_mat)
                    else:
                        # distance to center of array
                        distance_center = np.sqrt(np.power(self.cord_x - rx_transceiver.cord_x, 2) + np.power(
                            self.cord_y - rx_transceiver.cord_y, 2) + np.power(
                            self.cord_z - rx_transceiver.cord_z, 2))
                        # simplified array geometry, precoding based on angle
                        simplified_los_vec_fd = np.exp(
                            2j * np.pi * ((self.n_elements - 1) / 2 - idx) * self.wav_len_spacing
                            * sig_freq_vals / self.center_freq * (
                                    (rx_transceiver.cord_x - self.cord_x) / distance_center))
                        precoding_vec_fd = np.conjugate(simplified_los_vec_fd)

                # fill the precoding matrix
                precoding_mat_fd[idx, :] = precoding_vec_fd
        else:
            # set precoding vector based on provided channel mat coefficients
            channel_fd_mat_conjungate = np.conjugate(channel_fd_mat)
            # normalize rayleigh channel precoding coefficients
            precoding_mat_fd = np.exp(1j * np.angle(channel_fd_mat_conjungate))

        # apply precoding vectors to each tx node from matrix
        for idx, tx_transceiver in enumerate(self.array_elements):
            # select coefficients based on carrier frequencies
            tx_n_sc = tx_transceiver.modem.n_sub_carr
            precoding_mat_row_fd = precoding_mat_fd[idx, :]
            precoding_vec = np.ones(tx_transceiver.modem.n_fft, dtype=np.complex128)
            precoding_vec[1:(tx_n_sc // 2) + 1] = precoding_mat_row_fd[1:(tx_n_sc // 2) + 1]
            precoding_vec[-tx_n_sc // 2:] = precoding_mat_row_fd[-tx_n_sc // 2:]

            tx_transceiver.modem.set_precoding_vec(precoding_vec)
