from speedup import jit
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import modulation
import copy
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

    def plot_configuration(self, plot_3d=False):

        if plot_3d:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for transceiver in self.array_elements:
                ax.scatter(transceiver.cord_x, transceiver.cord_y, transceiver.cord_z, color="C0", marker='^')

            ax.set_title('Antenna array')
            ax.set_xlabel("X plane [m]")
            ax.set_ylabel("Y plane [m]")
            ax.set_zlabel("Z plane [m]")
            ax.grid()
            ax.set_axisbelow(True)
            plt.show()
        else:
            fig, ax = plt.subplots()
            for transceiver in self.array_elements:
                ax.scatter(transceiver.cord_x, transceiver.cord_y, color="C0", marker='^')
            ax.set_title('Antenna array')
            ax.set_xlabel("X plane [m]")
            ax.set_ylabel("Y plane [m]")
            ax.grid()
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()

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

    def set_precoding_single_point(self, rx_transceiver, channel_fd_mat=None, exact=False):
        if channel_fd_mat is None:
            for idx, tx_transceiver in enumerate(self.array_elements):
                # get frequency of each subcarrier
                sig_freq_vals = (torch.fft.fftfreq(self.array_elements[idx].modem.n_fft,
                                                   d=1 / self.array_elements[idx].modem.n_fft).numpy() *
                                 self.array_elements[idx].carrier_spacing + self.array_elements[idx].center_freq)

                if exact:
                    # distance to each TX
                    distance_tx = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                        tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                        tx_transceiver.cord_z - rx_transceiver.cord_z, 2))
                    channel_vec_fd = np.exp(2j * np.pi * distance_tx * (sig_freq_vals / scp.constants.c))

                else:
                    # distance to center of array
                    distance_center = np.sqrt(np.power(self.cord_x - rx_transceiver.cord_x, 2) + np.power(
                        self.cord_y - rx_transceiver.cord_y, 2) + np.power(
                        self.cord_z - rx_transceiver.cord_z, 2))
                    # simplified array geometry, precoding based on angle
                    channel_vec_fd = np.exp(2j * np.pi * ((self.n_elements - 1) / 2 - idx) * self.wav_len_spacing
                                            * sig_freq_vals / self.center_freq * (
                                                        (rx_transceiver.cord_x - self.cord_x) / distance_center))

                precoding_vec_fd = np.conjugate(channel_vec_fd)

                # select coefficients on carrier frequencies
                tx_n_sc = tx_transceiver.modem.n_sub_carr
                precoding_vec = np.ones(tx_transceiver.modem.n_fft, dtype=np.complex128)
                precoding_vec[1:(tx_n_sc // 2) + 1] = precoding_vec_fd[1:(tx_n_sc // 2) + 1]
                precoding_vec[-tx_n_sc // 2:] = precoding_vec_fd[-tx_n_sc // 2:]

                tx_transceiver.modem.set_precoding_vec(precoding_vec)
        # set precoding vector based on channel coefficients
        else:
            precoding_mat_fd = np.conjugate(channel_fd_mat)
            for idx, tx_transceiver in enumerate(self.array_elements):
                # select coefficients based on carrier frequencies
                tx_n_sc = tx_transceiver.modem.n_sub_carr
                precoding_mat_row_fd = precoding_mat_fd[idx, :]
                precoding_vec = np.ones(tx_transceiver.modem.n_fft, dtype=np.complex128)
                precoding_vec[1:(tx_n_sc // 2) + 1] = precoding_mat_row_fd[1:(tx_n_sc // 2) + 1]
                precoding_vec[-tx_n_sc // 2:] = precoding_mat_row_fd[-tx_n_sc // 2:]

                tx_transceiver.modem.set_precoding_vec(precoding_vec)
        #
