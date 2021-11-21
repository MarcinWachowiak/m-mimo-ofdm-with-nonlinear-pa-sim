from speedup import jit
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import modulation
import copy
import torch


class LinearArray:
    def __init__(self, n_elements, transceiver, center_freq, wav_len_spacing):
        self.n_elements = n_elements
        self.transceiver = transceiver
        self.center_freq = center_freq
        self.wav_len_spacing = wav_len_spacing
        self.array_elements = []
        self.array_center_x = 0
        self.array_center_y = 0
        self.array_center_z = 0

        if isinstance(self.transceiver, list) and len(self.transceiver) == self.n_elements and len(
                self.transceiver) == self.n_elements:
            # extend for custom list of transceivers
            for idx, base_modem in enumerate(self.transceiver):
                pass
        else:
            # antenna position vector centered around 0
            ant_vec = np.linspace(-(self.n_elements - 1) * self.wav_len_spacing / 2,
                                  (self.n_elements - 1) * self.wav_len_spacing / 2, self.n_elements)
            for idx in range(self.n_elements):
                tmp_transceiver = copy.deepcopy(self.transceiver)
                tmp_transceiver.cord_x = ant_vec[idx]
                tmp_transceiver.cord_y = 0
                tmp_transceiver.cord_z = 0
                self.array_elements.append(tmp_transceiver)

    def plot_configuration(self, plot_3d=False):
        fig, ax = plt.subplots()
        if plot_3d:
            ax = fig.add_subplot(projection='3d')
            for transceiver in self.array_elements:
                ax.scatter(transceiver.cord_x, transceiver.cord_y, transceiver.cord_z, color="C0", marker='^')

            ax.set_title('Antenna array')
            ax.set_xlabel("X plane [$\lambda$]")
            ax.set_ylabel("Y plane [$\lambda$]")
            ax.set_zlabel("Z plane [$\lambda$]")
            ax.grid()
            ax.set_axisbelow(True)
            plt.show()
        else:
            for transceiver in self.array_elements:
                ax.scatter(transceiver.cord_x, transceiver.cord_y, color="C0", marker='^')
            ax.set_title('Antenna array')
            ax.set_xlabel("X plane [$\lambda$]")
            ax.set_ylabel("Y plane [$\lambda$]")
            ax.grid()
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()

    def transmit(self, in_bits):
        out_sig_mat = np.empty([self.n_elements, len(in_bits) + self.transceiver.modem.cp_len], dtype=np.complex128)
        for idx, transceiver in enumerate(self.array_elements):
            out_sig_mat[idx, :] = transceiver.transmit(in_bits)
        return out_sig_mat

    def set_precoding_single_point(self, rx_transceiver, exact=False):

        for idx, tx_transceiver in enumerate(self.array_elements):
            # Leave subcarrier dependent precoding for more advanced filter model - for now all freqs delayed the same
            # ifft_freqs = torch.fft.fftfreq(tx_transceiver.modem.n_fft)
            # sub_carr_freqs = np.concatenate((ifft_freqs[-tx_transceiver.modem.n_sub_carr // 2:],
            #                                  ifft_freqs[1:(tx_transceiver.modem.n_sub_carr // 2) + 1]))
            # what about speed of light?
            c_constant = 1

            if exact:
                # distance to each TX
                distance_tx = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                    tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                    tx_transceiver.cord_z - rx_transceiver.cord_z, 2))
                precoding_vec = np.exp(2j * np.pi * distance_tx)
            else:
                # distance to center of array
                distance_center = np.sqrt(np.power(self.array_center_x - rx_transceiver.cord_x, 2) + np.power(
                    self.array_center_y - rx_transceiver.cord_y, 2) + np.power(
                    self.array_center_z - rx_transceiver.cord_z, 2))
                # simplified array geometry, precoding based on angle
                precoding_vec = np.exp(2j * np.pi * ((self.n_elements - 1) / 2 - idx) * self.wav_len_spacing
                                       * ((rx_transceiver.cord_x - self.array_center_x) / distance_center))
            tx_transceiver.modem.set_precoding_vec(precoding_vec)
