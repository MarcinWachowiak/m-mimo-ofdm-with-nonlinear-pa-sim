import copy

import numpy as np
import scipy as scp


class LinearArray:
    def __init__(self, n_elements, base_transceiver, center_freq, wav_len_spacing, cord_x=0, cord_y=0, cord_z=0):
        self.n_elements = n_elements
        self.base_transceiver = base_transceiver
        self.center_freq = center_freq
        self.wav_len_spacing = wav_len_spacing
        self.array_elements = []
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z

        if isinstance(self.base_transceiver, list) and len(self.base_transceiver) == self.n_elements and len(
                self.base_transceiver) == self.n_elements:
            # extend for custom list of transceivers
            for idx, base_modem in enumerate(self.base_transceiver):
                pass
        else:
            # antenna position vector centered around 0
            wavelength_at_freq = scp.constants.c / self.center_freq
            ant_vec = np.linspace(-(self.n_elements - 1) * self.wav_len_spacing * wavelength_at_freq / 2,
                                  (self.n_elements - 1) * self.wav_len_spacing * wavelength_at_freq / 2,
                                  self.n_elements)
            for idx in range(self.n_elements):
                tmp_transceiver = copy.deepcopy(self.base_transceiver)
                tmp_transceiver.cord_x = ant_vec[idx]
                tmp_transceiver.cord_y = 0
                tmp_transceiver.cord_z = self.cord_z
                self.array_elements.append(tmp_transceiver)

    def set_tx_power_lvls(self, tx_power_dbm, total=False):
        for tx in self.array_elements:
            if total:
                tx.set_tx_power_dbm(10 * (np.log10(10 ** (tx_power_dbm / 10) / len(self.array_elements))))
            else:
                tx.set_tx_power_dbm(tx_power_dbm)

    def transmit(self, in_bits, out_domain_fd=True, return_both=False):
        if out_domain_fd:
            out_sig_mat = np.empty([self.n_elements, self.base_transceiver.modem.n_fft],
                                   dtype=np.complex128)
        else:
            out_sig_mat = np.empty(
                [self.n_elements, self.base_transceiver.modem.n_fft + self.base_transceiver.modem.cp_len],
                dtype=np.complex128)
        if return_both:
            if out_domain_fd:
                clean_sig_mat = np.empty([self.n_elements, self.base_transceiver.modem.n_fft], dtype=np.complex128)
            else:
                clean_sig_mat = np.empty(
                    [self.n_elements, self.base_transceiver.modem.n_fft + self.base_transceiver.modem.cp_len],
                    dtype=np.complex128)

            for idx, tx_transceiver in enumerate(self.array_elements):
                out_sig_mat[idx, :], clean_sig_mat[idx, :] = tx_transceiver.transmit(in_bits,
                                                                                     out_domain_fd=out_domain_fd,
                                                                                     return_both=True)

            return out_sig_mat, clean_sig_mat
        else:
            for idx, tx_transceiver in enumerate(self.array_elements):
                out_sig_mat[idx, :] = tx_transceiver.transmit(in_bits, out_domain_fd=out_domain_fd,
                                                              return_both=return_both)
            return out_sig_mat

    def set_precoding_matrix(self, channel_mat_fd=None, mr_precoding=False):
        # set precoding vector based on provided channel mat coefficients
        # only the subcarriers are precoded, other normalization operations should be performed in regard to carrier pool
        tx_n_sc = self.base_transceiver.modem.n_sub_carr

        if not isinstance(channel_mat_fd, list):
            sc_channel_mat_fd = np.concatenate(
                (channel_mat_fd[:, 1:(tx_n_sc // 2) + 1], channel_mat_fd[:, -tx_n_sc // 2:]), axis=1)
            sc_channel_mat_fd_conjugate = np.conjugate(sc_channel_mat_fd)

            if mr_precoding is True:
                # normalize the precoding vector in regard to number of antennas and power
                # equal sum of TX power MR precoding
                precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate,
                                             np.sqrt(np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0)))
                # equal sum of RX power MR precoding
                # precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate, np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0))

            else:
                # take only phases into consideration
                # normalize channel precoding coefficients
                precoding_mat_fd = np.exp(1j * np.angle(sc_channel_mat_fd_conjugate))

            # apply precoding vectors to each tx node from matrix
            for idx, tx_transceiver in enumerate(self.array_elements):
                # select coefficients based on carrier frequencies
                tx_n_sc = tx_transceiver.modem.n_sub_carr
                precoding_mat_row_fd = precoding_mat_fd[idx, :]
                precoding_vec = np.concatenate((precoding_mat_row_fd[: tx_n_sc // 2], precoding_mat_row_fd[-tx_n_sc // 2:]), 0)
                tx_transceiver.modem.set_precoding(precoding_vec)
        # multiple user precoding
        else:
            n_users = len(channel_mat_fd)
            precoding_mat_fd = np.empty((n_users, self.n_elements, tx_n_sc), dtype=np.complex128)

            # old normalization for simple conjugate precoding
            # #multiple user TX power normalization factor
            # ant_norm_mat = np.empty((n_users, self.n_elements))
            # for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
            #     sc_channel_mat_fd = np.concatenate(
            #         (usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1], usr_chan_mat_fd[:, -tx_n_sc // 2:]), axis=1)
            #     ant_norm_mat[usr_idx, :] = np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=1)
            # ant_norm_coeff_vec = 1/np.sqrt(np.sum(ant_norm_mat, axis=0))

            for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                sc_channel_mat_fd = np.concatenate(
                    (usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1], usr_chan_mat_fd[:, -tx_n_sc // 2:]), axis=1)
                sc_channel_mat_fd_conjugate = np.conjugate(sc_channel_mat_fd)

                if mr_precoding is True:
                    # normalize the precoding vector in regard to number of antennas and power
                    # equal sum of TX power MR precoding
                    usr_precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate,
                                                     np.sqrt(np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0)))
                    # equal sum of RX power MR precoding
                    # precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate, np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0))

                else:
                    # take only phases into consideration
                    # normalize channel precoding coefficients
                    usr_precoding_mat_fd = np.exp(1j * np.angle(sc_channel_mat_fd_conjugate))
                precoding_mat_fd[usr_idx, :, :] = usr_precoding_mat_fd

            # normalize the precoding
            # #multiple user TX power normalization factor
            ant_norm_mat = np.empty((n_users, self.n_elements))
            for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                ant_norm_mat[usr_idx, :] = np.sum(np.power(np.abs(precoding_mat_fd[usr_idx, :, :]), 2), axis=1)
            ant_norm_coeff_vec = 1/np.sqrt(np.sum(ant_norm_mat, axis=0))

            # apply precoding matrix to each tx node
            for tx_idx, tx_transceiver in enumerate(self.array_elements):
                # select coefficients based on carrier frequencies
                mu_precoding_slice = precoding_mat_fd[:, tx_idx, :]
                precoding_mat = np.concatenate((mu_precoding_slice[:, :tx_n_sc // 2], mu_precoding_slice[:, -tx_n_sc // 2:]), axis=1)
                tx_transceiver.modem.set_precoding(precoding_mat * ant_norm_coeff_vec[tx_idx])

    def update_distortion(self, ibo_db, avg_sample_pow):
        # calculate the avg precoding gain only for the desired signal - withing the idx range of subcarriers
        # for idx, tx_transceiver in enumerate(self.array_elements):
        # select coefficients based on carrier frequencies
        tx_n_sc = self.base_transceiver.modem.n_sub_carr
        # Equal total TX power precoding distortion normalization
        n_users = self.base_transceiver.modem.n_users
        precoding_matrix = np.empty((self.n_elements, self.base_transceiver.modem.n_sub_carr), dtype=np.complex128)

        if n_users == 1:
            # get precoding matrix
            for idx, array_tx in enumerate(self.array_elements):
                precoding_matrix[idx, :] = array_tx.modem.precoding_mat
            sc_precoding_mat = np.concatenate(
                (precoding_matrix[:, 1:(tx_n_sc // 2) + 1], precoding_matrix[:, -tx_n_sc // 2:]), axis=1)
            avg_precoding_gain = np.average(
                np.divide(np.power(np.abs(sc_precoding_mat), 2), np.sum(np.power(np.abs(sc_precoding_mat), 2), axis=0)))
        else:
            # multiple user scenario
            for idx, array_tx in enumerate(self.array_elements):
                precoding_matrix[idx, :] = np.sum(array_tx.modem.precoding_mat, axis=0)
            sc_precoding_mat = np.concatenate(
                (precoding_matrix[:, 1:(tx_n_sc // 2) + 1], precoding_matrix[:, -tx_n_sc // 2:]), axis=1)
            avg_precoding_gain = np.average(
                np.divide(np.power(np.abs(sc_precoding_mat), 2), np.sum(np.power(np.abs(sc_precoding_mat), 2), axis=0)))
        # Equal total RX power precoding distortion normalization
        # sc_channel_mat = np.concatenate((channel_mat_fd[:, 1:(tx_n_sc // 2) + 1], channel_mat_fd[:, -tx_n_sc // 2:]), axis=1)
        #
        # avg_precoding_gain = np.average(np.divide(np.power(np.abs(sc_channel_mat), 2),
        #                                           np.power(np.sum(np.power(np.abs(sc_channel_mat), 2), axis=0), 2)))
        # print("AVG precoding gain: ", avg_precoding_gain)

        for idx, array_transceiver in enumerate(self.array_elements):
            array_transceiver.impairment.set_ibo(ibo_db)
            array_transceiver.impairment.set_avg_sample_power(avg_sample_pow * avg_precoding_gain)
