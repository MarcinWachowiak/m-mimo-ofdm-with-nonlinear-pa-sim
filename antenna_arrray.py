import copy

import numpy as np
import scipy as scp

import distortion


class LinearArray:
    def __init__(self, n_elements, base_transceiver, center_freq, wav_len_spacing, cord_x=0, cord_y=0, cord_z=0):
        self.n_elements = n_elements
        self.n_users = base_transceiver.modem.self.n_users
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

    def transmit(self, in_bits, out_domain_fd=True, return_both=False, skip_dist=False):
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
                                                                                     return_both=True,
                                                                                     skip_dist=skip_dist)

            return np.squeeze(out_sig_mat), np.squeeze(clean_sig_mat)
        else:
            for idx, tx_transceiver in enumerate(self.array_elements):
                out_sig_mat[idx, :] = tx_transceiver.transmit(in_bits, out_domain_fd=out_domain_fd,
                                                              return_both=return_both, skip_dist=skip_dist)
            return np.squeeze(out_sig_mat)

    def set_precoding_matrix(self, channel_mat_fd=None, mr_precoding=False, zf_precoding=False):
        # set precoding vector based on provided channel mat coefficients
        # only the subcarriers are precoded, other normalization operations should be performed in regard to carrier pool
        tx_n_sc = self.base_transceiver.modem.n_sub_carr

        if not isinstance(channel_mat_fd, list):
            sc_channel_mat_fd = np.concatenate(
                (channel_mat_fd[:, -tx_n_sc // 2:], channel_mat_fd[:, 1:(tx_n_sc // 2) + 1]), axis=1)
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
                precoding_vec = precoding_mat_fd[idx, :]
                tx_transceiver.modem.set_precoding(precoding_vec)

        # multiple user precoding
        else:
            precoding_mat_fd = np.empty((self.n_users, self.n_elements, tx_n_sc), dtype=np.complex128)

            # old normalization for simple conjugate precoding
            # #multiple user TX power normalization factor
            # ant_norm_mat = np.empty((self.n_users, self.n_elements))
            # for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
            #     sc_channel_mat_fd = np.concatenate(
            #         (usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1], usr_chan_mat_fd[:, -tx_n_sc // 2:]), axis=1)
            #     ant_norm_mat[usr_idx, :] = np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=1)
            # ant_norm_coeff_vec = 1/np.sqrt(np.sum(ant_norm_mat, axis=0))

            # calculate the normalizing factor K
            usrs_vects = np.empty((self.n_users, tx_n_sc))
            for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                sc_channel_mat_fd = np.concatenate(
                    (usr_chan_mat_fd[:, -tx_n_sc // 2:], usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1]), axis=1)
                usrs_vects[usr_idx, :] = np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0)
            nsc_power_normalzing_vec = np.sqrt(np.sum(usrs_vects, axis=0))

            for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                sc_channel_mat_fd = np.concatenate(
                    (usr_chan_mat_fd[:, -tx_n_sc // 2:], usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1]), axis=1)
                sc_channel_mat_fd_conjugate = np.conjugate(sc_channel_mat_fd)

                if mr_precoding:
                    # normalize the precoding vector in regard to number of antennas and power
                    # equal sum of TX power MR precoding
                    usr_precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate, nsc_power_normalzing_vec)
                    # equal sum of RX power MR precoding
                    # precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate, np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0))

                else:
                    # take only phases into consideration
                    # normalize channel precoding coefficients
                    usr_precoding_mat_fd = np.exp(1j * np.angle(sc_channel_mat_fd_conjugate))

                if zf_precoding:
                    try:
                        usr_precoding_mat_fd = np.sqrt(
                            self.n_elements - self.n_users) * sc_channel_mat_fd_conjugate * np.linalg.inv(
                            np.transpose(sc_channel_mat_fd) * sc_channel_mat_fd_conjugate)
                    except:
                        # try pseudoinverse if the standard one fails
                        usr_precoding_mat_fd = np.sqrt(
                            self.n_elements - self.n_users) * sc_channel_mat_fd_conjugate * np.linalg.pinv(
                            np.transpose(sc_channel_mat_fd) * sc_channel_mat_fd_conjugate)

                precoding_mat_fd[usr_idx, :, :] = usr_precoding_mat_fd

            # apply precoding matrix to each tx node
            for tx_idx, tx_transceiver in enumerate(self.array_elements):
                # select coefficients based on carrier frequencies
                mu_precoding_slice = precoding_mat_fd[:, tx_idx, :]
                tx_transceiver.modem.set_precoding(mu_precoding_slice)

    def update_distortion(self, ibo_db, avg_sample_pow, alpha_val=None):
        # calculate the avg precoding gain only for the desired signal - withing the idx range of subcarriers
        # for idx, tx_transceiver in enumerate(self.array_elements):
        # select coefficients based on carrier frequencies
        tx_n_sc = self.base_transceiver.modem.n_sub_carr
        # Equal total TX power precoding distortion normalization

        if self.n_users == 1:
            # get precoding matrix
            precoding_matrix = np.ones((self.n_elements, self.base_transceiver.modem.n_sub_carr), dtype=np.complex128)
            for idx, array_tx in enumerate(self.array_elements):
                if array_tx.modem.precoding_mat is not None:
                    precoding_matrix[idx, :] = array_tx.modem.precoding_mat
            avg_precoding_gain = np.average(np.power(np.abs(precoding_matrix), 2))
        else:
            # multiple user scenario
            precoding_matrix_pwr = np.ones((self.n_elements, self.base_transceiver.modem.n_sub_carr), dtype=np.float64)
            for idx, array_tx in enumerate(self.array_elements):
                if array_tx.modem.precoding_mat is not None:
                    precoding_matrix_pwr[idx, :] = np.sum(np.power(np.abs(array_tx.modem.precoding_mat), 2), axis=0)
            avg_precoding_gain = np.average(precoding_matrix_pwr)

        # Equal total RX power precoding distortion normalization
        # sc_channel_mat = np.concatenate((channel_mat_fd[:, 1:(tx_n_sc // 2) + 1], channel_mat_fd[:, -tx_n_sc // 2:]), axis=1)
        #
        # avg_precoding_gain = np.average(np.divide(np.power(np.abs(sc_channel_mat), 2),
        #                                           np.power(np.sum(np.power(np.abs(sc_channel_mat), 2), axis=0), 2)))
        # print("AVG precoding gain: ", avg_precoding_gain)

        for idx, array_transceiver in enumerate(self.array_elements):
            if isinstance(array_transceiver.impairment, distortion.ThirdOrderNonLin):
                array_transceiver.modem.alpha = alpha_val
                array_transceiver.impairment.set_toi(ibo_db)
            else:
                array_transceiver.modem.alpha = array_transceiver.modem.calc_alpha(ibo_db=ibo_db)
                array_transceiver.impairment.set_ibo(ibo_db)

            array_transceiver.impairment.set_avg_sample_power(avg_sample_pow * avg_precoding_gain)

    def get_avg_precoding_gain(self):

        if self.n_users == 1:
            # get precoding matrix
            precoding_matrix = np.empty((self.n_elements, self.base_transceiver.modem.n_sub_carr), dtype=np.complex128)
            for idx, array_tx in enumerate(self.array_elements):
                precoding_matrix[idx, :] = array_tx.modem.precoding_mat
            avg_precoding_gain = np.average(np.power(np.abs(precoding_matrix), 2))
        else:
            # multiple user scenario
            precoding_matrix_pwr = np.empty((self.n_elements, self.base_transceiver.modem.n_sub_carr), dtype=np.float64)
            for idx, array_tx in enumerate(self.array_elements):
                precoding_matrix_pwr[idx, :] = np.sum(np.power(np.abs(array_tx.modem.precoding_mat), 2), axis=0)
            avg_precoding_gain = np.average(precoding_matrix_pwr)

        return avg_precoding_gain

    def get_precoding_mat(self):
        if self.n_users == 1:
            precoding_matrix = np.empty((self.n_elements, self.base_transceiver.modem.n_sub_carr), dtype=np.complex128)
            for idx, array_tx in enumerate(self.array_elements):
                precoding_matrix[idx, :] = array_tx.modem.precoding_mat
        else:
            precoding_matrix = np.empty((self.n_elements, self.n_users, self.base_transceiver.modem.n_sub_carr),
                                        dtype=np.complex128)
            for idx, array_tx in enumerate(self.array_elements):
                precoding_matrix[idx, :, :] = array_tx.modem.precoding_mat

        return precoding_matrix
