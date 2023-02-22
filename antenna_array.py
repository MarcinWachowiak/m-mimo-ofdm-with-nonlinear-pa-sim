import copy
from abc import ABC
from typing import Union

import numpy as np
import scipy as scp
from numpy import ndarray

import distortion
import utilities
from transceiver import Transceiver


class AntennaArray(ABC):
    """
    Base class of the antenna array objects.

    :param n_elements: number of antenna elements in the array
    :param base_transceiver: transceiver object that will be copied for each antenna node
    :param center_freq: center frequency of the system, used for element spacing calculations
    :param wav_len_spacing: spacing of antenna expressed in fraction of the wavelength
    :param cord_x: x coordinate of the center of the array
    :param cord_y: y coordinate of the center of the array
    :param cord_z: z coordinate of the center of the array
    """

    def __init__(self, n_elements: int, base_transceiver: Transceiver, center_freq: int,
                 wav_len_spacing: float = 0.5, cord_x: float = 0, cord_y: float = 0, cord_z: float = 0):
        """
        Constructor method.
        """

        self.n_elements = n_elements
        self.n_users = base_transceiver.modem.n_users
        self.base_transceiver = base_transceiver
        self.center_freq = center_freq
        self.wav_len_spacing = wav_len_spacing
        self.array_elements = []
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z

    def set_tx_power_lvls(self, tx_power_dbm: float, total: bool = False) -> None:
        """
        [Unused] Set transmit power in dBm.

        :param tx_power_dbm: transmit power in dBm
        :param total: is the power value a cumulative sum of all elements TX power
        :return: None
        """

        for tx in self.array_elements:
            # is the power value a sum of all TX powers or per individual TX
            if total:
                tx.set_tx_power_dbm(10 * (np.log10(10 ** (tx_power_dbm / 10) / len(self.array_elements))))
            else:
                tx.set_tx_power_dbm(tx_power_dbm)

    def transmit(self, in_bits: ndarray, out_domain_fd: bool = True, return_both: bool = False, skip_dist: bool = False,
                 sum_usr_signals: bool = True) -> Union[tuple[ndarray, ndarray], ndarray]:
        """
        Transmit the input data with multiple antenna transceivers.

        :param in_bits: array of bits to be transmitted 1D for single user, 2D for multiple users
        :param out_domain_fd: flag if to return the signal in frequency domain or in time domain representation
        :param return_both: flag if both distorted and non-distorted signals should be returned
        :param skip_dist: flag if nonlinear processing should be skipped
        :param sum_usr_signals: flag in multiuser case if usr signals are summed together at the TX
        :return: multidimensional signal matrix or tuple of them with transmitted signal samples
            [usr_idx, tx_idx, sample_idx]
        """

        if sum_usr_signals is False and self.n_users == 1:
            sum_usr_signals = True

        if out_domain_fd:
            if sum_usr_signals:
                out_sig_mat = np.empty([self.n_elements, self.base_transceiver.modem.n_fft],
                                       dtype=np.complex128)
            else:
                out_sig_mat = np.empty([self.n_users, self.n_elements, self.base_transceiver.modem.n_fft],
                                       dtype=np.complex128)
        else:
            if sum_usr_signals:
                out_sig_mat = np.empty(
                    [self.n_elements, self.base_transceiver.modem.n_fft + self.base_transceiver.modem.cp_len],
                    dtype=np.complex128)
            else:
                out_sig_mat = np.empty(
                    [self.n_users, self.n_elements,
                     self.base_transceiver.modem.n_fft + self.base_transceiver.modem.cp_len],
                    dtype=np.complex128)

        if return_both:
            if out_domain_fd:
                if sum_usr_signals:
                    clean_sig_mat = np.empty([self.n_elements, self.base_transceiver.modem.n_fft], dtype=np.complex128)
                else:
                    clean_sig_mat = np.empty([self.n_users, self.n_elements, self.base_transceiver.modem.n_fft],
                                             dtype=np.complex128)
            else:
                if sum_usr_signals:
                    clean_sig_mat = np.empty(
                        [self.n_elements, self.base_transceiver.modem.n_fft + self.base_transceiver.modem.cp_len],
                        dtype=np.complex128)
                else:
                    clean_sig_mat = np.empty([self.n_users, self.n_elements,
                                              self.base_transceiver.modem.n_fft + self.base_transceiver.modem.cp_len],
                                             dtype=np.complex128)

            if sum_usr_signals:
                for idx, tx_transceiver in enumerate(self.array_elements):
                    out_sig_mat[idx, :], clean_sig_mat[idx, :] = tx_transceiver.transmit(in_bits,
                                                                                         out_domain_fd=out_domain_fd,
                                                                                         return_both=return_both,
                                                                                         skip_dist=skip_dist,
                                                                                         sum_usr_signals=sum_usr_signals)
                return np.squeeze(out_sig_mat), np.squeeze(clean_sig_mat)
            else:
                for tx_idx, tx_transceiver in enumerate(self.array_elements):
                    usr_signal_lst = tx_transceiver.transmit(in_bits, out_domain_fd=out_domain_fd,
                                                             return_both=return_both, skip_dist=skip_dist,
                                                             sum_usr_signals=sum_usr_signals)
                    for usr_idx in range(self.n_users):
                        out_sig_mat[usr_idx, tx_idx, :], clean_sig_mat[usr_idx, tx_idx, :] = usr_signal_lst[usr_idx]
                return out_sig_mat, clean_sig_mat
        else:
            if sum_usr_signals:
                for idx, tx_transceiver in enumerate(self.array_elements):
                    out_sig_mat[idx, :] = tx_transceiver.transmit(in_bits, out_domain_fd=out_domain_fd,
                                                                  return_both=return_both, skip_dist=skip_dist,
                                                                  sum_usr_signals=sum_usr_signals)
                return np.squeeze(out_sig_mat)
            else:
                for tx_idx, tx_transceiver in enumerate(self.array_elements):
                    usr_signal_lst = tx_transceiver.transmit(in_bits, out_domain_fd=out_domain_fd,
                                                             return_both=return_both,
                                                             skip_dist=skip_dist, sum_usr_signals=sum_usr_signals)
                    for usr_idx in range(self.n_users):
                        out_sig_mat[usr_idx, tx_idx, :] = usr_signal_lst[usr_idx]
                return out_sig_mat

    def set_precoding_matrix(self, channel_mat_fd: list[ndarray] = None, mr_precoding: bool = False,
                             zf_precoding: bool = False, update_distortion: bool = False,
                             sep_carr_per_usr: bool = False) -> None:
        """
        Set precoding vectors in each TX modulators based on provided channel mat coefficients.
        Only the subcarriers are precoded, other normalization operations should be performed in regard to carrier pool.

        :param channel_mat_fd: channel matrix used to calculate the precoding, for multiple users it is a list
            of channel matrices
        :param mr_precoding: flag for Maximum Ratio Transmission (MRT) precoding
        :param zf_precoding: flag for Zero Forcing (ZF) precoding
        :param update_distortion: flat if distortion maximum power should be updated after the precoding changes
            the average power to maintain constant IBO
        :param sep_carr_per_usr: flag if the users are allocated on separate sets of subcarriers, precoding is
            then applied only to selected sets of subcarriers
        :return: None
        """

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
            if not sep_carr_per_usr:
                precoding_mat_fd = np.empty((self.n_users, self.n_elements, tx_n_sc), dtype=np.complex128)

                # old normalization for simple conjugate precoding
                # #multiple user TX power normalization factor
                # ant_norm_mat = np.empty((self.n_users, self.n_elements))
                # for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                #     sc_channel_mat_fd = np.concatenate(
                #         (usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1], usr_chan_mat_fd[:, -tx_n_sc // 2:]), axis=1)
                #     ant_norm_mat[usr_idx, :] = np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=1)
                # ant_norm_coeff_vec = 1/np.sqrt(np.sum(ant_norm_mat, axis=0))

                # calculate the normalizing factor
                usrs_vects = np.empty((self.n_users, tx_n_sc))
                for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                    sc_channel_mat_fd = np.concatenate(
                        (usr_chan_mat_fd[:, -tx_n_sc // 2:], usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1]), axis=1)
                    usrs_vects[usr_idx, :] = np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0)
                nsc_power_normalzing_vec = np.sqrt(np.sum(usrs_vects, axis=0))

                if mr_precoding:
                    for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                        sc_channel_mat_fd = np.concatenate(
                            (usr_chan_mat_fd[:, -tx_n_sc // 2:], usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1]), axis=1)
                        sc_channel_mat_fd_conjugate = np.conjugate(sc_channel_mat_fd)

                        # normalize the precoding vector in regard to number of antennas and power
                        # equal sum of TX power MR precoding
                        usr_precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate, nsc_power_normalzing_vec)
                        # equal sum of RX power MR precoding
                        # precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate, np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0))
                        precoding_mat_fd[usr_idx, :, :] = usr_precoding_mat_fd

                elif zf_precoding:
                    per_usr_sc_chan_mat_fd = []
                    for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                        sc_channel_mat_fd = np.concatenate(
                            (usr_chan_mat_fd[:, -tx_n_sc // 2:], usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1]), axis=1)
                        per_usr_sc_chan_mat_fd.append(sc_channel_mat_fd)

                    mu_chan_mat_per_sc = []
                    for sc_idx in range(tx_n_sc):
                        mu_chan_mat_at_sc = []
                        for usr_idx in range(self.n_users):
                            mu_chan_mat_at_sc.append(per_usr_sc_chan_mat_fd[usr_idx][:, sc_idx])
                        mu_chan_mat_per_sc.append(np.vstack(mu_chan_mat_at_sc))

                    try:
                        for sc_idx in range(tx_n_sc):
                            tmp_mat = np.transpose(mu_chan_mat_per_sc[sc_idx])
                            usr_precoding_mat_fd = np.sqrt(self.n_elements - self.n_users) * np.matmul(
                                np.conjugate(tmp_mat), np.linalg.inv(
                                    np.matmul(np.transpose(tmp_mat), np.conjugate(tmp_mat))))
                            precoding_mat_fd[:, :, sc_idx] = np.transpose(usr_precoding_mat_fd)
                    except Exception as e:
                        pass
                        # try pseudoinverse if the standard one fails
                        for sc_idx in range(tx_n_sc):
                            tmp_mat = np.transpose(mu_chan_mat_per_sc[sc_idx])
                            usr_precoding_mat_fd = np.sqrt(self.n_elements - self.n_users) * np.matmul(
                                np.conjugate(tmp_mat), np.linalg.pinv(
                                    np.matmul(np.transpose(tmp_mat), np.conjugate(tmp_mat))))
                            precoding_mat_fd[:, :, sc_idx] = np.transpose(usr_precoding_mat_fd)

                    # normalize the ZF precoding to have unit power at each subcarrier
                    for sc_idx in range(tx_n_sc):
                        pow_norm_factor = np.sqrt(
                            np.sum(np.sum(np.power(np.abs(precoding_mat_fd[:, :, sc_idx]), 2), axis=0)))
                        precoding_mat_fd[:, :, sc_idx] = np.divide(precoding_mat_fd[:, :, sc_idx], pow_norm_factor)

                else:
                    # take only phases into consideration
                    for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                        sc_channel_mat_fd = np.concatenate(
                            (usr_chan_mat_fd[:, -tx_n_sc // 2:], usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1]), axis=1)
                        sc_channel_mat_fd_conjugate = np.conjugate(sc_channel_mat_fd)
                        # normalize channel precoding coefficients
                        usr_precoding_mat_fd = np.exp(1j * np.angle(sc_channel_mat_fd_conjugate))
                        precoding_mat_fd[usr_idx, :, :] = usr_precoding_mat_fd

                # apply precoding matrix to each tx node
                for tx_idx, tx_transceiver in enumerate(self.array_elements):
                    # select coefficients based on carrier frequencies
                    mu_precoding_slice = precoding_mat_fd[:, tx_idx, :]
                    tx_transceiver.modem.set_precoding(mu_precoding_slice)

            else:
                composed_chan_mat = None
                for usr_idx, usr_chan_mat_fd in enumerate(channel_mat_fd):
                    sc_channel_mat_fd = np.concatenate(
                        (usr_chan_mat_fd[:, -tx_n_sc // 2:], usr_chan_mat_fd[:, 1:(tx_n_sc // 2) + 1]), axis=1)
                    nsc_split_lst = np.hsplit(sc_channel_mat_fd, len(channel_mat_fd))
                    if composed_chan_mat is None:
                        composed_chan_mat = nsc_split_lst[usr_idx]
                    else:
                        composed_chan_mat = np.concatenate((composed_chan_mat, nsc_split_lst[usr_idx]), axis=1)

                composed_chan_mat_conjugate = np.conjugate(composed_chan_mat)
                if mr_precoding is True:
                    # normalize the precoding vector in regard to number of antennas and power
                    # equal sum of TX power MR precoding
                    precoding_mat_fd = np.divide(composed_chan_mat_conjugate,
                                                 np.sqrt(np.sum(np.power(np.abs(composed_chan_mat), 2), axis=0)))
                    # equal sum of RX power MR precoding
                    # precoding_mat_fd = np.divide(sc_channel_mat_fd_conjugate, np.sum(np.power(np.abs(sc_channel_mat_fd), 2), axis=0))

                else:
                    # take only phases into consideration
                    # normalize channel precoding coefficients
                    precoding_mat_fd = np.exp(1j * np.angle(composed_chan_mat_conjugate))

                # apply precoding vectors to each tx node from matrix
                for idx, tx_transceiver in enumerate(self.array_elements):
                    # select coefficients based on carrier frequencies
                    tx_n_sc = tx_transceiver.modem.n_sub_carr
                    precoding_vec = precoding_mat_fd[idx, :]
                    tx_transceiver.modem.set_precoding(precoding_vec)

        if update_distortion:
            # update the average signal power expected by the distortion after it has been changed by precoding
            # coefficients to maintain constant IBO
            self.update_distortion(ibo_db=self.array_elements[0].impairment.ibo_db,
                                   avg_sample_pow=self.array_elements[0].modem.avg_sample_power)

    def update_distortion(self, ibo_db: float, avg_sample_pow: float, alpha_val: float = None) -> None:
        """
        Update the average and maximum power expected by distortion: soft-limiter to maintain constant Input Backoff
        (IBO) after applied precoding.

        :param ibo_db: input backoff (IBO) value in [dB]
        :param avg_sample_pow: average power of the signal sample
        :param alpha_val: value of the alpha - shrinking coefficient
        :return: None
        """

        # calculate the avg precoding gain only for the desired signal - withing the idx range of subcarriers
        # for idx, tx_transceiver in enumerate(self.array_elements):
        # select coefficients based on carrier frequencies
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

        # update the ibo
        for idx, array_transceiver in enumerate(self.array_elements):
            # TODO: Third order Nonlinear distortion update needs to be implemented properly
            if isinstance(array_transceiver.impairment, distortion.ThirdOrderNonLin):
                array_transceiver.modem.alpha = alpha_val
                array_transceiver.impairment.set_toi(ibo_db)
            else:
                array_transceiver.modem.alpha = array_transceiver.modem.calc_alpha(ibo_db=ibo_db)
                array_transceiver.impairment.set_ibo(ibo_db)

            array_transceiver.impairment.set_avg_sample_power(avg_sample_pow * avg_precoding_gain)

    def get_avg_precoding_gain(self) -> float:
        """
        Calculate the average precoding power gain based on the precoding matrices in the array system.

        :return: average precoding power gain
        """

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

    def get_precoding_mat(self) -> ndarray:
        """
        Return the precoding matrix from all transceivers.

        :return: multidimensional array of precoding coefficients [usr_idx, tx_idx, sample_idx]
        """

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

    def update_n_users(self, n_users) -> None:
        """
        Update the number of users in the system (in each transceiver)

        :param n_users: number of users
        :return: None
        """
        self.n_users = n_users
        for idx, array_tx in enumerate(self.array_elements):
            array_tx.modem.n_users = n_users


class LinearArray(AntennaArray):
    """
    Uniform linear array (ULA) class.

    :param n_elements: number of antenna elements in the array
    :param base_transceiver: transceiver object that will be copied for each antenna node
    :param center_freq: center frequency of the system, used for element spacing calculations
    :param wav_len_spacing: spacing of antenna expressed in fraction of the wavelength
    :param cord_x: x coordinate of the center of the array
    :param cord_y: y coordinate of the center of the array
    :param cord_z: z coordinate of the center of the array
    """

    def __init__(self, n_elements: int, base_transceiver: Transceiver, center_freq: int, wav_len_spacing: float,
                 cord_x: float = 0, cord_y: float = 0, cord_z: float = 0):
        """
        Create a uniform linear antenna array along X axis.
        """

        super().__init__(n_elements, base_transceiver, center_freq, wav_len_spacing, cord_x, cord_y, cord_z)
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


class CircularArray(AntennaArray):
    """
    Uniform circular array (UCA) class.

    :param n_elements: number of antenna elements in the array
    :param base_transceiver: transceiver object that will be copied for each antenna node
    :param center_freq: center frequency of the system, used for element spacing calculations
    :param wav_len_spacing: spacing of antenna expressed in fraction of the wavelength
    :param cord_x: x coordinate of the center of the array
    :param cord_y: y coordinate of the center of the array
    :param cord_z: z coordinate of the center of the array
    """

    def __init__(self, n_elements: int, base_transceiver: Transceiver, center_freq: int, wav_len_spacing: float,
                 cord_x: float = 0, cord_y: float = 0, cord_z: float = 0):
        """
        Create a uniform circular antenna array on X-Y plane.
        """

        # TODO: add a switch for full circular or semicircular topology
        super().__init__(n_elements, base_transceiver, center_freq, wav_len_spacing, cord_x, cord_y, cord_z)
        # uniform circular array with radius specified by wav_len_spacing
        wavelength_at_freq = scp.constants.c / self.center_freq
        # spacing on radius <= lambda/2
        array_radius = wavelength_at_freq * (self.n_elements - 1) / (2 * np.pi)
        ant_pos_lst = utilities.pts_on_semicircum(r=array_radius, n=self.n_elements)
        for idx in range(self.n_elements):
            tmp_transceiver = copy.deepcopy(self.base_transceiver)
            tmp_transceiver.cord_x = ant_pos_lst[idx][0]
            tmp_transceiver.cord_y = ant_pos_lst[idx][1]
            tmp_transceiver.cord_z = self.cord_z
            self.array_elements.append(tmp_transceiver)


class PlanarRectangularArray(AntennaArray):
    """
    Uniform rectangular array class.

    :param n_elements_per_row: number of elements in a row
    :param n_elements_per_col: number of elements in a column
    :param base_transceiver: transceiver object that will be copied for each antenna node
    :param center_freq: center frequency of the system, used for element spacing calculations
    :param wav_len_spacing: spacing of antenna expressed in fraction of the wavelength
    :param cord_x: x coordinate of the center of the array
    :param cord_y: y coordinate of the center of the array
    :param cord_z: z coordinate of the center of the array
    """

    def __init__(self, n_elements_per_row: int, n_elements_per_col: int, base_transceiver: Transceiver, center_freq: int, wav_len_spacing: float,
                 cord_x: float = 0, cord_y: float = 0, cord_z: float = 0):
        """
        Create a uniform rectangular antenna array on X-Z plane.
        """

        n_elements = n_elements_per_row * n_elements_per_col
        super().__init__(n_elements, base_transceiver, center_freq, wav_len_spacing, cord_x, cord_y, cord_z)
        # antenna position vector centered around 0
        wavelength_at_freq = scp.constants.c / self.center_freq

        ant_vec_col = np.linspace(-(n_elements_per_col - 1) * self.wav_len_spacing * wavelength_at_freq / 2,
                                  (n_elements_per_col - 1) * self.wav_len_spacing * wavelength_at_freq / 2,
                                  n_elements_per_col)
        ant_vec_row = np.linspace(-(n_elements_per_row - 1) * self.wav_len_spacing * wavelength_at_freq / 2,
                                  (n_elements_per_row - 1) * self.wav_len_spacing * wavelength_at_freq / 2,
                                  n_elements_per_row)

        for col_coord in ant_vec_col:
            for row_coord in ant_vec_row:
                tmp_transceiver = copy.deepcopy(self.base_transceiver)
                tmp_transceiver.cord_x = col_coord
                tmp_transceiver.cord_y = 0
                tmp_transceiver.cord_z = self.cord_z + row_coord
                self.array_elements.append(tmp_transceiver)
