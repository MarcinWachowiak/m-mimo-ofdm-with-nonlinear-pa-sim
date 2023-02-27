import matlab.engine
import numpy as np
import torch
from scipy import constants as scp_constants
from numpy import ndarray
from transceiver import Transceiver


# TODO: Add a generic channel base class and introduce inheritance

class MisoLosFd:
    """
    Multiple-input single-output (MISO) line-of-sight (LOS) channel class.
    Includes only free-space path loss and phase shift calculated based on propagation distances.
    """

    def __init__(self):
        """
        Creates MISO LOS channel, without initializing the channel matrix.
        """
        self.channel_mat_fd = None

    def __str__(self):
        return "los"

    def get_channel_mat_fd(self) -> ndarray:
        """
        Returns the channel matrix in frequency domain.

        :return: matrix of channel coefficients in frequency domain
        """

        return self.channel_mat_fd

    def calc_channel_mat(self, tx_transceivers: list[Transceiver], rx_transceiver: Transceiver,
                         skip_attenuation: bool = False) -> None:
        """
        Calculates the channel coefficients matrix in frequency domain based on the channel model.

        :param tx_transceivers: list of transceiver objects of the antenna array
        :param rx_transceiver: receiver object
        :param skip_attenuation: flag if to skip the free-space path loss in calculations
        :return: None
        """

        # for each tx to rx get distance
        tx_rx_los_distances = np.empty(len(tx_transceivers))
        tx_ant_gains = np.empty(len(tx_transceivers))

        # get carrier frequencies
        sig_freq_vals = torch.fft.fftfreq(rx_transceiver.modem.n_fft, d=1 / rx_transceiver.modem.n_fft).numpy() \
                        * rx_transceiver.carrier_spacing + rx_transceiver.center_freq

        for idx, tx_transceiver in enumerate(tx_transceivers):
            tx_ant_gains[idx] = tx_transceiver.tx_ant_gain_db
            tx_rx_los_distances[idx] = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                tx_transceiver.cord_z - rx_transceiver.cord_z, 2))

        # shift phases of carriers accordingly to spatial relations
        fd_ph_shift_mat = np.exp(2j * np.pi * np.outer(tx_rx_los_distances, sig_freq_vals) / scp_constants.c)

        # multiply fd signals by attenuation matrix
        if not skip_attenuation:
            fd_freq_att_mat = np.sqrt(np.power(10, (tx_ant_gains[:, np.newaxis] + rx_transceiver.rx_ant_gain_db) / 10)) \
                              * (scp_constants.c / (4 * np.pi * np.outer(tx_rx_los_distances, sig_freq_vals)))
            calc_channel_mat_fd = np.multiply(fd_ph_shift_mat, fd_freq_att_mat)
        else:
            # then channel matrix consist of phase shits only
            calc_channel_mat_fd = fd_ph_shift_mat

        self.channel_mat_fd = calc_channel_mat_fd

    def propagate(self, in_sig_mat: ndarray, sum_signals: bool = True) -> ndarray:
        """
        Multiplies the input signal matrix by the channel matrix.
        Both should be in frequency domain.

        :param in_sig_mat: input signal matrix from antenna array
        :param sum_signals: flag if the output signals should be summed or not, to model single antenna receiver
        :return: signal matrix after the channel propagation propagation
        """

        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum_signals columns
        if sum_signals:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan


class MisoTwoPathFd:
    """
    Multiple-input single-output (MISO) two-path channel model.
    Apart from the line-of-sight path it includes also the reflection from the ground with the fixed coe
    """

    def __init__(self):
        """
        Creates MISO two-path channel, without initializing the channel matrix.
        """
        self.channel_mat_fd = None

    def __str__(self):
        return "two_path"

    def get_channel_mat_fd(self) -> ndarray:
        """
        Returns the channel matrix in frequency domain.

        :return: matrix of channel coefficients in frequency domain
        """

        return self.channel_mat_fd

    def calc_channel_mat(self, tx_transceivers: list[Transceiver], rx_transceiver: Transceiver,
                         skip_attenuation: bool = False) -> None:
        """
        Calculates the channel coefficients matrix in frequency domain taking into consideration two propagation paths.

        :param tx_transceivers: list of transceiver objects of the antenna array
        :param rx_transceiver: receiver object
        :param skip_attenuation: flag if to skip the free-space path loss in calculations
        :return: None
        """

        # get frequencies of subcarriers
        sig_freq_vals = torch.fft.fftfreq(rx_transceiver.modem.n_fft, d=1 / rx_transceiver.modem.n_fft).numpy() \
                        * rx_transceiver.carrier_spacing + rx_transceiver.center_freq

        los_distances = np.empty(len(tx_transceivers))
        sec_distances = np.empty(len(tx_transceivers))
        tx_ant_gains = np.empty(len(tx_transceivers))

        for idx, tx_transceiver in enumerate(tx_transceivers):
            tx_ant_gains[idx] = tx_transceiver.tx_ant_gain_db
            los_distances[idx] = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                tx_transceiver.cord_z - rx_transceiver.cord_z, 2))

            dim_ratio = (tx_transceiver.cord_z + rx_transceiver.cord_z) / (
                np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                    tx_transceiver.cord_y - rx_transceiver.cord_y, 2)))
            # angle of elevation (angle in relation to the ground plane) = 90 deg - angle of incidence
            angle_of_elev_rad = np.arctan(dim_ratio)

            incident_path_len = tx_transceiver.cord_z / np.sin(angle_of_elev_rad)
            reflected_path_len = rx_transceiver.cord_z / np.sin(angle_of_elev_rad)
            sec_distances[idx] = incident_path_len + reflected_path_len

        los_fd_shift_mat = np.exp(2j * np.pi * np.outer(los_distances, sig_freq_vals) / scp_constants.c)
        # TODO: include detailed calculation of reflection coefficient depending on the incidence angle etc.
        reflection_coeff = -1.0
        sec_fd_shift_mat = reflection_coeff * np.exp(2j * np.pi *
                                                     np.outer(sec_distances, sig_freq_vals) / scp_constants.c)

        if not skip_attenuation:
            los_fd_att_mat = np.sqrt(np.power(10, (tx_ant_gains[:, np.newaxis] + rx_transceiver.rx_ant_gain_db) / 10)) \
                             * (scp_constants.c / (4 * np.pi * np.outer(los_distances, sig_freq_vals)))
            sec_fd_att_mat = np.sqrt(np.power(10, (tx_ant_gains[:, np.newaxis] + rx_transceiver.rx_ant_gain_db) / 10)) \
                             * (scp_constants.c / (4 * np.pi * np.outer(sec_distances, sig_freq_vals)))

            los_fd_shift_mat = np.multiply(los_fd_shift_mat, los_fd_att_mat)
            sec_fd_shift_mat = np.multiply(sec_fd_shift_mat, sec_fd_att_mat)

        # combine two path coefficients without normalization
        self.channel_mat_fd = np.add(los_fd_shift_mat, sec_fd_shift_mat)

    def propagate(self, in_sig_mat: ndarray, sum_signals: bool = True) -> ndarray:
        """
        Multiplies the input signal matrix by the channel matrix.
        Both should be in frequency domain.

        :param in_sig_mat: input signal matrix from antenna array
        :param sum_signals: flag if the output signals should be summed or not, to model single antenna receiver
        :return: signal matrix after the channel propagation propagation
        """

        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum_signals columns
        if sum_signals:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan


class MisoRayleighFd:
    """
    Multiple-input single-output (MISO) Rayleigh channel model.
    The coefficients between antennas and subcarriers are independent identically distributed (IID) Rayleigh variables.
    Additionally, the IID Rayleigh coefficients are multiplied by the free space attenuation.

    :param tx_transceivers: list of transceiver objects of the antenna array
    :param rx_transceiver: receiver object
    :param seed: random number generator seed for generation of the IID Rayleigh coeffcicients
    """

    def __init__(self, tx_transceivers: list[Transceiver], rx_transceiver: Transceiver, seed: int = None):
        """
        Creates MISO Rayleigh channel, with initialization of the channel matrix.
        """

        self.n_inputs = len(tx_transceivers)
        self.fd_samp_size = tx_transceivers[0].modem.n_fft
        self.fd_att_mat = None
        self.los_fd_att_mat = None
        # seed for random channel coefficients
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)
        else:
            self.rng_gen = np.random.default_rng(1234)

        # get frequencies of subcarriers
        sig_freq_vals = torch.fft.fftfreq(rx_transceiver.modem.n_fft, d=1 / rx_transceiver.modem.n_fft).numpy() \
                        * rx_transceiver.carrier_spacing + rx_transceiver.center_freq

        los_distances = np.empty(len(tx_transceivers))
        tx_ant_gains = np.empty(len(tx_transceivers))

        for idx, tx_transceiver in enumerate(tx_transceivers):
            tx_ant_gains[idx] = tx_transceiver.tx_ant_gain_db
            los_distances[idx] = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                tx_transceiver.cord_z - rx_transceiver.cord_z, 2))
        self.los_fd_att_mat = np.sqrt(np.power(10, (tx_ant_gains[:, np.newaxis] + rx_transceiver.rx_ant_gain_db) / 10)) \
                              * (scp_constants.c / (4 * np.pi * np.outer(los_distances, sig_freq_vals)))
        # generate the IID coefficients and set the channel matrix
        self.set_channel_mat_fd()

    def __str__(self):
        return "rayleigh"

    def set_channel_mat_fd(self, channel_mat_fd: ndarray = None, skip_attenuation: bool = False) -> None:
        """
        Set the channel matrix if it is provided in the channel_mat_fd, else generate one IID and set it.

        :param channel_mat_fd: input channel matrix to be set in the object
        :param skip_attenuation: flag if to skip the free-space path loss in calculations
        :return: None
        """
        if channel_mat_fd is None:
            # generate rayleigh channel coefficients
            fd_rayleigh_coeffs = self.rng_gen.standard_normal(size=(self.n_inputs, self.fd_samp_size * 2)).view(
                dtype=np.complex128) / np.sqrt(2.0)
            if skip_attenuation:
                self.channel_mat_fd = fd_rayleigh_coeffs
            else:
                self.channel_mat_fd = np.multiply(fd_rayleigh_coeffs, self.los_fd_att_mat)
        else:
            self.channel_mat_fd = channel_mat_fd

    def get_channel_mat_fd(self) -> ndarray:
        """
        Returns the channel matrix in frequency domain.

        :return: matrix of channel coefficients in frequency domain
        """

        return self.channel_mat_fd

    def reroll_channel_coeffs(self, skip_attenuation: bool = False) -> None:
        """
        Generate new IID channel coefficients and replace the channel matrix with them.

        :param skip_attenuation: flag if to skip the free-space path loss in calculations
        :return: None
        """

        fd_rayleigh_coeffs = self.rng_gen.standard_normal(size=(self.n_inputs, self.fd_samp_size * 2)).view(
            dtype=np.complex128) / np.sqrt(2.0)
        if skip_attenuation:
            self.channel_mat_fd = fd_rayleigh_coeffs
        else:
            self.channel_mat_fd = np.multiply(fd_rayleigh_coeffs, self.los_fd_att_mat)

    def propagate(self, in_sig_mat: ndarray, sum_signals: bool = True) -> ndarray:
        """
        Multiplies the input signal matrix by the channel matrix.
        Both should be in frequency domain.

        :param in_sig_mat: input signal matrix from antenna array
        :param sum_signals: flag if the output signals should be summed or not, to model single antenna receiver
        :return: signal matrix after the channel propagation
        """

        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum_signals columns
        if sum_signals:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan


class MisoRandomPathsFd:
    """
    Multiple-input single-output (MISO) random paths channel model.
    Based on equation (62) in https://ieeexplore.ieee.org/document/8429913

    :param tx_transceivers: list of transceiver objects of the antenna array
    :param rx_transceiver: receiver object
    :param seed: random number generator seed for the random paths generation
    :param n_paths: number of propagation paths
    :param max_delay_spread: maximum delay spread of the propagation paths
    """

    def __init__(self, tx_transceivers: list[Transceiver], rx_transceiver: Transceiver, seed: int = None,
                 n_paths: int = 10, max_delay_spread: float = 1000e-9):
        """
        Create MISO Random paths channel model.
        """

        self.n_inputs = len(tx_transceivers)
        self.fd_samp_size = tx_transceivers[0].modem.n_fft
        self.n_paths = n_paths
        self.max_delay_spread = max_delay_spread
        # seed for random channel coefficients
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)
        else:
            self.rng_gen = np.random.default_rng(1234)

        # get frequencies of subcarriers
        sig_freq_vals = torch.fft.fftfreq(rx_transceiver.modem.n_fft, d=1 / rx_transceiver.modem.n_fft).numpy() \
                        * rx_transceiver.carrier_spacing + rx_transceiver.center_freq

        self.channel_mat_fd = np.ones((len(tx_transceivers), self.fd_samp_size), dtype=np.complex)

        angles_of_dep = self.rng_gen.uniform(low=-np.pi / 2.0, high=np.pi / 2, size=self.n_paths)
        tau_delays = self.rng_gen.uniform(low=0.0, high=self.max_delay_spread, size=self.n_paths)

        # default reference antenna position to first of the array
        ref_x, ref_y, ref_z = tx_transceivers[0].cord_x, tx_transceivers[0].cord_y, tx_transceivers[0].cord_z
        for tx_idx, tx_transceiver in enumerate(tx_transceivers):
            # relative distance to array center
            delta_m = np.sqrt(np.power(tx_transceiver.cord_x - ref_x, 2) + np.power(
                tx_transceiver.cord_y - ref_y, 2) + np.power(
                tx_transceiver.cord_z - ref_z, 2))
            for freq_idx, freq_val in enumerate(sig_freq_vals):
                path_coeffs = np.exp(
                    -2j * freq_val * (tau_delays + delta_m * np.sin(angles_of_dep / scp_constants.speed_of_light)))
                channel_coeff = 1 / np.sqrt(self.n_paths) * np.sum(path_coeffs)
                self.channel_mat_fd[tx_idx, freq_idx] = channel_coeff

    def __str__(self):
        return "random_paths"

    def get_channel_mat_fd(self) -> ndarray:
        """
        Returns the channel matrix in frequency domain.

        :return: matrix of channel coefficients in frequency domain
        """

        return self.channel_mat_fd

    def reroll_channel_coeffs(self, tx_transceivers: list[Transceiver]) -> None:
        """
        Generate new path parameters, calculate the channel coefficients from them and set the new channel matrix.

        :param tx_transceivers: list of transceiver objects of the antenna array
        :return: None
        """

        # get frequencies of subcarriers
        sig_freq_vals = torch.fft.fftfreq(tx_transceivers[0].modem.n_fft, d=1 / tx_transceivers[0].modem.n_fft).numpy() \
                        * tx_transceivers[0].carrier_spacing + tx_transceivers[0].center_freq

        angles_of_dep = self.rng_gen.uniform(low=-np.pi / 2.0, high=np.pi / 2, size=self.n_paths)
        tau_delays = self.rng_gen.uniform(low=0.0, high=self.max_delay_spread, size=self.n_paths)

        # default reference antenna position to first of the array
        ref_x, ref_y, ref_z = tx_transceivers[0].cord_x, tx_transceivers[0].cord_y, tx_transceivers[0].cord_z
        for tx_idx, tx_transceiver in enumerate(tx_transceivers):
            # relative distance to array center
            delta_m = np.sqrt(np.power(tx_transceiver.cord_x - ref_x, 2) + np.power(
                tx_transceiver.cord_y - ref_y, 2) + np.power(
                tx_transceiver.cord_z - ref_z, 2))
            for freq_idx, freq_val in enumerate(sig_freq_vals):
                path_coeffs = np.exp(
                    -2j * freq_val * (tau_delays + delta_m * np.sin(angles_of_dep / scp_constants.speed_of_light)))
                channel_coeff = 1 / np.sqrt(self.n_paths) * np.sum(path_coeffs)
                self.channel_mat_fd[tx_idx, freq_idx] = channel_coeff

    def propagate(self, in_sig_mat: ndarray, sum_signals: bool = True) -> ndarray:
        """
        Multiplies the input signal matrix by the channel matrix.
        Both should be in frequency domain.

        :param in_sig_mat: input signal matrix from antenna array
        :param sum_signals: flag if the output signals should be summed or not, to model single antenna receiver
        :return: signal matrix after the channel propagation
        """

        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum_signals columns
        if sum_signals:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan


class MisoQuadrigaFd:
    """
    Multiple-input single-output (MISO) Quadriga channel model.
    Uses https://quadriga-channel-model.de/ matlab packet.

    :param tx_transceivers: list of transceiver objects of the antenna array
    :param rx_transceiver: receiver object
    :param channel_model_str: scenario name from the Quadriga documentation
    :param start_matlab_eng: flag if to start the matlab engine (used in mp_model)
    """

    def __init__(self, tx_transceivers: list[Transceiver], rx_transceiver: Transceiver, channel_model_str: str,
                 start_matlab_eng: bool = True):
        """
        Create a Quadriga channel model.
        """

        self.tx_transceivers = tx_transceivers
        self.rx_transceiver = rx_transceiver

        self.n_ant_val = len(tx_transceivers)
        self.n_fft = tx_transceivers[0].modem.n_fft
        self.subcarr_spacing = tx_transceivers[0].carrier_spacing
        self.center_freq = tx_transceivers[0].center_freq
        self.channel_model_str = channel_model_str
        self.distance = np.sqrt(rx_transceiver.cord_x ** 2 + rx_transceiver.cord_y ** 2)

        if start_matlab_eng:
            self.meng = matlab.engine.start_matlab()
            # add directory containing quadriga channel wrappers to the path
            self.meng.addpath(r"../main_quadriga_channel")

            self.meng.rng(5)

            self.meng.qd_channel_env_setup(self.meng.double(self.n_ant_val), self.meng.double(self.n_fft),
                                           self.meng.double(self.subcarr_spacing), self.meng.double(self.center_freq),
                                           self.meng.double(self.distance), channel_model_str, nargout=0)

            self.channel_mat_fd = np.array(
                self.meng.qd_get_channel_mat(rx_transceiver.cord_x, rx_transceiver.cord_y, rx_transceiver.cord_z))

    def __str__(self):
        return "quadriga"

    def get_channel_mat_fd(self) -> ndarray:
        """
        Returns the channel matrix in frequency domain.

        :return: matrix of channel coefficients in frequency domain
        """

        return self.channel_mat_fd

    def calc_channel_mat(self, tx_transceivers: list[Transceiver], rx_transceiver: Transceiver) -> None:
        """
        Calculate/Generate channel matrix based on the channel model string.

        :param tx_transceivers: [unused] list of transceiver objects of the antenna array
        :param rx_transceiver: receiver object
        :return: None
        """
        self.channel_mat_fd = np.array(
            self.meng.qd_get_channel_mat(rx_transceiver.cord_x, rx_transceiver.cord_y, rx_transceiver.cord_z))

    def reroll_channel_coeffs(self, tx_transceivers: list[Transceiver], rx_transceiver: Transceiver) -> None:
        """
        Calculate/Generate new channel matrix based on the channel model string.

        :param tx_transceivers: [unused] list of transceiver objects of the antenna array
        :param rx_transceiver: receiver object
        :return: None
        """
        self.channel_mat_fd = np.array(
            self.meng.qd_get_channel_mat(rx_transceiver.cord_x, rx_transceiver.cord_y, rx_transceiver.cord_z))

    def propagate(self, in_sig_mat: ndarray, sum_signals: bool = True) -> ndarray:
        """
        Multiplies the input signal matrix by the channel matrix.
        Both should be in frequency domain.

        :param in_sig_mat: input signal matrix from antenna array
        :param sum_signals: flag if the output signals should be summed or not, to model single antenna receiver
        :return: signal matrix after the channel propagation
        """

        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum_signals columns
        if sum_signals:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan
