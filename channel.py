import numpy as np
import torch
from scipy import constants as scp_constants
import matlab.engine


class MisoLosFd:
    def __init__(self):
        self.channel_mat_fd = None

    def __str__(self):
        return "los"

    def get_channel_mat_fd(self):
        return self.channel_mat_fd

    def calc_channel_mat(self, tx_transceivers, rx_transceiver, skip_attenuation=False):
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

    def propagate(self, in_sig_mat, sum=True):
        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum columns
        if sum:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan


class MisoTwoPathFd:
    def __str__(self):
        return "two_path"

    def get_channel_mat_fd(self):
        return self.channel_mat_fd

    def calc_channel_mat(self, tx_transceivers, rx_transceiver, skip_attenuation=False):
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
        # TODO: include detailed calculation of reflection coefficient
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

    def propagate(self, in_sig_mat, sum=True):
        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum columns
        if sum:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan


class MisoRayleighFd:
    def __init__(self, tx_transceivers, rx_transceiver, seed=None):
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

        self.set_channel_mat_fd()

    def __str__(self):
        return "rayleigh"

    def set_channel_mat_fd(self, channel_mat_fd=None, skip_attenuation=False):
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

    def get_channel_mat_fd(self):
        return self.channel_mat_fd

    def reroll_channel_coeffs(self, skip_attenuation=False):
        fd_rayleigh_coeffs = self.rng_gen.standard_normal(size=(self.n_inputs, self.fd_samp_size * 2)).view(
            dtype=np.complex128) / np.sqrt(2.0)
        if skip_attenuation:
            self.channel_mat_fd = fd_rayleigh_coeffs
        else:
            self.channel_mat_fd = np.multiply(fd_rayleigh_coeffs, self.los_fd_att_mat)

    def propagate(self, in_sig_mat, sum=True):
        # channel in frequency domain
        # multiply signal by rayleigh channel coefficients in frequency domain
        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum columns
        if sum:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan


class MisoRandomPathsFd:
    def __init__(self, tx_transceivers, rx_transceiver, seed=None, n_paths=10, max_delay_spread=1000e-9):
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

    def get_channel_mat_fd(self):
        return self.channel_mat_fd

    def reroll_channel_coeffs(self, tx_transceivers):
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

    def propagate(self, in_sig_mat, sum=True):
        # channel in frequency domain
        # multiply signal by rayleigh channel coefficients in frequency domain
        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum columns
        if sum:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan

class MisoQuadrigaFd:
    def __init__(self, tx_transceivers, rx_transceiver, matlab_engine):
        self.matlab_engine = matlab_engine
        self.channel_mat_fd = np.array(self.matlab_engine.qd_get_channel_mat(rx_transceiver.cord_x, rx_transceiver.cord_y, rx_transceiver.cord_z))

    def __str__(self):
        return "quadriga"

    def get_channel_mat_fd(self):
        return self.channel_mat_fd

    def calc_channel_mat(self, tx_transceivers, rx_transceiver):
        self.channel_mat_fd = np.array(self.matlab_engine.qd_get_channel_mat(rx_transceiver.cord_x, rx_transceiver.cord_y, rx_transceiver.cord_z))

    def reroll_channel_coeffs(self, tx_transceivers, rx_transceiver):
        self.channel_mat_fd = np.array(self.matlab_engine.qd_get_channel_mat(rx_transceiver.cord_x, rx_transceiver.cord_y, rx_transceiver.cord_z))

    def propagate(self, in_sig_mat, sum=True):
        # channel in frequency domain
        # multiply signal by rayleigh channel coefficients in frequency domain
        fd_sigmat_after_chan = np.multiply(in_sig_mat, self.channel_mat_fd)
        # sum columns
        if sum:
            return np.sum(fd_sigmat_after_chan, axis=0)
        else:
            return fd_sigmat_after_chan