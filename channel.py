import abc

import numpy as np
import scipy as scp
import torch

from utilities import to_db, signal_power


class Channel(metaclass=abc.ABCMeta):
    def __init__(self, snr_db, is_complex, seed=None):
        self.snr_db = snr_db
        self.is_complex = is_complex
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)
        else:
            self.rng_gen = np.random.default_rng(1234)

    def set_snr(self, snr_db):
        self.snr_db = snr_db


# @jit(nopython=True) not useful for such small rng datasets
def _propagate_awgn(is_complex, snr_db, rng_gen, in_sig, avg_sample_pow):
    n_sampl = len(in_sig)
    noise_std = np.complex128(np.sqrt((int(is_complex) + 1) * avg_sample_pow / (10 ** (snr_db / 10))))
    if is_complex:
        noise = rng_gen.standard_normal((n_sampl, 2)).view(np.complex128)[:, 0] * noise_std * 0.5
    else:
        noise = rng_gen.standard_normal(n_sampl) * noise_std
    # check resultant SNR
    # print("SNR: ", to_db(signal_power(in_sig)/signal_power(noise)))

    return in_sig + noise


class AwgnTdTd(Channel):

    def __init__(self, snr_db, is_complex, seed=None):
        super().__init__(snr_db, is_complex, seed)

    def propagate(self, in_sig, avg_sample_pow):
        return _propagate_awgn(self.is_complex, self.snr_db, self.rng_gen, in_sig, avg_sample_pow)


class AwgnMisoLosTdFd(Channel):

    def __init__(self, n_inputs, snr_db, is_complex, seed=None):
        self.n_inputs = n_inputs
        super().__init__(snr_db, is_complex, seed)

    def propagate(self, tx_transceivers, rx_transceiver, in_sig_mat, avg_sample_pow = None, skip_noise=False, skip_attenuation=False):
        # channel in frequency domain
        # remove cp from in sig matrix
        no_cp_td_sig_mat = in_sig_mat[:, rx_transceiver.modem.cp_len:]
        # perform fft row wise
        no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(no_cp_td_sig_mat), norm="ortho").numpy()

        # get carrier frequencies
        sig_freq_vals = torch.fft.fftfreq(rx_transceiver.modem.n_fft, d=1 / rx_transceiver.modem.n_fft).numpy() \
                        * rx_transceiver.carrier_spacing + rx_transceiver.center_freq
        # for each tx to rx get distance
        tx_rx_los_distances = np.empty(len(tx_transceivers))
        tx_ant_gains = np.empty(len(tx_transceivers))

        for idx, tx_transceiver in enumerate(tx_transceivers):
            tx_ant_gains[idx] = tx_transceiver.tx_ant_gain_db
            tx_rx_los_distances[idx] = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                tx_transceiver.cord_z - rx_transceiver.cord_z, 2))

        fd_ph_shift_mat = np.exp(2j * np.pi * np.outer(tx_rx_los_distances, sig_freq_vals) / scp.constants.c)

        # multiply fd signals by attenuation matrix
        if not skip_attenuation:
            fd_freq_att_mat = np.sqrt(np.power(10, (tx_ant_gains[:, np.newaxis] + rx_transceiver.rx_ant_gain_db) / 10)) \
                              * (scp.constants.c / (4 * np.pi * np.outer(tx_rx_los_distances, sig_freq_vals)))
            no_cp_fd_sig_mat = np.multiply(no_cp_fd_sig_mat, fd_freq_att_mat)

        # shift phases of carriers accordingly to spatial relations
        fd_signal_at_point = np.multiply(no_cp_fd_sig_mat, fd_ph_shift_mat)
        # sum columns
        fd_signal_at_point = np.sum(fd_signal_at_point, axis=0)

        # TODO: add noise based on RX noise floor
        if not skip_noise:
            n_sampl = len(in_sig_mat)
            noise_std = np.complex128(np.sqrt((int(self.is_complex) + 1) * avg_sample_pow / (10 ** (self.snr_db / 10))))
            if self.is_complex:
                noise = self.rng_gen.standard_normal((n_sampl, 2)).view(np.complex128)[:, 0] * noise_std * 0.5
            else:
                noise = self.rng_gen.standard_normal(n_sampl) * noise_std

            # check resultant SNR
            # print("SNR: ", to_db(signal_power(in_sig_mat)/signal_power(noise)))

            fd_signal_at_point = np.sum(fd_signal_at_point, noise)

        return fd_signal_at_point


# TODO: Should Rayleigh channel include antenna gains?
class RayleighMisoTdFd(Channel):
    def __init__(self, n_inputs, fd_samp_size, snr_db, is_complex, seed=None):
        self.n_inputs = n_inputs
        self.fd_samp_size = fd_samp_size
        super().__init__(snr_db, is_complex, seed)

        self.set_channel_coeffs()

    def set_channel_coeffs(self, fd_chan_mat=None, avg=0, sigma=1):
        if fd_chan_mat is None:
            # generate rayleigh channel coefficients
            self.fd_chan_mat = self.rng_gen.normal(avg, sigma, size=(self.n_inputs, self.fd_samp_size * 2)).view(
                dtype=np.complex128)
        else:
            self.fd_chan_mat = fd_chan_mat

    def get_channel_coeffs(self):
        return self.fd_chan_mat

    def reroll_channel_coeffs(self, avg=0, sigma=1):
        self.fd_chan_mat = self.rng_gen.normal(avg, sigma, size=(self.n_inputs, self.fd_samp_size * 2)).view(
            dtype=np.complex128)

    def propagate(self, tx_transceivers, rx_transceiver, in_sig_mat, skip_noise=False, skip_attenuation=False):
        # channel in frequency domain
        # remove cp from in sig matrix
        no_cp_td_sig_mat = in_sig_mat[:, rx_transceiver.modem.cp_len:]
        # perform fft row wise
        no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(no_cp_td_sig_mat), norm="ortho").numpy()

        # multiply signal by rayleigh channel coefficients in frequency domain
        fd_sigmat_after_chan = np.multiply(no_cp_fd_sig_mat, self.fd_chan_mat)
        # attenuation is already in channel coefficients
        # TODO: add noise based on RX noise floor
        if not skip_noise:
            pass

        # sum columns
        return np.sum(fd_sigmat_after_chan, axis=0)


class AwgnMisoTwoPathTdFd(Channel):

    def __init__(self, n_inputs, snr_db, is_complex, seed=None):
        self.n_inputs = n_inputs
        super().__init__(snr_db, is_complex, seed)

    def propagate(self, tx_transceivers, rx_transceiver, in_sig_mat, skip_noise=False, skip_attenuation=False):
        # channel in frequency domain
        # remove cp from in sig matrix
        no_cp_td_sig_mat = in_sig_mat[:, rx_transceiver.modem.cp_len:]
        # perform fft row wise
        no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(no_cp_td_sig_mat), norm="ortho").numpy()

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
            # Change of phase, att at reflection point?
            sec_distances[idx] = incident_path_len + reflected_path_len

        los_fd_shift_mat = np.exp(2j * np.pi * np.outer(los_distances, sig_freq_vals) / scp.constants.c)
        # TODO: include detailed calculation of reflection coefficient
        reflection_coeff = -1.0
        sec_fd_shift_mat = reflection_coeff * np.exp(2j * np.pi * np.outer(sec_distances, sig_freq_vals) / scp.constants.c)

        if not skip_attenuation:
            los_fd_att_mat = np.sqrt(np.power(10, (tx_ant_gains[:, np.newaxis] + rx_transceiver.rx_ant_gain_db) / 10)) \
                             * (scp.constants.c / (4 * np.pi * np.outer(los_distances, sig_freq_vals)))
            sec_fd_att_mat = np.sqrt(np.power(10, (tx_ant_gains[:, np.newaxis] + rx_transceiver.rx_ant_gain_db) / 10)) \
                             * (scp.constants.c / (4 * np.pi * np.outer(sec_distances, sig_freq_vals)))

            los_fd_shift_mat = np.multiply(los_fd_shift_mat, los_fd_att_mat)
            sec_fd_shift_mat = np.multiply(sec_fd_shift_mat, sec_fd_att_mat)

        # combine two path coefficients without normalization
        combinded_fd_chan = np.add(los_fd_shift_mat, sec_fd_shift_mat)
        combinded_fd_sig = np.multiply(no_cp_fd_sig_mat, combinded_fd_chan)

        # TODO: add noise based on RX noise floor
        if not skip_noise:
            pass

        # sum columns
        return np.sum(combinded_fd_sig, axis=0)
