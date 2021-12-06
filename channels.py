import numpy as np
from utilities import to_db, signal_power
from speedup import jit
import torch
import scipy as scp

import matplotlib.pyplot as plt

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


class Awgn:

    def __init__(self, snr_db, is_complex, seed=None):
        self.snr_db = snr_db
        self.is_complex = is_complex
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def propagate(self, in_sig, avg_sample_pow):
        return _propagate_awgn(self.is_complex, self.snr_db, self.rng_gen, in_sig, avg_sample_pow)


class AwgnMisoLos:

    def __init__(self, n_inputs, snr_db, is_complex, seed=None):
        self.snr_db = snr_db
        self.is_complex = is_complex
        self.n_inputs = n_inputs
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def propagate(self, tx_transceivers, rx_transceiver, in_sig_mat, skip_noise=False, skip_attenuation=False):
        # channel in frequency domain
        # remove cp from in sig matrix
        no_cp_td_sig_mat = in_sig_mat[:, tx_transceivers[0].modem.cp_len:]
        # perform fft row wise
        no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(no_cp_td_sig_mat), norm="ortho").numpy()
        # for each tx get distance
        fd_ph_shift_mat = np.empty(no_cp_fd_sig_mat.shape, dtype=np.complex128)
        fd_freq_att_mat = np.empty(no_cp_fd_sig_mat.shape, dtype=np.complex128)

        for idx, tx_transceiver in enumerate(tx_transceivers):
            tx_rx_distance = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                tx_transceiver.cord_z - rx_transceiver.cord_z, 2))
            sig_freq_vals = torch.fft.fftfreq(tx_transceivers[idx].modem.n_fft,
                                              d=1 / tx_transceivers[idx].modem.n_fft).numpy() * tx_transceivers[
                                idx].carrier_spacing + tx_transceivers[idx].center_freq

            fd_ph_shift_mat[idx, :] = np.exp(2j * np.pi * tx_rx_distance * (sig_freq_vals / scp.constants.c))
            fd_freq_att_mat[idx, :] = np.sqrt(np.power(10, (tx_transceiver.tx_ant_gain_db + rx_transceiver.rx_ant_gain_db)/10)) \
                                        * (scp.constants.c/(4 * np.pi * tx_rx_distance * sig_freq_vals))
        # multiply fd signals by attenuation matrix
        if not skip_attenuation:
            no_cp_fd_sig_mat = np.multiply(no_cp_fd_sig_mat, fd_freq_att_mat)

        #shift phases of carriers accordingly to spatial relations
        fd_signal_at_point = np.multiply(no_cp_fd_sig_mat, fd_ph_shift_mat)

        # TODO: add noise based on RX noise floor
        if not skip_noise:
            pass

        # sum columns
        return np.sum(fd_signal_at_point, axis=0)

class RayleighMiso:
    def __init__(self, n_inputs, fd_samp_size, snr_db, is_complex, seed=None):
        self.snr_db = snr_db
        self.is_complex = is_complex
        self.n_inputs = n_inputs
        self.fd_samp_size = fd_samp_size
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)
        else:
            # generate seed based on something
            # TODO: add seed generation routine
            pass

        self.set_channel_coeffs()

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def set_channel_coeffs(self, fd_chan_mat=None, avg=0, sigma=1):
        if fd_chan_mat is None:
            # generate rayleigh channel coefficients
            self.fd_chan_mat = self.rng_gen.normal(avg, sigma, size=(self.n_inputs, self.fd_samp_size * 2)).view(dtype=np.complex128)
        else:
            self.fd_chan_mat = fd_chan_mat

    def get_channel_coeffs(self):
        return self.fd_chan_mat

    def reroll_channel_coeffs(self, avg=0, sigma=1):
        self.fd_chan_mat = self.rng_gen.normal(avg, sigma, size=(self.n_inputs, self.fd_samp_size * 2)).view(dtype=np.complex128)

    def propagate(self, tx_transceivers, rx_transceiver, in_sig_mat, skip_noise=False, skip_attenuation=False):
        # channel in frequency domain
        # remove cp from in sig matrix
        no_cp_td_sig_mat = in_sig_mat[:, tx_transceivers[0].modem.cp_len:]
        # perform fft row wise
        no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(no_cp_td_sig_mat), norm="ortho").numpy()

        #multiply signal by rayleigh channel coefficients in frequency domain
        fd_sigmat_after_chan = np.multiply(no_cp_fd_sig_mat, self.fd_chan_mat)

        # multiply fd signals by attenuation matrix
        if not skip_attenuation:
            fd_freq_att_mat = np.empty(no_cp_fd_sig_mat.shape, dtype=np.complex128)
            for idx, tx_transceiver in enumerate(tx_transceivers):
                tx_rx_distance = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                    tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                    tx_transceiver.cord_z - rx_transceiver.cord_z, 2))
                sig_freq_vals = torch.fft.fftfreq(tx_transceivers[idx].modem.n_fft,
                                                  d=1 / tx_transceivers[idx].modem.n_fft).numpy() * tx_transceivers[
                                    idx].carrier_spacing + tx_transceivers[idx].center_freq
                fd_freq_att_mat[idx, :] = np.sqrt(
                    np.power(10, (tx_transceiver.tx_ant_gain_db + rx_transceiver.rx_ant_gain_db) / 10)) \
                                          * (scp.constants.c / (4 * np.pi * tx_rx_distance * sig_freq_vals))
            fd_sigmat_after_chan = np.multiply(fd_sigmat_after_chan, fd_freq_att_mat)

        # TODO: add noise based on RX noise floor
        if not skip_noise:
            pass

        # sum columns
        return np.sum(fd_sigmat_after_chan, axis=0)


class AwgnMisoTwoPath:

    def __init__(self, n_inputs, snr_db, is_complex, seed=None):
        self.snr_db = snr_db
        self.is_complex = is_complex
        self.n_inputs = n_inputs
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def propagate(self, tx_transceivers, rx_transceiver, in_sig_mat, skip_noise=False, skip_attenuation=False):
        # channel in frequency domain
        # remove cp from in sig matrix
        no_cp_td_sig_mat = in_sig_mat[:, tx_transceivers[0].modem.cp_len:]
        # perform fft row wise
        no_cp_fd_sig_mat = torch.fft.fft(torch.from_numpy(no_cp_td_sig_mat), norm="ortho").numpy()
        #LOS path
        los_fd_shift_mat = np.empty(no_cp_fd_sig_mat.shape, dtype=np.complex128)
        los_fd_att_mat = np.empty(no_cp_fd_sig_mat.shape, dtype=np.complex128)
        #Second path
        sec_fd_shift_mat = np.empty(no_cp_fd_sig_mat.shape, dtype=np.complex128)
        sec_fd_att_mat = np.empty(no_cp_fd_sig_mat.shape, dtype=np.complex128)

        for idx, tx_transceiver in enumerate(tx_transceivers):
            sig_freq_vals = torch.fft.fftfreq(tx_transceivers[idx].modem.n_fft,
                                              d=1 / tx_transceivers[idx].modem.n_fft).numpy() * tx_transceivers[
                                idx].carrier_spacing + tx_transceivers[idx].center_freq

            los_distance = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                tx_transceiver.cord_z - rx_transceiver.cord_z, 2))

            los_fd_shift_mat[idx, :] = np.exp(2j * np.pi * los_distance * (sig_freq_vals / scp.constants.c))
            los_fd_att_mat[idx, :] = np.sqrt(np.power(10, (tx_transceiver.tx_ant_gain_db + rx_transceiver.rx_ant_gain_db)/10)) \
                                        * (scp.constants.c/(4 * np.pi * los_distance * sig_freq_vals))


            dim_ratio = (tx_transceiver.cord_z+rx_transceiver.cord_z)/(np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                tx_transceiver.cord_y - rx_transceiver.cord_y, 2)))
            # angle of elevation (angle in relation to the ground plane) = 90 deg - angle of incidence
            angle_of_elev_rad = np.arctan(dim_ratio)

            second_path_len = rx_transceiver.cord_z/np.sin(angle_of_elev_rad)
            first_path_len = tx_transceiver.cord_z/np.sin(angle_of_elev_rad)
            # Change of phase, att at reflection point?
            sec_fd_shift_mat[idx, :] = np.exp(2j * np.pi * (first_path_len+second_path_len) * (sig_freq_vals / scp.constants.c))
            #total path 1 + 2 attenuation
            sec_fd_att_mat[idx, :] = np.sqrt(np.power(10, (tx_transceiver.tx_ant_gain_db + rx_transceiver.rx_ant_gain_db) / 10)) \
                                     * (scp.constants.c / (4 * np.pi * (first_path_len+second_path_len) * sig_freq_vals))

        #shift phases of carriers accordingly to spatial relations
        if not skip_attenuation:
            los_fd_shift_mat = np.multiply(los_fd_shift_mat, los_fd_att_mat)
            sec_fd_shift_mat = np.multiply(sec_fd_shift_mat, sec_fd_att_mat)

        combinded_fd_chan = np.add(los_fd_shift_mat, sec_fd_shift_mat)

        combinded_fd_sig = np.multiply(no_cp_fd_sig_mat, combinded_fd_chan)

        # TODO: add noise based on RX noise floor
        if not skip_noise:
            pass

        # sum columns
        return np.sum(combinded_fd_sig, axis=0)