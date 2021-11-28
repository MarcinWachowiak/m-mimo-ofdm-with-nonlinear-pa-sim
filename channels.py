import numpy as np
from utilities import to_db, signal_power
from speedup import jit
import torch
import scipy as scp


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


class AwgnMiso:

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
        td_ph_shift_mat = np.empty(no_cp_td_sig_mat.shape, dtype=np.complex128)
        fd_freq_att_mat = np.empty(no_cp_td_sig_mat.shape, dtype=np.complex128)
        for idx, tx_transceiver in enumerate(tx_transceivers):
            tx_rx_distance = np.sqrt(np.power(tx_transceiver.cord_x - rx_transceiver.cord_x, 2) + np.power(
                tx_transceiver.cord_y - rx_transceiver.cord_y, 2) + np.power(
                tx_transceiver.cord_z - rx_transceiver.cord_z, 2))
            sig_freq_vals = torch.fft.fftfreq(tx_transceivers[idx].modem.n_fft,
                                              d=1 / tx_transceivers[idx].modem.n_fft).numpy() * tx_transceivers[
                                idx].carrier_spacing + tx_transceivers[idx].center_freq

            td_ph_shift_mat[idx, :] = np.exp(2j * np.pi * tx_rx_distance * sig_freq_vals / scp.constants.c)
            fd_freq_att_mat[idx, :] = np.sqrt(np.power(10, (tx_transceiver.tx_ant_gain_db + rx_transceiver.rx_ant_gain_db)/10)) \
                                      * (scp.constants.c/(4 * np.pi * tx_rx_distance * sig_freq_vals))
        # multiply fd signals by attenuation matrix
        if not skip_attenuation:
            no_cp_fd_sig_mat = np.multiply(no_cp_fd_sig_mat, fd_freq_att_mat)

        fd_ph_shift_mat = torch.fft.fft(torch.from_numpy(td_ph_shift_mat), norm="ortho").numpy()
        # normalize not to introduce additional attenuation
        fd_ph_shift_mat = fd_ph_shift_mat / np.abs(fd_ph_shift_mat)
        sig_at_point_mat = np.multiply(no_cp_fd_sig_mat, fd_ph_shift_mat)

        # TODO: add noise based on RX noise floor
        if not skip_noise:
            pass

        # sum columns
        return np.sum(sig_at_point_mat, axis=0)
