import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import objmode
from scipy import special as scp_special

from speedup import jit
from utilities import bitarray2dec, dec2bitarray, td_signal_power


# @jit(nopython=True)
def _modulate(constellation, n_bits_per_symbol, input_bits):
    mapfunc = np.vectorize(lambda i:
                           constellation[bitarray2dec(input_bits[i:i + n_bits_per_symbol])])
    baseband_symbols = mapfunc(np.arange(0, len(input_bits), n_bits_per_symbol))
    return baseband_symbols


# @jit(nopython=True)
def _demodulate(constellation, n_bits_per_symbol, input_symbols):
    index_list = np.abs(input_symbols - constellation[:, None]).argmin(0)
    demod_bits = dec2bitarray(index_list, n_bits_per_symbol)
    return demod_bits


# TODO: Separate constellations for RX and TX to ease the access and use of shrinking coeffs

class Modem:

    def __init__(self, constellation, reorder_as_gray=True):
        # constellation correcting coefficient - alpha
        self.alpha = 1
        self.constel_size = len(constellation)

        if reorder_as_gray:
            # generate gray codes from linear arange
            gray_codes = np.asarray([x ^ (x >> 1) for x in range(self.constel_size)])
            # remap constellation based on gray code indices
            self.constellation = np.array(constellation)[gray_codes.argsort()]

        else:
            self.constellation = constellation

    def modulate(self, input_bits):
        return _modulate(self._constellation, self.n_bits_per_symbol, input_bits)

    def demodulate(self, input_symbols):
        return _demodulate(self._constellation, self.n_bits_per_symbol, input_symbols)

    # TODO: optimize symbol detection
    def symbol_detection(self, input_symbols):
        index_list = np.abs(input_symbols - self.constellation[:, None]).argmin(0)
        return self.constellation[index_list]

    def plot_constellation(self):
        fig, ax = plt.subplots()
        ax.scatter(self.constellation.real, self.constellation.imag)

        for symbol in self.constellation:
            ax.text(symbol.real + .2, symbol.imag, self.demodulate(symbol))

        ax.set_title('Constellation')
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.grid()
        plt.show()

    # correct constellation shrinking in RX (soft limiter case)
    def correct_constellation(self, ibo_db):
        self.alpha = self.calc_alpha(ibo_db)
        self._constellation = self.alpha * self._constellation

    def calc_alpha(self, ibo_db):
        # gamma coefficient in Ochiai paper is a ratio of envelopes not powers!
        gamma = np.power(10, ibo_db / 20)
        alpha = 1 - np.exp(-np.power(gamma, 2)) + (np.sqrt(np.pi) * gamma / 2) * scp_special.erfc(gamma)
        # scale constellation
        return alpha

    def update_alpha(self, ibo_db):
        self.alpha = self.calc_alpha(ibo_db)

    @property
    def constellation(self):
        return self._constellation

    @constellation.setter
    def constellation(self, value):

        n_bits_per_symbol = np.log2(len(value))
        if n_bits_per_symbol != int(n_bits_per_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        self._constellation = np.array(value)
        self.avg_symbol_power = td_signal_power(self.constellation)
        self.constellation_size = self._constellation.size
        self.n_bits_per_symbol = int(n_bits_per_symbol)


class QamModem(Modem):

    def __init__(self, constel_size):
        # check if constellation size generates a square QAM
        n_symb = np.sqrt(constel_size)
        if n_symb != int(n_symb):
            raise ValueError('Constellation size must be a power of 2, only square QAM supported.')

        # generate centered around 0, equally spaced (by 2) PAM symbols, indexing from lower left corner
        pam_symb = np.arange(-n_symb + 1, n_symb, 2)
        # arrange into QAM
        constellation = np.tile(np.hstack((pam_symb, pam_symb[::-1])), int(n_symb) // 2) * 1j + pam_symb.repeat(n_symb)

        super().__init__(constellation)


# @jit(nopython=True)
def _tx_ofdm_symbol(mod_symbols, n_fft: int, n_sub_carr: int, cp_length: int):
    # generate OFDM symbol block - size given by n_sub_carr size

    if len(mod_symbols) != n_sub_carr:
        raise ValueError('mod_symbols length must match n_sub_carr value')

    # skip idx = 0 and fill the carriers from LR
    ofdm_sym_freq = np.zeros(n_fft, dtype=np.complex128)

    ofdm_sym_freq[-(n_sub_carr // 2):] = mod_symbols[0:n_sub_carr // 2]
    ofdm_sym_freq[1:(n_sub_carr // 2) + 1] = mod_symbols[n_sub_carr // 2:]
    tmp = (np.fft.fftfreq(n_fft))

    with objmode(ofdm_sym_time='complex128[:]'):
        ofdm_sym_time = torch.fft.ifft(torch.from_numpy(ofdm_sym_freq), norm="ortho").numpy()

    # add cyclic prefix
    return np.concatenate((ofdm_sym_time[-cp_length:], ofdm_sym_time))


# TODO: add input signal domain flag freq/time to skip fft
# @jit(nopython=True)
def _rx_ofdm_symbol(ofdm_symbol, n_fft: int, n_sub_carr: int, cp_length: int):
    # decode OFDM symbol block - size given by n_sub_carr size
    with objmode(ofdm_sym_freq='complex128[:]'):
        # skip cyclic prefix
        ofdm_sym_freq = torch.fft.fft(torch.from_numpy(ofdm_symbol[cp_length:]), norm="ortho").numpy()

    # extract and rearange data from LR
    return np.concatenate((ofdm_sym_freq[-n_sub_carr // 2:], ofdm_sym_freq[1:(n_sub_carr // 2) + 1]))


class OfdmQamModem(QamModem):

    def __init__(self, constel_size: int, n_fft: int, n_sub_carr: int, cp_len: int, n_users=1):
        super().__init__(constel_size)

        self.n_fft = n_fft
        self.n_sub_carr = n_sub_carr
        self.cp_len = cp_len
        self.n_bits_per_ofdm_sym = int(np.log2(constel_size) * n_sub_carr)
        self.avg_sample_power = self.ofdm_avg_sample_pow()
        self.precoding_mat = None
        self.n_users = n_users

    def set_precoding(self, precoding_mat):
        # each row represents channel to each user
        self.precoding_mat = precoding_mat

    def precode_symbols(self, in_symbols, precoding_mat=None):
        # apply precoding if any
        if precoding_mat is not None:
            return np.multiply(in_symbols, precoding_mat)
        else:
            return in_symbols

    def modulate(self, input_bits, get_symbols_only=False, separate_usrs_sig=False):
        if self.n_users == 1:
            modulated_symbols = _modulate(self._constellation, self.n_bits_per_symbol, input_bits)
            if get_symbols_only:
                return modulated_symbols
            else:
                precoded_symbols = np.squeeze(self.precode_symbols(modulated_symbols, self.precoding_mat))
                return _tx_ofdm_symbol(precoded_symbols, self.n_fft, self.n_sub_carr, self.cp_len)
        else:
            modulated_symbols = np.empty((self.n_users, self.n_sub_carr), dtype=np.complex128)
            for user_idx in range(self.n_users):
                modulated_symbols[user_idx, :] = _modulate(self._constellation, self.n_bits_per_symbol,
                                                           input_bits[user_idx, :])
            if separate_usrs_sig:
                mu_symbols = self.precode_symbols(modulated_symbols, self.precoding_mat)
                # return _tx_ofdm_symbol(combined_mu_symbols, self.n_fft, self.n_sub_carr, self.cp_len)
                tx_sig_per_usr = []
                for usr_idx in range(self.n_users):
                    tx_sig_per_usr.append(_tx_ofdm_symbol(mu_symbols[usr_idx, :], self.n_fft, self.n_sub_carr, self.cp_len))
                return tx_sig_per_usr

            if get_symbols_only:
                return modulated_symbols
            else:
                combined_mu_symbols = np.sum(self.precode_symbols(modulated_symbols, self.precoding_mat), axis=0)
                return _tx_ofdm_symbol(combined_mu_symbols, self.n_fft, self.n_sub_carr, self.cp_len)

    def demodulate(self, ofdm_symbol, get_symbols_only=False):
        baseband_symbols = _rx_ofdm_symbol(ofdm_symbol, self.n_fft, self.n_sub_carr, self.cp_len)
        if get_symbols_only:
            return baseband_symbols
        else:
            return _demodulate(self._constellation, self.n_bits_per_symbol, baseband_symbols)

    def symbols_to_bits(self, input_symbols):
        return _demodulate(self._constellation, self.n_bits_per_symbol, input_symbols)

    def ofdm_avg_sample_pow(self):
        return self.avg_symbol_power * (self.n_sub_carr / self.n_fft)
