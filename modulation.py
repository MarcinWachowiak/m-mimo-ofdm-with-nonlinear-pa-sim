import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import torch
from utilities import bitarray2dec, dec2bitarray, signal_power
from speedup import jit
from numba import objmode, types


def _modulate(constellation, n_bits_per_symbol, input_bits):
    mapfunc = np.vectorize(lambda i:
                           constellation[bitarray2dec(input_bits[i:i + n_bits_per_symbol])])
    baseband_symbols = mapfunc(np.arange(0, len(input_bits), n_bits_per_symbol))

    return baseband_symbols


def _demodulate(constellation, n_bits_per_symbol, input_symbols):
    index_list = np.abs(input_symbols - constellation[:, None]).argmin(0)
    demod_bits = dec2bitarray(index_list, n_bits_per_symbol)

    return demod_bits


class Modem:

    def __init__(self, constellation, reorder_as_gray=True):
        # constellation correcting coefficient - alpha
        self.alpha = 1

        if reorder_as_gray:
            constel_size = len(constellation)
            # generate gray codes from linear arange
            gray_codes = np.asarray([x ^ (x >> 1) for x in range(constel_size)])
            # remap constellation based on gray code indices
            self.constellation = np.array(constellation)[gray_codes.argsort()]

        else:
            self.constellation = constellation

    def modulate(self, input_bits):
        return _modulate(self._constellation, self.n_bits_per_symbol, input_bits)

    def demodulate(self, input_symbols):
        return _demodulate(self._constellation, self.n_bits_per_symbol, input_symbols)

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

    def correct_constellation(self, ibo_db):
        gamma = np.power(10, ibo_db / 10)
        alpha = 1 - np.exp(-np.power(gamma, 2)) + (np.sqrt(np.pi) * gamma/ 2) * scp.special.erfc(gamma)
        #scale constellation
        self.alpha = alpha
        self._constellation = alpha * self._constellation

    @property
    def constellation(self):
        return self._constellation

    @constellation.setter
    def constellation(self, value):

        n_bits_per_symbol = np.log2(len(value))
        if n_bits_per_symbol != int(n_bits_per_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        self._constellation = np.array(value)
        self.avg_symbol_power = signal_power(self.constellation)
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


@jit(nopython=True)
def tx_ofdm_symbol(mod_symbols, n_fft: int, n_sub_carr: int, cp_length: int):
    # generate OFDM symbol block - size given by n_sub_carr size
    if len(mod_symbols) != n_sub_carr:
        raise ValueError('mod_symbols length must match n_sub_carr value')

    # skip idx = 0 and fill the carriers from LR
    ofdm_sym_freq = np.zeros(n_fft, dtype=np.complex128)
    ofdm_sym_freq[1:(n_sub_carr // 2) + 1] = mod_symbols[n_sub_carr // 2:]
    ofdm_sym_freq[-(n_sub_carr // 2):] = mod_symbols[0:n_sub_carr // 2]

    with objmode(ofdm_sym_time='complex128[:]'):
        ofdm_sym_time = torch.fft.ifft(torch.from_numpy(ofdm_sym_freq), norm="ortho").numpy()

    # add cyclic prefix
    return np.concatenate((ofdm_sym_time[-cp_length:], ofdm_sym_time))

@jit(nopython=True)
def rx_ofdm_symbol(ofdm_symbol, n_fft: int, n_sub_carr: int, cp_length: int):
    # decode OFDM symbol block - size given by n_sub_carr size
    with objmode(ofdm_sym_freq='complex128[:]'):
        # skip cyclic prefix
        ofdm_sym_freq = torch.fft.fft(torch.from_numpy(ofdm_symbol[cp_length:]), norm="ortho").numpy()

    # extract and rearange data from LR boundaries
    return np.concatenate((ofdm_sym_freq[-n_sub_carr // 2:], ofdm_sym_freq[1:(n_sub_carr // 2) + 1]))
