import matplotlib.pyplot as plt
from numpy import arange, array, zeros, sqrt, log2, tile, exp, log, vectorize, \
     abs, asarray, hstack, concatenate
from numpy.fft import fft, ifft

from utilities import bitarray2dec, dec2bitarray, signal_power


class Modem:

    def __init__(self, constellation, reorder_as_gray=True):

        if reorder_as_gray:
            constel_size = len(constellation)
            # generate gray codes from linear arange
            gray_codes = asarray([x ^ (x >> 1) for x in range(constel_size)])
            # remap constellation based on gray code indices
            self.constellation = array(constellation)[gray_codes.argsort()]

        else:
            self.constellation = constellation

    def modulate(self, input_bits):

        mapfunc = vectorize(lambda i:
                            self._constellation[bitarray2dec(input_bits[i:i + self.n_bits_per_symbol])])

        baseband_symbols = mapfunc(arange(0, len(input_bits), self.n_bits_per_symbol))

        return baseband_symbols

    def demodulate(self, input_symbols, demod_type="hard", noise_var=0):

        if demod_type == 'hard':
            index_list = abs(input_symbols - self._constellation[:, None]).argmin(0)
            demod_bits = dec2bitarray(index_list, self.n_bits_per_symbol)

        elif demod_type == 'soft':
            demod_bits = zeros(len(input_symbols) * self.n_bits_per_symbol)
            for i in arange(len(input_symbols)):
                current_symbol = input_symbols[i]
                for bit_index in arange(self.n_bits_per_symbol):
                    llr_num = 0
                    llr_den = 0
                    for bit_value, symbol in enumerate(self._constellation):
                        if (bit_value >> bit_index) & 1:
                            llr_num += exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                        else:
                            llr_den += exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                    demod_bits[i * self.n_bits_per_symbol + self.n_bits_per_symbol - 1 - bit_index] = log(
                        llr_num / llr_den)
        else:
            raise ValueError('demod_type must be "hard" or "soft"')

        return demod_bits

    def plot_constellation(self):
        fig, ax = plt.subplots()
        ax.scatter(self.constellation.real, self.constellation.imag)

        for symbol in self.constellation:
            ax.text(symbol.real + .2, symbol.imag, self.demodulate(symbol, 'hard'))

        ax.set_title('Constellation')
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.grid()
        plt.show()

    @property
    def constellation(self):
        return self._constellation

    @constellation.setter
    def constellation(self, value):

        n_bits_per_symbol = log2(len(value))
        if n_bits_per_symbol != int(n_bits_per_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        self._constellation = array(value)

        self.avg_symbol_power = signal_power(self.constellation)
        self.constellation_size = self._constellation.size
        self.n_bits_per_symbol = int(n_bits_per_symbol)


class QamModem(Modem):

    def __init__(self, constel_size):
        # check if constellation size generates a square QAM
        n_symb = sqrt(constel_size)
        if n_symb != int(n_symb):
            raise ValueError('Constellation size must be a power of 2, only square QAM supported.')

        # generate centered around 0, equally spaced (by 2) PAM symbols, indexing from lower left corner
        pam_symb = arange(-n_symb + 1, n_symb, 2)
        # arrange into QAM
        constellation = tile(hstack((pam_symb, pam_symb[::-1])), int(n_symb) // 2) * 1j + pam_symb.repeat(n_symb)

        super().__init__(constellation)


def tx_ofdm_symbol(mod_symbols, n_fft: int, n_sub_carr: int, cp_length: int):
    # generate OFDM symbol block - size given by n_sub_carr size
    if len(mod_symbols) != n_sub_carr:
        raise ValueError('mod_symbols length must match n_sub_carr value')

    # skip idx = 0 and fill the carriers from LR
    ofdm_sym_freq = zeros(n_fft, dtype=complex)
    ofdm_sym_freq[1:(n_sub_carr // 2) + 1] = mod_symbols[n_sub_carr // 2:]
    ofdm_sym_freq[-(n_sub_carr // 2):] = mod_symbols[0:n_sub_carr // 2]
    ofdm_sym_time = ifft(ofdm_sym_freq, norm="ortho")

    if cp_length != 0:
        cyclic_prefix = ofdm_sym_time[-cp_length:]
    else:
        cyclic_prefix = []

    # add cyclic prefix
    return concatenate((cyclic_prefix, ofdm_sym_time))


def rx_ofdm_symbol(ofdm_symbol, n_fft: int, n_sub_carr: int, cp_length: int):
    # decode OFDM symbol block - size given by n_sub_carr size

    # skip cyclic prefix
    ofdm_sym_freq = fft(ofdm_symbol[cp_length:], norm="ortho")
    # extract and rearange data from LR boundaries
    return concatenate((ofdm_sym_freq[-n_sub_carr // 2:], ofdm_sym_freq[1:(n_sub_carr // 2) + 1]))

