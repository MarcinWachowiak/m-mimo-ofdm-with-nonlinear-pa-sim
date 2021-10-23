from bisect import insort

import matplotlib.pyplot as plt
from numpy import arange, array, zeros, pi, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron, inf, full, abs, newaxis, minimum, clip, fromiter
from numpy.fft import fft, ifft
from numpy.linalg import qr, norm
from sympy.combinatorics.graycode import GrayCode

from utilities import bitarray2dec, dec2bitarray, signal_power


class Modem:

    def __init__(self, constellation, reorder_as_gray=True):

        if reorder_as_gray:
            m = log2(len(constellation))
            gray_code_sequence = GrayCode(m).generate_gray()
            gray_code_sequence_array = fromiter((int(g, 2) for g in gray_code_sequence), int, len(constellation))
            self.constellation = array(constellation)[gray_code_sequence_array.argsort()]

        else:
            self.constellation = constellation

    def modulate(self, input_bits):

        mapfunc = vectorize(lambda i:
                            self._constellation[bitarray2dec(input_bits[i:i + self.num_bits_symbol])])

        baseband_symbols = mapfunc(arange(0, len(input_bits), self.num_bits_symbol))

        return baseband_symbols

    def demodulate(self, input_symbols, demod_type="hard", noise_var=0):

        if demod_type == 'hard':
            index_list = abs(input_symbols - self._constellation[:, None]).argmin(0)
            demod_bits = dec2bitarray(index_list, self.num_bits_symbol)

        elif demod_type == 'soft':
            demod_bits = zeros(len(input_symbols) * self.num_bits_symbol)
            for i in arange(len(input_symbols)):
                current_symbol = input_symbols[i]
                for bit_index in arange(self.num_bits_symbol):
                    llr_num = 0
                    llr_den = 0
                    for bit_value, symbol in enumerate(self._constellation):
                        if (bit_value >> bit_index) & 1:
                            llr_num += exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                        else:
                            llr_den += exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                    demod_bits[i * self.num_bits_symbol + self.num_bits_symbol - 1 - bit_index] = log(llr_num / llr_den)
        else:
            raise ValueError('demod_type must be "hard" or "soft"')

        return demod_bits

    def plot_constellation(self):
        """ Plot the constellation """
        fig, ax = plt.subplots()
        ax.scatter(self.constellation.real, self.constellation.imag)

        # for symb in self.constellation:
        #     plt.text(symb.real + .2, symb.imag, self.demodulate(symb, 'hard'))
        print(self.constellation)

        ax.set_title('Constellation')
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid()
        plt.show()

    @property
    def constellation(self):
        """ Constellation of the modem. """
        return self._constellation

    @constellation.setter
    def constellation(self, value):
        # Check value input
        num_bits_symbol = log2(len(value))
        if num_bits_symbol != int(num_bits_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        # Set constellation as an array
        self._constellation = array(value)

        # Update other attributes
        self.Es = signal_power(self.constellation)
        self.m = self._constellation.size
        self.num_bits_symbol = int(num_bits_symbol)


class QAMModem(Modem):
    """ Creates a Quadrature Amplitude Modulation (QAM) Modem object.

        Parameters
        ----------
        m : int
            Size of the PSK constellation.

        Attributes
        ----------
        constellation : 1D-ndarray of complex
                        Modem constellation. If changed, the length of the new constellation must be a power of 2.

        Es            : float
                        Average energy per symbols.

        m             : integer
                        Constellation length.

        num_bits_symb : integer
                        Number of bits per symbol.

        Raises
        ------
        ValueError
                        If the constellation is changed to an array-like with length that is not a power of 2.
                        If the parameter m would lead to an non-square QAM during initialization.
    """

    def __init__(self, m):
        """ Creates a Quadrature Amplitude Modulation (QAM) Modem object.

        Parameters
        ----------
        m : int
            Size of the QAM constellation. Must lead to a square QAM (ie sqrt(m) is an integer).

        Raises
        ------
        ValueError
                        If m would lead to an non-square QAM.
        """

        num_symb_pam = sqrt(m)
        if num_symb_pam != int(num_symb_pam):
            raise ValueError('m must lead to a square QAM.')

        pam = arange(-num_symb_pam + 1, num_symb_pam, 2)
        constellation = tile(hstack((pam, pam[::-1])), int(num_symb_pam) // 2) * 1j + pam.repeat(num_symb_pam)
        super().__init__(constellation)


def ofdm_tx(x, nfft, nsc, cp_length):
    """ OFDM Transmit signal generation """

    nfft = float(nfft)
    nsc = float(nsc)
    cp_length = float(cp_length)
    ofdm_tx_signal = array([])

    for i in range(0, shape(x)[1]):
        symbols = x[:, i]
        ofdm_sym_freq = zeros(nfft, dtype=complex)
        ofdm_sym_freq[1:(nsc / 2) + 1] = symbols[nsc / 2:]
        ofdm_sym_freq[-(nsc / 2):] = symbols[0:nsc / 2]
        ofdm_sym_time = ifft(ofdm_sym_freq)
        cp = ofdm_sym_time[-cp_length:]
        ofdm_tx_signal = concatenate((ofdm_tx_signal, cp, ofdm_sym_time))

    return ofdm_tx_signal


def ofdm_rx(y, nfft, nsc, cp_length):
    """ OFDM Receive Signal Processing """

    num_ofdm_symbols = int(len(y) / (nfft + cp_length))
    x_hat = zeros([nsc, num_ofdm_symbols], dtype=complex)

    for i in range(0, num_ofdm_symbols):
        ofdm_symbol = y[i * nfft + (i + 1) * cp_length:(i + 1) * (nfft + cp_length)]
        symbols_freq = fft(ofdm_symbol)
        x_hat[:, i] = concatenate((symbols_freq[-nsc / 2:], symbols_freq[1:(nsc / 2) + 1]))

    return x_hat
