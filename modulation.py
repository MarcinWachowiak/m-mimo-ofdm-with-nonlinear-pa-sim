import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import objmode
from scipy import special as scp_special
from numpy import ndarray

from speedup import jit
from utilities import bitarray2dec, dec2bitarray, td_signal_power


# @jit(nopython=True)
def modulate(constellation: ndarray, n_bits_per_symbol: int, input_bits: ndarray) -> ndarray:
    """
    Modulate the input bits into complex baseband symbols.

    :param constellation: vector of the reference complex modulation symbols - constellation
    :param n_bits_per_symbol: number of bits per symbol
    :param input_bits: vector of input bits to be mapped
    :return: complex baseband symbols
    """
    mapfunc = np.vectorize(lambda i:
                           constellation[bitarray2dec(input_bits[i:i + n_bits_per_symbol])])
    baseband_symbols = mapfunc(np.arange(0, len(input_bits), n_bits_per_symbol))
    return baseband_symbols


# Numba does significantly speed up this function
@jit(nopython=True)
def soft_decoding(constellation: ndarray, n_bits_per_symbol: int, input_symbols: ndarray,
                  noise_var_vec: ndarray) -> ndarray:
    """
    Perform soft-decision decoding of the received symbols, log-likelihood (LLR) demodulation.

    :param constellation: vector of the reference complex modulation symbols - constellation
    :param n_bits_per_symbol: number of bits per symbol
    :param input_symbols: received complex baseband input symbols
    :param noise_var_vec: vector specifying noise variance per each symbol (commonly a constant value across all symbols)
    :return: vector of LLR of the bits
    """
    demod_bits = np.zeros(len(input_symbols) * n_bits_per_symbol)
    for i in np.arange(len(input_symbols)):
        current_symbol = input_symbols[i]
        for bit_index in np.arange(n_bits_per_symbol):
            llr_num = 0
            llr_den = 0
            for bit_value, symbol in enumerate(constellation):
                if (bit_value >> bit_index) & 1:
                    llr_num += np.exp((-np.abs(current_symbol - symbol) ** 2) / noise_var_vec[i])
                else:
                    llr_den += np.exp((-np.abs(current_symbol - symbol) ** 2) / noise_var_vec[i])

            if np.real(llr_den) == 0:
                demod_bits[i * n_bits_per_symbol + n_bits_per_symbol - 1 - bit_index] = np.inf
            else:
                demod_bits[i * n_bits_per_symbol + n_bits_per_symbol - 1 - bit_index] = np.log(
                    np.abs(llr_num) / np.abs(llr_den))

    return demod_bits


# @jit(nopython=True)
def demodulate(constellation: ndarray, n_bits_per_symbol: int, input_symbols: ndarray, soft: bool = False,
               noise_var: float = 0.0) -> ndarray:
    """
    Demodulate the baseband symbols into data bits.

    :param constellation: vector of the reference complex modulation symbols - constellation
    :param n_bits_per_symbol: number of bits per symbol
    :param input_symbols: received complex baseband input symbols
    :param soft: flag if the detection should be soft or hard
    :param noise_var: vector specifying noise variance per each symbol (commonly a constant value across all symbols)
    :return: demodulated data bits vector (soft or hard detected)
    """
    if not soft:
        index_list = np.abs(input_symbols - constellation[:, None]).argmin(0)
        demod_bits = dec2bitarray(index_list, n_bits_per_symbol)
    else:
        if not isinstance(noise_var, np.ndarray):
            noise_var_vec = np.repeat(noise_var, len(input_symbols))
            demod_bits = soft_decoding(constellation=constellation, n_bits_per_symbol=n_bits_per_symbol,
                                       input_symbols=input_symbols,
                                       noise_var_vec=noise_var_vec)
        else:
            demod_bits = soft_decoding(constellation=constellation, n_bits_per_symbol=n_bits_per_symbol,
                                       input_symbols=input_symbols,
                                       noise_var_vec=noise_var)
    return demod_bits


class Modem:
    """
    Digital modem class.

    :param constellation: vector of the reference complex modulation symbols - constellation
    """

    def __init__(self, constellation: list, reorder_as_gray: bool = True):
        """
        Create a baseband digital modem.

        :param constellation: vector of the reference complex modulation symbols - constellation
        :param reorder_as_gray: flag if to reorder the constellation mapping according to gray code
        """

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

    def modulate(self, input_bits: ndarray) -> ndarray:
        """
        Modulate the input bits to complex baseband symbols.

        :param input_bits: input data bits vector
        :return: complex baseband symbols
        """
        return modulate(self._constellation, self.n_bits_per_symbol, input_bits)

    def demodulate(self, input_symbols: ndarray) -> ndarray:
        """
        Demodulate the input symbols to data information bits.

        :param input_symbols: received complex baseband symbols vector
        :return: demodulated data bits vector
        """
        return demodulate(self._constellation, self.n_bits_per_symbol, input_symbols)

    # TODO: optimize/speed up symbol detection and mapping
    def symbol_detection(self, input_symbols: ndarray) -> ndarray:
        """
        Hard detect symbols and output them.

        :param input_symbols: received complex baseband symbol vector
        :return: hard detected complex modulation symbols vector
        """
        index_list = np.abs(input_symbols - self.constellation[:, None]).argmin(0)
        return self.constellation[index_list]

    def plot_constellation(self) -> None:
        """
        Plot the constellation IQ plot.

        :return: None
        """
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
    def correct_constellation(self, ibo_db: float) -> None:
        """
        Shrink the constellation by the alpha coefficient.
        According to Bussgang theorem: y = ax + d

        :param ibo_db: input bak-off value in [dB]
        :return: None
        """
        self.alpha = self.calc_alpha(ibo_db)
        self._constellation = self.alpha * self._constellation

    def calc_alpha(self, ibo_db):
        """
        Calculate the value of the alpha shrinking coefficient.

        :param ibo_db: input bak-off value in [dB]
        :return: None
        """
        # gamma coefficient in Ochiai paper is a ratio of envelopes not powers!
        gamma = np.power(10, ibo_db / 20)
        alpha = 1 - np.exp(-np.power(gamma, 2)) + (np.sqrt(np.pi) * gamma / 2) * scp_special.erfc(gamma)
        # scale constellation
        return alpha

    def update_alpha(self, ibo_db: float) -> None:
        """
        Update the value of alpha shrinking coefficient.

        :param ibo_db: input bak-off value in [dB]
        :return: None
        """
        self.alpha = self.calc_alpha(ibo_db)

    @property
    def constellation(self):
        return self._constellation

    @constellation.setter
    def constellation(self, constelation_symb_lst: list):
        """
        Set the constellation and the parameters associated to it.

        :param constelation_symb_lst: list of ordered constellation symbols
        :return: None
        """

        n_bits_per_symbol = np.log2(len(constelation_symb_lst))
        if n_bits_per_symbol != int(n_bits_per_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        self._constellation = np.array(constelation_symb_lst)
        self.avg_symbol_power = td_signal_power(self.constellation)
        self.constellation_size = self._constellation.size
        self.n_bits_per_symbol = int(n_bits_per_symbol)


class QamModem(Modem):
    """
    QAM modem class.

    :param constel_size: size of the QAM constellation (has to be a power of some number)
    """

    def __init__(self, constel_size):
        """
        Create a QAM Modem object.
        """
        # check if constellation size generates a square QAM
        n_symb = np.sqrt(constel_size)
        if n_symb != int(n_symb):
            raise ValueError('Constellation size must be a power of some number, only square QAM supported.')

        # generate centered around 0, equally spaced (by 2) PAM symbols, indexing from lower left corner
        pam_symb = np.arange(-n_symb + 1, n_symb, 2)
        # arrange into QAM
        constellation = np.tile(np.hstack((pam_symb, pam_symb[::-1])), int(n_symb) // 2) * 1j + pam_symb.repeat(n_symb)

        super().__init__(constellation)


# @jit(nopython=True)
def _tx_ofdm_symbol(mod_symbols: ndarray, n_fft: int, n_sub_carr: int, cp_length: int) -> ndarray:
    """
    OFDM modulate the input baseband symbols, generate an OFDM frame.

    :param mod_symbols: vector of complex baseband symbols
    :param n_fft: size of the inverse Fourier transform (IFFT), total number of the subcarriers
    :param n_sub_carr: number of the data subcarriers
    :param cp_length: cyclic prefix length
    :return: OFDM modulated signal vector in time domain
    """

    # generate OFDM symbol block - size given by n_sub_carr size
    if len(mod_symbols) != n_sub_carr:
        raise ValueError('mod_symbols length must match n_sub_carr value')

    # skip idx = 0 and fill the carriers from LR
    ofdm_sym_freq = np.zeros(n_fft, dtype=np.complex128)

    ofdm_sym_freq[-(n_sub_carr // 2):] = mod_symbols[0:n_sub_carr // 2]
    ofdm_sym_freq[1:(n_sub_carr // 2) + 1] = mod_symbols[n_sub_carr // 2:]

    with objmode(ofdm_sym_time='complex128[:]'):
        ofdm_sym_time = torch.fft.ifft(torch.from_numpy(ofdm_sym_freq), norm="ortho").numpy()

    # add cyclic prefix
    return np.concatenate((ofdm_sym_time[-cp_length:], ofdm_sym_time))


# @jit(nopython=True)
def _rx_ofdm_symbol(ofdm_symbol: ndarray, n_fft: int, n_sub_carr: int, cp_length: int) -> ndarray:
    """
    Demodulate the OFDM frame into baseband symbols.

    :param ofdm_symbol: OFDM modulated signal vector in time domain
    :param n_fft: size of the Fourier transform (FFT), total number of the subcarriers
    :param n_sub_carr: number of the data subcarriers
    :param cp_length: cyclic prefix length
    :return: demodulated baseband symbols vector
    """
    # decode OFDM symbol block - size given by n_sub_carr size
    with objmode(ofdm_sym_freq='complex128[:]'):
        # skip cyclic prefix
        ofdm_sym_freq = torch.fft.fft(torch.from_numpy(ofdm_symbol[cp_length:]), norm="ortho").numpy()

    # extract and rearange data from LR
    return np.concatenate((ofdm_sym_freq[-n_sub_carr // 2:], ofdm_sym_freq[1:(n_sub_carr // 2) + 1]))


class OfdmQamModem(QamModem):
    """
    QAM OFDM modem class.

    :param constel_size: size of the QAM constellation
    :param n_fft: size of the Fourier transform (FFT), total number of the subcarriers
    :param n_sub_carr: number of the data subcarriers
    :param cp_len: cyclic prefix length
    :param n_users: number of users
    """

    def __init__(self, constel_size: int, n_fft: int, n_sub_carr: int, cp_len: int, n_users: int = 1):
        """
        Create a QAM OFDM modem object.
        """
        super().__init__(constel_size)

        self.n_fft = n_fft
        self.n_sub_carr = n_sub_carr
        self.cp_len = cp_len
        self.n_bits_per_ofdm_sym = int(np.log2(constel_size) * n_sub_carr)
        self.avg_sample_power = self.ofdm_avg_sample_pow()
        self.precoding_mat = None
        self.n_users = n_users

    def set_precoding(self, precoding_mat: ndarray) -> None:
        """
        Set the precoding vector, that multiplies the baseband symbols before the transmission.
        Precoding is only applied to data subcarriers.

        :param precoding_mat: matrix or array containing precoding coefficients
        :return:
        """
        # each row represents channel to each user
        self.precoding_mat = precoding_mat

    def precode_symbols(self, in_symbols: ndarray, precoding_mat: ndarray = None) -> ndarray:
        """
        Multiply the baseband symbols by the precoding coefficients if there are any.

        :param in_symbols: input baseband symbol vectors
        :param precoding_mat: precoding coefficients matrix or vector
        :return: precoded symbols vector or matrix
        """
        # apply precoding if any
        if precoding_mat is not None:
            return np.multiply(in_symbols, precoding_mat)
        else:
            return in_symbols

    def modulate(self, input_bits: ndarray, get_symbols_only: bool = False, sum_usr_signals: bool = True):
        """
        Map the input bits to symbols and modulate them with OFDM.

        :param input_bits: input data bits vector
        :param get_symbols_only: flag it to return only the modualted and precoded symbols without OFDM modualtion
        :param sum_usr_signals: flag if to sum the user signals in multi-user scenario
        :return: OFDM modulated signal vector
        """
        if self.n_users == 1:
            modulated_symbols = modulate(self._constellation, self.n_bits_per_symbol, input_bits)
            if get_symbols_only:
                return modulated_symbols
            else:
                precoded_symbols = np.squeeze(self.precode_symbols(modulated_symbols, self.precoding_mat))
                return _tx_ofdm_symbol(precoded_symbols, self.n_fft, self.n_sub_carr, self.cp_len)
        else:
            # multiuser scenario
            modulated_symbols = np.empty((self.n_users, self.n_sub_carr), dtype=np.complex128)
            for user_idx in range(self.n_users):
                modulated_symbols[user_idx, :] = modulate(self._constellation, self.n_bits_per_symbol,
                                                           input_bits[user_idx, :])

            if get_symbols_only:
                return modulated_symbols
            else:
                if sum_usr_signals:
                    combined_mu_symbols = np.sum(self.precode_symbols(modulated_symbols, self.precoding_mat), axis=0)
                    return _tx_ofdm_symbol(combined_mu_symbols, self.n_fft, self.n_sub_carr, self.cp_len)
                else:
                    mu_symbols = self.precode_symbols(modulated_symbols, self.precoding_mat)
                    # return _tx_ofdm_symbol(combined_mu_symbols, self.n_fft, self.n_sub_carr, self.cp_len)
                    tx_sig_per_usr = []
                    for usr_idx in range(self.n_users):
                        tx_sig_per_usr.append(
                            _tx_ofdm_symbol(mu_symbols[usr_idx, :], self.n_fft, self.n_sub_carr, self.cp_len))
                    return tx_sig_per_usr

    def demodulate(self, ofdm_symbol: ndarray, get_symbols_only: bool = False) -> ndarray:
        """
        Demodulate the OFDM frame/symbol.

        :param ofdm_symbol: OFDM modulated signal vector in time domain
        :param get_symbols_only: flag it to return only the demodulated complex symbols
        :return: demodulated data bits vector
        """
        baseband_symbols = _rx_ofdm_symbol(ofdm_symbol, self.n_fft, self.n_sub_carr, self.cp_len)
        if get_symbols_only:
            return baseband_symbols
        else:
            return demodulate(self._constellation, self.n_bits_per_symbol, baseband_symbols)

    def symbols_to_bits(self, input_symbols:ndarray) -> ndarray:
        """
        Demodulate the symbols into bits.

        :param input_symbols: vector of the complex baseband symbols
        :return: vector of detected data bits
        """
        return demodulate(self._constellation, self.n_bits_per_symbol, input_symbols)

    def soft_detection_llr(self, baseband_symbols, noise_var: float) -> ndarray:
        """
        Soft detect the input complex baseband symbols using LLR.

        :param baseband_symbols: vector of the complex baseband symbols
        :param noise_var: value of noise variance (assumes uniform noise variance across all symbols)
        :return: vector of soft-detected data bits
        """
        return demodulate(self._constellation, self.n_bits_per_symbol, baseband_symbols, soft=True,
                          noise_var=noise_var)

    def ofdm_avg_sample_pow(self) -> float:
        """
        Calculate the average sample power of the OFDM signal.

        :return: float
        """
        return self.avg_symbol_power * (self.n_sub_carr / self.n_fft)
