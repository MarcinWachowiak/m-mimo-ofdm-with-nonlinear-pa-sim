from abc import ABC

import numpy as np
from numpy import ndarray
from utilities import to_db, fd_signal_power


class Noise(ABC):
    """
    Noise base class.

    :param snr_db: signal-to-noise ratio expressed in [dB]
    :param noise_p_dbm: noise power expressed in [W]
    :param seed: seed for the random number generator
    """

    def __init__(self, snr_db: float, noise_p_dbm: float, seed: int):
        """
        Create a Noise base object.
        """
        self.snr_db = snr_db
        self.noise_p_dbm = noise_p_dbm
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)
        else:
            self.rng_gen = np.random.default_rng(1234)


class Awgn(Noise):
    """
    Additive white Gaussian noise class.

    :param snr_db: signal-to-noise ratio expressed in [dB]
    :param noise_p_dbm: power of the noise expressed in [W]
    :param seed: seed for the random number generator
    """

    def __init__(self, snr_db=None, noise_p_dbm=None, seed=None):
        """
        Create an additive white Gaussian noise object.
        """
        super().__init__(snr_db, noise_p_dbm, seed)

    def process(self, in_sig: ndarray, avg_sample_pow: float = 1, fixed_noise_power: bool = False,
                disp_data: bool = False) -> ndarray:
        """
        Add the noise to the input signal.

        :param in_sig: input signal vector
        :param avg_sample_pow: average sample power of the signal
        :param fixed_noise_power: flag if the noise power should be calculated based on fixed power value.
        :param disp_data: flag if to display the instantaneous SNR within the vector or frame
        :return:
        """
        n_sampl = len(in_sig)

        # use noise_p_dbm to generate noise - limited and constant noise power
        if fixed_noise_power:
            noise_std = np.complex128(np.sqrt(2 * 0.001 * 10 ** (self.noise_p_dbm / 10)))

        # use snr_db ratio to generate noise - unlimited noise power
        else:
            noise_std = np.complex128(np.sqrt(2 * avg_sample_pow / (10 ** (self.snr_db / 10))))

        noise = self.rng_gen.standard_normal((n_sampl, 2)).view(np.complex128)[:, 0] * noise_std * 0.5

        # check resultant SNR
        if disp_data:
            # Frequency domain
            # limit to subcarriers band
            n_nsc = 2048
            in_sig_nsc = np.concatenate((in_sig[-n_nsc // 2:], in_sig[1:(n_nsc // 2) + 1]))
            noise_nsc = np.concatenate((noise[-n_nsc // 2:], noise[1:(n_nsc // 2) + 1]))
            print("Signal power:[dBm]", to_db(fd_signal_power(in_sig_nsc) + 30))
            print("Noise power:[dBm]", to_db(fd_signal_power(noise_nsc)) + 30)
            print("SNR: ", to_db(fd_signal_power(in_sig_nsc) / fd_signal_power(noise_nsc)))
            #
            # print("Signal power:[dBm]", to_db(td_signal_power((in_sig)))+30)
            # print("Noise power:[dBm]", to_db(td_signal_power((noise)))+30)
            # print("SNR: ", to_db(td_signal_power((in_sig))/td_signal_power((noise))))

        return in_sig + noise
