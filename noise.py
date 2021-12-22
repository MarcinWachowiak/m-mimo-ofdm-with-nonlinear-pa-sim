import abc

import numpy as np


class Noise(metaclass=abc.ABCMeta):
    def __init__(self, snr_db, noise_p_dbm, seed):
        self.snr_db = snr_db
        self.noise_p_dbm = noise_p_dbm
        if seed is not None:
            self.seed = seed
            self.rng_gen = np.random.default_rng(seed)
        else:
            self.rng_gen = np.random.default_rng(1234)


class Awgn(Noise):
    def __init__(self, snr_db=None, noise_p_dbm=None, seed=None):
        super().__init__(snr_db, noise_p_dbm, seed)

    def process(self, in_sig, avg_sample_pow):
        n_sampl = len(in_sig)
        noise_std = np.complex128(np.sqrt(2 * avg_sample_pow / (10 ** (self.snr_db / 10))))
        noise = self.rng_gen.standard_normal((n_sampl, 2)).view(np.complex128)[:, 0] * noise_std * 0.5

        # check resultant SNR
        # print("SNR: ", to_db(signal_power(in_sig)/signal_power(noise)))
        return in_sig + noise

