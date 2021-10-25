from numpy.random import standard_normal
from numpy import sqrt, log10
from utilities import signal_power
from numpy.random import default_rng


class AwgnChannel():

    def __init__(self, snr_db, is_complex, seed=None):
        self.snr_db = snr_db
        self.is_complex = is_complex
        self.seed = seed
        self.rng_gen = default_rng(seed)

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def propagate(self, in_sig):
        n_sampl = len(in_sig)
        in_sig_pow = signal_power(in_sig)

        noise_std = sqrt((int(self.is_complex) + 1) * in_sig_pow / (10 ** (self.snr_db / 10)))

        if self.is_complex:
            noise = (self.rng_gen.standard_normal(n_sampl) + 1j * self.rng_gen.standard_normal(n_sampl)) * noise_std * 0.5
        else:
            noise = self.rng_gen.standard_normal(n_sampl) * noise_std

        # check final SNR
        # noise_pow = signal_power(noise)
        # res_snr = 10 * log10(in_sig_pow / noise_pow)
        # print(res_snr)

        return in_sig + noise
