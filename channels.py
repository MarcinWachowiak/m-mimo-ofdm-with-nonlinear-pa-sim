import numpy as np
from utilities import to_db
from speedup import jit


# @jit(nopython=True) not useful for such small rng datasets
def _propagate_awgn(is_complex, snr_db, rng_gen, in_sig, avg_symb_pow, n_sub_carr, n_fft):
    n_sampl = len(in_sig)
    in_sig_pow = avg_symb_pow * (n_sub_carr / n_fft)
    noise_std = np.complex128(np.sqrt((int(is_complex) + 1) * in_sig_pow / (10 ** (snr_db / 10))))

    if is_complex:
        noise = rng_gen.standard_normal((n_sampl, 2)).view(np.complex128)[:, 0] * noise_std * 0.5
    else:
        noise = rng_gen.standard_normal(n_sampl) * noise_std
    # check resultant SNR
    # print("SNR: ", to_db(signal_power(in_sig)/signal_power(noise)))

    return in_sig + noise


class AwgnChannel:

    def __init__(self, snr_db, is_complex, seed=None):
        self.snr_db = snr_db
        self.is_complex = is_complex
        self.seed = seed
        self.rng_gen = np.random.default_rng(seed)

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def propagate(self, in_sig, avg_symb_pow, n_sub_carr=1, n_fft=1):
        return _propagate_awgn(self.is_complex, self.snr_db, self.rng_gen, in_sig, avg_symb_pow, n_sub_carr, n_fft)
