import abc

import numpy as np

#
from utilities import to_db, fd_signal_power, td_signal_power, to_time_domain


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

    def process(self, in_sig, avg_sample_pow=1, fixed_noise_power=False, disp_data=False):
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
            # calculate SNR only for SC
            # nsc=2048
            # ofdm_sig_nsc_fd = np.concatenate((in_sig[-nsc // 2:], in_sig[1:(nsc // 2) + 1]))
            # noise_nsc = np.concatenate((noise[-nsc // 2:], noise[1:(nsc // 2) + 1]))
            # Frequency domain
            # print("Signal power:[dBm]", to_db(fd_signal_power(ofdm_sig_nsc_fd))+30)
            # print("Noise power:[dBm]", to_db(fd_signal_power(noise_nsc))+30)
            # print("SNR: ", to_db(fd_signal_power(ofdm_sig_nsc_fd)/fd_signal_power(noise_nsc)))
            #
            print("Signal power:[dBm]", to_db(td_signal_power(to_time_domain(in_sig)))+30)
            print("Noise power:[dBm]", to_db(td_signal_power(to_time_domain(noise)))+30)
            print("SNR: ", to_db(td_signal_power(to_time_domain(in_sig))/td_signal_power(to_time_domain(noise))))

        return in_sig + noise

