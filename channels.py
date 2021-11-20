import numpy as np
from utilities import to_db, signal_power
from speedup import jit


# @jit(nopython=True) not useful for such small rng datasets
def _propagate_awgn(is_complex, snr_db, rng_gen, in_sig, avg_sample_pow):
    n_sampl = len(in_sig)
    noise_std = np.complex128(np.sqrt((int(is_complex) + 1) * avg_sample_pow / (10 ** (snr_db / 10))))
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

    def propagate(self, in_sig, avg_sample_pow):
        return _propagate_awgn(self.is_complex, self.snr_db, self.rng_gen, in_sig, avg_sample_pow)


class AwgnMisoChannel:

    def __init__(self, n_inputs, snr_db, is_complex, seeds=None):
        self.snr_db = snr_db
        self.is_complex = is_complex
        self.n_inputs = n_inputs
        self.seeds = seeds
        self.awgn_chan_lst = []

        if isinstance(self.seeds, list) and len(self.seeds) == self.n_inputs:
            # custom list of seeds
            for idx, seed in enumerate(self.seeds):
                self.awgn_chan_lst.append(AwgnChannel(self.snr_db, self.is_complex, seed))
        else:
            # rng seed generator
            rng_seed_gen = np.random.default_rng(9876)
            for idx in range(self.n_inputs):
                # generate random seed
                self.awgn_chan_lst.append(
                    AwgnChannel(self.snr_db, self.is_complex, rng_seed_gen.integers(0, np.iinfo(np.int64).max)))

    def set_snr(self, snr_db):
        # update snr in every subchannel
        for chan in self.awgn_chan_lst:
            chan.set_snr(snr_db)

    def propagate(self, tx_transceivers, rx_transceiver, in_sig, skip_noise=False):

        out_sig_mat = np.empty(in_sig.shape, dtype=np.complex128)
        for idx, tx_transceiver in enumerate(tx_transceivers):
            if not skip_noise:
                out_sig = self.awgn_chan_lst[idx].propagate(in_sig[idx, :], tx_transceiver.modem.avg_sample_power)
            else:
                out_sig = in_sig[idx, :]

            distance = np.sqrt(np.power(tx_transceiver.cord_x-rx_transceiver.cord_x, 2) + np.power(tx_transceiver.cord_y-rx_transceiver.cord_y, 2) + np.power(tx_transceiver.cord_z-rx_transceiver.cord_z, 2))
            phase_shift = np.exp(-2j*np.pi*distance)
            out_sig_mat[idx:] = out_sig * phase_shift

        return np.sum(out_sig_mat, axis=0)
