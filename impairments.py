import numpy as np
import matplotlib.pyplot as plt
from speedup import jit
from utilities import to_db


@jit(nopython=True)
def _process_soft_lim(sat_pow, in_sig):
    return np.where(np.power(np.abs(in_sig), 2) <= sat_pow, in_sig, in_sig * np.sqrt(np.divide(sat_pow, np.power(np.abs(np.where(in_sig != 0, in_sig, 1)), 2))))


class SoftLimiter:

    def __init__(self, ibo_db, avg_symb_pow):
        self.ibo_db = ibo_db
        self.avg_symb_pow = avg_symb_pow
        self.sat_pow = np.power(10, ibo_db / 10) * avg_symb_pow

    def set_ibo(self, ibo_db):
        self.ibo_db = ibo_db
        self.sat_pow = np.power(10, ibo_db / 10) * self.avg_symb_pow

    def plot_characteristics(self, in_ampl_min=-10, in_ampl_max=10, step=0.1):
        in_sig_ampl = np.arange(in_ampl_min, in_ampl_max + step, step)
        out_sig_ampl = self.process(in_sig_ampl)

        fig, ax = plt.subplots(1, 1)
        ax.plot(in_sig_ampl, out_sig_ampl, label=self.ibo_db)
        ax.set_title("Soft limiter transfer characteristic")
        ax.set_xlabel("Input signal amplitude [V]")
        ax.set_ylabel("Output signal amplitude [V]")
        ax.legend(title="IBO [dB]")
        ax.grid()
        plt.tight_layout()
        #plt.savefig("figs/soft_lim_tf.pdf", dpi=600, bbox_inches='tight')
        plt.show()

    def process(self, in_sig):
        return _process_soft_lim(self.sat_pow, in_sig)


@jit(nopython=True)
def _process_rapp(sat_pow, p_hardness, in_sig):
    return in_sig / (np.power(1 + np.power(np.abs(in_sig) / np.sqrt(sat_pow), 2 * p_hardness), 1 / (2 * p_hardness)))


class Rapp:

    def __init__(self, ibo_db, avg_symb_pow, p_hardness):
        self.p_hardness = p_hardness
        self.ibo_db = ibo_db
        self.avg_symb_pow = avg_symb_pow
        self.sat_pow = np.power(10, ibo_db / 10) * avg_symb_pow

    def set_hardness(self, p_hardness):
        self.p_hardness = p_hardness

    def set_ibo(self, ibo_db):
        self.ibo_db = ibo_db
        self.sat_pow = np.power(10, ibo_db / 10) * self.avg_symb_pow

    def plot_characteristics(self, in_ampl_min=-10, in_ampl_max=10, step=0.1):
        in_sig_ampl = np.arange(in_ampl_min, in_ampl_max + step, step)
        out_sig_ampl = self.process(in_sig_ampl)

        fig, ax = plt.subplots(1, 1)
        ax.plot(in_sig_ampl, out_sig_ampl, label=self.ibo_db)
        ax.set_title("Rapp model transfer characteristic")
        ax.set_xlabel("Input signal amplitude [V]")
        ax.set_ylabel("Output signal amplitude [V]")
        ax.legend(title="IBO [dB]")
        ax.grid()
        plt.tight_layout()
        #plt.savefig("figs/rapp_lim_tf.pdf", dpi=600, bbox_inches='tight')
        plt.show()

    def process(self, in_sig):
        return _process_rapp(self.sat_pow, self.p_hardness, in_sig)


@jit(nopython=True)
def _process_toi(cubic_dist_coeff, in_sig):
    return in_sig - cubic_dist_coeff * np.power(in_sig, 3)


class ThirdOrderNonLin:

    def __init__(self, toi_db, avg_symb_pow):
        self.toi_db = toi_db
        self.avg_symb_pow = avg_symb_pow
        self.cubic_dist_coeff = 1 / (avg_symb_pow * np.power(10, (toi_db / 10)))

    def set_toi(self, toi_db):
        self.toi_db = toi_db
        self.cubic_dist_coeff = 1 / (self.avg_symb_pow * np.power(10, (toi_db / 10)))

    def plot_characteristics(self, in_ampl_min=-10, in_ampl_max=10, step=0.1):
        in_sig_ampl = np.arange(in_ampl_min, in_ampl_max + step, step)
        out_sig_ampl = self.process(in_sig_ampl)

        fig, ax = plt.subplots(1, 1)
        ax.plot(in_sig_ampl, out_sig_ampl, label=self.toi_db)
        # ax.plot(to_db(in_sig_ampl**2), to_db(out_sig_ampl**2), label=self.toi_db)
        # ax.plot(to_db(in_sig_ampl**2), to_db(in_sig_ampl**2) - to_db(out_sig_ampl**2), label="Difference")
        # ax.plot(to_db(in_sig_ampl**2), to_db(in_sig_ampl**2), label="Linear")

        ax.set_title("Third order distortion transfer characteristic")
        ax.set_xlabel("Input signal amplitude [V]")
        ax.set_ylabel("Output signal amplitude [V]")
        ax.legend(title="TOI [dB]")
        ax.grid()
        plt.tight_layout()
        #plt.savefig("figs/toi_tf.pdf", dpi=600, bbox_inches='tight')
        plt.show()

    def process(self, in_sig):
        return _process_toi(self.cubic_dist_coeff, in_sig)