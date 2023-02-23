import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import plot_settings

# TODO: create generic distortion class and introduce inheritance

# @jit(nopython=True)
def _process_soft_lim(sat_pow: float, in_sig: ndarray) -> ndarray:
    """
    Process the input signal by the soft-limited nonlinear transfer function.
    Numba enabled prototype.

    :param sat_pow: value of the maximum power - where the clipping happens
    :param in_sig: input signal vector
    :return: nonlinearly distorted output signal vector
    """
    return np.where(np.power(np.abs(in_sig), 2) <= sat_pow, in_sig,
                    in_sig * np.sqrt(np.divide(sat_pow, np.power(np.abs(np.where(in_sig != 0, in_sig, 1)), 2))))


class SoftLimiter:
    """
    Soft limiter / clipper nonlinear distortion class.

    :param ibo_db: input back-off value in [dB]
    :param avg_samp_pow: average signal sample power in time domain
    """

    def __init__(self, ibo_db: float, avg_samp_pow: float):
        """
        Create a soft limiter object.
        """

        self.ibo_db = ibo_db
        self.avg_samp_pow = avg_samp_pow
        self.sat_pow = np.power(10, ibo_db / 10) * avg_samp_pow

    def __str__(self):
        return "softlim"

    def set_ibo(self, ibo_db: float) -> None:
        """
        Set the input back-off value in [dB].

        :param ibo_db: input back-off value in [dB]
        :return: None
        """

        self.ibo_db = ibo_db
        self.sat_pow = np.power(10, ibo_db / 10) * self.avg_samp_pow

    def set_avg_sample_power(self, avg_samp_pow: float) -> None:
        """
        Set the average signal sample power.

        :param avg_samp_pow: average signal sample power in time domain
        :return: None
        """
        self.avg_samp_pow = avg_samp_pow
        self.sat_pow = np.power(10, self.ibo_db / 10) * avg_samp_pow

    def plot_characteristics(self, in_ampl_min: float = -10, in_ampl_max: float = 10, step: float = 0.1) -> None:
        """
        Plot the transfer characteristics of the soft limiter nonlinear distortion model.

        :param in_ampl_min: lower bound of the input signal amplitude
        :param in_ampl_max: upper bound of the input signal amplitude
        :param step: step of the incrementing the input signal amplitude
        :return: None
        """
        in_sig_ampl = np.arange(in_ampl_min, in_ampl_max + step, step)
        out_sig_ampl = self.process(in_sig_ampl)

        plot_settings.set_latex_plot_style(use_tex=True, fig_width_in=5.89572, fig_height_in=3)
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim([-5, 5])
        ax.plot(in_sig_ampl, out_sig_ampl, label=self.ibo_db)
        ax.set_title("Soft limiter transfer characteristic")
        ax.set_xlabel("Input signal amplitude [V]")
        ax.set_ylabel("Output signal amplitude [V]")
        ax.legend(title="IBO [dB]")
        props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        ax.text(0.63, 0.05, ("Average sample power = %2.1f [W]" % self.avg_samp_pow), transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom', bbox=props)
        ax.grid()
        plt.tight_layout()
        plt.savefig("../figs/msc_figs/soft_lim_tf_ibo%d.pdf" % self.ibo_db, dpi=600, bbox_inches='tight')
        plt.show()

    def process(self, in_sig: ndarray) -> ndarray:
        """
        Process the input signal by the nonlinear distortion function/model in time domain.

        :param in_sig: input signal vector
        :return: distorted signal vector
        """
        return _process_soft_lim(self.sat_pow, in_sig)


# @jit(nopython=True)
def _process_rapp(sat_pow: float, p_hardness: float, in_sig: ndarray):
    """
    Process the input signal by the Rapp nonlinear transfer function.
    Numba enabled prototype.

    :param sat_pow: value of the maximum power - where the clipping happens
    :param p_hardness: value of the p hardness coefficient
    :param in_sig: input signal vector
    :return: nonlinearly distorted output signal vector
    """

    return in_sig / (np.power(1 + np.power(np.abs(in_sig) / np.sqrt(sat_pow), 2 * p_hardness), 1 / (2 * p_hardness)))


class Rapp:
    """
    Rapp model of nonlinear distortion class.

    :param ibo_db: input back-off value in [dB]
    :param avg_samp_pow: average signal sample power in time domain
    :param p_hardness: value of the p hardness coefficient
    """

    def __init__(self, ibo_db: float, avg_samp_pow: float, p_hardness: float):
        """
        Create a Rapp nonlinear distortion object.
        """
        self.p_hardness = p_hardness
        self.ibo_db = ibo_db
        self.avg_samp_pow = avg_samp_pow
        self.sat_pow = np.power(10, ibo_db / 10) * avg_samp_pow

    def __str__(self):
        return "rapp"

    def set_hardness(self, p_hardness: float) -> None:
        """
        Set the p hardness coefficient.

        :param p_hardness: p hardness coeffcicient value
        :return: None
        """
        self.p_hardness = p_hardness

    def set_ibo(self, ibo_db: float) -> None:
        """
        Set the input back-off value in [dB].

        :param ibo_db: input back-off value in [dB]
        :return: None
        """

        self.ibo_db = ibo_db
        self.sat_pow = np.power(10, ibo_db / 10) * self.avg_samp_pow

    def set_avg_sample_power(self, avg_samp_pow: float) -> None:
        """
        Set the average signal sample power.

        :param avg_samp_pow: average signal sample power in time domain
        :return: None
        """

        self.sat_pow = np.power(10, self.ibo_db / 10) * avg_samp_pow

    def plot_characteristics(self, in_ampl_min: float = -10, in_ampl_max: float = 10, step: float = 0.1) -> None:
        """
        Plot the transfer characteristics of the Rapp nonlinear distortion model.

        :param in_ampl_min: lower bound of the input signal amplitude
        :param in_ampl_max: upper bound of the input signal amplitude
        :param step: step of the incrementing the input signal amplitude
        :return: None
        """

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
        plt.savefig("../figs/rapp_lim_tf_ibo%d.png" % self.ibo_db, dpi=600, bbox_inches='tight')
        plt.show()

    def process(self, in_sig: ndarray) -> ndarray:
        """
        Process the input signal by the nonlinear distortion function/model in time domain.

        :param in_sig: input signal vector
        :return: distorted signal vector
        """
        return _process_rapp(self.sat_pow, self.p_hardness, in_sig)


# @jit(nopython=True)
def _process_toi(cubic_dist_coeff: float, in_sig: ndarray) -> ndarray:
    """
    Process the input signal by the third order nonlinear transfer function.
    Numba enabled prototype.

    :param cubic_dist_coeff: coefficient multiplying the cubic term in the polynomial model
    :param in_sig: input signal vector
    :return: nonlinearly distorted output signal vector
    """
    return in_sig - cubic_dist_coeff * in_sig * np.power(np.abs(in_sig), 2)


# TODO: verify proper third order coefficient calculation
class ThirdOrderNonLin:
    """
    Third order memoryless polynomial nonlinear distortion class.

    :param toi_db: third order intercept point value in [dB] in relatio to average signal sample power
    :param avg_samp_pow: average signal sample power in time domain
    """
    def __init__(self, toi_db: float, avg_samp_pow: float):
        """
        Create a third order memoryless polynomial nonlinear distortion object.
        """
        self.avg_samp_pow = avg_samp_pow
        self.toi_db = toi_db
        self.cubic_dist_coeff = 1 / (np.power(10, (toi_db / 10))) / avg_samp_pow

    def __str__(self):
        return "toi"

    def set_toi(self, toi_db: float) -> None:
        """
        Set third order intercept point in [dB].

        :param toi_db: third order intercept point value in [dB] in relatio to average signal sample power
        :return: None
        """
        self.toi_db = toi_db
        self.cubic_dist_coeff = 1 / (np.power(10, (toi_db / 10))) / self.avg_samp_pow

    def set_avg_sample_power(self, avg_samp_pow: float) -> None:
        """
        Set the average signal sample power and update the cubic distortion coefficient.

        :param avg_samp_pow: average signal sample power in time domain in time domain
        :return: None
        """
        self.avg_samp_pow = avg_samp_pow
        self.cubic_dist_coeff = 1 / (np.power(10, (self.toi_db / 10))) / avg_samp_pow

    def plot_characteristics(self, in_ampl_min: float = -10, in_ampl_max: float = 10, step: float = 0.1) -> None:
        """
        Plot the transfer characteristics of the TOI nonlinear distortion model.

        :param in_ampl_min: lower bound of the input signal amplitude
        :param in_ampl_max: upper bound of the input signal amplitude
        :param step: step of the incrementing the input signal amplitude
        :return: None
        """

        in_sig_ampl = np.arange(in_ampl_min, in_ampl_max + step, step)
        out_sig_ampl = self.process(in_sig_ampl)

        fig, ax = plt.subplots(1, 1)
        # ax.plot(in_sig_ampl, out_sig_ampl, label=self.toi_db)
        ax.plot(in_sig_ampl, out_sig_ampl, label=self.toi_db)
        # ax.plot(to_db(in_sig_ampl ** 2), to_db(in_sig_ampl ** 2) - to_db(out_sig_ampl ** 2), label="Difference")
        # ax.plot(to_db(in_sig_ampl ** 2), to_db(in_sig_ampl ** 2), label="Linear")

        ax.set_title("Third order distortion transfer characteristic")
        ax.set_xlabel("Input signal amplitude [V]")
        ax.set_ylabel("Output signal amplitude [V]")
        ax.legend(title="TOI [dB]")
        ax.grid()
        plt.tight_layout()
        plt.savefig("../figs/toi_tf_toi%d.png" % self.toi_db, dpi=600, bbox_inches='tight')
        plt.show()

    def process(self, in_sig):
        return _process_toi(self.cubic_dist_coeff, in_sig)
