"""
Plot the nonlinear distortion model transfer characteristic,
output amplitude as a function of the input amplitude of the signal.
"""

# %%
import os
import sys

sys.path.append(os.getcwd())

import copy

import distortion
import modulation
import transceiver
from plot_settings import set_latex_plot_style


if __name__ == '__main__':

    set_latex_plot_style()

    # %%

    my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=256)

    my_distortion = distortion.SoftLimiter(ibo_db=3, avg_samp_pow=10)
    my_limiter2 = distortion.Rapp(0, my_mod.avg_sample_power, p_hardness=5)
    my_limiter3 = distortion.ThirdOrderNonLin(25, my_mod.avg_sample_power)

    my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                    center_freq=int(3.5e9), carrier_spacing=int(15e3))
    my_tx.impairment.plot_characteristics()
