# %%
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import distortion
import modulation
import noise
import transceiver
from plot_settings import set_latex_plot_style
from utilities import count_mismatched_bits, ebn0_to_snr, to_db

set_latex_plot_style()

# %%

my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=1024, cp_len=256)

my_distortion = distortion.SoftLimiter(ibo_db=3, avg_samp_pow=10)
my_limiter2 = distortion.Rapp(0, my_mod.avg_sample_power, p_hardness=5)
my_limiter3 = distortion.ThirdOrderNonLin(25, my_mod.avg_sample_power)

my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion))
my_tx.impairment.plot_characteristics()