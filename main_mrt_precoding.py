# antenna array evaluation
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import time

from utilities import count_mismatched_bits, snr_to_ebn0, ebn0_to_snr, to_db, ofdm_avg_sample_pow
import channels
import modulation
import impairments

from plot_settings import set_latex_plot_style

set_latex_plot_style()

print("Multi antenna processing init!")