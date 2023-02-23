# %%
import os
import sys

sys.path.append(os.getcwd())

import copy

import numpy as np

import channel
import distortion
import modulation
import transceiver
import utilities
from plot_settings import set_latex_plot_style
from utilities import pts_on_semicircum

set_latex_plot_style(use_tex=False, fig_width_in=7.0)
# %%
ibo_val_db = 3
n_snapshots = 100
n_points = 180
radial_distance = 300  # fix users position at radial/beampattern measurment distance
precoding_angle = 45
sel_psd_angle = 78

sel_ptx_idx = int(n_points / 180 * sel_psd_angle)

# PSD plotting params
psd_nfft = 4096
n_samp_per_seg = 1024

rx_points = pts_on_semicircum(r=radial_distance, n=n_points)
radian_vals = np.radians(np.linspace(0, 180, n_points + 1))
# %%
my_mod = modulation.OfdmQamModem(constel_size=64, n_fft=4096, n_sub_carr=2048, cp_len=128)
my_distortion = distortion.SoftLimiter(ibo_db=ibo_val_db, avg_samp_pow=my_mod.avg_sample_power)
my_tx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion),
                                center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))

my_rx = transceiver.Transceiver(modem=copy.deepcopy(my_mod), impairment=copy.deepcopy(my_distortion), cord_x=212,
                                cord_y=212, cord_z=1.5, center_freq=int(3.5e9),
                                carrier_spacing=int(15e3))

my_miso_chan = channel.MisoLosFd()

my_array = antenna_array.LinearArray(n_elements=128, base_transceiver=my_tx, center_freq=int(3.5e9),
                                      wav_len_spacing=0.5, cord_x=0, cord_y=0, cord_z=15)

utilities.plot_spatial_config(ant_array=my_array, rx_transceiver=my_rx, plot_3d=True)
