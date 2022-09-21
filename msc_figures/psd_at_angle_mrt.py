# %%
import os
import sys

import utilities

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from plot_settings import set_latex_plot_style
from utilities import to_db

set_latex_plot_style(use_tex=True, fig_width_in=5.89572)

filename = "psd_mrt_los_chan_ibo3_npoints1800_nsnap100_angle45_nant128"

psd_data = utilities.read_from_csv(filename=filename)

psd_data[0], psd_data[1] = zip(*sorted(zip(psd_data[0], psd_data[1])))
psd_data[2], psd_data[3] = zip(*sorted(zip(psd_data[2], psd_data[3])))

# normalize
# Plot PSD at the receiver
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(psd_data[0], to_db(psd_data[1]), label="Desired")
ax1.plot(psd_data[2], to_db(psd_data[3]), label="Distortion")

ax1.set_title("Relative power spectral density of transmitted OFDM signal at angle of 45$\degree$")
ax1.set_xlabel("Subcarrier index [-]")
ax1.set_ylabel("Relative power spectral density [dB]")
ax1.legend(title="Signals:")
ax1.grid()

plt.tight_layout()
plt.savefig("../figs/msc_figs/%s.pdf" % (filename), dpi=600, bbox_inches='tight')
plt.show()

# %%
# fading at 54.4
filename = "psd_mrt_los_chan_ibo3_npoints1800_nsnap100_angle54_nant128"

psd_data = utilities.read_from_csv(filename=filename)

psd_data[0], psd_data[1] = zip(*sorted(zip(psd_data[0], psd_data[1])))
psd_data[2], psd_data[3] = zip(*sorted(zip(psd_data[2], psd_data[3])))
psd_data[4], psd_data[5] = zip(*sorted(zip(psd_data[4], psd_data[5])))

# normalize
# Plot PSD at the receiver
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(psd_data[0], to_db(psd_data[1]), label="Desired")
ax1.plot(psd_data[2], to_db(psd_data[3]), label="Distortion")
ax1.plot(psd_data[4], to_db(psd_data[5]), label="No distortion", color="#595959")

ax1.set_title("Relative power spectral density of transmitted OFDM signal at angle of 54.4$\degree$")
ax1.set_xlabel("Subcarrier index [-]")
ax1.set_ylabel("Relative power spectral density [dB]")
ax1.legend(title="Signals:")
ax1.grid()

plt.tight_layout()
plt.savefig("../figs/msc_figs/%s.pdf" % (filename), dpi=600, bbox_inches='tight')
plt.show()
