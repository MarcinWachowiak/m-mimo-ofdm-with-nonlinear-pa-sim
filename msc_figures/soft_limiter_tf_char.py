# %%
import matplotlib.pyplot as plt
import numpy as np

import plot_settings


def process_soft_lim(sat_pow, in_sig):
    return np.where(np.power(np.abs(in_sig), 2) <= sat_pow, in_sig,
                    in_sig * np.sqrt(np.divide(sat_pow, np.power(np.abs(np.where(in_sig != 0, in_sig, 1)), 2))))


in_ampl_min = 0
in_ampl_max = 10
step = 0.1

in_sig_ampl = np.arange(in_ampl_min, in_ampl_max + step, step)
out_sig_ampl = process_soft_lim(sat_pow=25.0, in_sig=in_sig_ampl)

plot_settings.set_latex_plot_style(use_tex=True, fig_width_in=5.89572, fig_height_in=3)

fig, ax = plt.subplots(1, 1)

plt.xticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           [None, 0, None, None, None, None, "$P_{\mathrm{max}}$", None, None, None, None, None])
plt.yticks([-1, 0, 1, 2, 3, 4, 5, 6, 7], [None, 0, None, None, None, None, "$P_{\mathrm{max}}$", None, None])

ax.set_ylim([-1, 7])
ax.set_xlim([-1, 11])
ax.plot(in_sig_ampl, out_sig_ampl, linewidth=2)
ax.set_title("Soft limiter transfer function")
ax.set_xlabel("Input signal power")
ax.set_ylabel("Output signal power")

# plt.tick_params(
#     axis='both',  # changes apply to the x-axis
#     which='both',  # both major and minor ticks are affected
#     bottom=False,  # ticks along the bottom edge are off
#     top=False,  # ticks along the top edge are off
#     right=False,
#     left=False,
#     labelbottom=False,
#     labelleft=False)  # labels along the bottom edge are off

ax.grid()
plt.tight_layout()
plt.savefig("../figs/msc_figs/soft_lim_tf.pdf", dpi=600, bbox_inches='tight')
plt.show()
