import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

K_antennas = 10
alpha = np.deg2rad(-60) # np.linspace(-90, 90, n_samples, endpoint=True)
beta = np.deg2rad(60) # np.linspace(-90, 90, n_samples, endpoint=True)
dist_alpha_k = np.zeros(K_antennas)
dist_beta_k = np.zeros(K_antennas)

arccos_arg_periodize = lambda val_a: val_a - 2.0 if val_a > 1.0 else (val_a + 2.0 if val_a < -1.0 else val_a)

for k_idx in range(K_antennas):
    dist_alpha_k[k_idx] = np.arccos((np.cos(alpha - 2*np.pi*k_idx/K_antennas))) + 2*np.pi*k_idx/K_antennas
    dist_beta_k[k_idx] = np.arccos(arccos_arg_periodize(2*np.cos(beta - 2*np.pi*k_idx/K_antennas) - np.cos(alpha - 2*np.pi*k_idx/K_antennas))) + 2*np.pi*k_idx/K_antennas

print("Alpha dist angles: ", np.rad2deg(dist_alpha_k))
print("Beta dist angles: ", np.rad2deg(dist_beta_k))

# for alpha_idx, alpha_val in enumerate(alpha):
#     for beta_idx, beta_val in enumerate(beta):
#         dist_alpha[alpha_idx, beta_idx] = np.rad2deg(
#             np.arcsin(arcsin_arg_periodize(2 * np.sin(np.deg2rad(alpha_val)) - np.sin(np.deg2rad(beta_val)))))
#         dist_beta[alpha_idx, beta_idx] = np.rad2deg(
#             np.arcsin(arcsin_arg_periodize(2 * np.sin(np.deg2rad(beta_val)) - np.sin(np.deg2rad(alpha_val)))))

# # replace NaN
# dist_alpha = np.nan_to_num(dist_alpha, nan=-100)
# dist_beta = np.nan_to_num(dist_beta, nan=-100)

# fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
# sel_alpha_idx = 0
# plt.tight_layout()
# ax.plot(beta, dist_alpha[sel_alpha_idx, :], label="Dist alpha")
# ax.plot(beta, dist_beta[sel_alpha_idx, :], label="Dist beta")
# ax.grid(True)
# ax.legend()
# plt.show()

# %%
# x_and_y_ticks = np.linspace(-90, 90, 7, endpoint=True)
# fig, axs = plt.subplots(1, 2)
# # fig.set_tight_layout(True)
# im1 = axs[0].imshow(dist_alpha, extent=[-90, 90, -90, 90], vmin=-90, vmax=90, interpolation='None')
# axs[0].set_title("Distortion from alpha user [°]")
# axs[0].set_xlabel("Alpha [°]")
# axs[0].set_ylabel("Beta [°]")
# axs[0].set_xticks(x_and_y_ticks)
# axs[0].set_yticks(x_and_y_ticks)
# divider = make_axes_locatable(axs[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im1, cax=cax, orientation='vertical', ticks=x_and_y_ticks)
#
# im2 = axs[1].imshow(dist_beta, extent=[-90, 90, -90, 90], vmin=-90, vmax=90, interpolation='None')
# axs[1].set_xlabel("Alpha [°]")
# axs[1].set_ylabel("Beta [°]")
# axs[1].set_title("Distortion from beta user [°]")
# axs[1].set_xticks(x_and_y_ticks)
# axs[1].set_yticks(x_and_y_ticks)
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im1, cax=cax, orientation='vertical', ticks=x_and_y_ticks)
# plt.tight_layout()
# plt.savefig("../figs/multiuser/distortion_directions_eval/distortion_angles_prediction.png", dpi=600,
#             bbox_inches='tight')
#
# plt.show()
