"""
Test script to predict the beamforming directions of the distortions signal in the presence of multiple users
for uniform linear antenna array [Unused and requires investigation].
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == '__main__':

    n_samples = 181
    alpha = np.linspace(-90, 90, n_samples, endpoint=True)
    beta = np.linspace(-90, 90, n_samples, endpoint=True)

    m_alpha, m_beta = np.meshgrid(alpha, beta)

    dist_alpha = np.zeros((n_samples, n_samples))
    dist_beta = np.zeros((n_samples, n_samples))

    arcsin_arg_periodize = lambda val_a: val_a - 2.0 if val_a > 1.0 else (val_a + 2.0 if val_a < -1.0 else val_a)

    for alpha_idx, alpha_val in enumerate(alpha):
        for beta_idx, beta_val in enumerate(beta):
            dist_alpha[alpha_idx, beta_idx] = np.rad2deg(
                np.arcsin(arcsin_arg_periodize(2 * np.sin(np.deg2rad(alpha_val)) - np.sin(np.deg2rad(beta_val)))))
            dist_beta[alpha_idx, beta_idx] = np.rad2deg(
                np.arcsin(arcsin_arg_periodize(2 * np.sin(np.deg2rad(beta_val)) - np.sin(np.deg2rad(alpha_val)))))

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
    x_and_y_ticks = np.linspace(-90, 90, 7, endpoint=True)
    fig, axs = plt.subplots(1, 2)
    # fig.set_tight_layout(True)
    im1 = axs[0].imshow(dist_alpha, extent=[-90, 90, -90, 90], vmin=-90, vmax=90, interpolation='None')
    axs[0].set_title("Distortion from alpha user [°]")
    axs[0].set_xlabel("Alpha [°]")
    axs[0].set_ylabel("Beta [°]")
    axs[0].set_xticks(x_and_y_ticks)
    axs[0].set_yticks(x_and_y_ticks)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical', ticks=x_and_y_ticks)

    im2 = axs[1].imshow(dist_beta, extent=[-90, 90, -90, 90], vmin=-90, vmax=90, interpolation='None')
    axs[1].set_xlabel("Alpha [°]")
    axs[1].set_ylabel("Beta [°]")
    axs[1].set_title("Distortion from beta user [°]")
    axs[1].set_xticks(x_and_y_ticks)
    axs[1].set_yticks(x_and_y_ticks)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical', ticks=x_and_y_ticks)
    plt.tight_layout()
    plt.savefig("../figs/multiuser/distortion_directions_eval/distortion_angles_prediction.png", dpi=600,
                bbox_inches='tight')

    plt.show()
