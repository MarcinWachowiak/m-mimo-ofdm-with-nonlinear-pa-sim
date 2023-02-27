"""
Master's thesis plotting script.
Plot the evolution of the number of mobile subscribers and global mobile data traffic over the years according
to the Ericsson's reports
"""

import matplotlib.pyplot as plt
import numpy as np

from plot_settings import set_latex_plot_style

if __name__ == '__main__':

        set_latex_plot_style(use_tex=True, fig_width_in=5.89572, fig_height_in=3.20)

        # %%
        # plot user growth
        y_usr = [6084.265, 6198.8, 6328.789, 6426.262, 6521.513, 6612.575, 6698.486]
        y_usr = np.array(y_usr)
        y_usr = y_usr / 1000
        x_years = np.arange(2021, 2028, 1)

        fig, ax = plt.subplots()

        plt.bar(x_years, y_usr, width=0.65, alpha=.75)

        ax.set_axisbelow(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis=u'both', which=u'both')
        plt.title("Mobile subscribers")
        plt.ylabel("Billions of mobile subscribers")
        plt.xlabel("Year")
        plt.ylim([5.5, 7.0])
        ax.set_yticks([5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0])
        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig("../figs/msc_figs/ericcson_mobile_usr_growth.pdf", dpi=600, bbox_inches='tight')
        plt.show()

        # %%
        # plot mobile traffic growth
        y_rest = [60.24, 72.44, 84.128, 96.647, 106.079, 113.756, 113.841]
        y_5g = [6.901, 18.121, 36.945, 61.426, 91.871, 126.544, 168.567]

        x_years = np.arange(2021, 2028, 1)

        fig, ax = plt.subplots()

        plt.stackplot(x_years, y_rest, y_5g, alpha=.75)

        ax.set_axisbelow(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(axis=u'both', which=u'both')
        plt.title("Global mobile data traffic")
        plt.ylabel("Exa bytes per month $(10^{18}$B)")
        plt.xlabel("Year")
        ax.text(0.82, 0.55, '5G',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                transform=ax.transAxes)

        ax.text(0.753, 0.175, '2G/3G/4G',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                transform=ax.transAxes)

        plt.ylim([0, 300])
        plt.xlim([2021, 2027])
        plt.xticks(x_years)

        # ax.tick_params(axis='x', which='major')

        plt.tight_layout()
        plt.grid()
        plt.savefig("../figs/msc_figs/ericsson_global_mobile_data.pdf", dpi=600, bbox_inches='tight')
        plt.show()
