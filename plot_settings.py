from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler, rc


def set_latex_plot_style(use_tex: bool = False, fig_width_in: float = 7.0, fig_height_in: float = None) -> None:
    """
    Sets the scientific style of plots.
    Makes matplotlib use LaTex typesetting and custom color palette.

    :param use_tex: flag if to use LaTex for typesetting
    :param fig_width_in: width of the figures in inches
    :param fig_height_in: height of the figures in inches, if not specified calculated from golden ratio based on width
    :return: None
    """
    CB_color_cycle = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79',
                      '#CFCFCF']
    rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)

    # uncomment only for final hq plots with latex fonts - latex fonts highly slow down plotting
    if use_tex:
        rc('text', usetex=True)
        rc('text.latex', preamble=r'\usepackage{gensymb}')
        rcParams["font.family"] = ["Latin Modern Roman"]

    # fig_width_pt = 426.0  # Get this from LaTeX using \showthe\columnwidth result:
    # inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    # fig_width_in = 3.5  # width in inches
    if fig_height_in is None:
        fig_height_in = fig_width_in * golden_mean  # height in inches

    fig_size = [fig_width_in, fig_height_in]

    params = {'backend': 'Qt5Agg',
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'figure.figsize': fig_size}
    rcParams['path.simplify'] = True

    rcParams.update(params)
    matplotlib.use("Qt5Agg")


def reset_color_cycle():
    """
    Resets the matplotlib color cycle to the default.

    :return: None
    """
    plt.gca().set_prop_cycle(None)
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)
