from math import sqrt

from matplotlib import rcParams, cycler, rc


def set_latex_plot_style(use_tex=False, fig_width_in=7.0):
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
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
    fig_height = fig_width_in * golden_mean  # height in inches
    fig_size = [fig_width_in, fig_height]

    params = {'backend': 'Qt5Agg',
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'figure.figsize': fig_size}
    rcParams['path.simplify'] = True

    rcParams.update(params)
