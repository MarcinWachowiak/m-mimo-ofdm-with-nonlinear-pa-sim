from matplotlib import rcParams, cycler
from math import sqrt


def set_latex_plot_style():
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)

    # rc('text', usetex=True)
    # rc('text.latex', preamble=r'\usepackage{gensymb}')

    fig_width_pt = 426.0  # Get this from LaTeX using \showthe\columnwidth result:
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]

    params = {'backend': 'Qt5Agg',
              'axes.labelsize': 10,
              'font.size': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'figure.figsize': fig_size}

    rcParams.update(params)
    rcParams["font.family"] = ["Latin Modern Roman"]
    rcParams['path.simplify'] = True
