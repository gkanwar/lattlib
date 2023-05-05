### Configures matplotlib for paper-quality plots (Latex fonts, font sizes, etc)

import matplotlib.pyplot as plt

pix_to_pt = 72/100

latex_config = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
    'font.serif': ['Computer Modern Roman']
}

def load_latex_config():
    load_basic_config()
    plt.rcParams.update(latex_config)

basic_config = {
    'lines.linewidth': 1.0 * pix_to_pt,
    'lines.markeredgewidth': 1.0 * pix_to_pt,
    'savefig.bbox': 'tight',
    'axes.labelsize': 9,
    'axes.linewidth': 0.5 * pix_to_pt,
    'lines.markersize': 2,
    'errorbar.capsize': 1,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.5 * pix_to_pt,
    'xtick.minor.width': 0.5 * pix_to_pt,
    'ytick.major.width': 0.5 * pix_to_pt,
    'ytick.minor.width': 0.5 * pix_to_pt,
    'legend.fontsize': 8,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': '0.8',
    'legend.facecolor': 'white',
    'legend.edgecolor': '0.5',
    'patch.linewidth': 0.5 * pix_to_pt,
    'legend.handlelength': 1.0,
    'legend.fontsize': 8,
    'font.family': 'serif'
}

def load_basic_config():
    plt.rcParams.update(basic_config)

def inner_ylabel(ax, label, *, x, y, labelpad, **kwargs):
    ax.set_ylabel(label, x=x, y=y, labelpad=labelpad, rotation='horizontal', fontsize=10, **kwargs)
