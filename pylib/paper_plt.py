### Configures matplotlib for paper-quality plots (Latex fonts, font sizes, etc)

import matplotlib.pyplot as plt

pix_to_pt = 72/100
def load_latex_config():
    plt.rcParams['errorbar.capsize'] = 1
    plt.rcParams['lines.linewidth'] = 1.0 * pix_to_pt
    plt.rcParams['lines.markeredgewidth'] = 1.0 * pix_to_pt
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.linewidth'] = 0.5 * pix_to_pt
    plt.rcParams['lines.markersize'] = 2
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.major.width'] = 0.5 * pix_to_pt
    plt.rcParams['xtick.minor.width'] = 0.5 * pix_to_pt
    plt.rcParams['ytick.major.width'] = 0.5 * pix_to_pt
    plt.rcParams['ytick.minor.width'] = 0.5 * pix_to_pt
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['legend.framealpha'] = '0.8'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = '0.5'
    plt.rcParams['patch.linewidth'] = 0.5 * pix_to_pt
    plt.rcParams['legend.handlelength'] = 1.0
    plt.rcParams['legend.fontsize'] = 8

def inner_ylabel(ax, label, *, x, y, labelpad, **kwargs):
    ax.set_ylabel(label, x=x, y=y, labelpad=labelpad, rotation='horizontal', fontsize=10, **kwargs)
