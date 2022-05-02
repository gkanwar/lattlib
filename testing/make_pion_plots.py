import argparse
import os
from pylib.analysis import mean, rmean, log_meff, acosh_meff, bootstrap
from pybin.schwinger_hmc import handle_bc_arg
from schwinger.schwinger import *
import matplotlib.pyplot as plt

# open corr files, bootstrap, writeout boostrapped with uncertainties

def asinh_meff(x):
    corr = mean(x)
    return np.arcsinh((corr[2:] - corr[:-2])/(2*corr[1:-1]))

constructions = ['g5-g5', 'g2g5-g2g5', 'g5-g2g5', 'g2g5-g5']
meffs = {
    'g5-g5': (log_meff, 'log'),
    'g2g5-g2g5': (log_meff, 'log'),
    'g5-g2g5': (asinh_meff, 'asinh'),
    'g2g5-g5': (asinh_meff, 'asinh'),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HMC for Schwinger')
    # general params
    parser.add_argument('--seed', type=int)
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--n_skip', type=int, required=True)
    parser.add_argument('--n_therm', type=int, required=True)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--n_leap', type=int, default=20)
    # action params
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--gauge_obc_x', action="store_true")
    parser.add_argument('--beta', type=float)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--conn_weight', type=float, default=1.0)
    parser.add_argument('--disc_weight', type=float, default=0.0)
    parser.add_argument('--xspace', type=int, default=1)
    args = parser.parse_args()
    print("args = {}".format(args))

    if len(args.tag) > 0:
        args.tag = "_" + args.tag
    L = [args.Lx, args.Lt]
    Nd = len(L)
    Ns = 2**(int(Nd/2))
    shape = tuple([Nd] + list(L))
    gauge_bc = handle_bc_arg(args.gauge_obc_x, args.type)
    if args.type == "two_flavor":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        action = TwoFlavorAction(args.beta, args.kappa, gauge_bc = gauge_bc)
        compute_dirac = "wilson"
    elif args.type == "one_flavor":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        action = OneFlavorAction(args.beta, args.kappa, gauge_bc = gauge_bc)
        compute_dirac = "wilson"
    elif args.type == "exact_1flav_staggered":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        m0 = m0_from_kappa(args.kappa, Nd)
        action = ExactStaggeredAction(args.beta, m0, Nf=1)
        compute_dirac = "staggered"
    elif args.type == "exact_2flav_staggered":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        m0 = m0_from_kappa(args.kappa, Nd)
        action = ExactStaggeredAction(args.beta, m0, Nf=2)
        compute_drac = "staggered"
    else:
        print("Unknown action type {}".format(args.type))
        sys.exit(1)

    path = '{:s}/{:s}/gaugeBC{:d}{:d}/'.format(args.tag, action.make_tag(), gauge_bc[0], gauge_bc[1])
    path_anls = 'anls/' + path
    path_plots = 'plots/' + path
    os.makedirs(path_anls, exist_ok=True)
    os.makedirs(path_plots, exist_ok=True)
    prefix = 'u1_{:s}_N{:d}_skip{:d}_therm{:d}_{:d}_{:d}_gaugeBC{:d}{:d}{:s}'.format(
        action.make_tag(), args.Ncfg, args.n_skip, args.n_therm,
        args.Lx, args.Lt,
        gauge_bc[0], gauge_bc[1],
        args.tag)
    prefix_anls = path_anls + prefix
    prefix_plots = path_plots + prefix

    booted_corr = []
    booted_meff = []
    tsteps = args.Lt
    # setup plots
    # corr plot
    corr_fig, corr_ax = plt.subplots()
    corr_ax.set_title(r'$C(t)$ ' + prefix)
    corr_ax.set_xlabel("$t$")
    corr_ax.set_ylabel("$C(t)$")
    corr_ax.set_yscale('log')
    # meff plot
    meff_fig, meff_ax = plt.subplots()
    meff_ax.set_title(r'$M_{\mathrm{eff}}(t)$ ' + prefix)
    meff_ax.set_xlabel("$t$")
    meff_ax.set_ylabel("$M(t)$")
    meff_ax.set_yscale('linear')
    for construction in constructions:
        print('construction', construction)
        # read in correlation file
        fname = prefix_anls + '_meson_boot_Ct_' + construction + '.npy'
        booted_corr = np.load(fname)
        print('Read Cts from {}'.format(fname))
        print(booted_corr)
        # read in Meff file
        fname = prefix_anls + '_meson_boot_meff_' + meffs[construction][1] + '_' + construction + '.npy'
        booted_meff = np.load(fname)
        print('Read meff from {}'.format(fname))
        print(booted_meff)
        # plot correlator
        print('Plotting Ct')
        t = np.arange(tsteps)
        nt = len(t)
        ct_mean = booted_corr[:nt, 0]
        ct_err = booted_corr[:nt, 1]
        print('t len', len(t))
        print('ct mean len', len(ct_mean))
        print('ct err len', len(ct_err))
        corr_ax.errorbar(t,
                        ct_mean,
                        yerr=ct_err,
                        linestyle='None', marker='o',
                        label = construction)
        ## plot effective mass
        print('Plotting Meff')
        if meffs[construction][1] == 'log':
            t = np.arange(0.5, tsteps-1)
        elif (meffs[construction][1] == 'acosh') or (meffs[construction][1] == 'asinh'):
            t = np.arange(1.5, tsteps-1)
        else:
            assert(False) # unknown meff function
        nt = len(t)
        meff_mean = booted_meff[:nt, 0]
        meff_err = booted_meff[:nt, 1]
        print('t len', len(t))
        print('meff mean len', meff_mean)
        print('meff err len', meff_err)
        meff_ax.errorbar(t,
                        meff_mean,
                        yerr=meff_err,
                        linestyle='None', marker='o',
                        label=construction + '_' + meffs[construction][1])
    # save corr fig
    corr_ax.legend(loc='upper center')
    corr_fig_fname = prefix_plots + '_corr_bc_comparison.png'
    corr_fig.savefig(corr_fig_fname)
    # save meff fig
    meff_ax.legend(loc='upper center')
    meff_fig_fname = prefix_plots + '_meff_bc_comparison.png'
    meff_fig.savefig(meff_fig_fname)
