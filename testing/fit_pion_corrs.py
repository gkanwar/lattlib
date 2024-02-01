import argparse
import os
import numpy as np
from pylib.analysis import mean, rmean, log_meff, acosh_meff, bootstrap
from pybin.schwinger_hmc import handle_bc_arg
from schwinger.schwinger import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import eig

def moverg(kappa, beta):
    return (1/(2*kappa)-2)*np.sqrt(beta)

def asinh_meff(x):
    corr = mean(x)
    return np.arcsinh((corr[2:] - corr[:-2])/(2*corr[1:-1]))

def cosh_corr(nt, aM, a4Z, Nt):
    return (a4Z/aM) * np.exp(-Nt * aM / 2.) * np.cosh((Nt/2. - nt) * aM)

def sinh_corr(nt, aM, a4Z, Nt):
    return ((a4Z)/aM) * np.exp(-Nt * aM / 2.) * np.sinh((Nt/2. - nt) * aM)

constructions = ['g5-g5', 'g2g5-g2g5']# 'g2-g2', 'g5-g5', 'g2g5-g2g5']#, 'g5-g2g5', 'g2g5-g5']
meffs = {
    'g5-g5': (log_meff, 'log'),
    'g2g5-g2g5': (log_meff, 'log'),
    'g5-g2g5': (asinh_meff, 'asinh'),
    'g2g5-g5': (asinh_meff, 'asinh'),
    'g0-g0': (log_meff, 'log'),
    'g2-g2': (log_meff, 'log'),
    'g0-g2': (asinh_meff, 'asinh'),
    'g2-g0': (asinh_meff, 'asinh'),
    }
corrs = {
    'g5-g5':        (cosh_corr, 'cosh'),
    'g2g5-g2g5':    (cosh_corr, 'cosh'),
    'g5-g2g5':      (sinh_corr, 'sinh'),
    'g2g5-g5':      (lambda nt, M, Z1, Z2, Nt: -sinh_corr(nt, M, Z1, Z2, Nt), '-sinh'),
    'g0-g0':        (cosh_corr, 'cosh'),
    'g2-g2':        (cosh_corr, 'cosh'),
    'g2-g0':        (sinh_corr, 'sinh'),
    'g0-g2':        (lambda nt, M, Z1, Z2, Nt: -sinh_corr(nt, M, Z1, Z2, Nt), '-sinh'),
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
    Nt = args.Lt
    shape = tuple([Nd] + list(L))
    gauge_bc = handle_bc_arg(args.gauge_obc_x, args.type)

    corr_fits = {
        'g5-g5':        (lambda nt, M, Z : cosh_corr(nt, M, Z, Nt), 'cosh'),
        'g2g5-g2g5':    (lambda nt, M, Z : cosh_corr(nt, M, Z, Nt), 'cosh'),
        'g5-g2g5':      (lambda nt, M, Z:  -sinh_corr(nt, M, Z, Nt), 'sinh'),
        'g2g5-g5':      (lambda nt, M, Z:  sinh_corr(nt, M, Z, Nt), 'sinh'),
        'g0-g0':        (lambda nt, M, Z : cosh_corr(nt, M, Z, Nt), 'cosh'),
        'g2-g2':        (lambda nt, M, Z : cosh_corr(nt, M, Z, Nt), 'cosh'),
        'g0-g2':        (lambda nt, M, Z:  -sinh_corr(nt, M, Z, Nt), 'sinh'),
        'g2-g0':        (lambda nt, M, Z:  sinh_corr(nt, M, Z, Nt), 'sinh'),
        }

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
    # find m/g
    mg = moverg(args.kappa, args.beta)
    print(f'm/g {mg:1.3f}')
    booted_corr = []
    booted_meff = []
    tsteps = args.Lt
    t = np.arange(0.0, tsteps)
    # FIXME HARDCODED T RANGE
    fit_range = np.r_[11:15, 34:38]
    # setup plots
    # meff plot
    meff_fig, meff_ax = plt.subplots()
    meff_ax.set_title(r'$M_{\mathrm{eff}}(t)$ ' + r'$m/g=$' + f'{mg:1.2f}, ' + r'$x=$' +  f'{args.beta:1.2f} ' + f'{args.Lx}x{args.Lt}')
    meff_ax.set_xlabel("$t$")
    meff_ax.set_ylabel("$M(t)$")
    meff_ax.set_yscale('linear')
    print("hardcoded fit range: ", fit_range)
    for construction in constructions:
        print('construction', construction)
        # read in correlation file
        fname = prefix_anls + '_meson_boot_Ct_' + construction + '.npy'
        booted_corr = np.load(fname)
        print('Read Cts from {}'.format(fname))
        # read in Meff file
        fname = prefix_anls + '_meson_boot_meff_' + meffs[construction][1] + '_' + construction + '.npy'
        booted_meff = np.load(fname)
        print('read meff from {}'.format(fname))
        ct_mean = booted_corr[:Nt, 0]
        ct_err = booted_corr[:Nt, 1]
        model = corr_fits[construction][0]
        model_name = corr_fits[construction][1]
        if model_name == 'cosh':
            p0 = [1.0, 1.0]         # aM, a4Z
        elif model_name == 'sinh':
            p0 = [1.0, 1.0]         # aM, a4Z
        elif model_name == 'naive':
            p0 = None
        else:
            raise "don't know this fit model"
        #print(ct_mean[fit_range])
        #print(ct_err[fit_range])
        if (model_name == 'naive'):
            xdata = t
            ydata = np.real(ct_mean)
            sigma = np.real(ct_err)
        else:
            xdata = t[fit_range]
            ydata = np.real(ct_mean[fit_range])
            sigma = np.real(ct_err[fit_range])
            f = corr_fits[construction][0]
            popt, pcov = curve_fit(f,
                                    xdata, ydata,
                                    p0 = p0,
                                    sigma = sigma,
                                    absolute_sigma = False,
                                    check_finite = False
                                    )
            print("aM:", popt[0],  "+/-", pcov[0, 0]**0.5)
            print("Zs:", popt[1],  "+/-", pcov[1, 1]**0.5)
            print("covariance: ", pcov)
            print("diagonalized covariance: ", pcov)
            errbar = np.real(np.sqrt(np.max(eig(pcov)[0])))
            # find chi-squared through residuals
            ndofs = len(xdata) - len(popt)
            r = ydata - f(xdata, *popt)
            chisq = sum((r/sigma)**2)
            reduced_chisq = chisq / ndofs
            print("chi squared", chisq)
            print("reduced chi squared", reduced_chisq)
        # plot effective mass
        meff_mean = booted_meff[:Nt, 0]
        meff_err = booted_meff[:Nt, 1]
        errbar=np.max(abs(meff_mean[fit_range]))- np.min(abs(meff_mean[fit_range]))
        if (model_name == 'naive'):
            fit = ydata             # get from meff instead
        else:
            fit = f(t, *popt)
            print("data", ydata)
            print("fit", f(xdata, *popt))
        if meffs[construction][1] == 'log':
            t_meff = np.arange(0.5, tsteps-1)
            meff_fit = np.log(fit[:-1]/fit[1:])
            if (model_name == 'naive'):
                meff_fit = 0.5 * (meff_fit[6]-meff_fit[-6]) * np.ones(len(meff_fit))
                meff_fit_val = meff_fit[6]
                meff_fit_err = meff_err[6] * np.ones(len(meff_err))
            else:
                meff_fit_val = popt[0]
                meff_fit_err = errbar
        elif (meffs[construction][1] == 'acosh') or (meffs[construction][1] == 'asinh'):
            t_meff = np.arange(1.5, tsteps-1)
            if meffs[construction][1] == 'acosh':
                meff_fit = np.arccosh((fit[2:] + fit[:-2])/(2*fit[1:-1]))
            else:
                meff_fit = np.arcsinh((fit[2:] - fit[:-2])/(2*fit[1:-1]))
            if (model_name == 'naive'):
                meff_fit = 0.5 * (meff_fit[6]-meff_fit[-6]) * np.ones(len(meff_fit))
                meff_fit_err = meff_err[6] * np.ones(len(meff_fit_err))
            else:
                meff_fit_val = popt[0]
                meff_fit_err = errbar
        else:
            assert(False) # unknown meff function
        meff_ax.errorbar(t_meff,
                        meff_mean,
                        yerr=meff_err,
                        linestyle='None', marker='o', markersize=2.0,
                        label=construction + '_' + meffs[construction][1])
        #meff_ax.errorbar(t_meff,
        #                    meff_fit,
        #                    yerr=meff_fit_err * np.ones(len(meff_fit)),
        #                    linestyle='-', marker=None,
        #                    label=f'{construction} {model_name} meff={meff_fit_val:1.4f}+/-{meff_fit_err:1.4f}')
    meff_ax.legend(loc='upper right')
    meff_fig_fname = prefix_plots + '_meff_bc.png'
    plt.show()
