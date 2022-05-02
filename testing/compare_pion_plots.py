import argparse
import os
from pylib.analysis import mean, rmean, log_meff, acosh_meff, bootstrap
from pybin.schwinger_hmc import handle_bc_arg
from schwinger.schwinger import *
import matplotlib.pyplot as plt
import numpy as np

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

Ngoups = 2
Lt = 24

action_type_list = ['one_flavor', 'one_flavor']
gauge_obc_list = [(0, 1), (1, 1)]
Lx_list = [16, 16]

beta_list = [2.0, 2.0]
kappa_list = [1.0, 1.0]

tag_list = ['testing', 'testing']
ncfgs_list = [100, 100]
nskip_list = [10, 10]
ntherm_list = [40, 40]
tau_list = [0.1, 0.1]
nleap_list = [20, 20]


# some default values if not specified
dflt['plot_name'] = 'test_compare'
dflt['beta'] = 2.0
dflt['kappa'] = 1.0
dflt['tag'] = 'testing'
dflt['ncfgs'] = 100
dflt['ntherm'] = 40
dflt['tau'] = 0.1
dflt['nleap'] = 20

def compare_pion_plots(plot_name,
                       action_type_list,
                       gauge_bc_list,
                       Lx_list, Lt,
                       beta_list=None, kappa_list=None,
                       tag_list=None,
                       ncfgs_list=None,
                       ntherm_list=None, tau_list=None, nleap_list=None)
    assert(len(action_type_list)>0)
    assert(len(gauge_bc_list)>0)
    assert(len(ldim_list)>0)

    # set defaults if not specified
    if len(plot_name)==0: plot_name   = dflt['plot_name']
    if beta_list==None:   beta_list   = dflt['beta']   * np.ones(len(action_type_list))
    if kappa_list==None:  kappa_list  = dflt['kappa']  * np.ones(len(action_type_list))
    if tag_list==None:    tag_list    = dflt['tag']    * np.ones(len(action_type_list))
    if ncfgs_list==None:  ncfgs_list  = dflt['ncfgs']  * np.ones(len(action_type_list))
    if ntherm_list==None: ntherm_list = dflt['ntherm'] * np.ones(len(action_type_list))
    if tau_list==None:    tau_list    = dflt['tau']    * np.ones(len(action_type_list))
    if nleap_list==None:  nleap_list  = dflt['nleap']  * np.ones(len(action_type_list))

    lists = [action_type_list, gauge_bc_list, ldim_lst, beta_list, kappa_list,
            tag_list, ncfgs_list, ntherm_list, tau_list, nleap_list]
    it = iter(lists)
    ngroups = len(next(it))
    if not all(len(l) == ngroups for l in it):
        raise ValueError('not all parameter lists have same length!')

    # set out path
    path_plots = 'plots/'
    os.makedirs(path_plots, exist_ok=True)
    prefix_plots = path_plots + plot_name

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

    for group in range(len(ngroups)):
        tag = tag_list[group]
        if len(tag) > 0:
            tag = "_" + tag
        Lx = ldim_list[group][0]
        L = [Lx, Lt]
        Nd = len(L)
        Ns = 2**(int(Nd/2))
        shape = tuple([Nd] + list(L))

        # configure action
        action_type = action_type_list[group]
        beta = beta_list[group]
        kappa = kappa_list[group]
        gauge_bc = gauge_bc_list[group]
        if action_type == "two_flavor":
            action = TwoFlavorAction(args.beta, args.kappa, gauge_bc = gauge_bc)
        elif args.type == "one_flavor":
            action = OneFlavorAction(args.beta, args.kappa, gauge_bc = gauge_bc)
        else:
            print("Unknown action type {}".format(args.type))
            sys.exit(1)

        # configure remaining params
        ncfg = ncfgs_list[group]
        nskip = nskip_list[group]
        ntherm = ntherm_list[group]
        tau = tau_list[group]
        nskip = nskip_list[group]

        # set input path
        path = '{:s}/{:s}/gaugeBC{:d}{:d}/'.format(tag,
                                                    action.make_tag(),
                                                    gauge_bc[0], gauge_bc[1])
        path_anls = 'anls/' + path
        os.makedirs(path_anls, exist_ok=True)
        prefix = 'u1_{:s}_N{:d}_skip{:d}_therm{:d}_{:d}_{:d}_gaugeBC{:d}{:d}{:s}'.format(
            action.make_tag(), ncfg, nskip, ntherm,
            Lx, Lt,
            gauge_bc[0], gauge_bc[1],
            args.tag)
        prefix_anls = path_anls + prefix

        booted_corr = []
        booted_meff = []
        tsteps = Lt
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
        corr_fig_fname = prefix_plots + '.png'
        corr_fig.savefig(corr_fig_fname)
        # save meff fig
        meff_ax.legend(loc='upper center')
        meff_fig_fname = prefix_plots + '.png'
        meff_fig.savefig(meff_fig_fname)
