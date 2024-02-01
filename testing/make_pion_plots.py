import argparse

import os
from pybin.schwinger_hmc import handle_bc_arg
from schwinger.schwinger import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import eig

corr_trange = [0, 20]
meff_trange = [0, 19]

mean = lambda x: np.mean(x, axis=0)
rmean = lambda x: np.real(np.mean(x, axis=0))
imean = lambda x: np.imag(np.mean(x, axis=0))
amean = lambda x: np.abs(np.mean(x, axis=0))

def moverg(kappa, beta):
    return (1/(2*kappa)-2)*np.sqrt(beta)

def log_meff(x):
    corr = mean(x)
    return np.log(corr[:-1] / corr[1:])

def asinh_meff(x):
    corr = mean(x)
    return np.arcsinh((corr[:-2] - corr[2:])/(2*corr[1:-1]))

def acosh_meff(x):
    corr = mean(x)
    return np.arccosh((corr[:-2] + corr[2:])/(2*corr[1:-1]))

def cosh_corr(nt, aM, a4Z, Nt):
    return (a4Z/aM) * np.exp(-Nt * aM / 2.) * np.cosh((Nt/2. - nt) * aM)

def sinh_corr(nt, aM, a4Z, Nt):
    return ((a4Z)/aM) * np.exp(-Nt * aM / 2.) * np.sinh((Nt/2. - nt) * aM)

constructions = ['g5-g5', 'g2g5-g2g5', 'g5-g2g5', 'g2g5-g5']#, 'g2-g2', 'g2-g0', 'g0-g2']
colors = ['maroon', 'tomato', 'darkblue', 'darkgreen']# , 'cadetblue', 'plum']
construction_color = {}
for color, construction in zip(colors, constructions):
    construction_color[construction] = color
meffs = {
    'g5-g5': (acosh_meff, 'acosh'),
    'g2g5-g2g5': (acosh_meff, 'acosh'),
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
    'g2g5-g5':      (sinh_corr, 'sinh'), #(lambda nt, M, Z1, Z2, Nt: -sinh_corr(nt, M, Z1, Z2, Nt), '-sinh'),
    'g0-g0':        (cosh_corr, 'cosh'),
    'g2-g2':        (cosh_corr, 'cosh'),
    'g2-g0':        (sinh_corr, 'sinh'),
    'g0-g2':        (sinh_corr, 'sinh') #(lambda nt, M, Z1, Z2, Nt: -sinh_corr(nt, M, Z1, Z2, Nt), '-sinh'),
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
    Lx = args.Lx
    Lt = args.Lt
    L = [Lx, Lt]
    Nd = len(L)
    Ns = 2**(int(Nd/2))
    shape = tuple([Nd] + list(L))
    gauge_bc = handle_bc_arg(args.gauge_obc_x, args.type)

    corr_fits = {
        'g5-g5':        (lambda nt, M, Z : cosh_corr(nt, M, Z,  Lt), 'cosh'),
        'g2g5-g2g5':    (lambda nt, M, Z : cosh_corr(nt, M, Z,  Lt), 'cosh'),
        'g5-g2g5':      (lambda nt, M, Z:  sinh_corr(nt, M, Z, Lt), 'sinh'), # flip?
        'g2g5-g5':      (lambda nt, M, Z:  sinh_corr(nt, M, Z,  Lt), 'sinh'),
        'g0-g0':        (lambda nt, M, Z : cosh_corr(nt, M, Z,  Lt), 'cosh'),
        'g2-g2':        (lambda nt, M, Z : cosh_corr(nt, M, Z,  Lt), 'cosh'),
        'g0-g2':        (lambda nt, M, Z:  sinh_corr(nt, M, Z, Lt), 'sinh'), # flip?
        'g2-g0':        (lambda nt, M, Z:  sinh_corr(nt, M, Z,  Lt), 'sinh'),
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
    fit_range = np.r_[3:8]
    # setup plots
    # corr plot
    corr_fig, corr_ax = plt.subplots()
    corr_ax.set_title(r'$C(t)$ ' + prefix)
    corr_ax.set_xlabel("$t$")
    corr_ax.set_ylabel("$C(t)$")
    corr_ax.set_yscale('log') # FIXME
    #meff plot
    meff_fig, meff_ax = plt.subplots()
    meff_ax.set_title(r'$M_{\mathrm{eff}}(t)$ ' + prefix)
    meff_ax.set_xlabel("$t$")
    meff_ax.set_ylabel("$M(t)$")
    meff_ax.set_yscale('linear')
    for construction in constructions:
        print('construction', construction)
        # read in correlation files #FIXME
        fname_conn = prefix_anls + '_meson_boot_Ct_conn_' + construction + '.npy'
        #fname_conn_cov = prefix_anls + '_meson_boot_Ct_conn_cov_' + construction + '.npy'
        booted_corr_conn = np.load(fname_conn)
        # booted_corr_conn_cov = np.load(fname_conn_cov)
        print('Read Cts from {}'.format(fname_conn))
        #fname_disc = prefix_anls + '_meson_boot_Ct_disc_' + construction + '.npy'
        #fname_disc_cov = prefix_anls + '_meson_boot_Ct_disc_cov_' + construction + '.npy'
        #booted_corr_disc = np.load(fname_disc)
        #booted_corr_disc_cov = np.load(fname_disc_cov)
        #print('Read Cts from {}'.format(fname_disc))
        # read in Meff file
        fname = prefix_anls + '_meson_boot_meff_' + meffs[construction][1] + '_' + construction + '.npy'
        booted_meff = np.load(fname)
        print('read meff (from connected) from {}'.format(fname))
        #fname = prefix_anls + '_meson_boot_meff_sub_' + meffs[construction][1] + '_' + construction + '.npy'
        #booted_meff_sub = np.load(fname)
        #print('read meff (subtracted) from {}'.format(fname))
        # plot correlator
        print('Plotting Ct')
        nskip = 1
        t = np.arange(nskip, Lt)
        # booted data loaded in range nskip:nt
        nt = Lt
        ct_conn_mean = booted_corr_conn[nskip:, 0]
        ct_conn_err  = booted_corr_conn[nskip:, 1]
        #ct_conn_cov = booted_corr_conn_cov[nskip:, nskip:]
        #ct_disc_mean = booted_corr_disc[:, 0]
        #ct_disc_err  = booted_corr_disc[:, 1]
        #ct_disc_cov = booted_corr_disc_cov[:, :]
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
        xdata = t[fit_range]
        ydata = np.real(ct_conn_mean[fit_range])
        sigma1d = np.real(ct_conn_err[fit_range])
        #sigma2d = np.real(ct_conn_cov[3:8, 3:8])
        # f = corr_fits[construction][0]
        # popt, pcov = curve_fit(f,
        #                         xdata, ydata,
        #                         p0 = p0,
        #                         sigma = sigma2d,
        #                         absolute_sigma = False,
        #                         check_finite = False
        #                         )
        # print("aM:", popt[0],  "+/-", pcov[0, 0]**0.5)
        # print("Zs:", popt[1],  "+/-", pcov[1, 1]**0.5)
        # fit = f(t, *popt)
        # # print("covariance: ", pcov)
        # # print("diagonalized covariance: ", pcov)
        # # errbar = np.real(np.sqrt(np.max(eig(pcov)[0])))
        # # find chi-squared through residuals
        # ndofs = len(xdata) - len(popt)
        # r = ydata - f(xdata, *popt)
        # chisq = sum((r/sigma1d)**2)
        # reduced_chisq = chisq / ndofs
        # print("chi squared", chisq)
        # print("reduced chi squared", reduced_chisq)
        # print('t len', len(t))
        # print(ct_conn_mean)
        # corr_ax.errorbar(t,
        #                ct_disc_mean,
        #                yerr=ct_disc_err,
        #                linestyle='None', marker='x',
        #                color = construction_color[construction],
        #                alpha = 1.0,
        #                label = construction + " (disc)")
        print("number of timesteps in correlator plot:", len(t))
        print("number of correlation function values:", len(ct_conn_mean))
        l = corr_trange[0]
        r = corr_trange[1]
        corr_ax.errorbar(t[l:r],
                         ct_conn_mean[l:r],
                         yerr=ct_conn_err[l:r],
                         linestyle='None', marker='o', alpha = 0.5,
                         color = construction_color[construction],
                         label = construction + " (conn)")
        # plot effective mass
        print('Plotting Meff')
        if meffs[construction][1] == 'log':
            t_meff = np.arange(nskip+0.5, Lt-1)
        elif (meffs[construction][1] == 'acosh') or (meffs[construction][1] == 'asinh'):
            t_meff = np.arange(nskip+1.0, Lt-1)
        else:
            assert(False) # unknown meff function
        meff_mean = booted_meff[nskip:, 0]
        meff_err = booted_meff[nskip:, 1]
        #meff_sub_mean = booted_meff_sub[nskip:, 0]
        #meff_sub_err = booted_meff_sub[nskip:, 1]
        print('meff mean', meff_mean)
        print('meff err', meff_err)
        print("data", ydata)
        # print("fit", f(xdata, *popt))
        # print(fit.shape)
        # meff_fit = np.log(fit[:-1]/fit[1:])
        # meff_fit_val = popt[0]
        # meff_fit_err = pcov[0,0]**0.5
        print("number of timesteps in meff plot:", len(t_meff))
        print("number of effective mass values:", len(meff_mean))
        l = meff_trange[0]
        r = meff_trange[1]
        meff_ax.errorbar(t_meff[l:r],
                        meff_mean[l:r],
                        yerr=meff_err[l:r],
                        linestyle='None', marker='o', alpha = 0.5,
                        color = construction_color[construction],
                        label=construction + '_' + meffs[construction][1])
        #meff_ax.errorbar(t,
        #                    meff_fit,
        #                    yerr=meff_fit_err * np.ones(len(meff_fit)),
        #                    color = construction_color[construction],
        #                    alpha = 0.5,
        #                    linestyle='-', marker=None,
        #                    label=f'{construction} meff={meff_fit_val:1.4f}+/-{meff_fit_err:1.4f}')
        #meff_ax.errorbar(t,
        #                meff_sub_mean,
        #                yerr=meff_sub_err,
        #                linestyle='None', marker='x', alpha = 1.0,
        #                color = construction_color[construction],
        #                label=construction + '_' + meffs[construction][1] + " (subtracted)")
    # PLOT LINEAR COMBINATIONS
    # read
    fname_11 = prefix_anls + '_meson_boot_Ct_conn_' + 'g5-g5' + '.npy'
    fname_22 = prefix_anls + '_meson_boot_Ct_conn_' + 'g2g5-g2g5' + '.npy'
    fname_12 = prefix_anls + '_meson_boot_Ct_conn_' + 'g5-g2g5' + '.npy'
    fname_21 = prefix_anls + '_meson_boot_Ct_conn_' + 'g2g5-g5' + '.npy'
    booted_corr_11 = np.load(fname_11)
    booted_corr_22 = np.load(fname_22)
    booted_corr_12 = np.load(fname_12)
    booted_corr_21 = np.load(fname_21)
    # get
    ct_mean_11 = booted_corr_11[nskip:, 0]
    ct_err_11  = booted_corr_11[nskip:, 1]
    ct_mean_22 = booted_corr_22[nskip:, 0]
    ct_err_22  = booted_corr_22[nskip:, 1]
    ct_mean_12 = booted_corr_12[nskip:, 0]
    ct_err_12  = booted_corr_12[nskip:, 1]
    ct_mean_21 = -booted_corr_21[nskip:, 0]     # sign!
    ct_err_21  = booted_corr_21[nskip:, 1]
    # coeffs
    a1 = 1 #0.586742
    a2 = 0 # 0.223623
    # print(t)
    # print("11")
    # print(ct_mean_11[:24])
    # print(ct_mean_11[24:])
    # print("22")
    # print(ct_mean_22[:24])
    # print(ct_mean_22[24:])
    # print("12")
    # print(ct_mean_12[:24])
    # print(ct_mean_12[24:])
    # print("21")
    # print(ct_mean_21[:24])
    # print(ct_mean_21[24:])
    ct_mean_comb_1 = (a1**2 * ct_mean_11 + a2**2 * ct_mean_22 + (a1*a2) * ct_mean_12 + (a2*a1) * ct_mean_21)
    ct_mean_comb_2 = (a2**2 * ct_mean_11 + a1**2 * ct_mean_22 + (a2*a1) * ct_mean_12 + (a1*a2) * ct_mean_21)
    # print("comb 1")
    # print(ct_mean_comb_1[:24])
    # print(ct_mean_comb_1[24:])
    # print("comb 2")
    # print(ct_mean_comb_2[:24])
    # print(ct_mean_comb_2[24:])
    # print("number of timesteps in combined correlator plot:", len(t))
    # print("number of combined correlation function values:", len(ct_mean_comb_1), " and ", len(ct_mean_comb_2))
    l = corr_trange[0]
    r = corr_trange[1]
    # corr_ax.errorbar(t[l:r],
    #                 ct_mean_comb_1[l:r],
    #                 linestyle='None', marker='s', alpha = 0.5,
    #                 color = 'k',
    #                 label = f'({a1}, {a2})')
    # corr_ax.errorbar(t[l:r],
    #                 ct_mean_comb_2[l:r],
    #                 linestyle='None', marker='d', alpha = 0.5,
    #                 color = 'magenta',
    #                 label = f'({a2}, {a1})')
    # # save corr fig
    corr_ax.legend()# bbox_to_anchor=(1.05, 1), loc='upper left')
    corr_fig_fname = prefix_plots + '_corr_src-nog2.png'
    # plt.tight_layout()
    #corr_fig.savefig(corr_fig_fname)
    # save meff fig
    l = meff_trange[0]
    r = meff_trange[1]
    #meff_mean_comb1 = np.log(ct_mean_comb_1[:-1]/ct_mean_comb_1[1:])
    #meff_mean_comb2 = np.log(ct_mean_comb_2[:-1]/ct_mean_comb_2[1:])
    #print("meff comb 1")
    #print(meff_mean_comb1[:23])
    #print(meff_mean_comb1[23:])
    #print("meff comb 2")
    #print(meff_mean_comb2[:23])
    #print(meff_mean_comb2[23:])
    # t_meff = np.arange(nskip+0.5, Lt-1)
    # meff_ax.errorbar(t_meff[l:r],
    #                 meff_mean_comb1[l:r],
    #                 linestyle='None', marker='s', alpha = 0.5,
    #                 color = 'k',
    #                 label= f'({a1}, {a2})')
    # meff_ax.errorbar(t_meff[l:r],
    #                 meff_mean_comb2[l:r],
    #                 linestyle='None', marker='d', alpha = 0.5,
    #                 color = 'magenta',
    #                 label= f'({a2}, {a1})')
    meff_ax.legend()# bbox_to_anchor=(1.05, 1), loc='upper left')
    meff_fig_fname = prefix_plots + '_meff_src-sub-nog2.png'
    #V plt.tight_layout()
    plt.show()
