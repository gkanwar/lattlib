import argparse
import os
from schwinger import *
from schwinger_hmc import handle_bc_arg
import matplotlib.pyplot as plt

# open corr files, bootstrap, writeout boostrapped with uncertainties

# Standard bootstrapping fns
mean = lambda x: np.mean(x, axis=0)
rmean = lambda x: np.real(np.mean(x, axis=0))
imean = lambda x: np.imag(np.mean(x, axis=0))
amean = lambda x: np.abs(np.mean(x, axis=0))

def log_meff(corr):
    return np.log(corr[:, :-1] / corr[:, 1:])

def acosh_meff(corr):
    return np.arccosh((corr[:, :-2] + corr[:, 2:])/(2*corr[:, 1:-1]))

def asinh_meff(corr):
    return np.arcsinh((corr[:, :-2] - corr[:, 2:])/(2*corr[:, 1:-1]))

def make_stn_f(*, N_inner_boot, f):
    def stn(x):
        mean, err = bootstrap(x, Nboot=N_inner_boot, f=f)
        stn = np.abs(mean) / np.abs(err)
        return stn
    return stn

def bootstrap_gen(*samples, Nboot):
    n = len(samples[0])
    np.random.seed(5)
    for i in range(Nboot):
        inds = np.random.randint(n, size=n)
        yield tuple(s[inds] for s in samples)

def bootstrap(*samples, Nboot, f):
    boots = []
    for x in bootstrap_gen(*samples, Nboot=Nboot):
        boots.append(f(*x))
    return boots #np.mean(boots, axis=0), np.std(boots, axis=0)

def covar_from_boots(boots):
    boots = np.array(boots)
    Nboot = boots.shape[0]
    means = np.mean(boots, axis=0, keepdims=True)
    deltas = boots - means
    return np.tensordot(deltas, deltas, axes=(0,0)) / (Nboot-1)

gsrcs = ['5', '25']
gsnks = ['5', '25']
csrcs = ['l', 'n', 'n2']
csnks = ['l', 'n', 'n2']
constructions = []
for gsrc in gsrcs:
    for gsnk in gsnks:
        for csrc in csrcs:
            for csnk in csnks:
                constructions.append('g' + gsrc + '-' + 'g' + gsnk + '_' + csrc + csnk)
#constructions = [
#                    'g5-g5-ll', 'g2g5-g2g5-ll', 'g2g5-g5-ll', 'g5-g2g5-ll',
#                    'g5-g5-nl', 'g2g5-g2g5-nl', 'g2g5-g5-nl', 'g5-g2g5-nl',
#                    'g5-g5-ln', 'g2g5-g2g5-ln', 'g2g5-g5-ln', 'g5-g2g5-ln',
#                    'g5-g5-nn', 'g2g5-g2g5-nn', 'g2g5-g5-nn', 'g5-g2g5-nn'
#                ]
#colors = [
#         'maroon', 'tomato',
#         'sandybrown', 'lightcoral',
#         #'darkblue'#, 'darkgreen',
#         #'cadetblue', 'plum'
#        ]
# construction_color = {}
# for color, construction in zip(colors, constructions):
#     construction_color[construction] = color
meffs = {
    'g5-g5': (acosh_meff, 'acosh'),
    'g25-g25': (acosh_meff, 'acosh'),
    'g5-g25': (asinh_meff, 'asinh'),
    'g25-g5': (asinh_meff, 'asinh'),
    'g0-g0': (log_meff, 'log'),
    'g2-g2': (log_meff, 'log'),
    'g0-g2': (asinh_meff, 'asinh'),
    'g2-g0': (asinh_meff, 'asinh'),
    }
g0 = pauli(0)
g5 = -pauli(3)
g2 = pauli(2)
g2g5 = np.matmul(g2, g5)
g = {}
g['0'] = pauli(0)
g['5'] = -pauli(3)
g['2']= pauli(2)
g['25'] = np.matmul(g['2'], g['5'])
vac_funs = {
    'g5-g5':     lambda vac: np.einsum('ab,ba', vac, g5)   * np.einsum('ab,ba', vac, g2 @ np.conj( g5.T   ) @ g2),
    'g25-g25': lambda vac: np.einsum('ab,ba', vac, g2g5) * np.einsum('ab,ba', vac, g2 @ np.conj( g2g5.T ) @ g2),
    'g5-g25':   lambda vac: np.einsum('ab,ba', vac, g5)   * np.einsum('ab,ba', vac, g2 @ np.conj( g2g5.T ) @ g2),
    'g25-g5':   lambda vac: np.einsum('ab,ba', vac, g2g5) * np.einsum('ab,ba', vac, g2 @ np.conj( g5.T   ) @ g2),
    'g0-g0':     lambda vac: np.einsum('ab,ba', vac, g0)   * np.einsum('ab,ba', vac, g2 @ np.conj( g0.T   ) @ g2),
    'g2-g2':     lambda vac: np.einsum('ab,ba', vac, g2)   * np.einsum('ab,ba', vac, g2 @ np.conj( g2.T   ) @ g2),
    'g0-g2':     lambda vac: np.einsum('ab,ba', vac, g0)   * np.einsum('ab,ba', vac, g2 @ np.conj( g2.T   ) @ g2),
    'g2-g0':     lambda vac: np.einsum('ab,ba', vac, g2)   * np.einsum('ab,ba', vac, g2 @ np.conj( g0.T   ) @ g2)
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
    path_meas = 'meas/' + path
    path_anls = 'anls/' + path
    os.makedirs(path_meas, exist_ok=True)
    os.makedirs(path_anls, exist_ok=True)
    prefix = 'u1_{:s}_N{:d}_skip{:d}_therm{:d}_{:d}_{:d}_gaugeBC{:d}{:d}{:s}'.format(
        action.make_tag(), args.Ncfg, args.n_skip, args.n_therm,
        args.Lx, args.Lt,
        gauge_bc[0], gauge_bc[1],
        args.tag)
    prefix_meas = path_meas + prefix
    prefix_anls = path_anls + prefix

    # TODO reorganize into h5py files
    corrs = []
    corr_mean = []
    corr_err = []
    vac = []
    vac_mean = []
    vac_err = []
    meff_mean = []
    meff_err = []
    covar = []
    # read in vac file
    fname_vac = prefix_meas + '_vac.npy'
    #vacs = np.load(fname_vac)
    #vacs_boot = bootstrap(vacs, Nboot = args.Ncfg, f = lambda x : mean(x))
    #vacs_mean  = np.mean(vacs_boot, axis=0)
    #vacs_err   = np.std(vacs_boot, axis=0, ddof=1)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax4.set_xlabel(r'$\log(C(t))/\bar(C)(t))$ ')
    ax1.set_title(prefix)
    th = [2, 6, 10, 14]
    Nboot = args.Ncfg if args.Ncfg < 1000 else 1000
    for gsrc in gsrcs:
        for gsnk in gsnks:
            for csrc in csrcs:
                for csnk in csnks:
                    gstruct = 'g' + gsrc + '-' + 'g' + gsnk
                    construction = gstruct + '_' + csrc + csnk
                    print('construction', construction)
                    #vac_mean = vac_funs[construction](vacs_mean)
                    #vac_err  = vac_funs[construction](vacs_err)
                    #print('vac', vac_mean**2)
                    # read in correlation file
                    # FIXME
                    fname_conn = prefix_meas + '_meson_Ct_conn_' + construction + '.npy'
                    fname_disc = prefix_meas + '_meson_Ct_disc_' + construction + '.npy'
                    corrs_conn = np.load(fname_conn)
                    corrs_disc = 2 * np.load(fname_disc)            # disco weight
                    corrs = corrs_conn + corrs_disc #FIXME
                    corrs_conn_sample_mean = np.mean(corrs_conn, axis=0)
                    corrs_conn_sample_err  = np.std(corrs_conn, axis=0, ddof = 1)
                    corrs_disc_sample_mean = np.mean(corrs_disc, axis=0)
                    corrs_disc_sample_err  = np.std(corrs_disc, axis=0, ddof = 1)
                    corrs_sample_mean = np.mean(corrs, axis=0)
                    corrs_sample_err  = np.std(corrs, axis=0, ddof = 1)
                    print("corrs_conn mean", corrs_conn_sample_mean)
                    print("corrs_conn err", corrs_conn_sample_err)
                    #print("corrs_disc", corrs_disc_sample_mean)
                    # bootstrap estimates
                    corrs_conn_boot     = np.array(bootstrap(corrs_conn,
                                                             Nboot = Nboot,
                                                             f = lambda x : mean(x)))
                    corrs_disc_boot     = np.array(bootstrap(corrs_disc,
                                                             Nboot = Nboot,
                                                             f = lambda x : mean(x)))        # vacuum-subtracted
                    corrs_boot          = np.array(bootstrap(corrs,
                                                             Nboot = Nboot,
                                                             f = lambda x : mean(x)))
                    corr_conn_mean      = np.mean(corrs_conn_boot, axis=0)
                    corr_conn_covar     = covar_from_boots(corrs_conn_boot)
                    corr_conn_err       = np.sqrt(np.diag(corr_conn_covar))
                    corr_disc_mean      = np.mean(corrs_disc_boot, axis=0)
                    corr_disc_covar     = covar_from_boots(corrs_disc_boot)
                    corr_disc_err       = np.sqrt(np.diag(corr_disc_covar))
                    corr_mean           = np.mean(corrs_boot, axis=0)
                    corr_covar          = covar_from_boots(corrs_boot)
                    corr_err            = np.sqrt(np.diag(corr_covar))
                    # ax1.hist(np.log(corrs_conn_boot[:, th[0]]/corr_conn_mean[th[0]]),
                    #         label=f'{construction}, $nt={th[0]}$', color=construction_color[construction],
                    #         density=True, alpha=0.5)
                    # ax2.hist(np.log(corrs_conn_boot[:, th[1]]/corr_conn_mean[th[1]]),
                    #         label=f'{construction}, $nt={th[1]}$',
                    #         color=construction_color[construction],
                    #         density=True, alpha=0.5)
                    # ax3.hist(np.log(corrs_conn_boot[:, th[2]]/corr_conn_mean[th[2]]),
                    #         color=construction_color[construction],
                    #         label=f'{construction}, $nt={th[2]}$', density=True, alpha=0.5)
                    # ax4.hist(np.log(corrs_conn_boot[:, th[3]]/corr_conn_mean[th[3]]),
                    #         color=construction_color[construction],
                    #         label=f'{construction}, $nt={th[3]}$', density=True, alpha=0.5)
                    # print("corrs_disc_sub", corr_disc_sub_mean)
                    #print("corrs_conn", corr_conn_mean)
                    # bootstrap effective mass
                    # meffs_sub_boot = np.array(bootstrap(corrs,
                    #                                Nboot = Nboot,
                    #                                f = meffs[construction][0]))
                    # meff_sub_mean = np.mean(meffs_sub_boot, axis=0)
                    # meff_sub_covar = covar_from_boots(meffs_sub_boot)
                    # meff_sub_err = np.sqrt(np.diag(meff_sub_covar))
                    print("corr_conn_boot_shape", corrs_conn_boot.shape)

                    meffs_boot = meffs[gstruct][0](corrs_conn_boot)
                    #np.log(corrs_conn_boot[:,:-1]/corrs_conn_boot[:,1:])

                                # meffs[construction [0]
                                #np.array(bootstrap(corrs_conn,
                                 #                   Nboot = Nboot,
                                 #                   f = meffs[construction][0]))
                    meff_mean = np.mean(meffs_boot, axis=0)
                    print("meff mean", meff_mean)
                    # meff_covar = covar_from_boots(meffs_boot)
                    meff_err = np.std(meffs_boot, axis=0)
                    print("meff err", meff_err)
                    # print("meff covar err", meff_err)
                    # write out files
                    # FIXME wow what a mess I made
                    # corr_conn_boot = np.vstack((corr_conn_mean, corr_conn_err)).T
                    # corr_disc_boot = np.vstack((corr_disc_mean, corr_disc_err)).T
                    # corr_boot = np.vstack((corr_mean, corr_err)).T
                    # meff_sub_boot = np.vstack((meff_sub_mean, meff_sub_err)).T
                    meff_boot = np.vstack((meff_mean, meff_err)).T
                    #
                    fname_conn = prefix_anls + '_meson_boot_Ct_conn_' + construction + '_' + '.npy'
                    #fname_disc = prefix_anls + '_meson_boot_Ct_disc_' + construction + '.npy'
                    #fname = prefix_anls + '_meson_boot_Ct_' + construction + '.npy'
                    #fname_conn_cov = prefix_anls + '_meson_boot_Ct_conn_cov_' + construction + '.npy'
                    #fname_disc_cov = prefix_anls + '_meson_boot_Ct_disc_cov_' + construction + '.npy'
                    #fname_cov = prefix_anls + '_meson_boot_Ct_cov_' + construction + '.npy'
                    np.save(fname_conn, np.array(corrs_conn_boot))
                    #np.save(fname_disc, np.array(corr_disc_boot))
                    #np.save(fname, np.array(corr_boot))
                    #np.save(fname_conn_cov, np.array(corr_conn_covar))
                    #np.save(fname_disc_cov, np.array(corr_disc_covar))
                    #np.save(fname_cov, np.array(corr_covar))
                    print("Wrote booted conn Cts to {}".format(fname_conn))
                    fname = prefix_anls + '_meson_boot_meff_' + meffs[gstruct][1] + '_' + construction + '_' + '.npy'
                    #fname_sub = prefix_anls + '_meson_boot_meff_sub_' + meffs[construction][1] + '_' + construction + '.npy'
                    #fname_cov = prefix_anls + '_meson_boot_meff_cov_' + meffs[construction][1] + '_' + construction + '.npy'
                    #fname_sub_cov = prefix_anls + '_meson_boot_meff_sub_cov_' + meffs[construction][1] + '_' + construction + '.npy'
                    np.save(fname, np.array(meff_boot))
                    #np.save(fname_sub, np.array(meff_sub_boot))
                    #np.save(fname_cov, np.array(meff_covar))
                    #np.save(fname_sub_cov, np.array(meff_sub_covar))
                    print("Wrote booted Meffs (from connecteds) to {}".format(fname))
    #ax1.set_xlim([-1, 1])
    #ax2.set_xlim([-1, 1])
    #ax3.set_xlim([-1, 1])
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.tight_layout()
    # plt.show()
    #fig.savefig("histo_large.png")


