import argparse
import os
from pylib.analysis import mean, rmean, log_meff, acosh_meff, bootstrap
from pybin.schwinger_hmc import handle_bc_arg
from schwinger.schwinger import *

# open corr files, bootstrap, writeout boostrapped with uncertainties

def asinh_meff(x):
    corr = mean(x)
    return np.arcsinh((corr[:-2] - corr[2:])/(2*corr[1:-1]))

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
    meff_mean = []
    meff_err = []
    for construction in constructions:
        print('construction', construction)
        # read in correlation file
        fname = prefix_meas + '_meson_Ct_' + construction + '.npy'
        corrs = np.load(fname)
        print('Read Cts from {}'.format(fname))
        # bootstrap correlation function
        corr_mean, corr_err = bootstrap(corrs, Nboot = args.Ncfg, f = lambda x : mean(x))
        # bootstrap effective mass
        meff_mean, meff_err = bootstrap(corrs, Nboot = args.Ncfg, f = meffs[construction][0])
        # write out files
        corr_boot = np.vstack((corr_mean, corr_err)).T
        meff_boot = np.vstack((meff_mean, meff_err)).T
        fname = prefix_anls + '_meson_boot_Ct_' + construction + '.npy'
        np.save(fname, np.array(corr_boot))
        print("Wrote booted Cts to {}".format(fname))
        fname = prefix_anls + '_meson_boot_meff_' + meffs[construction][1] + '_' + construction + '.npy'
        np.save(fname, np.array(meff_boot))
        print("Wrote booted Meffs to {}".format(fname))
