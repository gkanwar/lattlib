"""
HMC for (complex) scalar fields in arbitrary dimensions.
"""

import argparse

from scalar_field.scalar_field_hmc import *


def main(out_prefix, init_cfg, action, eps, n_leap, iters, skip, therm):
    ensemble, actions = hmc_ensemble(init_cfg, action, eps, n_leap, iters, skip, therm)
    print('Ensemble shape = {}.'.format(np.array(ensemble).shape))
    fname = out_prefix+'.npy'
    print('Writing ensemble to {}.'.format(fname))
    np.save(fname, np.array(ensemble))
    fname = out_prefix+'.S.npy'
    print('Writing actions to {}.'.format(fname))
    np.save(fname, np.array(actions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMC ensemble for scalar field.")
    # action
    parser.add_argument('--m2', type=float, required=True,
                        help='Lattice M^2 setting inverse spacing')
    parser.add_argument('--lam', type=float, default=0.0,
                        help='Coeffient lambda of phi^4 term')
    parser.add_argument('--complex_type', action='store_true',
                        help='Use complex phi fields')
    # logistics
    parser.add_argument('--out_prefix', type=str, required=True,
                        help='Output prefix for ensemble')
    parser.add_argument('--eps', type=float, required=True,
                        help='HMC traj eps')
    parser.add_argument('--n_leap', type=int, required=True,
                        help='HMC traj leaps')
    parser.add_argument('--iters', type=int, required=True,
                        help='Number of measurements')
    parser.add_argument('--therm', type=int, required=True,
                        help='Number of therm iters')
    parser.add_argument('--skip', type=int, required=True,
                        help='Number skipped between meas')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--seed_file', type=str)
    # lattice
    parser.add_argument('dims', metavar='d', type=int, nargs='+',
                        help='Dimensions')
    args = parser.parse_args()
    print('Running with args = {}'.format(args))
    # parse args
    action = Action([ScalarKineticMassTerm(args.m2)])
    if args.lam != 0.0:
        action.terms.append(ScalarPhi4Term(args.lam))
    print('Using action = {}'.format(action))
    if args.seed is None:
        args.seed = np.random.randint(np.iinfo('uint32').max)
        print("Generated random seed = {}".format(args.seed))
    np.random.seed(args.seed)
    print("Using seed = {}.".format(args.seed))
    if args.seed_file is not None:
        print("Loading seed cfg from {}".format(args.seed_file))
        dtype = np.complex128 if args.complex_type else np.float64
        phi = np.fromfile(
            args.seed_file, dtype=dtype, count=np.prod(args.dims))
        phi = phi.reshape(args.dims)
    else:
        print("Generating hot start cfg.")
        phi = 0.3*np.random.normal(size=args.dims)
        if args.complex_type:
            phi = phi + 1j * 0.3*np.random.normal(size=args.dims)
    # run
    main(args.out_prefix, phi, action, args.eps, args.n_leap,
         args.iters, args.skip, args.therm)
    # final logs
    print('Again, args = {}'.format(args))

