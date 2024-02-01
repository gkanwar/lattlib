import argparse
import numpy as np
import sys

from gauge_theory.gauge_theory_hmc import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HMC for U(1)/SU(N) gauge theory')
    # general params
    parser.add_argument('--seed', type=int)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--n_skip', type=int, required=True)
    parser.add_argument('--n_therm', type=int, required=True)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--n_leap', type=int, default=20)
    parser.add_argument('--init_cfg', type=str)
    parser.add_argument('--Nc', type=int, required=True)
    parser.add_argument('--compute_topo', action='store_true',
                        help='Additionally compute and save topo charge Q')
    # action params
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--Zn', type=int)
    parser.add_argument('--beta_prec', type=int, default=2)
    # lattice
    parser.add_argument('dims', metavar='d', type=int, nargs='+')
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()

    # handle params
    if len(args.tag) > 0:
        args.tag = "_" + args.tag
    if args.seed is None:
        args.seed = np.random.randint(np.iinfo('uint32').max)
        print("Generated random seed = {}".format(args.seed))
    np.random.seed(args.seed)
    print("Using seed = {}.".format(args.seed))
    L = args.dims
    Nd = len(L)
    shape = tuple([Nd] + list(L) + [args.Nc,args.Nc])
    if args.init_cfg is None:
        print('Generating warm init cfg.')
        init_cfg_A = 0.1*(np.random.normal(size=shape) + 1j*np.random.normal(size=shape))
        gauge_proj_hermitian(init_cfg_A)
        gauge_proj_traceless(init_cfg_A)
        cfg = gauge_expm(1j * init_cfg_A)
        # FORNOW: random center rotation
        index = tuple([Nd-1] + [slice(None)]*(Nd-1) + [0])
        cfg[index] *= np.exp(1j * np.random.randint(args.Nc) * 2*np.pi/args.Nc)
    else:
        print('Loading init cfg from {}.'.format(args.init_cfg))
        cfg = np.fromfile(args.init_cfg, dtype=np.complex128)
        cfg = cfg.reshape(shape)
    tot_steps = args.Ncfg * args.n_skip
    if args.type == "pure_gauge":
        assert(args.beta is not None)
        action = PureGaugeAction(beta=args.beta, beta_prec=args.beta_prec)
    elif args.type == "pure_gauge_zn":
        assert(args.beta is not None)
        assert(args.Zn is not None)
        action = PureGaugeModZnAction(beta=args.beta, beta_prec=args.beta_prec, Zn=args.Zn)
    else:
        print("Unknown action type {}".format(args.type))
        sys.exit(1)

    # do the thing!
    res = run_hmc(L, tot_steps, args.n_skip, args.n_therm,
                  args.tau, args.n_leap, action, cfg,
                  should_compute_topo=args.compute_topo)
    cfgs = res['cfgs']
    plaqs = res['plaqs']
    acts = res['acts']
    if args.compute_topo:
        topos = res['topos']

    # write stuff out
    group_tag = 'u1' if args.Nc == 1 else 'su{:d}'.format(args.Nc)
    prefix = '{:s}_{:s}_N{:d}_skip{:d}_therm{:d}_{:s}{:s}'.format(
        group_tag, action.make_tag(), args.Ncfg, args.n_skip, args.n_therm,
        '_'.join(map(str, L)), args.tag)
    print(f'Writing observables to {prefix}.*.npy')
    np.save(f'{prefix}.npy', np.array(cfgs))
    np.save(f'{prefix}.plaq.npy', np.array(plaqs))
    np.save(f'{prefix}.S.npy', np.array(acts))
    if args.compute_topo:
        np.save(f'{prefix}.topo.npy', np.array(topos))

    print("TIME ensemble gen {:.2f}s".format(time.time()-start))
