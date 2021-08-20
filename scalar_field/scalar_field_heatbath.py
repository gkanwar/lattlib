"""
Complex scalar field in Nd using Numpy to handle multi-dim lattices.
"""

from scalar_field import *

import argparse
import numpy as np
import pickle
import sys
import time
import tqdm

def local_update(phi, action, eps):
    dims = phi.shape
    reject = 0
    it = np.nditer(phi, flags=['multi_index'])
    while not it.finished:
        x = it.multi_index
        old_val = np.copy(it[0])
        old_S = action.local_action(x, phi)
        # TEST:
        # old_S_tot = np.sum(action.action(phi))
        phi[x] = old_val + np.complex(
            eps*(np.random.random()*2.0 - 1.0),
            eps*(np.random.random()*2.0 - 1.0))
        new_S = action.local_action(x, phi)
        # TEST:
        # new_S_tot = np.sum(action.action(phi))
        # assert(np.isclose(new_S_tot - old_S_tot, new_S - old_S))

        delta_S = new_S - old_S
        if np.min([1.0, np.exp(-delta_S)]) < np.random.random():
            phi[x] = old_val
            reject += 1
        it.iternext()
    # Return the accept fraction
    return (phi.size - reject) / float(phi.size)

def metropolis_ensemble(init_cfg, action, eps, iters, skip, therm):
    dims = init_cfg.shape
    phi = init_cfg
    print("Running {} thermalization iters.".format(therm))
    start = time.time()
    for _ in tqdm.tqdm(range(skip*therm)):
        local_update(phi, action, eps)
    print("Thermalization done in {:.3g}s.".format(time.time() - start))
    accept = 0.0
    N_update = 0
    out = []
    Sout = []
    print("Running {} measurement iters.".format(iters))
    start = time.time()
    for it in tqdm.tqdm(range(iters)):
        for _ in range(skip):
            accept_new = local_update(phi, action, eps)
            accept += accept_new
            N_update += 1
        S = np.sum(action.action(phi))
        out.append(np.copy(phi))
        Sout.append(S)
        if it % 10 == 0:
            print("It {}/{} ({:.3g}s)".format(it, iters, time.time() - start))
            print("S = {}".format(S))
            print("Acc {}".format(accept / float(N_update)))
    accept = accept / float(N_update)
    print("Finished measurement iters.")
    print("Total accept rate is {}.".format(accept))
    print("Total time taken {:.3g}s.".format(time.time() - start))
    return accept, out, Sout

def main(out_prefix, init_cfg, action, eps, iters, skip, therm):
    _,ensemble, actions = metropolis_ensemble(init_cfg, action, eps, iters, skip, therm)
    print('Ensemble shape = {}.'.format(np.array(ensemble).shape))
    fname = out_prefix+'.dat'
    print('Writing ensemble to {}.'.format(fname))
    np.array(ensemble).tofile(fname)
    fname = out_prefix+'.S.dat'
    print('Writing actions to {}.'.format(fname))
    np.array(actions).tofile(fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metropolis ensemble for SHO.")
    # action
    parser.add_argument('--m2', type=float, required=True,
                        help='Lattice M^2 setting inverse spacing')
    parser.add_argument('--lam', type=float, default=0.0,
                        help='Coeffient lambda of phi^4 term')
    # logistics
    parser.add_argument('--out_prefix', type=str, required=True,
                        help='Output prefix for ensemble')
    parser.add_argument('--eps', type=float, required=True,
                        help='Step eps for metropolis')
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
        phi = np.fromfile(
            args.seed_file, dtype=np.complex128, count=np.prod(args.dims))
    else:
        print("Generating hot start cfg.")
        phi_r = np.random.normal(size=args.dims)
        phi_i = np.random.normal(size=args.dims)
        phi = phi_r+ np.complex(0,1)*phi_i
        
    # run
    main(args.out_prefix, phi, action, args.eps,
         args.iters, args.skip, args.therm)
    # final logs
    print('Again, args = {}'.format(args))
