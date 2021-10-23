"""
Compute correlators for 2D Ising model.
"""

import argparse
import numpy as np
import time
import tqdm

def compute_twopt_vev(c, alpha, beta):
    alpha_term = alpha * np.sum(c, axis=0)
    beta_term = beta * np.sum(
        c * np.roll(c, -1, axis=0) * np.roll(c, -2, axis=0), axis=0)
    op = alpha_term + beta_term
    assert(len(op.shape) == 1) # projected to just time
    Lt = op.shape[0]
    twopt = np.zeros((Lt,), dtype=np.float64)
    for dt in range(Lt):
        twopt[dt] = np.sum(op * np.roll(op, -dt, axis=0)) / Lt
    return twopt, op

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute 2D Ising corrs on ensemble.')
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True,
                        help='One sigma coeff.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Three sigma coeff.')
    args = parser.parse_args()
    print('Running with args = {}'.format(args))

    fname = args.tag + '.dat'
    print('Reading ensemble from {}'.format(fname))
    cfgs = np.fromfile(fname, dtype=np.float64).reshape(
        args.Ncfg, args.Lx, args.Lt)
    twopts = []
    vevs = []
    start = time.time()
    for c in tqdm.tqdm(cfgs):
        twopt, vev = compute_twopt_vev(c, args.alpha, args.beta)
        twopts.append(twopt)
        vevs.append(vev)
    twopts = np.array(twopts)
    vevs = np.array(vevs)
    print('Done all corrs in {:.1f}s'.format(time.time() - start))
    print('twopts shape = {}'.format(twopts.shape))
    print('vevs shape = {}'.format(vevs.shape))
    fname = args.tag + '.twopt_a{:.2f}_b{:.2f}.dat'.format(args.alpha, args.beta)
    twopts.tofile(fname)
    print('Wrote twopts to {}.'.format(fname))
    fname = args.tag + '.vev_a{:.2f}_b{:.2f}.dat'.format(args.alpha, args.beta)
    vevs.tofile(fname)
    print('Wrote vevs to {}.'.format(fname))
