"""
XY model HMC simulation.
"""

from xy_model.xy_model import *

import argparse
import numpy as np
import sys
import time
import tqdm

def sample_pi(shape):
    return np.random.normal(size=shape)

def update_x_with_p(cfg, pi, action, t, dt):
    cfg += dt * pi
    cfg %= 2*np.pi
def update_p_with_x(cfg, pi, action, t, dt):
    F = action.force(cfg) #TODO: t dependence?
    pi += dt * F

# Mutates cfg, pi according to leapfrog update
def leapfrog_update(cfg, pi, action, tau, n_leap):
    print("Leapfrog  update")
    start = time.time()
    dt = tau / n_leap
    update_x_with_p(cfg, pi, action, 0, dt / 2)
    for i in range(n_leap-1):
        update_p_with_x(cfg, pi, action, i*dt, dt)
        update_x_with_p(cfg, pi, action, (i+0.5)*dt, dt)
    update_p_with_x(cfg, pi, action, (n_leap-1)*dt, dt)
    update_x_with_p(cfg, pi, action, (n_leap-0.5)*dt, dt / 2)
    print("TIME leapfrog {:.2f}s".format(time.time() - start))

def hmc_ensemble(init_cfg, action, tau, n_leap, iters, skip, therm):
    cfg = init_cfg
    # HMC
    cfgs = []
    actions = []
    total_acc = 0
    start = time.time()
    for i in tqdm.tqdm(range(-skip*therm, skip*iters)):
        print("MC step {} / {} ({:.4f}s)".format(i+1, skip*iters, time.time()-start))
        old_cfg = np.copy(cfg)
        old_S = np.sum(action.action(old_cfg, verbose=True))
        old_pi = sample_pi(cfg.shape)
        old_K = np.sum(np.abs(old_pi)**2) / 2
        old_H = old_S + old_K

        new_pi = np.copy(old_pi)
        cfg = np.copy(cfg)
        leapfrog_update(cfg, new_pi, action, tau, n_leap)

        new_S = np.sum(action.action(cfg, verbose=True))
        new_K = np.sum(np.abs(new_pi)**2) / 2
        new_H = new_S + new_K

        delta_H = new_H - old_H
        print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
        print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))
        print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))

        # metropolis step
        acc = 0
        if np.random.random() < np.exp(-delta_H):
            acc = 1
            S = new_S
        else:
            cfg = old_cfg
            S = old_S
        print("Acc {:.5g} (changed {})".format(min(1.0, np.exp(-delta_H)), acc))
        if i >= 0: total_acc += acc
        
        # save cfg
        if i >= 0 and i % skip == 0:
            print("Saving cfg!")
            cfgs.append(cfg)
            actions.append(S)
    print("MC finished.")
    print("Total acc {:.4f}".format(total_acc / (skip*iters)))
    print("Total time taken {:.3g}s.".format(time.time() - start))
    return cfgs, actions

def main(out_prefix, init_cfg, action, tau, n_leap, iters, skip, therm):
    ensemble, actions = hmc_ensemble(init_cfg, action, tau, n_leap, iters, skip, therm)
    print('Ensemble shape = {}.'.format(np.array(ensemble).shape))
    fname = out_prefix+'.dat'
    print('Writing ensemble to {}.'.format(fname))
    np.array(ensemble).tofile(fname)
    fname = out_prefix+'.S.dat'
    print('Writing actions to {}.'.format(fname))
    np.array(actions).tofile(fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMC ensemble for XY.")
    # action
    parser.add_argument('--beta', type=float, required=True,
                        help='Lattice coupling beta')
    # logistics
    parser.add_argument('--out_prefix', type=str, required=True,
                        help='Output prefix for ensemble')
    parser.add_argument('--tau', type=float, required=True,
                        help='HMC traj tau')
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
    action = Action([XYCosTerm(args.beta)])
    print('Using action = {}'.format(action))
    if args.seed is None:
        args.seed = np.random.randint(np.iinfo('uint32').max)
        print("Generated random seed = {}".format(args.seed))
    np.random.seed(args.seed)
    print("Using seed = {}.".format(args.seed))
    if args.seed_file is not None:
        print("Loading seed cfg from {}".format(args.seed_file))
        theta = np.fromfile(
            args.seed_file, dtype=np.float64, count=np.prod(args.dims))
        theta = theta.reshape(args.dims)
    else:
        print("Generating hot start cfg.")
        theta = 0.3*np.random.normal(size=args.dims) % (2*np.pi)
    # run
    main(args.out_prefix, theta, action, args.tau, args.n_leap,
         args.iters, args.skip, args.therm)
    # final logs
    print('Again, args = {}'.format(args))
