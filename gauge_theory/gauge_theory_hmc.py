import argparse
import numpy as np
import sys
import time
import tqdm

from .gauge_theory import *


def hmc_update(cfg, action, tau, n_leap, verbose=True):
    old_cfg = np.copy(cfg)
    old_S = action.init_traj(old_cfg)
    old_pi = sample_pi(cfg.shape)
    old_K = np.real(np.sum(np.trace(old_pi @ old_pi, axis1=-1, axis2=-2)) / 2)
    old_H = old_S + old_K

    cfg = np.copy(cfg)
    new_pi = np.copy(old_pi)
    leapfrog_update(cfg, new_pi, action, tau, n_leap, verbose=verbose)

    new_S = action.compute_action(cfg)
    new_K = np.real(np.sum(np.trace(new_pi @ new_pi, axis1=-1, axis2=-2)) / 2)
    new_H = new_S + new_K

    delta_H = new_H - old_H
    if verbose:
        print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))
        print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
        print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))
        
    # metropolis step
    acc = 0
    if np.random.random() < np.exp(-delta_H):
        acc = 1
        S = new_S
    else:
        cfg = old_cfg
        S = old_S
    if verbose:
        print("Acc {:.5g} (changed {})".format(min(1.0, np.exp(-delta_H)), acc))
    return cfg, S, acc
    

"""
Run a full HMC sequence of `n_step` trajectories, saving configurations and info
every `n_skip` trajectories. Run is preceeded by `n_therm` thermalization
trajectories. Each trajectory is characterized by integration time `tau`
integrated by a leapfrog integrator employing `n_leap` steps.

Returns: A dict `res` with keys 'cfgs' and 'plaqs', and optionally 'topos'.
"""
def run_hmc(L, n_step, n_skip, n_therm, tau, n_leap, action, cfg,
            should_compute_topo=False):
    if should_compute_topo:
        if len(L) == 2:
            compute_topo = topo_charge_density_2d
        elif len(L) == 4:
            compute_topo = topo_charge_density_4d
        else:
            raise NotImplementedError()
    V = np.prod(L)
    Nplaq = len(L) * (len(L) - 1) // 2
    Nd = len(L)
    Nc = cfg.shape[-1]
    # MC updates
    total_acc = 0
    cfgs = []
    plaqs = []
    acts = []
    topos = []
    with tqdm.tqdm(total = n_therm + n_step, postfix='Acc: ???') as t:
        for i in range(-n_therm, n_step):
            print("MC step {} / {}".format(i+1, n_step))
            cfg, S, acc = hmc_update(cfg, action, tau, n_leap)
            if i >= 0:
                total_acc += acc
                t.postfix = 'Acc: {:.3f}'.format(total_acc / (i+1))

            # avg plaq
            plaq = np.sum(np.real(closed_plaqs(cfg))) / (Nplaq * V)
            print("Average plaq = {:.6g}".format(plaq))
            # action
            act = action.compute_action(cfg)
            # topo Q
            if should_compute_topo:
                topo = np.sum(compute_topo_u1_2d(cfg))
                print("Topo = {:d}".format(int(round(topo))))
            else:
                topo = None

            # save cfg
            if i >= 0 and i % n_skip == 0:
                print("Saving cfg!")
                cfgs.append(cfg)
                plaqs.append(plaq)
                acts.append(act)
                if should_compute_topo: topos.append(topo)
            t.update()
            
    print("MC finished.")
    print("Total acc {:.4f}".format(total_acc / n_step))
    res = {
        'cfgs': cfgs,
        'plaqs': plaqs,
        'acts': acts
    }
    if should_compute_topo:
        res['topos'] = topos
    return res
