from .scalar_field import *

import numpy as np
import time
import tqdm

def sample_pi(shape, dtype):
    out = np.random.normal(size=shape)
    if np.issubdtype(dtype, np.complexfloating):
        out = out + 1j * np.random.normal(size=shape)
    return out

def update_x_with_p(cfg, pi, action, t, dt):
    cfg += dt * pi
def update_p_with_x(cfg, pi, action, t, dt):
    F = action.force(cfg) #TODO: t dependence?
    pi += dt * F

# Mutates cfg, pi according to leapfrog update
def leapfrog_update(cfg, pi, action, dt, n_leap, verbose=False):
    if verbose: print("Leapfrog  update")
    start = time.time()
    update_x_with_p(cfg, pi, action, 0, dt / 2)
    for i in range(n_leap-1):
        update_p_with_x(cfg, pi, action, i*dt, dt)
        update_x_with_p(cfg, pi, action, (i+0.5)*dt, dt)
    update_p_with_x(cfg, pi, action, (n_leap-1)*dt, dt)
    update_x_with_p(cfg, pi, action, (n_leap-0.5)*dt, dt / 2)
    if verbose: print("TIME leapfrog {:.2f}s".format(time.time() - start))

def hmc_update(cfg, action, eps, n_leap, verbose=True):
    old_cfg = np.copy(cfg)
    old_S = np.sum(action.action(old_cfg, verbose=verbose))
    old_pi = sample_pi(cfg.shape, cfg.dtype)
    old_K = np.sum(np.abs(old_pi)**2) / 2
    old_H = old_S + old_K

    new_pi = np.copy(old_pi)
    cfg = np.copy(cfg)
    leapfrog_update(cfg, new_pi, action, eps, n_leap, verbose=verbose)

    new_S = np.sum(action.action(cfg, verbose=verbose))
    new_K = np.sum(np.abs(new_pi)**2) / 2
    new_H = new_S + new_K

    delta_H = new_H - old_H
    if verbose:
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
    if verbose:
        print("Acc {:.5g} (changed {})".format(min(1.0, np.exp(-delta_H)), acc))
    results = {
        'cfg': cfg,
        'S': S,
        'acc': acc,
        'deltaH': delta_H
    }
    return results

def hmc_ensemble(init_cfg, action, eps, n_leap, iters, skip, therm):
    cfg = init_cfg
    # HMC
    cfgs = []
    actions = []
    total_acc = 0
    total_dH = 0
    start = time.time()
    with tqdm.tqdm(total = skip*(therm+iters), postfix='Acc: ???, DeltaH: ???') as t:
        for i in range(-skip*therm, skip*iters):
            # print("MC step {} / {} ({:.4f}s)".format(i+1, skip*iters, time.time()-start))
            res = hmc_update(cfg, action, eps, n_leap)
            cfg, S, acc = res['cfg'], res['S'], res['acc']
            if i >= 0:
                total_acc += acc
                total_dH += res['deltaH']

            # save cfg
            if i >= 0 and i % skip == 0:
                print("Saving cfg!")
                cfgs.append(cfg)
                actions.append(S)
                t.postfix = 'Acc: {:.3f}, DeltaH: {:.3f}'.format(total_acc / (i+1), total_dH / (i+1))
            t.update()
    print("MC finished.")
    print("Total acc {:.4f}".format(total_acc / (skip*iters)))
    print("Mean delta H {:.4f}".format(total_dH / (skip*iters)))
    print("Total time taken {:.1f}s.".format(time.time() - start))
    return cfgs, actions

def aug_hmc_ensemble(init_cfg, action, eps, n_leap, iters, skip, therm):
    cfg = init_cfg
    # HMC
    cfgs = []
    actions = []
    total_acc = 0
    total_dH = 0
    start = time.time()
    with tqdm.tqdm(total = skip*(therm+iters), postfix='Acc: ???, DeltaH: ???') as t:
        for i in range(-skip*therm, skip*iters):
            # print("MC step {} / {} ({:.4f}s)".format(i+1, skip*iters, time.time()-start))
            res = hmc_update(cfg, action, eps, n_leap)
            cfg, S, acc = res['cfg'], res['S'], res['acc']
            if i >= 0:
                total_acc += acc
                total_dH += res['deltaH']
            # Augmented: propose sign flip, Metropolis A/R
            prop_cfg = -cfg
            prop_S = np.sum(action.action(prop_cfg))
            if np.random.random() < np.exp(-prop_S + S):
                cfg = prop_cfg
                S = prop_S

            # save cfg
            if i >= 0 and i % skip == 0:
                print("Saving cfg!")
                cfgs.append(cfg)
                actions.append(S)
                t.postfix = 'Acc: {:.3f}, DeltaH: {:.3f}'.format(total_acc / (i+1), total_dH / (i+1))
            t.update()
    print("MC finished.")
    print("Total acc {:.4f}".format(total_acc / (skip*iters)))
    print("Mean delta H {:.4f}".format(total_dH / (skip*iters)))
    print("Total time taken {:.1f}s.".format(time.time() - start))
    return cfgs, actions
