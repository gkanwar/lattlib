"""
Wolff cluster updates for the XY model.

Note: PhysRevD.45.2098 proposes using the Wolff cluster updates on the 1D XY
model resulting from gauge fixing U(1) gauge theory in 2D. To relate the
theories, keep beta identical and set the number of spins = Lx*Ly.
"""

import numpy as np
import tqdm
from .xy_model import *

def flip_spin(th, *, th_r):
    return (2*th_r - th) % (2*np.pi)

def get_neighbors(x, *, shape):
    ns = []
    for mu in range(len(shape)):
        ns.append(tuple((x[nu]+1) % shape[nu] if nu == mu else x[nu] for nu in range(len(shape))))
        ns.append(tuple((x[nu]-1) % shape[nu] if nu == mu else x[nu] for nu in range(len(shape))))
    return ns

def cluster_update(cfg, *, beta):
    th_r = 2*np.pi * np.random.random() # represent unit vec r by angle
    mark = np.zeros(cfg.shape, dtype=np.uint8)
    x = tuple(np.random.randint(L_mu) for L_mu in cfg.shape)
    mark[x] = 1
    queue = [x]
    while len(queue) > 0:
        x = queue.pop()
        th_x = cfg[x]
        ns = get_neighbors(x, shape=cfg.shape)
        for n in ns:
            if mark[n]: continue
            th_n = cfg[n]
            p = 1 - np.exp(min(0, -2*beta*np.sin(th_x - th_r)*np.sin(th_n - th_r)))
            if np.random.random() < p:
                mark[n] = 1
                queue.append(n)
    cfg[mark == 1] = flip_spin(cfg[mark == 1], th_r=th_r)
    return cfg, mark

def run_cluster_mcmc(cfg, *, beta, N_cfg, n_therm, n_skip, custom_step=None):
    action = Action([XYCosTerm(beta)])
    ensemble = []
    clusters = []
    actions = []
    for i in tqdm.tqdm(range(-n_therm*n_skip, N_cfg*n_skip)):
        cfg, cluster = cluster_update(cfg, beta=beta)
        # useful for e.g. flipping gauge fixing dir for XY 1D <-> U(1) 2D mapping
        if custom_step is not None:
            cfg = custom_step(cfg)
        if i >= 0 and (i+1) % n_skip == 0:
            ensemble.append(np.copy(cfg))
            clusters.append(cluster)
            actions.append(np.sum(action.action(cfg)))
    assert len(ensemble) == N_cfg
    return {
        'cfgs': np.array(ensemble),
        'clusters': np.array(clusters),
        'actions': np.array(actions)
    }

