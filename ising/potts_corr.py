"""
Compute correlators for 2D Potts model (either directly or random cluster).
"""

from cftp import *

import argparse
import math
import networkx as nx
import numpy as np
import time
import tqdm

def compute_twopt(c, use_extra_ops):
    op = np.sum(c, axis=0)
    assert(len(op.shape) == 1) # projected to just time
    Lt = op.shape[0]
    ops = [np.ones((Lt,), dtype=np.complex128), op]
    # twopt = np.zeros((Lt,), dtype=np.complex128)
    # for dt in range(Lt):
    #     twopt[dt] = np.sum(np.conj(op) * np.roll(op, -dt, axis=0)) / Lt
    # return twopt, op
    if use_extra_ops:
        # (sigma_z^x)^2 = (sigma_z^x)^*
        ops.append(np.sum(np.conj(c), axis=0))
        # sigma_z^x (sigma_z^(x+1))^*
        ops.append(np.sum(c * np.roll(np.conj(c), -1, axis=0), axis=0))
        # (sigma_z^x)^* sigma_z^(x+1)
        ops.append(np.sum(np.conj(c) * np.roll(c, -1, axis=0), axis=0))
        # sigma_z^x (sigma_z^(x+2))^*
        ops.append(np.sum(c * np.roll(np.conj(c), -2, axis=0), axis=0))
        # (sigma_z^x)^* sigma_z^(x+2)
        ops.append(np.sum(np.conj(c) * np.roll(c, -2, axis=0), axis=0))
        # sigma_z^x sigma_z^(x+1)
        ops.append(np.sum(c * np.roll(c, -1, axis=0), axis=0))
        # (sigma_z^x)^* (sigma_z^(x+1))^*
        ops.append(np.sum(np.conj(c * np.roll(c, -1, axis=0)), axis=0))
        
    N_ops = len(ops)
    twopt = np.zeros((Lt,N_ops,N_ops), dtype=np.complex128)
    for dt in range(Lt):
        for src,src_op in enumerate(ops):
            for snk,snk_op in enumerate(ops):
                twopt[dt,src,snk] = np.sum(np.conj(src_op) * np.roll(snk_op, -dt, axis=0)) / Lt
    return twopt

def compute_twopt_cluster(c_and_mag):
    c = c_and_mag[:2]
    mag = c_and_mag[2]
    Lt = c.shape[-1]
    g = make_cluster_graph(c)
    g = extend_graph_with_ghost(g, mag)
    comps = nx.connected_components(g)
    twopt = np.zeros((Lt,), dtype=np.complex128)
    mag_count = 0
    for comp in comps:
        counts = np.zeros((Lt,), dtype=np.complex128)
        ghost_comp = 'ghost' in comp
        for x in comp:
            if x == 'ghost': continue
            counts[x[-1]] += 1
            if ghost_comp: mag_count += 1
        for dt in range(Lt):
            twopt[dt] += np.sum(counts * np.roll(counts, -dt, axis=0)) / Lt
    vev = mag_count/Lt
    return twopt, vev

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute 2D Potts corrs on ensemble.')
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--n_state', type=int, required=True)
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--use_clusters', action='store_true')
    parser.add_argument('--use_extra_ops', action='store_true')
    args = parser.parse_args()
    print('Running with args = {}'.format(args))

    if args.use_clusters:
        fname = args.tag + '.cluster.dat'
        print('Reading cluster ensemble from {}'.format(fname))
        cfgs = np.fromfile(fname, dtype=np.float64).reshape(
            args.Ncfg, 2+1, args.Lx, args.Lt)
    else:
        fname = args.tag + '.dat'
        print('Reading ensemble from {}'.format(fname))
        cfgs = np.fromfile(fname, dtype=np.float64).reshape(
            args.Ncfg, args.Lx, args.Lt)
        cfgs = np.exp(2j * math.pi * cfgs / args.n_state)
    twopts = []
    vevs = []
    start = time.time()
    for c in tqdm.tqdm(cfgs):
        if args.use_clusters:
            twopt,vev = compute_twopt_cluster(c)
            vevs.append(vev)
        else:
            twopt = compute_twopt(c, args.use_extra_ops)
        twopts.append(twopt)
    twopts = np.array(twopts)
    vevs = np.array(vevs)
    print('Done all corrs in {:.1f}s'.format(time.time() - start))
    print('twopts shape = {}'.format(twopts.shape))
    print('vevs shape = {}'.format(vevs.shape))
    if args.use_clusters:
        fname = args.tag + '.twopt_clusters.dat'
    else:
        fname = args.tag + '.twopt.dat'
    twopts.tofile(fname)
    print('Wrote twopts to {}.'.format(fname))
    if args.use_clusters:
        fname = args.tag + '.vev_clusters.dat'
    else:
        fname = args.tag + '.vev.dat'
    vevs.tofile(fname)
    print('Wrote vevs to {}.'.format(fname))
