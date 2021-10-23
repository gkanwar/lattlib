"""
Sample clusters for 2D Potts model.
"""

from cftp import *

import argparse
import networkx as nx
import numpy as np
import time
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample potts cfgs from clusters.')
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--n_state', type=int, required=True)
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    args = parser.parse_args()
    print('Running with args = {}'.format(args))

    fname = args.tag + '.cluster.dat'
    print('Reading cluster ensemble from {}'.format(fname))
    cfgs = np.fromfile(fname, dtype=np.float64).reshape(
        args.Ncfg, 2+1, args.Lx, args.Lt)

    L = (args.Lx, args.Lt)
    spin_cfgs = []
    start = time.time()
    for c_and_mag in tqdm.tqdm(cfgs):
        c = c_and_mag[:2]
        mag = c_and_mag[2]
        g = make_cluster_graph(c)
        g = extend_graph_with_ghost(g, mag)
        comps  = nx.connected_components(g)
        spin_cfg = np.zeros(L)
        for comp in comps:
            if 'ghost' in comp:
                val = 0
            else:
                val = np.random.choice(np.arange(args.n_state))
            for x in comp:
                if x == 'ghost': continue
                spin_cfg[x] = val
        spin_cfgs.append(spin_cfg)
    print('Potts resampling complete!')
    print('Total time = {:.1f}s'.format(time.time() - start))
    fname = args.tag + '.dat'
    np.array(spin_cfgs).tofile(fname)
    print('Wrote Potts spin cfgs to file {}'.format(fname))
