"""
Compute the Potts actions on cfgs in the given ensemble.
"""

import argparse
import numpy as np

# Action is normalized/shifted to equal Ising for identical beta.
def action(beta, cfg):
    assert(len(cfg.shape) == 2)
    return (beta * (2*(np.roll(cfg, -1, axis=0) == cfg)-1) +
            beta * (2*(np.roll(cfg, -1, axis=1) == cfg)-1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute Potts actions on cfgs in the given ensemble.')
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--beta', type=float, required=True)
    args = parser.parse_args()
    fname = args.tag + '.dat'
    cfgs = np.fromfile(fname, dtype=np.float64).reshape(
        args.Ncfg, args.Lx, args.Lt)
    actions = []
    for c in cfgs:
        S = np.sum(action(args.beta, c))
        print('S = {}'.format(S))
        actions.append(S)
    fname = args.tag + '.S.dat'
    np.array(actions).tofile(fname)
    print('Wrote actions to {}.'.format(fname))
