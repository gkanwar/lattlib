### Read XY configs, measure correlators, write two-point function files.

import argparse
import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, required=True)
parser.add_argument('--Ncfg', type=int, required=True)
parser.add_argument('shape', metavar='d', type=int, nargs='+')
args = parser.parse_args()

# NOTE: Might be cleaner to use .npy, but these are in .dat format to interop with Mathematica
fname = args.prefix + '.dat'
cfgs = np.fromfile(fname, dtype=np.float64).reshape((args.Ncfg,) + tuple(args.shape))
corrs = []
Lt = args.shape[-1]
for cfg in tqdm.tqdm(cfgs):
    corr = np.zeros((Lt,), dtype=np.float64)
    for dt in range(Lt):
        corr[dt] = np.mean(np.cos(cfg - np.roll(cfg, -dt, axis=-1)))
    corrs.append(corr)
np.array(corrs).tofile(args.prefix + '.twopt.dat')
