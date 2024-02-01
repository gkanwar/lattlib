"""
Measure two-point function given scalar field ensemble.
"""

from scalar_field.scalar_field import *

import argparse
import numpy as np
import os
import sys
import time

def compute_twopts(cfgs, verbose):
    start = time.time()
    corrs = []
    Nd = len(cfgs.shape[1:])
    spatial_axes = tuple(range(0,Nd-1))
    for i,c in enumerate(cfgs):
        if verbose and i % 10 == 0:
            print('Cfg {} / {} ({:.2f}s)'.format(
                i+1, cfgs.shape[0], time.time()-start))
        corr = np.array(all_corrs(c, xspace, tspace))
        # corr = np.sum(np.mean(corr, axis=0), axis=spatial_axes) # mom proj
        corr = np.mean(corr, axis=0)
        corrs.append(corr)
    print('Two-points done. '
          'Total time = {:.2f}s.'.format(time.time()-start))
    return corrs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Measure two-pt function for scalar field theory')
    # ensemble path
    parser.add_argument('--path', type=str, help='Path to .npy format ensemble')
    parser.add_argument('--tag', type=str, help='Prefix for .dat format ensemble')
    parser.add_argument('--Ncfg', type=int)
    parser.add_argument('--use_real64', action='store_true')
    parser.add_argument('--use_real32', action='store_true')
    # corr params
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('dims', metavar='d', type=int, nargs='+')
    args = parser.parse_args()
    print('Running with args = {}'.format(args))

    if args.path is not None: # for .npy ensembles
        print('Reading ensemble from {}.'.format(args.path))
        cfgs = np.load(args.path)
        dtype = cfgs.dtype
        args.tag = os.path.splitext(args.path)[0]
    else: # for (older format) .dat ensembles
        assert args.tag is not None
        assert args.Ncfg is not None
        assert args.dims is not None
        if args.use_real32:
            fname = args.tag + '.cfgs.dat'
            dtype = np.float32
        else:
            fname = args.tag + '.dat'
            dtype = np.float64 if args.use_real64 else np.complex128
        print('Reading ensemble from {}.'.format(fname))
        cfgs = np.fromfile(fname, dtype=dtype)
        cfgs = cfgs.reshape(args.Ncfg, *args.dims)
    corrs = compute_twopts(cfgs, verbose=not args.quiet)
    twopt = np.array(corrs)
    print('twopt shape {}'.format(twopt.shape))
    fname = args.tag + '.twopt.dat'
    print('Writing two-pts to {}.'.format(fname))
    twopt.tofile(fname)
            
