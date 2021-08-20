import argparse
import numpy as np
import time

from xy_model.xy_cluster import run_cluster_mcmc

def np_wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def check_plaq_reconstruct(links, plaqs):
    plaqs2 = (
        links[:,0] + np.roll(links[:,1], -1, axis=1) +
        np.roll(-links[:,0], -1, axis=2) - links[:,1]) % (2*np.pi)
    diff_plaqs = np_wrap(plaqs2 - plaqs)
    assert np.allclose(diff_plaqs, np.zeros_like(diff_plaqs)), \
        f'Plaq reconstruction failed, diff =\n{diff_plaqs}'

def convert_plaqs_to_links(plaqs):
    Nd = len(plaqs.shape)-1
    assert Nd == 2, plaqs.shape
    links = np.zeros((plaqs.shape[0], Nd) + plaqs.shape[1:])
    links[:,1] = np.cumsum(np.insert(plaqs[:,:-1,:], 0, 0, axis=1), axis=1) % (2*np.pi)
    rem_plaqs = -(plaqs[:,-1,:] + links[:,1,-1,:])
    links[:,0,-1] = np.cumsum(np.insert(rem_plaqs[:,:-1], 0, 0, axis=1), axis=1) % (2*np.pi)
    check_plaq_reconstruct(links, plaqs)
    return np.exp(1j * links)

def convert_xy_to_u1(cfgs, *, shape):
    assert len(shape) == 2, 'specialized for 2d U(1)'
    assert np.prod(shape) == cfgs.shape[-1]
    polya = 2*np.pi*np.random.random()
    flat_plaqs = cfgs - np.roll(cfgs, -1, axis=1)
    shaped_plaqs = flat_plaqs.reshape((cfgs.shape[0],) + shape)
    for i in range(shaped_plaqs.shape[1]):
        if i % 2 == 0: continue
        shaped_plaqs[:,i] = shaped_plaqs[:,i,::-1]
    return convert_plaqs_to_links(shaped_plaqs)

def flip_xy_snake(cfg, *, shape):
    assert len(cfg.shape) == 1, 'must be 1D XY config'
    assert np.prod(shape) == cfg.shape[0], 'volumes must agree'
    flat_plaqs = cfg - np.roll(cfg, -1)
    shaped_plaqs = flat_plaqs.reshape(shape)
    for i in range(shaped_plaqs.shape[0]):
        if i % 2 == 0: continue
        shaped_plaqs[i] = shaped_plaqs[i,::-1]
    shaped_plaqs = -np.transpose(shaped_plaqs)
    for i in range(shaped_plaqs.shape[0]):
        if i % 2 == 0: continue
        shaped_plaqs[i] = shaped_plaqs[i,::-1]
    flat_plaqs_2 = shaped_plaqs.flatten()
    return np.cumsum(np.insert(flat_plaqs_2[:-1], 0, 2*np.pi*np.random.random())) % (2*np.pi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Wolff cluster MCMC for 2d pure gauge U(1)')
    # general params
    parser.add_argument('--seed', type=int)
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--n_skip', type=int, required=True)
    parser.add_argument('--n_therm', type=int, required=True)
    parser.add_argument('--init_cfg', type=str)
    parser.add_argument('--tag', type=str, default="")
    # action params
    parser.add_argument('--beta', type=float)
    parser.add_argument('--beta_prec', type=int, default=2)
    # lattice
    parser.add_argument('dims', metavar='d', type=int, nargs='+')
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()
    # handle params
    if len(args.tag) > 0:
        args.tag = "_" + args.tag
    if args.seed is None:
        args.seed = np.random.randint(np.iinfo('uint32').max)
        print("Generated random seed = {}".format(args.seed))
    np.random.seed(args.seed)
    print("Using seed = {}.".format(args.seed))
    assert len(args.dims) == 2, 'cluster algorithm is specialized for 2D U(1) gauge theory'
    args.dims = tuple(args.dims)
    shape = (np.prod(args.dims),) # flatten for U(1) <-> XY mapping
    if args.init_cfg is None:
        print('Generating hot init cfg.')
        cfg = 2*np.pi*np.random.random(size=shape)
    else:
        print('Loading init cfg from {}.'.format(args.init_cfg))
        cfg = np.fromfile(args.init_cfg, dtype=np.float64)
        cfg = cfg.reshape(shape)

    # do the thing!
    res = run_cluster_mcmc(cfg, beta=args.beta, N_cfg=args.Ncfg,
                           n_therm=args.n_therm, n_skip=args.n_skip,
                           custom_step=lambda cfg: flip_xy_snake(cfg, shape=args.dims))
    cfgs = res['cfgs']
    clusters = res['clusters']
    actions = res['actions']
    topo = np.sum(np_wrap(cfgs - np.roll(cfgs, -1, axis=1)), axis=1) / (2*np.pi)
    # convert to U(1) 2D
    cfgs = convert_xy_to_u1(cfgs, shape=args.dims)

    # write stuff out
    beta_str = ('b{:.'+str(args.beta_prec)+'f}').format(args.beta)
    prefix = (
        f'u1_2d_{beta_str}_N{args.Ncfg}_skip{args.n_skip}_therm{args.n_therm}_'
        + '_'.join(map(str, args.dims)) + args.tag)
    print(f'Writing observables to {prefix}.*npy')
    np.save(f'{prefix}.npy', cfgs)
    np.save(f'{prefix}.clusters.npy', clusters)
    np.save(f'{prefix}.S.npy', actions)
    np.save(f'{prefix}.topo.npy', topo)
    

    print(f'TIME ensemble gen {time.time()-start:.2f}s')
