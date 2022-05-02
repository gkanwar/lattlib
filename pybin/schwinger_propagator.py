import argparse
import os
import math
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
import tqdm
import time
from schwinger_hmc import handle_bc_arg
from schwinger.schwinger import *

def get_coord_index(x, L):
    assert(len(L) == 2)
    return x[0] * L[1] + x[1]

def make_prop_src(all_srcs, L):
    V = np.prod(L)
    N_src = len(all_srcs)
    all_inds = list(map(lambda x: get_coord_index(x, L), all_srcs))
    src = np.zeros((NS*V,NS*N_src), dtype=np.complex128)
    count = 0
    for ind in all_inds:
        for a in range(NS):
            src[NS*ind+a,count+a] = 1.0
        count += NS
    assert(count == NS*N_src)
    return src

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute props for Schwinger')
    # general params
    parser.add_argument('--seed', type=int)
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--n_skip', type=int, required=True)
    parser.add_argument('--n_therm', type=int, required=True)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--n_leap', type=int, default=20)
    # action params
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--gauge_obc_x', action="store_true")
    parser.add_argument('--beta', type=float)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--conn_weight', type=float, default=1.0)
    parser.add_argument('--disc_weight', type=float, default=0.0)
    parser.add_argument('--xspace', type=int, default=1)
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()

    if len(args.tag) > 0:
        args.tag = "_" + args.tag
    L = [args.Lx, args.Lt]
    Nd = len(L)
    Ns = 2**(int(Nd/2))
    shape = tuple([Nd] + list(L))
    gauge_bc = handle_bc_arg(args.gauge_obc_x, args.type)
    if args.type == "two_flavor":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        action = TwoFlavorAction(args.beta, args.kappa, gauge_bc = gauge_bc)
        compute_dirac = "wilson"
    elif args.type == "one_flavor":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        action = OneFlavorAction(args.beta, args.kappa, gauge_bc = gauge_bc)
        compute_dirac = "wilson"
    elif args.type == "exact_1flav_staggered":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        m0 = m0_from_kappa(args.kappa, Nd)
        action = ExactStaggeredAction(args.beta, m0, Nf=1)
        compute_dirac = "staggered"
    elif args.type == "exact_2flav_staggered":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        m0 = m0_from_kappa(args.kappa, Nd)
        action = ExactStaggeredAction(args.beta, m0, Nf=2)
        compute_drac = "staggered"
    else:
        print("Unknown action type {}".format(args.type))
        sys.exit(1)

    path = '{:s}/{:s}/gaugeBC{:d}{:d}/'.format(args.tag, action.make_tag(), gauge_bc[0], gauge_bc[1])
    path_ens = 'ens/' + path
    path_meas = 'meas/' + path
    os.makedirs(path_ens, exist_ok=True)
    os.makedirs(path_meas, exist_ok=True)
    prefix = 'u1_{:s}_N{:d}_skip{:d}_therm{:d}_{:d}_{:d}_gaugeBC{:d}{:d}{:s}'.format(
        action.make_tag(), args.Ncfg, args.n_skip, args.n_therm,
        args.Lx, args.Lt,
        gauge_bc[0], gauge_bc[1],
        args.tag)
    prefix_ens = path_ens + prefix
    prefix_meas = path_meas + prefix
    fname = prefix_ens + '.npy'
    # get config files
    cfgs = np.load(fname)

    # TODO: reorganize into h5py files
    constructions = ['g5-g5', 'g2g5-g2g5', 'g5-g2g5', 'g2g5-g5']
    if compute_dirac == "wilson":
        assert args.kappa is not None, "kappa required"
        #make_D = lambda cfg: dirac_op(cfg,
        #                              kappa=args.kappa,
        #                              sign=1,
        #                              fermion_bc=action.fermion_bc).toarray()
        Cts = {}                                # accumulator over cfgs for zero-momentum, source-averaged pion C(t)
        for construction in constructions:
            Cts[construction] = []              # shape = (Lt, )
        for icfg in range(len(cfgs)):
            print("Working on config %d/%d" % (icfg + 1, len(cfgs)))
            cfg = cfgs[icfg]
            M = dirac_op(cfg, kappa=args.kappa, sign=1, fermion_bc=action.fermion_bc)
            if args.type == 'one_flavor':
                Mdag = dirac_op(cfg, kappa=args.kappa, sign=-1, fermion_bc=action.fermion_bc)
                K = Mdag @ M
                a0 = action.a0
                r = action.r
                musq = action.musq
            Lx, Lt = L
            C_src_avg_acc = {}                 # accumulator over sources for pion C(x, t) (source averaging before momentum projection)
            for construction in constructions:
                C_src_avg_acc[construction] = 0
            # loop over sources in the bulk
            boundary_layer = int(Lt/4)
            # TODO add disconnecteds
            for x0 in range (int(Lx/2), int(Lx/2) + 1):
                print(x0)
            #for x0 in range(boundary_layer, Lx-boundary_layer):
                for t0 in range(Lt):
                    print("source at time ", t0)
                    src = make_prop_src([(x0, t0)], L)
                    if args.type == 'two_flavor':
                        prop = sp.sparse.linalg.spsolve(M, src)
                        resid = sp.linalg.norm(M @ prop - src)
                        print("Resid = {}".format(resid))
                    elif args.type == 'one_flavor':
                        prop = np.copy(src)
                        for s in range(Ns):
                            print("spin component ", s)
                            psi, info = stupid_multishift_cg(K, musq, src[:, s])
                            for k in range(len(musq)):
                                if info[k] > 0:
                                    print(f'WARNING RHMC (term {k}): CG failed to converge after {info[k]} iters')
                                    resid = np.linalg.norm(K @ psi[k] + musq[k] * psi[k]  - src[:, s])
                                    print('... residual (abs):', resid)
                                    print('... residual (rel):', resid / np.linalg.norm(src[:, s]))
                                elif info[k] < 0:
                                    print(f'WARNING RHMC (term {k}): CG illegal input or breakdown ({info[k]})')
                                prop[:, s] += r[k] * psi[k]
                            prop[:, s] *= a0
                    else:
                        raise
                    prop = prop.reshape(L + [Ns, Ns])
                    prop = np.roll(prop, (-x0,-t0), axis=(0,1))
                    # Find antiprop
                    #
                    # prop = G(y|x)_{cb}
                    # antiprop = G(x|y)_{ad}
                    #
                    # G(y|x)_{cb} = g5_{cd} Gdag(x|y)_{da} g5_{ab} =>
                    # Gdag(x|y)_{da} = g5dag_{dc} G(y|x)_{cb} g5dag_{ba} =>
                    # antiprop  = G(x|y)_{ad} = [Gdag(x|y)_{da}]^dag = (g5 G(y|x) g5)^dag
                    #           = g5_{ab} G(y|x)^*_{bc} g5_{cd}
                    #           = g5 * (prop^*)^{T in spin} * g5
                    #
                    # Basis choice:
                    # g_1 (Euclidean) = -i g_1 (Minkowski) = pauli_x
                    # g_2 (Euclidean) = g_0 (Minkowski)    = pauli_y
                    #  => g_5 (Euclidean) = i g_1 g_2 (Euclidean) = - pauli_z
                    # N.B the factor of i for chiral projector in Euclidean space in 2d
                    g5 = -pauli(3)
                    g2 = pauli(2)
                    g2g5 = np.matmul(g2, g5)
                    antiprop = np.einsum(
                        'ab,xybc,cd->xyad',
                        g5,
                        np.swapaxes(np.conj(prop), axis1=-1, axis2=-2),
                        g5)
                    # for source O propto gi, sink Obar propto pm gj
                    # C = mp tr(prop * gi * antiprop * gj)
                    meson_corr_conn = lambda p1, p2: -np.einsum(            # n.b. sign
                        'xyab,bc,xycd,da -> xy', prop, p1, antiprop, p2)
                    meson_corr = lambda p1, p2: args.conn_weight * meson_corr_conn(p1, p2) #+ args.disc_weight * meson_corr_disc(p0, p2)
                    # check that for g5 g5, connected correlator equal to
                    # elementwise product prop * prop^*
                    assert np.allclose(
                        meson_corr_conn(g5, g5),
                        -np.sum(np.abs(prop**2), axis=(2,3))
                    )
                    # project into zero momentum
                    # increment source accumulators by C(x, t) from this (x0, t0) source, project onto zero-momentum
                    C_src_avg_acc['g5-g5']      = C_src_avg_acc['g5-g5']     + np.mean(  meson_corr(g5,   g2 @ np.conj( g5.T   ) @ g2), axis=0 )
                    C_src_avg_acc['g2g5-g2g5']  = C_src_avg_acc['g2g5-g2g5'] + np.mean(  meson_corr(g2g5, g2 @ np.conj( g2g5.T ) @ g2), axis=0 )
                    C_src_avg_acc['g5-g2g5']    = C_src_avg_acc['g5-g2g5']   + np.mean(  meson_corr(g5,   g2 @ np.conj( g2g5.T ) @ g2), axis=0 )
                    C_src_avg_acc['g2g5-g5']    = C_src_avg_acc['g2g5-g5']   + np.mean(  meson_corr(g2g5, g2 @ np.conj( g5.T   ) @ g2), axis=0 )
                    # project onto zero momentum by summing over x within each t slice
                    for construction in constructions:
                        Ct = C_src_avg_acc[construction]
                        Cts[construction].append(Ct)
    elif compute_dirac == "staggered":
        assert args.kappa is not None, "kappa required"
        m0 = m0_from_kappa(args.kappa, Nd)
        make_D = lambda cfg: make_op_matrix(L, lambda psi: apply_staggered_D(psi, U=cfg, m0=m0))
        raise NotImplementedError('staggered not implemented')
    else:
        raise RuntimeError(f"Dirac op type {args.compute_dirac} not supported")
    for construction in ['g5-g5', 'g2g5-g2g5', 'g5-g2g5', 'g2g5-g5']:
        fname = prefix_meas + '_meson_Ct_' + construction + '.npy'
        np.save(fname, np.array(Cts[construction]))
        print("Wrote Cts to {}".format(fname))
