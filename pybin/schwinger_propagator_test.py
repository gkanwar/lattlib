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
    start = time.time()
    Lx = 48
    Lt = 48
    L = [Lx, Lt]
    Nd = len(L)
    Ns = 2**(int(Nd/2))
    shape = tuple([Nd] + list(L))
    beta = 1.0
    kappa = 0.2
    gauge_bc = (1, 1)
    Ncfg = 1
    n_skip = 0
    n_therm = 0
    tag = '_testinv'
    action = TwoFlavorAction(beta, kappa, gauge_bc = gauge_bc)
    compute_dirac = "wilson"
    path = '{:s}/{:s}/gaugeBC{:d}{:d}/'.format(tag,
                                                action.make_tag(),
                                                gauge_bc[0],
                                                gauge_bc[1])
    path_ens = 'ens/' + path
    path_meas = 'meas/' + path
    os.makedirs(path_ens, exist_ok=True)
    os.makedirs(path_meas, exist_ok=True)
    prefix = 'u1_{:s}_N{:d}_skip{:d}_therm{:d}_{:d}_{:d}_gaugeBC{:d}{:d}{:s}'.format(
        action.make_tag(), Ncfg, n_skip, n_therm,
        Lx, Lt,
        gauge_bc[0], gauge_bc[1],
        tag)
    prefix_ens = path_ens + prefix
    prefix_meas = path_meas + prefix
    fname = prefix_ens + '.npy'
    # generate config file
    cfgs = [np.ones((Nd, Lx, Lt))]#[np.random.rand(Nd, Lx, Lt)]
    constructions = ['g5-g5', 'g2g5-g2g5', 'g5-g2g5', 'g2g5-g5', 'g0-g0', 'g2-g2', 'g0-g2', 'g2-g0']
    if compute_dirac == "wilson":
        assert kappa is not None, "kappa required"
        make_D = lambda cfg: dirac_op(cfg,
                                      kappa=kappa,
                                      sign=1,
                                      fermion_bc=action.fermion_bc).toarray()
        vac = []
        Cts_conn = {}                                # accumulator over cfgs for zero-momentum, source-averaged pion C(t)
        Cts_disc = {}                                # accumulator over cfgs for zero-momentum, source-averaged pion C(t)
        for construction in constructions:
            Cts_conn[construction] = []              # shape = (Lt, )
            Cts_disc[construction] = []              # shape = (Lt, )
        for icfg in range(len(cfgs)):
            print("Working on config %d/%d" % (icfg + 1, len(cfgs)))
            cfg = cfgs[icfg]
            D = make_D(cfg)
            Dinv = np.linalg.inv(D)             # all to all propagagtor
            Dinv = np.reshape(Dinv, L + [NS] + L + [NS])
            #M = dirac_op(cfg, kappa=args.kappa, sign=1, fermion_bc=action.fermion_bc)
            #if args.type == 'one_flavor':
            #    Mdag = dirac_op(cfg, kappa=args.kappa, sign=-1, fermion_bc=action.fermion_bc)
            #    K = Mdag @ M
            #    a0 = action.a0
            #    r = action.r
            #    musq = action.musq
            Lx, Lt = L
            vac.append(np.mean(np.einsum('xyaxyb -> xyab', Dinv), axis=(0, 1)))
            C_conn_src_avg_acc = {}                 # accumulator over sources for pion C(x, t) (source averaging before momentum projection)
            C_disc_src_avg_acc = {}                 # accumulator over sources for pion C(x, t) (source averaging before momentum projection)
            for construction in constructions:
                C_conn_src_avg_acc[construction] = 0
                C_disc_src_avg_acc[construction] = 0
            # loop over sources in the bulk
            boundary_layer = int(Lt/4)
            nsources = 0
            for x0 in range(int(Lx/2), int(Lx/2) + 1):
            #for x0 in range(boundary_layer, Lx-boundary_layer):
                for t0 in range(Lt):
                    nsources += 1
                    #print("source at time ", t0)
                    #src = make_prop_src([(x0, t0)], L)
                    # propagator from fixed (x, t, any spin) to any point,
                    # shifted back by (x, t)
                    prop = np.roll(Dinv[:,:,:,x0,t0,:], (-x0,-t0), axis=(0,1))
                    #if args.type == 'two_flavor':
                    #    prop = sp.sparse.linalg.spsolve(M, src)
                    #    resid = sp.linalg.norm(M @ prop - src)
                    #    print("Resid = {}".format(resid))
                    #elif args.type == 'one_flavor':
                    #   prop = np.copy(src)
                    #   for s in range(Ns):
                    #       print("spin component ", s)
                    #       psi, info = stupid_multishift_cg(K, musq, src[:, s])
                    #       for k in range(len(musq)):
                    #           if info[k] > 0:
                    #               print(f'WARNING RHMC (term {k}): CG failed to converge after {info[k]} iters')
                    #               resid = np.linalg.norm(K @ psi[k] + musq[k] * psi[k]  - src[:, s])
                    #               print('... residual (abs):', resid)
                    #               print('... residual (rel):', resid / np.linalg.norm(src[:, s]))
                    #           elif info[k] < 0:
                    #               print(f'WARNING RHMC (term {k}): CG illegal input or breakdown ({info[k]})')
                    #           prop[:, s] += r[k] * psi[k]
                    #       prop[:, s] *= a0
                    #else:
                    #    raise
                    #prop = prop.reshape(L + [Ns, Ns])
                    #prop = np.roll(prop, (-x0,-t0), axis=(0,1))
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
                    g0 = pauli(0)
                    g5 = -pauli(3)
                    g2 = pauli(2)
                    g2g5 = np.matmul(g2, g5)
                    antiprop = np.einsum(
                        'ab,xybc,cd->xyad',
                        g5,
                        np.swapaxes(np.conj(prop), axis1=-1, axis2=-2),
                        g5)
                    disc_src = Dinv[x0,t0,:,x0,t0,:]
                    disc_snk = np.roll(Dinv[:,:,:,:,:,:], (-x0,-t0,-x0,-t0), axis=(0,1,3,4))
                    # for source O propto gi, sink Obar propto pm gj
                    # C = mp tr(prop * gi * antiprop * gj)
                    meson_corr_disc = lambda p1, p2: np.einsum(
                        'ab,ba,xycxyd,dc -> xy', disc_src, p1, disc_snk, p2)
                    C_conn_src_avg_acc[construction] = 0
                    meson_corr_conn = lambda p1, p2: -np.einsum(            # n.b. sign
                        'xyab,bc,xycd,da -> xy', prop, p1, antiprop, p2)
                    # use args.conn_weight, args_disk_weight
                    # elementwise product prop * prop^*
                    assert np.allclose(
                        meson_corr_conn(g5, g5),
                        -np.sum(np.abs(prop**2), axis=(2,3))
                    )
                    # project into zero momentum
                    # increment source accumulators by C(x, t) from this (x0, t0) source, project onto zero-momentum
                    # vector
                    C_conn_src_avg_acc['g5-g5']      = C_conn_src_avg_acc['g5-g5']     + np.mean(  meson_corr_conn(g5,   g2 @ np.conj( g5.T   ) @ g2), axis=0 )
                    C_conn_src_avg_acc['g2g5-g2g5']  = C_conn_src_avg_acc['g2g5-g2g5'] + np.mean(  meson_corr_conn(g2g5, g2 @ np.conj( g2g5.T ) @ g2), axis=0 )
                    C_conn_src_avg_acc['g5-g2g5']    = C_conn_src_avg_acc['g5-g2g5']   + np.mean(  meson_corr_conn(g5,   g2 @ np.conj( g2g5.T ) @ g2), axis=0 )
                    C_conn_src_avg_acc['g2g5-g5']    = C_conn_src_avg_acc['g2g5-g5']   + np.mean(  meson_corr_conn(g2g5, g2 @ np.conj( g5.T   ) @ g2), axis=0 )
                    # scalar
                    C_conn_src_avg_acc['g0-g0']      = C_conn_src_avg_acc['g0-g0']     + np.mean(  meson_corr_conn(g0, g2 @ np.conj(  g0.T ) @ g2), axis=0 )
                    C_conn_src_avg_acc['g2-g2']      = C_conn_src_avg_acc['g2-g2']     + np.mean(  meson_corr_conn(g2, g2 @ np.conj(  g2.T ) @ g2), axis=0 )
                    C_conn_src_avg_acc['g0-g2']      = C_conn_src_avg_acc['g0-g2']     + np.mean(  meson_corr_conn(g0, g2 @ np.conj(  g2.T ) @ g2), axis=0 )
                    C_conn_src_avg_acc['g2-g0']      = C_conn_src_avg_acc['g2-g0']     + np.mean(  meson_corr_conn(g2, g2 @ np.conj(  g0.T ) @ g2), axis=0 )
                    # vector
                    C_disc_src_avg_acc['g5-g5']      = C_disc_src_avg_acc['g5-g5']     + np.mean(  meson_corr_disc(g5,   g2 @ np.conj( g5.T   ) @ g2), axis=0 )
                    C_disc_src_avg_acc['g2g5-g2g5']  = C_disc_src_avg_acc['g2g5-g2g5'] + np.mean(  meson_corr_disc(g2g5, g2 @ np.conj( g2g5.T ) @ g2), axis=0 )
                    C_disc_src_avg_acc['g5-g2g5']    = C_disc_src_avg_acc['g5-g2g5']   + np.mean(  meson_corr_disc(g5,   g2 @ np.conj( g2g5.T ) @ g2), axis=0 )
                    C_disc_src_avg_acc['g2g5-g5']    = C_disc_src_avg_acc['g2g5-g5']   + np.mean(  meson_corr_disc(g2g5, g2 @ np.conj( g5.T   ) @ g2), axis=0 )
                    # scalar
                    C_disc_src_avg_acc['g0-g0']      = C_disc_src_avg_acc['g0-g0']     + np.mean(  meson_corr_disc(g0, g2 @ np.conj(  g0.T ) @ g2), axis=0 )
                    C_disc_src_avg_acc['g2-g2']      = C_disc_src_avg_acc['g2-g2']     + np.mean(  meson_corr_disc(g2, g2 @ np.conj(  g2.T ) @ g2), axis=0 )
                    C_disc_src_avg_acc['g0-g2']      = C_disc_src_avg_acc['g0-g2']     + np.mean(  meson_corr_disc(g0, g2 @ np.conj(  g2.T ) @ g2), axis=0 )
                    C_disc_src_avg_acc['g2-g0']      = C_disc_src_avg_acc['g2-g0']     + np.mean(  meson_corr_disc(g2, g2 @ np.conj(  g0.T ) @ g2), axis=0 )
            # save source-averaged momentum-projected stuff
            for construction in constructions:
                Ct_conn = C_conn_src_avg_acc[construction] / nsources
                Ct_disc = C_disc_src_avg_acc[construction] / nsources
                Cts_conn[construction].append(Ct_conn)
                Cts_disc[construction].append(Ct_disc)
    elif compute_dirac == "staggered":
        assert kappa is not None, "kappa required"
        m0 = m0_from_kappa(kappa, Nd)
        make_D = lambda cfg: make_op_matrix(L, lambda psi: apply_staggered_D(psi, U=cfg, m0=m0))
        raise NotImplementedError('staggered not implemented')
    else:
        raise RuntimeError(f"Dirac op type {args.compute_dirac} not supported")
    fname_vac = prefix_meas + '_vac.npy'
    np.save(fname_vac, np.array(vac))
    print("Wrote vac to {}".format(fname_vac))
    for construction in constructions:
        fname = prefix_meas + '_meson_Ct_conn_' + construction + '.npy'
        np.save(fname, np.array(Cts_conn[construction]))
        print("Wrote connected Cts to {}".format(fname))
        fname = prefix_meas + '_meson_Ct_disc_' + construction + '.npy'
        np.save(fname, np.array(Cts_disc[construction]))
        print("Wrote disconnected Cts to {}".format(fname))
