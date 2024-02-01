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
from schwinger import *

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
    Lx, Lt = L
    # define gamma functions
    # Basis choice:
    # g_1 (Euclidean) = -i g_1 (Minkowski) = pauli_x
    # g_2 (Euclidean) = g_0 (Minkowski)    = pauli_y
    #  => g_5 (Euclidean) = i g_1 g_2 (Euclidean) = - pauli_z
    # N.B the factor of i for chiral projector in Euclidean space in 2d
    g = {}
    g['0'] = pauli(0)
    g['5'] = -pauli(3)
    g['2']= pauli(2)
    g['25'] = np.matmul(g['2'], g['5'])
    # do the thing
    gsrcs = ['5', '2', '25']
    gsnks = ['5', '2', '25']
    csrcs = ['l', 'n', 'n2'] # l = local, n = non-local 1 link, n2 = non-local 2 link
    csnks = ['l', 'n', 'n2']
    constructions = []
    for gsrc in gsrcs:
        for gsnk in gsnks:
            for csrc in csrcs:
                for csnk in csnks:
                    constructions.append('g' + gsrc + '-' + 'g' + gsnk + '_' + csrc + csnk)
    print(constructions)
    if compute_dirac == "wilson":
        assert args.kappa is not None, "kappa required"
        make_D = lambda cfg: dirac_op(cfg,
                                      kappa=args.kappa,
                                      sign=1,
                                      fermion_bc=action.fermion_bc).toarray()
        Cts_conn = {}                                # accumulator over cfgs for zero-momentum, source-averaged pion C(t)
        Cts_disc = {}                                # accumulator over cfgs for zero-momentum, source-averaged pion C(t)
        for construction in constructions:
            Cts_conn[construction] = []              # shape = (Lt, )
            Cts_disc[construction] = []              # shape = (Lt, )
        print("Computing vacuum contributions...")
        vac = np.zeros((Lx, Lt, Ns, Ns), dtype=complex)
        for icfg in range(len(cfgs)):
            print("Working on config %d/%d" % (icfg + 1, len(cfgs)))
            cfg = cfgs[icfg]
            D = make_D(cfg)
            Dinv = np.linalg.inv(D)                   # all to all propagagtor
            Dinv = np.reshape(Dinv, L + [NS] + L + [NS])
            vac += np.einsum('xyaxyb -> xyab', Dinv)
        vac = vac / len(cfgs)
        assert(vac.shape == (Lx, Lt, Ns, Ns))
        print("Computing connected & disconnected contributions, subtracting vacuum contributions...")
        for icfg in range(len(cfgs)):
            print("Working on config %d/%d" % (icfg + 1, len(cfgs)))
            cfg = cfgs[icfg]
            D = make_D(cfg)
            Dinv = np.linalg.inv(D)                 # all to all propagagtor
            Dinv = np.reshape(Dinv, L + [NS] + L + [NS])
            #M = dirac_op(cfg, kappa=args.kappa, sign=1, fermion_bc=action.fermion_bc)
            C_conn_src_avg_acc = {}                 # accumulator over sources for pion C(x, t) (source averaging before momentum projection)
            C_disc_src_avg_acc = {}                 # accumulator over sources for pion C(x, t) (source averaging before momentum projection)
            for construction in constructions:
                C_conn_src_avg_acc[construction] = 0
                C_disc_src_avg_acc[construction] = 0
            # loop over sources, optionally only in the "bulk" for spatial OBC
            boundary_layer = int(Lx/4)
            nsources = 0
            #for x0 in range(int(Lx/2), int(Lx/2) + 1):
            prop = {}
            opp_prop = {}
            antiprop = {}
            meson_corr_conn = {}
            meson_corr_disc = {}
            for x0 in range(boundary_layer, Lx-boundary_layer):
                for t0 in range(Lt):
                    nsources += 1
                    #src = make_prop_src([(x0, t0)], L)
                    # propagator from fixed (x, t, any spin) to any point,
                    # shifted back by (x, t) & multiplied by links at source and sink
                    # n = non-local with link from x to x+a
                    # n2 = non-local with link from x-a to x+a
                    # l = local
                    # prop_src_snk
                    opp_prop['ll']   = np.roll(Dinv[:,:,:,x0, t0,:], (-x0,-t0), axis=(0,1))             # src at 0, snk at x-x0
                    opp_prop['nl']   = np.roll(Dinv[:,:,:,(x0+1) % Lx, t0,:], (-x0,-t0), axis=(0,1))    # src at 0+a, snk at x-x0
                    opp_prop['n2l']  = opp_prop['nl']
                    opp_prop['ln']   = np.roll(opp_prop['ll'],  (-1), axis=0)                           # src at 0, snk at x-x0+a
                    opp_prop['ln2']  = opp_prop['ln']
                    opp_prop['nn']   = np.roll(opp_prop['nl'],  (-1), axis=0)                           # snk at x-x0+a, src at 0+a
                    opp_prop['n2n']  = opp_prop['nn']
                    opp_prop['nn2']  = opp_prop['nn']
                    opp_prop['n2n2'] = opp_prop['nn']
                    u = np.roll(np.conj(cfg[0]), (-x0, -t0), axis=(0, 1))
                    prop['ll']   = np.roll(Dinv[:,:,:,x0, t0,:], (-x0,-t0), axis=(0,1))                 # src at 0, snk at x-x0
                    prop['nl']   = cfg[0, x0, t0] * prop['ll']                                          # U0(0) * (src at 0), snk at x-x0
                    # U0(0) * U0(0-a) * (src at 0-a), snk at x-x0
                    prop['n2l']  = cfg[0, x0, t0] * cfg[0, (x0-1) % Lx, t0] * np.roll(Dinv[:,:,:,(x0-1) % Lx, t0,:], (-x0,-t0), axis=(0,1))
                    prop['ln']   = np.einsum('xy,xyab->xyab', u, prop['ll'])                            # src at 0, U0(x-x0)^dag * (snk at x-x0)
                    prop['ln2']  = np.einsum('xy,xyab->xyab', u, np.roll(prop['ln'], (+1), axis=0))     # src at 0, U0(x-x0)^dag * [U0(x-x0-a)^dag * (snk at x-x0-a)]
                    prop['nn']   = np.einsum('xy,xyab->xyab', u, prop['nl'])                            # U0(0) * (src at 0), U0(x-x0)^dag * (snk at x-x0)
                    prop['n2n']  = np.einsum('xy,xyab->xyab', u, prop['n2l'])                           # U0(0) * U0(0-a) * (src at 0-a), U0(x-x0) * (snk at x-x0)
                    prop['nn2']  = cfg[0, x0, t0] * prop['ln2']                                         # U0(0) * (src at 0), U0(x-x0)^dag * [U0(x-x0-a)^dag * (snk at x-x0-a)]
                    prop['n2n2'] = np.einsum('xy,xyab->xyab', u, np.roll(prop['n2n'], (+1), axis=0))    # U0(0) * U0(0-a) * (src at 0-a), U0(x-x0)^dag * [U0(x-x0-a)^dag * (snk at x-x0-a)]
                    # prop = sp.sparse.linalg.spsolve(M, src)
                    #resid = sp.linalg.norm(M @ prop - src)
                    #print("Resid = {}".format(resid))
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
                    for csrc in csrcs:
                        for csnk in csnks:
                            antiprop[csrc + csnk] = np.einsum(
                                'ab,xybc,cd->xyad',
                                g['5'],
                                np.swapaxes(np.conj(opp_prop[csrc + csnk]), axis1=-1, axis2=-2),
                                g['5'])
                            # disk is wrong
                            disc = np.einsum('xyaxyb->xyab', Dinv)
                            disc_sub = disc - vac
                            disc_sub_src = disc_sub[x0, t0, :, :]
                            disc_sub_snk = np.roll(disc_sub, (-x0, -t0), axis=(0, 1))
                            # for source O propto gi, sink Obar propto pm gj
                            # C = mp tr(prop * gi * antiprop * gj)
                            meson_corr_disc[csrc + csnk] = lambda p1, p2: np.einsum(
                                'ab,ba,xycd,dc -> xy', disc_sub_src, p1, disc_sub_snk, p2)
                            meson_corr_conn[csrc + csnk] = lambda p1, p2: -np.einsum(            # n.b. sign
                                'xyab,bc,xycd,da -> xy', prop[csrc+csnk], p1, antiprop[csrc+csnk], p2)
                            # use args.conn_weight, args_disk_weight
                            # meson_corr = lambda p1, p2: meson_corr_conn(p1, p2) + meson_corr_disc(p1, p2)
                    # check that for g5 g5, connected correlator equal to
                    # elementwise product prop * prop^*
                    # assert np.allclose(
                    #     meson_corr_conn(g5, g5),
                    #     -np.sum(np.abs(prop**2), axis=(2,3))
                    # )
                    # project into zero momentum
                    # increment source accumulators by C(x, t) from this (x0, t0) source
                    # vector
                    l = -x0 + boundary_layer
                    r = -x0 + (Lx-boundary_layer)
                    for gsrc in gsrcs:
                        for gsnk in gsnks:
                            for csrc in csrcs:
                                for csnk in csnks:
                                    constr = 'g' + gsrc + '-' + 'g' + gsnk + '_' + csrc + csnk
                                    C_conn_src_avg_acc[constr]   = C_conn_src_avg_acc[constr] + np.sum(meson_corr_conn[csrc+csnk](g[gsrc],   g['2'] @ np.conj( g[gsnk].T   ) @ g['2'])[range(l,r), :], axis=0)
                                    # C_conn_src_avg_acc['g2g5-g2g5']  = C_conn_src_avg_acc['g2g5-g2g5'] + np.sum(meson_corr_conn(g2g5, g2 @ np.conj( g2g5.T ) @ g2)[range(l,r), :], axis=0)
                                    # C_conn_src_avg_acc['g5-g2g5']    = C_conn_src_avg_acc['g5-g2g5']   + np.sum(meson_corr_conn(g5,   g2 @ np.conj( g2g5.T ) @ g2)[range(l,r), :], axis=0)
                                    # C_conn_src_avg_acc['g2g5-g5']    = C_conn_src_avg_acc['g2g5-g5']   + np.sum(meson_corr_conn(g2g5, g2 @ np.conj( g5.T   ) @ g2)[range(l,r), :], axis=0)
                                    # # # scalar
                                    # # C_conn_src_avg_acc['g0-g0']      = C_conn_src_avg_acc['g0-g0']     + np.sum(meson_corr_conn(g0, g2 @ np.conj(  g0.T ) @ g2)[range(l,r), :], axis=0)
                                    # # C_conn_src_avg_acc['g2-g2']      = C_conn_src_avg_acc['g2-g2']     + np.sum(meson_corr_conn(g2, g2 @ np.conj(  g2.T ) @ g2)[range(l,r), :], axis=0)
                                    # # C_conn_src_avg_acc['g0-g2']      = C_conn_src_avg_acc['g0-g2']     + np.sum(meson_corr_conn(g0, g2 @ np.conj(  g2.T ) @ g2)[range(l,r), :], axis=0)
                                    # # C_conn_src_avg_acc['g2-g0']      = C_conn_src_avg_acc['g2-g0']     + np.sum(meson_corr_conn(g2, g2 @ np.conj(  g0.T ) @ g2)[range(l,r), :], axis=0)
                                    C_disc_src_avg_acc[constr]   = C_disc_src_avg_acc[constr] + np.sum(meson_corr_conn[csrc+csnk](g[gsrc],   g['2'] @ np.conj( g[gsnk].T   ) @ g['2'])[range(l,r), :], axis=0)
                                    # # vector
                                    # C_disc_src_avg_acc['g5-g5']      = C_disc_src_avg_acc['g5-g5']     + np.sum(meson_corr_disc(g5,   g2 @ np.conj( g5.T   ) @ g2)[range(l,r), :], axis=0)
                                    # C_disc_src_avg_acc['g2g5-g2g5']  = C_disc_src_avg_acc['g2g5-g2g5'] + np.sum(meson_corr_disc(g2g5, g2 @ np.conj( g2g5.T ) @ g2)[range(l,r), :], axis=0)
                                    # C_disc_src_avg_acc['g5-g2g5']    = C_disc_src_avg_acc['g5-g2g5']   + np.sum(meson_corr_disc(g5,   g2 @ np.conj( g2g5.T ) @ g2)[range(l,r), :], axis=0)
                                    # C_disc_src_avg_acc['g2g5-g5']    = C_disc_src_avg_acc['g2g5-g5']   + np.sum(meson_corr_disc(g2g5, g2 @ np.conj( g5.T   ) @ g2)[range(l,r), :], axis=0)
                                    # # scalar
                                    # C_disc_src_avg_acc['g0-g0']      = C_disc_src_avg_acc['g0-g0']     + np.sum(meson_corr_disc(g0, g2 @ np.conj(  g0.T ) @ g2)[range(l,r), :], axis=0)
                                    # C_disc_src_avg_acc['g2-g2']      = C_disc_src_avg_acc['g2-g2']     + np.sum(meson_corr_disc(g2, g2 @ np.conj(  g2.T ) @ g2)[range(l,r), :], axis=0)
                                    # C_disc_src_avg_acc['g0-g2']      = C_disc_src_avg_acc['g0-g2']     + np.sum(meson_corr_disc(g0, g2 @ np.conj(  g2.T ) @ g2)[range(l,r), :], axis=0)
                                    # C_disc_src_avg_acc['g2-g0']      = C_disc_src_avg_acc['g2-g0']     + np.sum(meson_corr_disc(g2, g2 @ np.conj(  g0.T ) @ g2)[range(l,r), :], axis=0)
                # save source-averaged momentum-projected stuff
            for construction in constructions:
                Ct_conn = C_conn_src_avg_acc[construction] / nsources
                Ct_disc = C_disc_src_avg_acc[construction] / nsources
                Cts_conn[construction].append(Ct_conn)
                Cts_disc[construction].append(Ct_disc)
    elif compute_dirac == "staggered":
        assert args.kappa is not None, "kappa required"
        m0 = m0_from_kappa(args.kappa, Nd)
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
        print("Wrote vacuum-subtracted disconnected Cts to {}".format(fname))
