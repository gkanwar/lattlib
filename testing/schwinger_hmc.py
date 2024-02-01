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
from schwinger import *


def hmc_update(cfg, action, tau, n_leap, verbose=True):
    old_cfg = np.copy(cfg)
    old_S = action.init_traj(old_cfg)
    old_pi = sample_pi(cfg.shape)
    old_K = np.sum(old_pi*old_pi) / 2
    old_H = old_S + old_K

    cfg = np.copy(cfg)
    new_pi = np.copy(old_pi)
    leapfrog_update(cfg, new_pi, action, tau, n_leap, verbose=verbose)

    new_S = action.compute_action(cfg)
    new_K = np.sum(new_pi*new_pi) / 2
    new_H = new_S + new_K

    delta_H = new_H - old_H
    if verbose:
        print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))
        print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
        print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))

    # metropolis step
    acc = 0
    if np.random.random() < np.exp(-delta_H):
        acc = 1
        S = new_S
        K = new_K
    else:
        cfg = old_cfg
        S = old_S
        K = old_K
    if verbose:
        print("Acc {:.5g} (changed {})".format(min(1.0, np.exp(-delta_H)), acc))

    return cfg, S, acc, K

def run_hmc(L, n_step, n_skip, n_therm, tau, n_leap, action, cfg, *, topo_hop_freq=0, gauge_bc=(1, 1)):
    Nd = len(L)
    V = np.prod(L)
    shape = tuple([Nd] + list(L))
    pure_gauge_action = PureGaugeAction(beta=action.beta, gauge_bc=gauge_bc)

    # MC updates
    total_acc = 0
    hop_acc = 0
    hop_props = 0
    cfgs = []
    plaqs = []
    topos = []
    rates = []   # acceptance rates
    lmoms = []   # link momenta
    with tqdm.tqdm(total = n_therm + n_step, postfix='Acc: ???, Q: ???') as t:
        for i in tqdm.tqdm(range(-n_therm, n_step)):
            print("MC step {} / {}".format(i+1, n_step))
            cfg, S, acc, K = hmc_update(cfg, action, tau, n_leap)
            if i >= 0: total_acc += acc
            if topo_hop_freq > 0 and i % topo_hop_freq == 0:
                assert Nd == 2
                assert L[0] == L[1]
                Nf = 2
                hop_props += 1
                dQ = np.random.randint(-1, 2)
                print('Topo hop proposal dQ', dQ)
                dU = make_topo_cfg(L[0], dQ)
                new_cfg = cfg * dU # NOTE: This only works because Abelian
                # new_S = action.compute_action(new_cfg)
                old_S_marginal = pure_gauge_action.compute_action(cfg)
                new_S_marginal = pure_gauge_action.compute_action(new_cfg)
                _, old_logdetD = np.linalg.slogdet(dirac_op(cfg, kappa=action.kappa).toarray())
                _, new_logdetD = np.linalg.slogdet(dirac_op(new_cfg, kappa=action.kappa).toarray())
                old_S_marginal -= Nf * old_logdetD
                new_S_marginal -= Nf * new_logdetD
                print('delta S =', new_S_marginal - old_S_marginal)
                print('delta (-log det D) =', Nf*(old_logdetD - new_logdetD))
                print('old logdetD =', np.real(old_logdetD), 'new logdetD =', np.real(new_logdetD))
                if np.random.random() < np.exp(-new_S_marginal + old_S_marginal):
                    print('accepted')
                    cfg = new_cfg
                    hop_acc += 1
                else:
                    print('rejected')

            # avg plaq
            plaq = np.sum(np.real(ensemble_plaqs(cfg, gauge_bc=gauge_bc))) / V
            print("Average plaq = {:.6g}".format(plaq))
            # topo Q
            topo = np.sum(compute_topo(cfg, gauge_bc=gauge_bc))
            Q = int(round(topo))
            print("Topo = {:d}".format(Q))

            # save cfg
            if i >= 0 and i % n_skip == 0:
                print("Saving cfg!")
                cfgs.append(cfg)
                plaqs.append(plaq)
                topos.append(topo)
                rates.append(total_acc / (i+1))
                lmoms.append(K)
                t.postfix = 'Acc: {:.3f}, Q: {:d}'.format(total_acc / (i+1), Q)
            t.update()

    print("MC finished.")
    print("Total acc {:.4f}".format(total_acc / n_step))
    if topo_hop_freq > 0:
        print("Total hop acc {:.4f}".format(hop_acc / hop_props))
    return cfgs, plaqs, topos, rates, lmoms

def handle_bc_arg(gauge_obc_x, action_type):
    if (gauge_obc_x):
        if (action_type == 'two_flavor') or (action_type == 'one_flavor') or (action_type == 'pure_gauge'):
            gauge_bc = (0, 1)       # open in space, periodic in time
        else:
            print("OBC not supported for action type {}".format(action_type))
            sys.exit(1)
    else:
        gauge_bc = (1, 1)       # periodic in space, periodic in time
    return gauge_bc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HMC for Schwinger')
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
    parser.add_argument('--init_cfg', type=str)
    parser.add_argument('--topo_hop_freq', type=int, default=0)
    # action params
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--gauge_obc_x', action="store_true")
    parser.add_argument('--beta', type=float)
    parser.add_argument('--kappa', type=float)
    # atm these are hard coded in schwinger.py
    # parser.add_argument('--rhmc_poly_deg', type=int)
    # parser.add_argument('--rhmc_smallest', type=float)
    # parser.add_argument('--rhmc_largest', type=float)
    parser.add_argument('--reweight_dt', type=int)
    parser.add_argument('--conn_weight', type=float, default=1.0)
    parser.add_argument('--disc_weight', type=float, default=0.0)
    parser.add_argument('--xspace', type=int, default=4)
    parser.add_argument('--eps_reg', type=float)
    parser.add_argument('--polya_x', type=int)
    parser.add_argument('--theta_i', type=float)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--compute_dirac', type=str, default="",
                        help="Which Dirac op to compute on cfgs, if any")
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()

    if len(args.tag) > 0:
        args.tag = "_" + args.tag
    if args.seed is None:
        args.seed = np.random.randint(np.iinfo('uint32').max)
        print("Generated random seed = {}".format(args.seed))
    np.random.seed(args.seed)
    print("Using seed = {}.".format(args.seed))
    L = [args.Lx, args.Lt]
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    if args.init_cfg is None:
        print('Generating warm init cfg.')
        init_cfg_A = 0.4*np.random.normal(size=shape)
        cfg = np.exp(1j * init_cfg_A)
    elif args.init_cfg == 'hot':
        print('Generating hot init cfg.')
        init_cfg_A = 2*np.pi*np.random.random(size=shape)
        cfg = np.exp(1j * init_cfg_A)
    else:
        print('Loading init cfg from {}.'.format(args.init_cfg))
        cfg = np.load(args.init_cfg)
        cfg = cfg.reshape(shape)
    tot_steps = args.Ncfg * args.n_skip
    gauge_bc = handle_bc_arg(args.gauge_obc_x, args.type)
    if args.type == "pure_gauge":
        assert(args.beta is not None)
        action = PureGaugeAction(args.beta, gauge_bc = gauge_bc)
    elif args.type == "two_flavor":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        action = TwoFlavorAction(args.beta, args.kappa, gauge_bc = gauge_bc)
    elif args.type == "one_flavor":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        action = OneFlavorAction(args.beta, args.kappa, gauge_bc = gauge_bc)
    elif args.type == "exact_1flav_staggered":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        m0 = m0_from_kappa(args.kappa, Nd)
        action = ExactStaggeredAction(args.beta, m0, Nf=1)
    elif args.type == "exact_2flav_staggered":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        m0 = m0_from_kappa(args.kappa, Nd)
        action = ExactStaggeredAction(args.beta, m0, Nf=2)
    elif args.type == "exact_2flav_wilson":
        assert(args.beta is not None)
        assert(args.kappa is not None)
        action = ExactWilsonAction(args.beta, args.kappa, Nf=2)
    else:
        print("Unknown action type {}".format(args.type))
        sys.exit(1)

    # do the thing!
    cfgs, plaqs, topos, rates, lmoms = run_hmc(L, tot_steps, args.n_skip, args.n_therm,
                                 args.tau, args.n_leap, action, cfg,
                                 topo_hop_freq=args.topo_hop_freq,
                                 gauge_bc=gauge_bc)

    # write stuff out
    path = 'ens/{:s}/{:s}/gaugeBC{:d}{:d}/'.format(args.tag, action.make_tag(), gauge_bc[0], gauge_bc[1])
    os.makedirs(path, exist_ok=True)
    prefix = path + 'u1_{:s}_N{:d}_skip{:d}_therm{:d}_{:d}_{:d}_gaugeBC{:d}{:d}{:s}'.format(
        action.make_tag(), args.Ncfg, args.n_skip, args.n_therm,
        args.Lx, args.Lt,
        gauge_bc[0], gauge_bc[1],
        args.tag)
    fname = prefix + '.npy'
    np.save(fname, cfgs)
    print("Wrote ensemble to {}".format(fname))
    fname = prefix + '.plaq.npy'
    np.save(fname, plaqs)
    print("Wrote plaqs to {}".format(fname))
    fname = prefix + '.topo.npy'
    np.save(fname, topos)
    print("Wrote topos to {}".format(fname))
    fname = prefix + '.acc_rates.npy'
    np.save(fname, rates)
    print("Wrote acceptance rates to {}".format(fname))
    fname = prefix + '.lmom.npy'
    np.save(fname, lmoms)
    print("Wrote link momenta to {}".format(fname))

    # Compute meson correlation functions
    if args.compute_dirac == "":
        print("Skipping propagator calcs")
    else:
        if args.compute_dirac == "wilson":
            assert args.kappa is not None, "kappa required"
            make_D = lambda cfg: dirac_op(cfg,
                                          kappa=args.kappa,
                                          sign=1,
                                          fermion_bc=action.fermion_bc).toarray()
            Cts = []    # accumulator for zero-momentum pion C(t) over all configs
            for cfg in cfgs:
                D = make_D(cfg)
                Dinv = np.linalg.inv(D)             # all to all propagagtor
                Dinv = np.reshape(Dinv, L + [NS] + L + [NS])
                Lx, Lt = L
                C_src_avg = 0   # for averaging over all sources
                # and, for each, shift the source to the origin
                for x in range(Lx):
                    for t in range(Lt):
                        # propagator from fixed (x, t, any spin) to any point,
                        # shifted back by (x, t)
                        prop = np.roll(Dinv[:,:,:,x,t,:], (-x,-t), axis=(0,1))
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
                        antiprop = np.einsum(
                            'ab,xybc,cd->xyad', pauli(3),
                            np.swapaxes(np.conj(prop), axis1=-1, axis2=-2), pauli(3))
                        meson_corr = lambda p1, p2: np.einsum(
                            'xyab,bc,xycd,da -> xy', prop, p1, antiprop, p2)
                        assert np.allclose(
                            meson_corr(pauli(3), pauli(3)),
                            np.sum(np.abs(prop**2), axis=(2,3)))
                        pion_corr = meson_corr(pauli(3), pauli(3))
                        # TODO momentum-project before source averaging
                        C_src_avg = C_src_avg + pion_corr / (Lx*Lt)
                # project onto zero momentum by summing over x within each y slice
                Ct = np.mean(C_src_avg, axis=0)
                Cts.append(Ct)
        elif args.compute_dirac == "staggered":
            assert args.kappa is not None, "kappa required"
            m0 = m0_from_kappa(args.kappa, Nd)
            make_D = lambda cfg: make_op_matrix(L, lambda psi: apply_staggered_D(psi, U=cfg, m0=m0))
            raise NotImplementedError('staggered not implemented')
        else:
            raise RuntimeError(f"Dirac op type {args.compute_dirac} not supported")

        fname = prefix + '.meson_Ct.npy'
        np.save(fname, np.array(Cts))
        print("Wrote Cts to {}".format(fname))
    print("TIME ensemble gen {:.2f}s".format(time.time()-start))
