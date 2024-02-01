### Schwinger staggered code converted from Jax to bare Numpy

import functools
# from jax.config import config
# config.update("jax_enable_x64", True)
# import jax
# import jax.numpy as np
# import numpy as onp
import numpy as np
import time
import tqdm.auto as tqdm

def make_eta(shape):
    Nd = len(shape)
    eta_mu = []
    tot_n = np.zeros(shape)
    for mu in range(Nd):
        eta_mu.append((-1)**tot_n)
        broadcast_shape = (1,)*mu + (shape[mu],) + (1,)*(Nd-mu-1)
        tot_n += np.reshape(np.arange(shape[mu]), broadcast_shape)
    eta_5 = (-1)**tot_n
    eta = np.stack(eta_mu, axis=0)
    assert eta.shape == (Nd,) + shape
    assert eta_5.shape == shape
    return eta, eta_5

### NOTE: Staggered D is a non-negative operator, we can simulate with just
### det(D) instead of det(DD*).
def _apply_D(psi, U, sign, m0):
    Nd, latt_shape = U.shape[0], U.shape[1:]
    assert len(latt_shape) == Nd
    assert psi.shape[-Nd:] == latt_shape # batching allowed leftmost
    # bare mass term
    out = m0 * psi
    eta_mu, eta_5 = make_eta(latt_shape)
    # staggered derivative
    for mu in range(Nd):
        ax = -(Nd-mu)
        out = out + sign * eta_mu[mu] * (
            U[mu]*np.roll(psi, -1, axis=ax) -
            np.roll(np.conj(U[mu])*psi, 1, axis=ax)) / 2
    return out
def apply_D(psi, *, U, sign, m0):
    return _apply_D(psi, U, sign, m0)

def get_normsq(x, axes):
    return np.sum(np.abs(x)**2, axis=axes, keepdims=True)
def get_norm(x, axes):
    return np.sqrt(get_normsq(x, axes))
def get_inner(x, y, axes):
    return np.sum(np.conj(x)*y, axis=axes, keepdims=True)

### NOTE: This version is real slow due to Jax dispatch speed
# def cg(A, psi, *, eps, max_iter, batched=False, verbose=False):
#     # handle batching
#     if batched:
#         axes = tuple(range(1,len(psi.shape)))
#     else:
#         axes = tuple(range(len(psi.shape)))
#     # main CG
#     psi_norm = get_norm(psi, axes)
#     _start = time.time()
#     x = np.zeros_like(psi)
#     r = psi - A(x)
#     p = r
#     resid_sq = get_normsq(r, axes)
#     _Ap_time = 0
#     _pAp_time = 0
#     _xr_time = 0
#     _p_time = 0
#     for k in range(max_iter):
#         old_resid_sq = resid_sq
#         _start_Ap = time.time()
#         Ap = A(p)
#         _Ap_time += time.time() - _start_Ap
#         _start_pAp = time.time()
#         pAp = get_inner(p, Ap, axes)
#         _pAp_time += time.time() - _start_pAp
#         _start_xr = time.time()
#         alpha = resid_sq / pAp
#         x = x + alpha*p
#         r = r - alpha*Ap
#         resid_sq = get_normsq(r, axes)
#         _xr_time += time.time() - _start_xr
#         if np.all(np.sqrt(resid_sq)/psi_norm < eps): break
#         if verbose: print(f'CG resid = {np.sqrt(resid_sq.flatten())}')
#         _start_p = time.time()
#         beta = resid_sq / old_resid_sq
#         p = r + beta*p
#         _p_time += time.time() - _start_p
#     # if verbose:
#     if True:
#         _time = time.time() - _start
#         resid = np.sqrt(resid_sq)
#         print(f'CG TIME: {1000*_time:.1f}ms')
#         print(f'CG Ap TIME: {1000*_Ap_time:.1f}ms')
#         print(f'CG pAp TIME: {1000*_pAp_time:.1f}ms')
#         print(f'CG xr TIME: {1000*_xr_time:.1f}ms')
#         print(f'CG p TIME: {1000*_p_time:.1f}ms')
#         print(f'CG SOLVE resid: {resid.flatten()}, iters: {k}')
#     if k == max_iter-1:
#         raise RuntimeError('CG failed to converge!')
#     return x

### NOTE: Still really slow on cg_step for some reason?!
def cg_step(Ap, p, x, r, resid_sq, psi_norm, eps, axes):
    old_resid_sq = resid_sq
    pAp = get_inner(p, Ap, axes)
    alpha = resid_sq / pAp
    x = x + alpha*p
    r = r - alpha*Ap
    resid_sq = get_normsq(r, axes)
    beta = resid_sq / old_resid_sq
    p = r + beta*p
    done = np.all(np.sqrt(resid_sq)/psi_norm < eps)
    return x, r, p, resid_sq, done
def cg_init(Ax, psi, axes):
    r = psi - Ax
    p = r
    resid_sq = get_normsq(r, axes)
    psi_norm = get_norm(psi, axes)
    return r, p, resid_sq, psi_norm
def cg(A, psi, *, eps, max_iter, batched=False, verbose=False):
    # handle batching
    if batched:
        axes = tuple(range(1,len(psi.shape)))
    else:
        axes = tuple(range(len(psi.shape)))
    # main CG
    _start = time.time()
    x = np.zeros_like(psi)
    Ax = A(x)
    r, p, resid_sq, psi_norm = cg_init(Ax, psi, axes)
    _Ap_time = 0
    _step_time = 0
    for k in range(max_iter):
        _start_Ap = time.time()
        Ap = A(p)
        _Ap_time += time.time() - _start_Ap
        _start_step = time.time()
        x, r, p, resid_sq, done = cg_step(Ap, p, x, r, resid_sq, psi_norm, eps, axes)
        _step_time += time.time() - _start_step
        if done: break
        if verbose: print(f'CG resid = {np.sqrt(resid_sq.flatten())}')
    # if True:
    if verbose:
        _time = time.time() - _start
        resid = np.sqrt(resid_sq)
        print(f'CG TIME: {1000*_time:.1f}ms')
        print(f'CG Ap TIME: {1000*_Ap_time:.1f}ms')
        print(f'CG step TIME: {1000*_step_time:.1f}ms')
        print(f'CG SOLVE resid: {resid.flatten()}, iters: {k}')
    if k == max_iter-1:
        raise RuntimeError(f'CG failed to converge! Resid={np.sqrt(resid_sq)}')
    return x

### Quick check that ops look right
def _make_op_matrix(shape, D):
    assert len(shape) == 2, 'specialized to 2D'
    rows = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            psi = np.zeros(shape)
            psi[x,y] = 1.0
            rows.append(D(psi).flatten())
    return np.stack(rows, axis=0)

def _test_cg():
    latt_shape = (4,4)
    A = lambda psi: (
        5.0*psi - np.roll(psi, -1, axis=-2) - np.roll(psi, 1, axis=-2)
        -np.roll(psi, -1, axis=-1) - np.roll(psi, 1, axis=-1))
    A_mat = _make_op_matrix(latt_shape, A)
    print(A_mat, np.linalg.eigvals(A_mat))
    
    np.random.seed(1234)
    psi = np.random.normal(size=(2,)+latt_shape)
    x = cg(A, psi, eps=1e-8, max_iter=10, batched=True, verbose=True)
    assert np.allclose(A(x), psi)
    print('[PASSED test_cg]')
if __name__ == '__main__': _test_cg()

    
### NOTE: Jax provides a built-in CG for a linear operator. Let's use it.
### NOTE: ...except turns out it's much slower than our version?!
# def cg(A, psi, *, eps, max_iter, batched=False, verbose=False):
#     return jax.scipy.sparse.linalg.cg(A, psi, tol=eps, maxiter=max_iter)[0]

def cgne(A, Ax, psi, *, eps, max_iter, batched=False, verbose=False):
    AAx = lambda x: A(Ax(x))
    return Ax(cg(AAx, psi, eps=eps, max_iter=max_iter, batched=batched, verbose=verbose))

def _test_cgne():
    latt_shape = (4,4)
    Nd = len(latt_shape)
    shape = (Nd,) + latt_shape
    beta = 1.0
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    psi = np.random.normal(size=latt_shape) + 1j*np.random.normal(size=latt_shape)
    m0 = 1e-4
    D = lambda x: apply_D(x, U=cfg, sign=1, m0=m0)
    Dx = lambda x: apply_D(x, U=cfg, sign=-1, m0=m0)
    eta = cgne(D, Dx, psi, eps=1e-8, max_iter=1000, verbose=True)
    assert np.allclose(D(eta), psi)
    print('[PASSED test_cgne]')
if __name__ == '__main__': _test_cgne()

def meas_plaqs(U):
    assert(len(U.shape) == 2+1) # 1+1D ensemble
    U0, U1 = U[0], U[1]
    a = U0
    b = np.roll(U1, -1, axis=0)
    c = np.conj(np.roll(U0, -1, axis=1))
    d = np.conj(U1)
    return a*b*c*d

def gauge_action(U, *, beta):
    return -beta * np.sum(np.real(meas_plaqs(U)))

def gauge_deriv_and_act(U, *, beta, verbose=False):
    # specialized to U(1)
    plaqs = meas_plaqs(U)
    dS_0 = plaqs + np.conj(np.roll(plaqs, 1, axis=1))
    dS_1 = np.conj(plaqs) + np.roll(plaqs, 1, axis=0)
    dS = np.stack((dS_0, dS_1), axis=0)
    dS = -1j * beta * (dS - np.conj(dS)) / 2
    if verbose: print("gauge_force {:.8f}".format(-np.mean(np.abs(dS))))
    return dS, -beta * np.sum(np.real(plaqs))


def _test_gauge_deriv():
    latt_shape = (4,4)
    Nd = len(latt_shape)
    shape = (Nd,) + latt_shape
    beta = 1.0
    init_cfg_A = 0.3*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    old_S = gauge_action(cfg, beta=beta)

    # Random perturbation
    d = 0.0000001
    dA = d*np.random.normal(size=shape)
    F, old_S_2 = gauge_deriv_and_act(cfg, beta=beta)
    dS_thy = np.sum(dA * F)
    assert np.allclose(old_S_2, old_S)

    new_cfg = cfg * np.exp(1j * dA)
    new_S = gauge_action(new_cfg, beta=beta)
    dS_emp = new_S - old_S
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    print("ratio = {:.8g}".format(dS_thy / dS_emp))
    assert np.isclose(dS_thy/dS_emp, 1.0)
    print('[PASSED test_gauge_deriv]')
if __name__ == "__main__": _test_gauge_deriv()


### S_pf = \sum_{i=1,2} phi_i (D Dx)^{-1} phi_i / 2
Npf = 2
def pf_action(phi, U, *, m0):
    assert phi.shape[0] == Npf, 'wrong number of PF components'
    D = lambda x: apply_D(x, U=U, sign=1, m0=m0)
    Dx = lambda x: apply_D(x, U=U, sign=-1, m0=m0)
    A = lambda x: D(Dx(x))
    phi = phi[0] + 1j*phi[1] # to complex
    start = time.time()
    psi = cg(A, phi, eps=1e-8, max_iter=1000, batched=False)
    return np.sum(np.conj(phi) * psi)

def sample_pf(U, *, m0):
    assert Npf == 2, 'sample_pf specialized for 2 PF flavors'
    latt_shape = U.shape[1:]
    chi = np.random.normal(size=latt_shape) + 1j*np.random.normal(size=latt_shape)
    D = lambda x: apply_D(x, U=U, sign=1, m0=m0)
    D_chi = D(chi)
    phi = np.stack((np.real(D_chi), np.imag(D_chi)), axis=0)
    return phi / np.sqrt(2)


# deriv of dirac op w.r.t. U, conjugated by spinors zeta, psi.
#     zeta^dag dD / dU_mu psi
# NOTE: significantly simplified under the assumption of U(1) gauge theory
def _deriv_D(zeta, psi, U, sign):
    Nd, latt_shape = U.shape[0], U.shape[1:]
    # assert len(latt_shape) == Nd
    # assert zeta.shape[-Nd:] == latt_shape
    # assert psi.shape[-Nd:] == latt_shape
    zeta = np.conj(zeta)
    deriv = []
    eta_mu, eta_5 = make_eta(latt_shape)
    for mu in range(Nd):
        ax = -(Nd-mu)
        deriv.append(1j * sign * np.sum(
            (eta_mu[mu] * zeta * U[mu]*np.roll(psi, -1, axis=ax) +
             np.roll(eta_mu[mu] * zeta, -1, axis=ax) * np.conj(U[mu])*psi) / 2,
            axis=tuple(range(len(psi.shape)-Nd)))) # sum over batch dims
    deriv = np.stack(deriv, axis=0)
    assert deriv.shape == U.shape
    return deriv
def deriv_D(zeta, psi, *, U, sign):
    return _deriv_D(zeta, psi, U, sign)

def _test_deriv_D():
    latt_shape = (4,4)
    Nd = len(latt_shape)
    shape = (Nd,) + latt_shape
    pf_shape = (Npf,) + latt_shape
    beta = 1.0
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    zeta = np.random.normal(size=pf_shape) + 1j*np.random.normal(size=pf_shape)
    psi = np.random.normal(size=pf_shape) + 1j*np.random.normal(size=pf_shape)
    m0 = 1e-4
    sign = 1
    old_zDp = np.sum(np.conj(zeta) * apply_D(psi, U=cfg, sign=sign, m0=m0))

    # Random perturbation
    d = 0.000001
    dA = d*np.random.normal(size=shape)
    F = deriv_D(zeta, psi, U=cfg, sign=sign)
    dD_thy = np.sum(dA * F)

    new_cfg = cfg * np.exp(1j * dA)
    new_zDp = np.sum(np.conj(zeta) * apply_D(psi, U=new_cfg, sign=sign, m0=m0))
    dD_emp = new_zDp - old_zDp
    print("dD (thy.) = {:.5g}".format(dD_thy))
    print("dD (emp.) = {:.5g}".format(dD_emp))
    print("ratio = {:.8g}".format(dD_thy / dD_emp))
    assert np.isclose(dD_thy/dD_emp, 1.0)
    print('[PASSED test_deriv_D]')
if __name__ == "__main__": _test_deriv_D()

def pf_deriv_and_act(phi, U, *, m0):
    D = lambda x: apply_D(x, U=U, sign=1, m0=m0)
    Dx = lambda x: apply_D(x, U=U, sign=-1, m0=m0)
    A = lambda x: D(Dx(x))
    phi = phi[0] + 1j*phi[1] # as complex
    start = time.time()
    psi = cg(A, phi, eps=1e-8, max_iter=1000, batched=False)
    # print(f'TIME pf_deriv CG {time.time()-start:.2f}s')
    Dx_psi = Dx(psi)
    deriv_1 = deriv_D(psi, Dx_psi, U=U, sign=1)
    deriv_2 = deriv_D(Dx_psi, psi, U=U, sign=-1)
    return -(deriv_1 + deriv_2), np.sum(np.conj(phi) * psi)

def _test_pf_deriv():
    np.random.seed(1234)
    latt_shape = (4,4)
    Nd = len(latt_shape)
    shape = (Nd,) + latt_shape
    beta = 1.0
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    m0 = 1.0
    phi = sample_pf(cfg, m0=m0)
    old_S = pf_action(phi, cfg, m0=m0)

    # Random perturbation
    d = 0.0000001
    dA = d*np.random.normal(size=shape)
    F, old_S_2 = pf_deriv_and_act(phi, cfg, m0=m0)
    dS_thy = np.sum(dA * F)
    assert np.allclose(old_S_2, old_S)

    new_cfg = cfg * np.exp(1j * dA)
    new_S = pf_action(phi, new_cfg, m0=m0)
    dS_emp = new_S - old_S
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    print("ratio = {:.8g}".format(dS_thy / dS_emp))
    assert np.isclose(dS_thy/dS_emp, 1.0)
    print('[PASSED test_pf_deriv]')
if __name__ == "__main__": _test_pf_deriv()


class DynamicAction:
    def __init__(self, terms, refresh_pf):
        self.actions = []
        self.derivs = []
        for act,deriv in terms:
            self.actions.append(act)
            self.derivs.append(deriv)
        self.refresh_pf = refresh_pf
        self.phi = None
    def compute_action(self, U):
        assert self.phi is not None
        tot = 0
        for act in self.actions:
            start = time.time()
            tot = tot + act(self.phi, U)
            print(f'TIME compute_act {time.time()-start:.2f}s')
        # assert np.isclose(np.real(tot), tot)
        return np.real(tot)
    def force_and_act(self, U):
        assert self.phi is not None
        tot = 0
        tot_S = 0
        for deriv in self.derivs:
            d, S = deriv(self.phi, U)
            tot = tot + d
            tot_S = tot_S + S
        return -tot, tot_S
    def init_traj(self, U):
        start = time.time()
        self.phi = self.refresh_pf(U)
        print(f'TIME init traj refresh_pf {time.time()-start:.2f}s')
        start = time.time()
        S = self.compute_action(U)
        print(f'TIME init traj action {time.time()-start:.2f}s')
        return S
    __call__ = compute_action

def sample_pi(shape):
    return np.random.normal(size=shape)

def update_x_with_p(cfg, pi, dt):
    return cfg * np.exp(1j * dt * pi)
def update_p_with_x(cfg, pi, action, dt):
    F, S = action.force_and_act(cfg)
    return pi + np.real(dt * F), S

SHORT_CUT = True
SHORT_CUT_THRESH = 10.0
def leapfrog_update(cfg, pi, action, dt, n_leap, *, H0=0.0, verbose=True):
    if verbose: print("Leapfrog  update")
    _start = time.time()
    dHs = []
    ### TST
    pi, S = update_p_with_x(cfg, pi, action, dt/2)
    for i in range(n_leap-1):
        cfg = update_x_with_p(cfg, pi, dt)
        old_pi = pi
        pi, S = update_p_with_x(cfg, pi, action, dt)
        # need midpoint for symplectic energy estimate
        mid_pi = (pi + old_pi)/2
        dHs.append((S + np.sum(mid_pi**2)/2) - H0)
        if SHORT_CUT and np.abs(dHs[-1]) > SHORT_CUT_THRESH:
            print(f'Shortcut: {np.real(dHs)}')
            return None
    cfg = update_x_with_p(cfg, pi, dt)
    old_pi = pi
    pi, S = update_p_with_x(cfg, pi, action, dt/2)
    dHs.append((S + np.sum(pi**2)/2) - H0)
    if SHORT_CUT and np.abs(dHs[-1]) > SHORT_CUT_THRESH:
        print(f'Shortcut: {np.real(dHs)}')
        return None
    ## STS
    # cfg = update_x_with_p(cfg, pi, dt / 2)
    # for i in range(n_leap-1):
    #     pi, S = update_p_with_x(cfg, pi, action, dt)
    #     Hs.append(S + np.sum(pi**2)/2)
    #     cfg = update_x_with_p(cfg, pi, dt)
    # pi, S = update_p_with_x(cfg, pi, action, dt)
    # Hs.append(S + np.sum(pi**2)/2)
    # cfg = update_x_with_p(cfg, pi, dt / 2)
    if verbose: print("TIME leapfrog {:.2f}s".format(time.time() - _start))
    return cfg, pi, np.real(dHs)

def hmc_update(cfg, action, dt, n_leap, *, verbose=True):
    _start = time.time()
    old_cfg = cfg
    old_S = action.init_traj(cfg)
    print(f'old_S = {old_S:.5g}')
    old_pi = sample_pi(cfg.shape)
    old_K = np.sum(old_pi**2) / 2
    old_H = old_S + old_K
    print(f'TIME init traj {time.time()-_start:.2f}s')

    leap_res = leapfrog_update(cfg, old_pi, action, dt, n_leap, H0=old_H, verbose=verbose)
    if leap_res is None: # short-cut rejection
        print('SHORTCUT REJECTION')
        return old_cfg, old_S, 0
    
    cfg, new_pi, dHs = leap_res
    if False: # check reversibility
        print('Checking reversibility')
        cfg_check, _, _ = leapfrog_update(cfg, -new_pi, action, dt, n_leap, H0=old_H, verbose=False)
        assert np.allclose(old_cfg, cfg_check)
    if verbose:
        print(f'dHs = {dHs}')

    _start = time.time()
    new_S = action.compute_action(cfg)
    new_K = np.sum(new_pi**2) / 2
    new_H = new_S + new_K

    delta_H = new_H - old_H
    if verbose:
        print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))
        print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
        print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))
    print(f'TIME finish traj {time.time()-_start:.2f}s')

    # metropolis step
    acc = 0
    if np.random.random() < np.exp(-delta_H):
        acc = 1
        S = new_S
    else:
        cfg = old_cfg
        S = old_S
    if verbose:
        print("Acc {:.5g} (changed {})".format(min(1.0, np.exp(-delta_H)), acc))

    return cfg, S, acc


if __name__ == '__main__':
    L = 16
    beta = 1.0
    m0 = 0.01
    
    schwinger_action = DynamicAction([
        (lambda phi,U: gauge_action(U, beta=beta),
         lambda phi,U: gauge_deriv_and_act(U, beta=beta)),
        (lambda phi,U: pf_action(phi, U, m0=m0),
         lambda phi,U: pf_deriv_and_act(phi, U, m0=m0))],
        lambda U: sample_pf(U, m0=m0))
    
    latt_shape = (L,L)
    Nd = len(latt_shape)
    shape = (Nd,) + latt_shape
    init_A = 1.0*np.random.normal(size=shape)
    U = np.exp(1j * init_A)

    # TEST
    # schwinger_action.init_traj(U)
    # S = schwinger_action.compute_action(U)
    # F = schwinger_action.force(U)
    # print('S', S)
    # print('F', F)

    acc_rate = 0
    N_step = 1000
    for i in tqdm.tqdm(range(20)):
        U, S, acc = hmc_update(U, schwinger_action, 0.01, 50)
        print('Gauge act = ', gauge_action(U, beta=beta))
    print('Burn-in complete')
    ens = []
    acts = []
    for i in tqdm.tqdm(range(N_step)):
        U, S, acc = hmc_update(U, schwinger_action, 0.05, 20)
        acc_rate += acc / N_step
        ens.append(U)
        acts.append(S)
    print(f'Final acc rate {acc_rate}')
    np.save('tmp.npy', ens)
    np.save('tmp.S.npy', acts)
