### Small explicit test of using Jax to run Schwinger on GPUs

import functools
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import numpy as onp
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
@functools.partial(jax.jit, static_argnums=(2,3))
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

@functools.partial(jax.jit, static_argnums=1)
def get_normsq(x, axes):
    return np.sum(np.abs(x)**2, axis=axes, keepdims=True)
@functools.partial(jax.jit, static_argnums=1)
def get_norm(x, axes):
    return np.sqrt(get_normsq(x, axes))
@functools.partial(jax.jit, static_argnums=2)
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
@functools.partial(jax.jit, static_argnums=(6,7))
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
@functools.partial(jax.jit, static_argnums=2)
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
        raise RuntimeError('CG failed to converge!')
    return x

### Quick check that ops look right
def _make_op_matrix(shape, D):
    assert len(shape) == 2, 'specialized to 2D'
    rows = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            psi = onp.zeros(shape)
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
    
    onp.random.seed(1234)
    psi = onp.random.normal(size=(2,)+latt_shape)
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
    init_cfg_A = 0.7*onp.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    psi = onp.random.normal(size=latt_shape) + 1j*onp.random.normal(size=latt_shape)
    m0 = 1e-4
    D = lambda x: apply_D(x, U=cfg, sign=1, m0=m0)
    Dx = lambda x: apply_D(x, U=cfg, sign=-1, m0=m0)
    eta = cgne(D, Dx, psi, eps=1e-8, max_iter=1000, verbose=True)
    assert np.allclose(D(eta), psi)
    print('[PASSED test_cgne]')
if __name__ == '__main__': _test_cgne()

@jax.jit
def meas_plaqs(U):
    assert(len(U.shape) == 2+1) # 1+1D ensemble
    U0, U1 = U[0], U[1]
    a = U0
    b = np.roll(U1, -1, axis=0)
    c = np.conj(np.roll(U0, -1, axis=1))
    d = np.conj(U1)
    return a*b*c*d

@functools.partial(jax.jit, static_argnums=1)
def _gauge_action(U, beta):
    return -beta * np.sum(np.real(meas_plaqs(U)))
def gauge_action(U, *, beta):
    return _gauge_action(U, beta)

@functools.partial(jax.jit, static_argnums=1)
def _gauge_deriv(U, beta):
    # specialized to U(1)
    plaqs = meas_plaqs(U)
    dS_0 = plaqs + np.conj(np.roll(plaqs, 1, axis=1))
    dS_1 = np.conj(plaqs) + np.roll(plaqs, 1, axis=0)
    dS = np.stack((dS_0, dS_1), axis=0)
    dS = -1j * beta * (dS - np.conj(dS)) / 2
    return dS
def gauge_deriv(U, *, beta, verbose=False):
    dS = _gauge_deriv(U, beta)
    if verbose: print("gauge_force {:.8f}".format(-np.mean(np.abs(dS))))
    return dS


def _test_gauge_deriv():
    latt_shape = (4,4)
    Nd = len(latt_shape)
    shape = (Nd,) + latt_shape
    beta = 1.0
    init_cfg_A = 0.3*onp.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    old_S = gauge_action(cfg, beta=beta)

    # Random perturbation
    d = 0.0000001
    dA = d*onp.random.normal(size=shape)
    F = gauge_deriv(cfg, beta=beta)
    dS_thy = np.sum(dA * F)

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
    phi = phi + 0j # to complex
    start = time.time()
    print('pf_action CG')
    psi = cg(A, phi, eps=1e-8, max_iter=1000, batched=True)
    print(f'TIME pf_action CG {time.time()-start:.2f}s')
    return np.sum(np.conj(phi) * psi) / 2

def sample_pf(U, *, m0):
    latt_shape = U.shape[1:]
    eta = onp.random.normal(size=(Npf,) + latt_shape)
    D = lambda x: apply_D(x, U=U, sign=1, m0=m0)
    return np.real(D(eta)) # TODO: wasteful, should decompose re/im


# deriv of dirac op w.r.t. U, conjugated by spinors zeta, psi.
#     zeta^dag dD / dU_mu psi
# NOTE: significantly simplified under the assumption of U(1) gauge theory
@functools.partial(jax.jit, static_argnums=3)
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
    init_cfg_A = 0.7*onp.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    zeta = onp.random.normal(size=pf_shape) + 1j*onp.random.normal(size=pf_shape)
    psi = onp.random.normal(size=pf_shape) + 1j*onp.random.normal(size=pf_shape)
    m0 = 1e-4
    sign = 1
    old_zDp = np.sum(np.conj(zeta) * apply_D(psi, U=cfg, sign=sign, m0=m0))

    # Random perturbation
    d = 0.000001
    dA = d*onp.random.normal(size=shape)
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

def pf_deriv(phi, U, *, m0):
    D = lambda x: apply_D(x, U=U, sign=1, m0=m0)
    Dx = lambda x: apply_D(x, U=U, sign=-1, m0=m0)
    A = lambda x: D(Dx(x))
    phi = phi + 0j # as complex
    start = time.time()
    eta = cg(A, phi, eps=1e-8, max_iter=1000, batched=True)
    print(f'TIME pf_deriv CG {time.time()-start:.2f}s')
    Dx_eta = Dx(eta)
    deriv_1 = deriv_D(eta, Dx_eta, U=U, sign=1)
    deriv_2 = deriv_D(Dx_eta, eta, U=U, sign=-1)
    return -(deriv_1 + deriv_2)/2

def _test_pf_deriv():
    onp.random.seed(1234)
    latt_shape = (4,4)
    Nd = len(latt_shape)
    shape = (Nd,) + latt_shape
    beta = 1.0
    init_cfg_A = 0.7*onp.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    m0 = 1.0
    phi = sample_pf(cfg, m0=m0)
    old_S = pf_action(phi, cfg, m0=m0)

    # Random perturbation
    d = 0.0000001
    dA = d*onp.random.normal(size=shape)
    F = pf_deriv(phi, cfg, m0=m0)
    dS_thy = np.sum(dA * F)

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
    def force(self, U):
        assert self.phi is not None
        tot = 0
        for deriv in self.derivs:
            tot = tot + deriv(self.phi, U)
        return -tot
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
    return onp.random.normal(size=shape)

@functools.partial(jax.jit, static_argnums=2)
def update_x_with_p(cfg, pi, dt):
    return cfg * np.exp(1j * dt * pi)
def update_p_with_x(cfg, pi, action, dt):
    F = action.force(cfg)
    return pi + np.real(dt * F)

def leapfrog_update(cfg, pi, action, tau, n_leap, *, verbose=True):
    if verbose: print("Leapfrog  update")
    _start = time.time()
    dt = tau / n_leap
    ### STS
    # cfg = update_x_with_p(cfg, pi, dt / 2)
    # for i in range(n_leap-1):
    #     pi = update_p_with_x(cfg, pi, action, dt)
    #     cfg = update_x_with_p(cfg, pi, dt)
    # pi = update_p_with_x(cfg, pi, action, dt)
    # cfg = update_x_with_p(cfg, pi, dt / 2)
    ### TST
    pi = update_p_with_x(cfg, pi, action, dt/2)
    for i in range(n_leap-1):
        cfg = update_x_with_p(cfg, pi, dt)
        pi = update_p_with_x(cfg, pi, action, dt)
    cfg = update_x_with_p(cfg, pi, dt)
    pi = update_p_with_x(cfg, pi, action, dt/2)
    if verbose: print("TIME leapfrog {:.2f}s".format(time.time() - _start))
    return cfg, pi

def hmc_update(cfg, action, tau, n_leap, *, verbose=True):
    _start = time.time()
    old_cfg = cfg
    old_S = action.init_traj(cfg)
    old_pi = sample_pi(cfg.shape)
    old_K = np.sum(old_pi*old_pi) / 2
    old_H = old_S + old_K
    print(f'TIME init traj {time.time()-_start:.2f}s')

    cfg, new_pi = leapfrog_update(cfg, old_pi, action, tau, n_leap, verbose=verbose)

    _start = time.time()
    new_S = action.compute_action(cfg)
    new_K = np.sum(new_pi*new_pi) / 2
    new_H = new_S + new_K

    delta_H = new_H - old_H
    if verbose:
        print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))
        print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
        print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))
    print(f'TIME finish traj {time.time()-_start:.2f}s')

    # metropolis step
    acc = 0
    if onp.random.random() < np.exp(-delta_H):
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
    m0 = 0.1
    
    schwinger_action = DynamicAction([
        (lambda phi,U: gauge_action(U, beta=beta),
         lambda phi,U: gauge_deriv(U, beta=beta)),
        (lambda phi,U: pf_action(phi, U, m0=m0),
         lambda phi,U: pf_deriv(phi, U, m0=m0))],
        lambda U: sample_pf(U, m0=m0))
    
    latt_shape = (L,L)
    Nd = len(latt_shape)
    shape = (Nd,) + latt_shape
    init_A = 1.0*onp.random.normal(size=shape)
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
        U, S, acc = hmc_update(U, schwinger_action, 1.0, 10)
    print('Burn-in complete')
    for i in tqdm.tqdm(range(N_step)):
        U, S, acc = hmc_update(U, schwinger_action, 0.2, 50)
        acc_rate += acc / N_step
    print(f'Final acc rate {acc_rate}')
