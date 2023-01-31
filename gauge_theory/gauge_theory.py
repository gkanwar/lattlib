### SU(N) gauge theory utils.

import numpy as np
import scipy as sp
import scipy.linalg
import time

def gauge_adj(cfg):
    return np.conj(np.swapaxes(cfg, axis1=-1, axis2=-2))
def gauge_proj_hermitian(cfg):
    cfg *= 0.5
    cfg += gauge_adj(cfg)
    return cfg
def gauge_proj_traceless(cfg):
    Nc = cfg.shape[-1]
    if Nc == 1: # for Nc=1, do not remove overall phase
        return cfg
    diag_shift = np.trace(cfg, axis1=-1, axis2=-2) / Nc
    for c in range(Nc):
        cfg[...,c,c] -= diag_shift
    return cfg

def gauge_expm(A): # A must be anti-hermitian, traceless
    Nc = A.shape[-1]
    if Nc == 2: # specialized to 2x2
        a11 = A[...,0,0]
        a12 = A[...,0,1]
        Delta = np.sqrt(a11**2 - np.abs(a12)**2)
        out = np.zeros_like(A)
        out[...,0,0] = np.cosh(Delta) + a11*np.sinh(Delta)/Delta
        out[...,0,1] = a12*np.sinh(Delta)/Delta
        out[...,1,0] = -np.conj(a12)*np.sinh(Delta)/Delta
        out[...,1,1] = np.cosh(Delta) - a11*np.sinh(Delta)/Delta
        return out
    elif Nc == 1:
        return np.exp(A)
    elif Nc == 3:
        # cayley-hamilton for 3x3 hermitian traceless
        # See [hep-lat/0311018] for details
        A = A / 1j # we will compute exp(1j*A) for A hermitian
        assert np.allclose(np.trace(A, axis1=-1, axis2=-2), np.zeros(A.shape[:-2])), \
            'A must be traceless'
        A2 = A @ A
        c1 = np.real(np.trace(A2, axis1=-1, axis2=-2) / 2)
        A3 = A2 @ A
        c0 = np.real(np.trace(A3, axis1=-1, axis2=-2) / 3)
        flips = c0 < 0
        c0 = np.abs(c0)
        c0_max = 2*(c1/3)**(3/2)
        th = np.arccos(c0/c0_max)
        u = np.sqrt(c1/3) * np.cos(th/3)
        w = np.sqrt(c1) * np.sin(th/3)
        xi0 = np.sin(w)/w # TODO: stabilize
        h0 = (u**2 - w**2) * np.exp(2j*u) + np.exp(-1j*u) * (
            8*u**2*np.cos(w) + 2j*u*(3*u**2 + w**2)*xi0
        )
        h1 = 2*u*np.exp(2j*u) - np.exp(-1j*u) * (
            2*u*np.cos(w) - 1j*(3*u**2 - w**2)*xi0
        )
        h2 = np.exp(2j*u) - np.exp(-1j*u)*(np.cos(w) + 3j*u*xi0)
        denom = 9*u**2 - w**2
        f0 = h0 / denom
        f1 = h1 / denom
        f2 = h2 / denom
        f0[flips] = np.conjugate(f0[flips])
        f1[flips] = -np.conjugate(f1[flips])
        f2[flips] = np.conjugate(f2[flips])
        f0 = f0[...,np.newaxis,np.newaxis]
        f1 = f1[...,np.newaxis,np.newaxis]
        f2 = f2[...,np.newaxis,np.newaxis]
        I = np.broadcast_to(np.identity(Nc), A.shape)
        return f0*I + f1*A + f2*A2
    else:
        P = A
        M = np.broadcast_to(np.identity(Nc), A.shape) + P
        for i in range(2,20):
            P = (A @ P) / i
            M += P
        return M
if __name__ == "__main__":
    np.random.seed(1234)
    A = np.random.normal(size=(2,2)) + 1j * np.random.normal(size=(2,2))
    A = 0.5 * (A + np.conj(np.transpose(A)))
    A -= np.identity(2) * np.trace(A) / 2
    U2 = sp.linalg.expm(1j * A).reshape(1,1,1,2,2)
    A = A.reshape(1,1,1,2,2)
    U1 = gauge_expm(1j * A)
    assert(np.allclose(U1, U2))
    print('[PASSED gauge_expm 2x2]')
if __name__ == '__main__':
    np.random.seed(1234)
    batch_size = 128
    A = np.random.normal(size=(batch_size,3,3)) + 1j * np.random.normal(size=(batch_size,3,3))
    A = 0.5 * (A + np.conj(np.swapaxes(A, axis1=-1, axis2=-2)))
    A -= np.identity(3) * np.trace(A, axis1=-1, axis2=-2)[...,np.newaxis,np.newaxis] / 3
    U2 = np.array([sp.linalg.expm(1j * Ai) for Ai in A])
    U1 = gauge_expm(1j * A)
    assert(np.allclose(U1, U2))
    print('[PASSED gauge_expm 3x3]')
    

def open_plaqs_above(cfg, mu, nu):
    cfg0, cfg1 = cfg[mu], cfg[nu]
    a = cfg0
    b = np.roll(cfg1, -1, axis=mu)
    c = gauge_adj(np.roll(cfg0, -1, axis=nu))
    d = gauge_adj(cfg1)
    return a @ b @ c @ d
def open_plaqs_below(cfg, mu, nu):
    cfg0, cfg1 = cfg[mu], cfg[nu]
    a = cfg0
    b = gauge_adj(np.roll(np.roll(cfg1, -1, axis=mu), 1, axis=nu))
    c = gauge_adj(np.roll(cfg0, 1, axis=nu))
    d = np.roll(cfg1, 1, axis=nu)
    return a @ b @ c @ d
def open_plaqs_below_behind(cfg, mu, nu):
    cfg0, cfg1 = cfg[mu], cfg[nu]
    a = gauge_adj(np.roll(cfg0, 1, axis=mu))
    b = gauge_adj(np.roll(np.roll(cfg1, 1, axis=mu), 1, axis=nu))
    c = np.roll(np.roll(cfg0, 1, axis=mu), 1, axis=nu)
    d = np.roll(cfg1, 1, axis=nu)
    return a @ b @ c @ d
if __name__ == "__main__":
    shape = (2,4,4,2,2) # (Nd,L,L,Nc,Nc)
    init_cfg_A = 0.3*(np.random.normal(size=shape) + 1j*np.random.normal(size=shape))
    gauge_proj_hermitian(init_cfg_A)
    gauge_proj_traceless(init_cfg_A)
    cfg = gauge_expm(1j * init_cfg_A)
    assert(np.allclose(
        np.conj(np.trace(open_plaqs_above(cfg, 0, 1),
                         axis1=-1, axis2=-2)),
        np.trace(np.roll(open_plaqs_below(cfg, 0, 1), -1, axis=1),
                 axis1=-1, axis2=-2)
        ))
    assert(np.allclose(
        np.conj(np.trace(open_plaqs_above(cfg, 0, 1),
                         axis1=-1, axis2=-2)),
        np.trace(np.roll(np.roll(
            open_plaqs_below_behind(cfg, 0, 1), -1, axis=1), -1, axis=0),
                 axis1=-1, axis2=-2)
        ))
    print('[PASSED open_plaqs 2x2]')
    
def closed_plaqs(cfg):
    out = np.zeros(cfg.shape[1:-2], dtype=np.float64)
    Nd = cfg.shape[0]
    Nc = cfg.shape[-1]
    for mu in range(Nd-1):
        for nu in range(mu+1, Nd):
            out += np.real(np.trace(
                open_plaqs_above(cfg, mu, nu), axis1=-1, axis2=-2)) / Nc
    return out

def gauge_force(cfg):
    F = np.zeros(cfg.shape, dtype=np.complex128)
    Nd = cfg.shape[0]
    Nc = cfg.shape[-1]
    # TODO: Remove double computation
    for mu in range(Nd):
        for nu in range(Nd):
            if mu == nu: continue
            if mu < nu:
                F[mu] += open_plaqs_above(cfg, mu, nu) + open_plaqs_below(cfg, mu, nu)
            else:
                F[mu] += gauge_adj(open_plaqs_above(cfg, nu, mu)) + open_plaqs_below(cfg, mu, nu)
    F = -1j * (F - gauge_adj(F)) / (2 * Nc)
    if Nc > 1: gauge_proj_traceless(F)
    # print("gauge_force {:.8f}".format(np.mean(np.abs(F))))
    return F
# TEST:
def test_gauge_force():
    print("test_gauge_force")
    np.random.seed(5678)
    L = [4,4,4]
    Nc = 2
    Nd = len(L)
    shape = tuple([Nd] + list(L) + [Nc,Nc])
    beta = 2.0
    init_cfg_A = 0.3*(np.random.normal(size=shape)+1j*np.random.normal(size=shape))
    gauge_proj_hermitian(init_cfg_A)
    gauge_proj_traceless(init_cfg_A)
    cfg = gauge_expm(1j * init_cfg_A)
    old_S = -beta * np.sum(np.real(closed_plaqs(cfg)))

    # Random perturbation
    d = 0.00000001
    dA = d*np.random.normal(size=shape)
    gauge_proj_hermitian(dA)
    gauge_proj_traceless(dA)
    F = beta * gauge_force(cfg)
    dS_thy = np.sum(np.trace(dA @ F, axis1=-1, axis2=-2))

    new_cfg = gauge_expm(1j * dA) @ cfg
    new_S = -beta * np.sum(np.real(closed_plaqs(new_cfg)))
    dS_emp = new_S - old_S
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    ratio = dS_thy / dS_emp
    print("ratio = {:.8g}".format(ratio))
    assert(np.isclose(ratio, 1.0, 1e-4))
    print("[PASSED gauge_force 2x2]")
if __name__ == "__main__": test_gauge_force()

# Sample momentum
def sample_pi(shape):
    pi = np.random.normal(size=shape) + 1j*np.random.normal(size=shape)
    gauge_proj_hermitian(pi)
    gauge_proj_traceless(pi)
    return pi

class Action(object):
    def compute_action(self, cfg):
        raise NotImplementedError()
    def init_traj(self, cfg):
        raise NotImplementedError()
    def force(self, cfg, t):
        raise NotImplementedError()
    def make_tag(self):
        raise NotImplementedError()

class PureGaugeAction(Action):
    def __init__(self, *, beta, beta_prec):
        self.beta = beta
        self.beta_prec = beta_prec
    def compute_action(self, cfg):
        return -self.beta * np.sum(np.real(closed_plaqs(cfg)))
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        return self.beta * gauge_force(cfg)
    def make_tag(self):
        return ('w_b{:.'+str(self.beta_prec)+'f}').format(self.beta)

class PureGaugeModZnAction(Action):
    def __init__(self, *, beta, beta_prec, Zn):
        self.beta = beta
        self.beta_prec = beta_prec
        self.Zn = Zn
    def compute_action(self, cfg):
        P = closed_plaqs(cfg)
        P_target = (np.angle(P) + np.pi/self.Zn) % (2*np.pi / self.Zn) - np.pi/self.Zn
        real_P = np.abs(P) * np.cos(P_target)
        return -self.beta * np.sum(real_P)
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        return self.beta * gauge_force(cfg)
    def make_tag(self):
        return ('w_zn{:d}_b{:.'+str(self.beta_prec)+'f}').format(self.Zn, self.beta)

def update_x_with_p(cfg, pi, action, t, dt):
    np.copyto(cfg, gauge_expm(1j * dt * pi) @ cfg)
def update_p_with_x(cfg, pi, action, t, dt):
    F = action.force(cfg, t)
    pi -= dt * F

# Mutates cfg, pi according to leapfrog update
def leapfrog_update(cfg, pi, action, tau, n_leap, verbose=True):
    if verbose: print("Leapfrog  update")
    start = time.time()
    dt = tau / n_leap
    update_x_with_p(cfg, pi, action, 0, dt / 2)
    for i in range(n_leap-1):
        update_p_with_x(cfg, pi, action, i*dt, dt)
        update_x_with_p(cfg, pi, action, (i+0.5)*dt, dt)
    update_p_with_x(cfg, pi, action, (n_leap-1)*dt, dt)
    update_x_with_p(cfg, pi, action, (n_leap-0.5)*dt, dt / 2)
    if verbose: print("TIME leapfrog {:.2f}s".format(time.time() - start))

def compute_topo_u1_2d(cfg):
    assert cfg.shape[0] == 2
    assert cfg.shape[-1] == 1
    return np.sum(np.angle(closed_plaqs(cfg))) / (2*np.pi)

def topo_charge_density_2d(cfg):
    Nc = cfg.shape[-1]
    Nd = cfg.shape[0]
    assert Nc == 1
    assert Nd == 2
    plaq = open_plaqs_above(cfg, 0, 1)[...,0,0]
    topo = np.angle(plaq)
    return topo

def topo_charge_density_4d(cfg):
    topo = 0
    orients = [
        (0,1,2,3),
        (0,2,3,1),
        (0,3,1,2)
    ]
    for mu,nu,rho,sig in orients:
        F_munu = (
            open_plaqs_above(cfg, mu, nu) +
            gauge_adj(open_plaqs_below(cfg, mu, nu)) +
            open_plaqs_above(cfg, nu, mu) +
            open_plaqs_below_behind(cfg, mu, nu)
            ) / 2
        F_rhosig = (
            open_plaqs_above(cfg, rho, sig) +
            gauge_adj(open_plaqs_below(cfg, rho, sig)) +
            open_plaqs_above(cfg, sig, rho) +
            open_plaqs_below_behind(cfg, rho, sig)
            ) / 2
        F_munu = gauge_proj_traceless((F_munu - gauge_adj(F_munu)) / 2)
        F_rhosig = gauge_proj_traceless((F_rhosig - gauge_adj(F_rhosig)) / 2)
        topo = topo - np.trace(F_munu @ F_rhosig, axis1=-1, axis2=-2)
    return np.real(topo) / (4 * np.pi**2)
