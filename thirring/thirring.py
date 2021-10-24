### Thirring utils.

import math
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
import time

NS = 2
CG_TOL = 1e-12
CG_MAXITER = 1000

def wrap(vals):
    return (vals + math.pi) % (2*math.pi) - math.pi

PAULIS = [
    np.array([[1,0], [0,1]], dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128)
]
def pauli(i):
    return PAULIS[i]

# Fixed 2-spinor options
g_plus = (pauli(1) + 1j * pauli(2))/2
g_minus = (pauli(1) - 1j * pauli(2))/2

## 4-spinor version for testing!
def gamma(i):
    if i == -1: # ident
        return np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]], dtype=np.complex128)
    elif i == 0:
        return np.array([
            [0,0,0,1j],
            [0,0,1j,0],
            [0,-1j,0,0],
            [-1j,0,0,0]], dtype=np.complex128)
    elif i == 1:
        return np.array([
            [0,0,0,-1],
            [0,0,1,0],
            [0,1,0,0],
            [-1,0,0,0]], dtype=np.complex128)
    elif i == 2:
        return np.array([
            [0,0,1j,0],
            [0,0,0,-1j],
            [-1j,0,0,0],
            [0,1j,0,0]], dtype=np.complex128)
    elif i == 3:
        return np.array([
            [0,0,1,0],
            [0,0,0,1],
            [1,0,0,0],
            [0,1,0,0]], dtype=np.complex128)
    elif i == 5:
        return np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,-1,0],
            [0,0,0,-1]], dtype=np.complex128)
    else:
        assert(False) # unknown gamma

def get_coord_index(x, L):
    assert(len(L) == 2)
    return x[0] * L[1] + x[1]
def index_to_coord(i, L):
    assert(len(L) == 2)
    return (i // L[1], i % L[1])
# TEST:
def _test_indices():
    print("test_indices")
    L_test = [4,4]
    for i in range(np.prod(L_test)):
        assert(i == get_coord_index(index_to_coord(i, L_test), L_test))
    print("GOOD\n")
if __name__ == "__main__": _test_indices()

def dirac_op(cfg, kappa, sign=1):
    # print("Making dirac op...")
    start = time.time()
    Nd, L = cfg.shape[0], cfg.shape[1:]
    assert(Nd == 2 and len(L) == 2)
    V = np.prod(L)
    cfg0, cfg1 = cfg[0], cfg[1]

    indptr = []
    indices = []
    data = []
    for i in range(V):
        x = index_to_coord(i, L)
        fwd = list(x)
        fwd[0] = (fwd[0] + 1) % L[0]
        fwd_sign = 1 # spatial PBC
        bwd = list(x)
        bwd[0] = (bwd[0] - 1) % L[0]
        bwd_sign = 1 # spatial PBC
        up = list(x)
        up[1] = (up[1] + 1) % L[1]
        up_sign = -1 if up[1] == 0 else 1 # temporal APBC
        down = list(x)
        down[1] = (down[1] - 1) % L[1]
        down_sign = -1 if down[1] == L[1]-1 else 1 # temporal APBC
        link_fwd = fwd_sign*cfg0[x]
        link_bwd = bwd_sign*np.conj(cfg0[tuple(bwd)])
        link_up = up_sign*cfg1[x]
        link_down = down_sign*np.conj(cfg1[tuple(down)])
        j_fwd = get_coord_index(fwd, L)
        j_bwd = get_coord_index(bwd ,L)
        j_up = get_coord_index(up, L)
        j_down = get_coord_index(down, L)

        if NS == 2: # paulis for 2-spinors
            j_blocks = [(i, pauli(0)),
                        (j_fwd, -kappa * link_fwd * (pauli(0) - sign*pauli(1))),
                        (j_bwd, -kappa * link_bwd * (pauli(0) + sign*pauli(1))),
                        (j_up, -kappa * link_up * (pauli(0) - sign*pauli(2))),
                        (j_down, -kappa * link_down * (pauli(0) + sign*pauli(2)))]
        elif NS == 4: # gammas for 4-spinors
            j_blocks = [(i, gamma(-1)),
                        (j_fwd, -kappa * link_fwd * (gamma(-1) - sign*gamma(0))),
                        (j_bwd, -kappa * link_bwd * (gamma(-1) + sign*gamma(0))),
                        (j_up, -kappa * link_up * (gamma(-1) - sign*gamma(1))),
                        (j_down, -kappa * link_down * (gamma(-1) + sign*gamma(1)))]
        else: assert(False) # unknown NS
        j_blocks.sort(key=lambda x: x[0])
        indptr.append(len(indices))
        for j,block in j_blocks:
            indices.append(j)
            data.append(block)
    indptr.append(len(indices))
    data = np.array(data, dtype=np.complex128)
    indptr = np.array(indptr)
    indices = np.array(indices)
    out = sp.sparse.bsr_matrix((data, indices, indptr), shape=(NS*V,NS*V))
    # print("TIME dirac op {:.2f}s".format(time.time() - start))
    rescale = 1/(2*kappa)
    return rescale*out

def apply_wilson_D(psi, *, U, kappa, sign=1):
    Nd, L = U.shape[0], U.shape[1:]
    assert(Nd == 2 and len(L) == 2)
    orig_shape = psi.shape
    psi = psi.reshape(L + (NS,))
    out = psi + 0.0j # ensure complex dtype
    ### FORNOW
    for mu in range(Nd):
        pauli_fwd = pauli(0) - sign*pauli(mu+1)
        pauli_bwd = pauli(0) + sign*pauli(mu+1)
        psi_fwd = np.einsum('ij,...j->...i', pauli_fwd, psi)
        psi_bwd = np.einsum('ij,...j->...i', pauli_bwd, psi)
        psi_fwd = U[mu,...,np.newaxis] * np.roll(psi_fwd, -1, axis=mu)
        psi_bwd = np.roll(np.conj(U[mu,...,np.newaxis]) * psi_bwd, 1, axis=mu)
        if mu == Nd-1: # APBC in time
            ind_fwd = (slice(None),)*(Nd-1) + (-1,)
            psi_fwd[ind_fwd] *= -1
            ind_bwd = (slice(None),)*(Nd-1) + (0,)
            psi_bwd[ind_bwd] *= -1
        out += (-kappa) * (psi_fwd + psi_bwd)
    out /= 2*kappa # rescale
    return out.reshape(orig_shape)

# NOTE: Much slower on a 16x16 lattice. Maybe wins at larger volumes?
def dirac_op_implicit(cfg, kappa, sign=1):
    Nd, L = cfg.shape[0], cfg.shape[1:]
    assert len(L) == 2, 'specialized to 2D'
    V = L[0] * L[1]
    matvec = lambda psi: apply_wilson_D(psi, U=cfg, kappa=kappa, sign=sign)
    rmatvec = lambda psi: apply_wilson_D(psi, U=cfg, kappa=kappa, sign=-sign)
    return sp.sparse.linalg.LinearOperator(
        shape=(V*NS,V*NS), matvec=matvec, rmatvec=rmatvec, dtype=cfg.dtype)

# turn a linear operator defined by a function `D` into a matrix for lattice shape `shape` plus spinor DOFs
def make_op_spinor_matrix(shape, D):
    assert len(shape) == 2, 'specialized to 2D'
    cols = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            for s in range(NS):
                psi = np.zeros(shape + (NS,))
                psi[x,y,s] = 1.0
                cols.append(D(psi).flatten())
    return np.stack(cols, axis=1)

def _test_wilson_D():
    kappa = 0.25
    L = [4,4]
    V = int(np.prod(L))
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    D = lambda psi: apply_wilson_D(psi, U=cfg, kappa=kappa)
    Dmat = make_op_spinor_matrix(cfg.shape[1:], D)
    Dmat2 = dirac_op(cfg, kappa=kappa).toarray()
    print(Dmat[:4,:4])
    print(Dmat2[:4,:4])
    assert np.allclose(Dmat, Dmat2), 'Wilson Dirac ops do not match'
    print('[PASSED test_wilson_D]')
if __name__ == '__main__': _test_wilson_D()

# Derivative of Wilson D with respect to input bosonic field, given gradients matrix
# NOTE: This is not efficient due to O(V^2) memory/compute, but works for now.
def deriv_wilson_D(grad_vec, *, U, kappa):
    Nd, latt_shape = U.shape[0], U.shape[1:]
    grad_vec = np.reshape(grad_vec, latt_shape + (NS,) + latt_shape + (NS,))
    re_grad_vec = np.real(grad_vec)
    im_grad_vec = -np.imag(grad_vec)
    assert NS == 2, 'specialized for NS = 2'
    assert Nd == 2, 'specialized for 2D'
    grad_U = np.zeros_like(U)
    Lx, Lt = latt_shape
    pauli1_fwd = pauli(0) - pauli(1)
    pauli1_bwd = pauli(0) + pauli(1)
    pauli2_fwd = pauli(0) - pauli(2)
    pauli2_bwd = pauli(0) + pauli(2)
    for x in range(Lx):
        for t in range(Lt):
            grad_U[0,x,t] += (1/2) * np.sum(
                re_grad_vec[x,t,:,(x+1)%Lx,t,:] * np.imag(pauli1_fwd * U[0,x,t]) +
                im_grad_vec[x,t,:,(x+1)%Lx,t,:] * (-np.real(pauli1_fwd * U[0,x,t])))
            grad_U[0,x,t] += (-1/2) * np.sum(
                re_grad_vec[(x+1)%Lx,t,:,x,t,:] * np.imag(pauli1_bwd * np.conj(U[0,x,t])) -
                im_grad_vec[(x+1)%Lx,t,:,x,t,:] * np.real(pauli1_bwd * np.conj(U[0,x,t])))
            grad_U[1,x,t] += (1/2) * np.sum(
                re_grad_vec[x,t,:,x,(t+1)%Lt,:] * np.imag(pauli2_fwd * U[1,x,t]) +
                im_grad_vec[x,t,:,x,(t+1)%Lt,:] * (-np.real(pauli2_fwd * U[1,x,t])))
            grad_U[1,x,t] += (-1/2) * np.sum(
                re_grad_vec[x,(t+1)%Lt,:,x,t,:] * np.imag(pauli2_bwd * np.conj(U[1,x,t])) -
                im_grad_vec[x,(t+1)%Lt,:,x,t,:] * np.real(pauli2_bwd * np.conj(U[1,x,t])))
    # APBC in time
    grad_U[1,:,Lt-1] *= -1
    return grad_U

def _test_deriv_wilson():
    print("test_deriv_wilson")
    kappa = 0.25
    L = [2,2]
    V = int(np.prod(L))
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    D = lambda psi: apply_wilson_D(psi, U=cfg, kappa=kappa)
    Dmat = make_op_spinor_matrix(cfg.shape[1:], D)
    print('Dmat', Dmat.shape)
    print(Dmat)
    for i in range(V*NS):
        for j in range(V*NS):
            grad_vec = np.zeros_like(Dmat)
            a, b = np.random.normal(size=2)
            grad_vec[i,j] = a - 1j * b
            compute_S = lambda Dmat: a * np.real(Dmat[i,j]) + b * np.imag(Dmat[i,j])
            old_S = compute_S(Dmat)
            F = deriv_wilson_D(grad_vec, U=cfg, kappa=kappa)
            d = 0.000001
            dA = d*np.random.normal(size=shape)
            dS_thy = np.sum(dA * F)
            new_cfg = cfg * np.exp(1j * dA)
            new_D = lambda psi: apply_wilson_D(psi, U=new_cfg, kappa=kappa)
            new_Dmat = make_op_spinor_matrix(new_cfg.shape[1:], new_D)
            new_S = compute_S(new_Dmat)
            print('new_S', new_S)
            dS_emp = new_S - old_S
            if dS_emp == 0.0:
                assert np.isclose(dS_emp, 0.0), f'dS_emp = {dS_emp} != 0'
                continue
            ratio = dS_thy / dS_emp
            print("dS (thy.) = {:.5g}".format(dS_thy))
            print("dS (emp.) = {:.5g}".format(dS_emp))
            print("ratio = {:.8g}".format(ratio))
            assert np.isclose(ratio, 1.0, rtol=1e-4), f'ratio failed for (i,j) = ({i},{j})'
    print("[PASSED test_deriv_wilson]\n")
if __name__ == "__main__": _test_deriv_wilson()


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

def m0_from_kappa(kappa, Nd):
    return 1/(2*kappa) - Nd

### NOTE: Staggered D has positive determinant, we can simulate with just
### det(D) instead of det(DD*).
def apply_staggered_D(psi, *, U, m0):
    Nd, latt_shape = U.shape[0], U.shape[1:]
    assert len(latt_shape) == Nd
    assert psi.shape[-Nd:] == latt_shape # batching allowed leftmost
    # bare mass term
    out = m0 * psi
    eta_mu, _ = make_eta(latt_shape)
    # staggered derivative
    for mu in range(Nd):
        ax = -(Nd-mu)
        out = out + eta_mu[mu] * (
            U[mu]*np.roll(psi, -1, axis=ax) -
            np.roll(np.conj(U[mu])*psi, 1, axis=ax)) / 2
    return out

# turn a linear operator defined by a function `D` into a matrix with shape `shape`
def make_op_matrix(shape, D):
    assert len(shape) == 2, 'specialized to 2D'
    cols = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            psi = np.zeros(shape)
            psi[x,y] = 1.0
            cols.append(D(psi).flatten())
    return np.stack(cols, axis=1)

# Derivative of staggered D with respect to input bosonic field, given gradients matrix
# NOTE: This is not efficient due to O(V^2) memory/compute, but works for now.
def deriv_staggered_D(grad_vec, *, U, m0):
    Nd, latt_shape = U.shape[0], U.shape[1:]
    eta_mu, eta_5 = make_eta(latt_shape)
    grad_vec = np.reshape(grad_vec, latt_shape + latt_shape)
    re_grad_vec = np.real(grad_vec)
    im_grad_vec = -np.imag(grad_vec)
    assert Nd == 2, 'specialized for 2D'
    grad_U = np.zeros_like(U)
    Lx, Lt = latt_shape
    for x in range(Lx):
        for t in range(Lt):
            grad_U[0,x,t] += (eta_mu[0,x,t] / 2) * (
                re_grad_vec[x,t,(x+1)%Lx,t] * (-np.imag(U[0,x,t])) +
                im_grad_vec[x,t,(x+1)%Lx,t] * np.real(U[0,x,t]))
            grad_U[0,x,t] += (-eta_mu[0,(x+1)%Lx,t] / 2) * (
                re_grad_vec[(x+1)%Lx,t,x,t] * (-np.imag(U[0,x,t])) +
                im_grad_vec[(x+1)%Lx,t,x,t] * (-np.real(U[0,x,t])))
            grad_U[1,x,t] += (eta_mu[1,x,t] / 2) * (
                re_grad_vec[x,t,x,(t+1)%Lt] * (-np.imag(U[1,x,t])) +
                im_grad_vec[x,t,x,(t+1)%Lt] * np.real(U[1,x,t]))
            grad_U[1,x,t] += (-eta_mu[1,x,(t+1)%Lt] / 2) * (
                re_grad_vec[x,(t+1)%Lt,x,t] * (-np.imag(U[1,x,t])) +
                im_grad_vec[x,(t+1)%Lt,x,t] * (-np.real(U[1,x,t])))
    return grad_U

def _test_deriv_staggered():
    print("test_deriv_staggered")
    m0 = 0.0
    L = [2,2]
    V = int(np.prod(L))
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    D = lambda psi: apply_staggered_D(psi, U=cfg, m0=m0)
    Dmat = make_op_matrix(cfg.shape[1:], D)
    print('Dmat')
    print(Dmat)
    for i in range(V):
        for j in range(V):
            grad_vec = np.zeros_like(Dmat)
            a, b = np.random.normal(size=2)
            grad_vec[i,j] = a - 1j * b
            compute_S = lambda Dmat: a * np.real(Dmat[i,j]) + b * np.imag(Dmat[i,j])
            old_S = compute_S(Dmat)
            F = deriv_staggered_D(grad_vec, U=cfg, m0=m0)
            d = 0.000001
            dA = d*np.random.normal(size=shape)
            dS_thy = np.sum(dA * F)
            new_cfg = cfg * np.exp(1j * dA)
            new_D = lambda psi: apply_staggered_D(psi, U=new_cfg, m0=m0)
            new_Dmat = make_op_matrix(new_cfg.shape[1:], new_D)
            new_S = compute_S(new_Dmat)
            print('new_S', new_S)
            dS_emp = new_S - old_S
            if dS_emp == 0.0:
                assert np.isclose(dS_emp, 0.0), f'dS_emp = {dS_emp} != 0'
                continue
            ratio = dS_thy / dS_emp
            print("dS (thy.) = {:.5g}".format(dS_thy))
            print("dS (emp.) = {:.5g}".format(dS_emp))
            print("ratio = {:.8g}".format(ratio))
            assert np.isclose(ratio, 1.0, rtol=1e-4), f'ratio failed for (i,j) = ({i},{j})'
    print("[PASSED test_deriv_staggered]\n")
if __name__ == "__main__": _test_deriv_staggered()


# Do it the stupid way first.
def det_sparse(M):
    start = time.time()
    out = sp.linalg.det(M.todense())
    # print("det = {:.6g} (TIME {:.2f}s)".format(out, time.time()-start))
    return out

def bosonic_force(cfg, verbose=False):
    F = -1j * (cfg - np.conj(cfg)) / 2
    if verbose: print("bosonic_force {:.8f}".format(np.mean(np.abs(F))))
    return F
# TEST:
def _test_bosonic_force():
    print("test_bosonic_force")
    L = [4,4]
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    beta = 2.0
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    old_S = -beta * np.sum(np.real(cfg))

    # Random perturbation
    d = 0.000001
    dA = d*np.random.normal(size=shape)
    F = beta * bosonic_force(cfg)
    dS_thy = np.sum(dA * F)

    new_cfg = cfg * np.exp(1j * dA)
    new_S = -beta * np.sum(np.real(new_cfg))
    dS_emp = new_S - old_S
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    print("ratio = {:.8g}\n".format(dS_thy / dS_emp))
if __name__ == "__main__": _test_bosonic_force()

# Sample momentum
def sample_pi(shape):
    return np.random.normal(size=shape)

# Sample phi ~ exp(-phi^dag (Mdag M)^(-1) phi)
def sample_pf(Mdag):
    chi = math.sqrt(0.5) * (np.random.normal(size=Mdag.shape[0]) + 1j * np.random.normal(size=Mdag.shape[0]))
    return Mdag @ chi

# phi^dag (Mdag M)^(-1) phi
def pf_action(M, Mdag, phi):
    MxMi_phi, info = sp.sparse.linalg.cg(Mdag @ M, phi, tol=CG_TOL, atol=CG_TOL, maxiter=CG_MAXITER)
    if info > 0:
        print(f'WARNING PF action: CG failed to converge after {info} iters')
        resid = np.linalg.norm(Mdag @ M @ MxMi_phi - phi)
        print('... residual (abs):', resid)
        print('... residual (rel):', resid / np.linalg.norm(phi))
    elif info < 0:
        print(f'WARNING PF action: CG illegal input or breakdown ({info})')
    return np.real(np.conj(phi) @ MxMi_phi)
    # return np.real(np.conj(phi) @ np.linalg.inv((Mdag @ M).toarray()) @ phi)

# deriv of dirac op w.r.t. U, conjugated by spinors zeta, psi.
# Returns Re[ zeta^dag dD / dA_mu psi ]
def deriv_dirac_op(zeta, psi, cfg, kappa):
    Nd = len(cfg.shape) - 1
    deriv = np.zeros(cfg.shape, dtype=np.complex128)
    zeta = zeta.reshape(tuple(list(cfg.shape[1:]) + [NS]))
    psi = psi.reshape(tuple(list(cfg.shape[1:]) + [NS]))
    for mu in range(cfg.shape[0]):
        cfg_sign = np.copy(cfg[mu])
        if mu == Nd-1: # temporal
            cfg_sign[:,-1] *= -1 # APBC
        else: # spatial
            cfg_sign[-1] *= 1 # PBC
        lhs1 = np.conj(zeta)
        lhs2 = np.conj(psi)
        if NS == 2:
            rhs1 = np.einsum('ab,xt,xtb->xta',
                             pauli(0) - pauli(mu+1), cfg_sign, np.roll(psi, -1, axis=mu))
            rhs2 = np.einsum('ab,xt,xtb->xta',
                             pauli(0) + pauli(mu+1), cfg_sign, np.roll(zeta, -1, axis=mu))
        elif NS == 4:
            rhs1 = np.einsum('ab,xt,xtb->xta',
                             gamma(-1) - gamma(mu), cfg_sign, np.roll(psi, -1, axis=mu))
            rhs2 = np.einsum('ab,xt,xtb->xta',
                             gamma(-1) + gamma(mu), cfg_sign, np.roll(zeta, -1, axis=mu))
        else: assert(False) # unknown NS
        deriv[mu] = (np.einsum('xta,xta->xt', lhs1, rhs1) +
                     np.einsum('xta,xta->xt', lhs2, rhs2))
        deriv[mu] *= (-kappa/2)
    deriv = (deriv - np.conj(deriv)) / 2
    deriv *= 2j # mysterious factor copied from qlua
    deriv *= 1/(2*kappa) # Chroma rescale
    return deriv

def pf_force(M, Mdag, cfg, phi, kappa):
    psi, info = sp.sparse.linalg.cg(Mdag @ M, phi, tol=CG_TOL, atol=CG_TOL, maxiter=CG_MAXITER)
    if info > 0:
        print(f'WARNING PF force: CG failed to converge after {info} iters')
        resid = np.linalg.norm(Mdag @ M @ psi - phi)
        print('... residual (abs):', resid)
        print('... residual (rel):', resid / np.linalg.norm(phi))
    elif info < 0:
        print(f'WARNING PF force: CG illegal input or breakdown ({info})')
    # psi = np.linalg.inv((Mdag @ M).toarray()) @ phi
    zeta = M @ psi
    dD_dA = deriv_dirac_op(zeta, psi, cfg, kappa)
    F = -2 * dD_dA
    # print("pf_force {:.8f}".format(np.mean(np.abs(F))))
    return F
# TEST:
def test_pf_force():
    print("test_pf_force")
    L = [4,4]
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    kappa = 0.10
    M = dirac_op(cfg, kappa, sign=1)
    Mdag = dirac_op(cfg, kappa, sign=-1)
    phi = sample_pf(Mdag)
    old_S = pf_action(M, Mdag, phi)

    # Random perturbation
    d = 0.000001
    dA = d*np.random.normal(size=shape)
    F = pf_force(M, Mdag, cfg, phi, kappa)
    dS_thy = np.sum(dA * F)

    new_cfg = cfg * np.exp(1j * dA)
    M = dirac_op(new_cfg, kappa, sign=1)
    Mdag = dirac_op(new_cfg, kappa, sign=-1)
    new_S = pf_action(M, Mdag, phi)
    dS_emp = new_S - old_S
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    print("ratio = {:.8g}\n".format(dS_thy / dS_emp))
if __name__ == "__main__": test_pf_force()

class Action(object):
    def compute_action(self, cfg):
        raise NotImplementedError()
    def init_traj(self, cfg):
        raise NotImplementedError()
    def force(self, cfg, t):
        raise NotImplementedError()
    def make_tag(self):
        raise NotImplementedError()

class PureBosonicAction(Action):
    def __init__(self, beta):
        self.beta = beta
    def compute_action(self, cfg):
        return -self.beta * np.sum(np.real(cfg))
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        return self.beta * bosonic_force(cfg)
    def make_tag(self):
        return 'w_b{:.2f}'.format(self.beta)

class TwoFlavorAction(Action):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
        self.phi = None
    def compute_action(self, cfg, M=None, Mdag=None):
        if M is None: M = dirac_op(cfg, self.kappa, sign=1)
        if Mdag is None: Mdag = dirac_op(cfg, self.kappa, sign=-1)
        return (-self.beta * np.sum(np.real(cfg))
                + pf_action(M, Mdag, self.phi))
    def init_traj(self, cfg):
        M = dirac_op(cfg, self.kappa, sign=1)
        Mdag = dirac_op(cfg, self.kappa, sign=-1)
        self.phi = sample_pf(Mdag)
        return self.compute_action(cfg, M, Mdag)
    # TODO: Naming following qlua, but this may actually be dS/dA. Should check.
    def force(self, cfg, t):
        M = dirac_op(cfg, self.kappa, sign=1)
        Mdag = dirac_op(cfg, self.kappa, sign=-1)
        F_g = self.beta * bosonic_force(cfg)
        F_pf = pf_force(M, Mdag, cfg, self.phi, self.kappa)
        return F_g + F_pf
    def make_tag(self):
        return 'tf_b{:.2f}_k{:.3f}'.format(self.beta, self.kappa)


# Gradient of determinant of matrix wrt inputs.
# See: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
# Also: https://github.com/pytorch/pytorch/blob/e3d75b84/torch/csrc/autograd/FunctionsManual.cpp#L2606
def grad_logdet(M):
    return np.transpose(np.linalg.inv(M))

def _test_grad_logdet():
    print("test_grad_logdet")
    shape = (8,8)
    A = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    M = A @ np.conj(np.transpose(A))
    old_logdetM = np.log(np.linalg.det(M))
    
    d = 0.0000001
    dM = d*(np.random.normal(size=shape) + 1j * np.random.normal(size=shape))
    deriv = grad_logdet(M)
    dlogdetM_thy = np.sum(dM * deriv)

    new_M = M + dM
    new_logdetM = np.log(np.linalg.det(new_M))
    dlogdetM_emp = new_logdetM - old_logdetM
    ratio = dlogdetM_thy / dlogdetM_emp
    print("dlogdetM (thy.) = {:.5g}".format(dlogdetM_thy))
    print("dlogdetM (emp.) = {:.5g}".format(dlogdetM_emp))
    print("ratio = {:.8g}".format(ratio))
    assert np.isclose(ratio, 1.0)
    print("[PASSED test_grad_logdet]\n")
if __name__ == "__main__": _test_grad_logdet()

def _test_grad_logdet2():
    print("test_grad_logdet2")
    m0 = 0.0
    L = [2,2]
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    D = lambda psi: apply_staggered_D(psi, U=cfg, m0=m0)
    Dmat = make_op_matrix(cfg.shape[1:], D)
    print('Dmat')
    print(Dmat)
    old_S = np.log(np.linalg.det(Dmat))
    print('old_S', old_S)

    # Random perturbation
    d = 0.0000001
    dA = d*np.random.normal(size=shape)
    F = deriv_staggered_D(grad_logdet(Dmat), U=cfg, m0=m0)
    dS_thy = np.sum(dA * F)

    new_cfg = cfg * np.exp(1j * dA)
    D = lambda psi: apply_staggered_D(psi, U=new_cfg, m0=m0)
    Dmat = make_op_matrix(new_cfg.shape[1:], D)
    new_S = np.log(np.linalg.det(Dmat))
    print('new_S', new_S)
    dS_emp = new_S - old_S
    ratio = dS_thy / dS_emp
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    print("ratio = {:.8g}".format(ratio))
    assert np.isclose(ratio, 1.0)
    print("[PASSED test_grad_logdet]\n")
if __name__ == "__main__":
    with np.printoptions(linewidth=200):
        _test_grad_logdet2()

# Directly uses the determinant weights
class ExactStaggeredAction(Action):
    def __init__(self, beta, m0, Nf):
        self.beta = beta
        self.m0 = m0
        self.Nf = Nf
    def compute_action(self, cfg):
        D = lambda psi: apply_staggered_D(psi, U=cfg, m0=self.m0)
        Dmat = make_op_matrix(cfg.shape[1:], D)
        S_g = -self.beta * np.sum(np.real(cfg))
        S_f = -self.Nf * np.log(np.linalg.det(Dmat))
        return S_f + S_g
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        F_g = self.beta * bosonic_force(cfg)
        D = lambda psi: apply_staggered_D(psi, U=cfg, m0=self.m0)
        Dmat = make_op_matrix(cfg.shape[1:], D)
        F_f = -self.Nf * deriv_staggered_D(grad_logdet(Dmat), U=cfg, m0=self.m0)
        return F_g + F_f
    def make_tag(self):
        return 'exstg_Nf{:d}_b{:.2f}_m{:.3f}'.format(self.Nf, self.beta, self.m0)

class ExactWilsonAction(Action):
    def __init__(self, beta, kappa, Nf=2):
        self.beta = beta
        self.kappa = kappa
        if Nf % 2 != 0:
            raise RuntimeError('Exact Wilson action requires an even number of flavors')
        self.Nf = Nf
    def compute_action(self, cfg):
        D = lambda psi: apply_wilson_D(psi, U=cfg, kappa=self.kappa)
        Dmat = make_op_spinor_matrix(cfg.shape[1:], D)
        S_g = -self.beta * np.sum(np.real(cfg))
        sign, logdet = np.linalg.slogdet(Dmat)
        S_f = -self.Nf * logdet  # np.log(np.abs(np.linalg.det(Dmat)))
        return S_f + S_g
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        F_g = self.beta * bosonic_force(cfg)
        D = lambda psi: apply_wilson_D(psi, U=cfg, kappa=self.kappa)
        Dmat = make_op_spinor_matrix(cfg.shape[1:], D)
        F_f = -self.Nf * deriv_wilson_D(grad_logdet(Dmat), U=cfg, kappa=self.kappa)
        return F_g + F_f
    def make_tag(self):
        return 'exwils_Nf{:d}_b{:.2f}_k{:.3f}'.format(self.Nf, self.beta, self.kappa)

def update_x_with_p(cfg, pi, action, t, dt):
    cfg *= np.exp(1j * dt * pi)
def update_p_with_x(cfg, pi, action, t, dt):
    F = action.force(cfg, t)
    pi -= np.real(dt * F)

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
