### Schwinger utils.

import math
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.special import ellipk, ellipj
import sys
import time

NS = 2
CG_TOL = 1e-12
CG_MAXITER = 1000
RHMC_POLY_DEG = 20
RHMC_SMALLEST = 1e-5
RHMC_LARGEST = 1000
debug_default = False
verbose_default = True
test_default=True

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

def stupid_multishift_cg(bsr_matrix, shifts, source, tol=CG_TOL, atol=CG_TOL, maxiter=CG_MAXITER):
    phi = source
    assert(bsr_matrix.shape[0]==bsr_matrix.shape[1])
    dim = bsr_matrix.shape[0]
    psi_arr = []
    info_arr = []
    for shift in shifts:
        A = bsr_matrix + shift*sp.sparse.identity(dim)
        psi, info = sp.sparse.linalg.cg(A, phi, tol=tol, atol=atol, maxiter=maxiter)
        if info > 0:
            print(f'WARNING multishift CG: CG failed to converge after {info} iters')
            resid = np.linalg.norm(A @ psi - phi)
            print('... residual (abs):', resid)
            print('... residual (rel):', resid / np.linalg.norm(phi))
        elif info < 0:
            print(f'WARNING multishift CG: CG illegal input or breakdown ({info})')
        psi_arr.append(psi)
        info_arr.append(info)
    return np.array(psi_arr), np.array(info_arr)

def _test_stupid_multishift_cg():
    latt_shape = (15, 15)
    V = latt_shape[0] * latt_shape[1]
    NS = 2
    nshifts = 20
    # define operator
    # N.B.  inversions will be done for A = dirac_op(...), which is a bsr matrix of shape (NS*V, NS*V)
    # the way of defining A below is just for the convenience in the test only
    A = lambda psi: (
        5.0*psi - np.roll(psi, -1, axis=-2) - np.roll(psi, 1, axis=-2)
        -np.roll(psi, -1, axis=-1) - np.roll(psi, 1, axis=-1))
    A_mat = make_op_spinor_matrix(latt_shape, A)
    if V < 5: print(A_mat, np.linalg.eigvals(A_mat))
    shifts = np.random.rand(nshifts)
    np.random.seed(1234)
    # N.B.  inversions will with pf sources as (NS * V)-dimensional complex vectors
    #       this way of defining pf below is for convenience in the test only
    phi = np.random.normal(size=latt_shape+(NS,)) + 1j * np.random.normal(size=latt_shape+(NS,))
    psi_arr, info_arr = stupid_multishift_cg(
        sp.sparse.bsr_matrix(A_mat),                        # convert to BSR matrix of shape (NS*V, NS*V)
        shifts,
        phi.flatten(),                                      # flatten to (NS*V)-dimensional complex vector
        maxiter=1000)
    # N.B.  in application, fermions will be kept as (NS * V)-dimensional complex vectors;
    #       this reshaping is for convenience in the test only, to apply A to psi
    psi_arr = psi_arr.reshape((nshifts,) + latt_shape + (NS, ))
    for psi, shift in zip(psi_arr, shifts):
        #check = A(psi) + shift*psi
        #print("residue", np.linalg.norm(check-phi))
        assert np.allclose(A(psi) + shift*psi, phi)
    print('[PASSED test_stupid_multishift_cg]')
if __name__ == '__main__': _test_stupid_multishift_cg()

def plaqs_bc_weights(cfg_shape, gauge_bc):
    assert(len(cfg_shape) == 2+1)
    Nd, L = cfg_shape[0], cfg_shape[1:]
    assert(len(gauge_bc)==Nd)
    gauge_weights = np.ones(shape=L)
    gauge_weights[L[0]-1, ::] *= gauge_bc[0]
    gauge_weights[::, L[1]-1] *= gauge_bc[1]
    return gauge_weights

# TODO: deprecate naming
def ensemble_plaqs(cfg, gauge_bc=(1, 1)):
    assert(len(cfg.shape) == 2+1) # 1+1D ensemble
    cfg0, cfg1 = cfg[0], cfg[1]
    a = cfg0
    b = np.roll(cfg1, -1, axis=0)
    c = np.conj(np.roll(cfg0, -1, axis=1))
    d = np.conj(cfg1)
    bc_weights = plaqs_bc_weights(cfg.shape, gauge_bc)
    return a*b*c*d * bc_weights

def wrap(vals):
    return (vals + math.pi) % (2*math.pi) - math.pi

def compute_topo(cfg, gauge_bc=(1, 1)):
    assert(len(cfg.shape) == 2+1)
    cfg0, cfg1 = cfg[0], cfg[1]
    a = np.imag(np.log(cfg0))
    b = np.imag(np.log(np.roll(cfg1, -1, axis=0)))
    c = np.imag(np.log(np.conj(np.roll(cfg0, -1, axis=1))))
    d = np.imag(np.log(np.conj(cfg1)))
    bc_weights = plaqs_bc_weights(cfg.shape, gauge_bc)
    return wrap(a+b+c+d) / (2*math.pi) * bc_weights

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

# sign=1 computes Wilson D, sign=-1, Wilson D^d
# D = C * (1 - kappa * H), H hopping matrix, C explicit constant
def dirac_op(cfg, kappa, sign=1, fermion_bc=(1, -1)):
    # print("Making dirac op...")
    start = time.time()
    Nd, L = cfg.shape[0], cfg.shape[1:]
    assert(Nd == 2 and len(L) == 2)
    V = np.prod(L)
    cfg0, cfg1 = cfg[0], cfg[1]
    #
    # Arrays to construct the Wilson Dirac operator as a matrix in the Block Sparse Row (BSR) format,
    # where each element represents a datum with spin indices (Ns x Ns-dimensional) corresponding to
    # a link term from site n to site m, where m is one of the nearest neighbors of n;
    # the vectors n and m are stored in their flattened form as indices i and j. e.g. in 2d, think
    #
    #                       j = index(n+(0, 1))
    #                           ^
    #                           |
    # j = index(n-(1, 0)) <- i = index(n) -> j = index(n+(1, 0))
    #                           |
    #                           v
    #                       j = index(n-(0, 1))
    #
    # + the diagonal term i->i
    #
    # Here i indexes the sparse rows, and j, the columns of the Dirac operator.
    # For a given i, the data (i, j) ordered by j with j ranging over the nearest neighbors of i + i itself,
    # constitute a block "column".
    #
    indptr = []     # indtptr[i] points to the index of the first value of the i-th sparse row
                    # i.e., indptr[i:i+1] gives a range of indices from which to access the i-th block "column".
                    # since each site has the same number of nearest neighbors, indptr will be
                    #   e.g. 0, 5, 10, ... in 2d or 0, 9, 18, 27, ... in 4d
                    # i.e., all block "columns" have the same size.
    indices = []    # stores block "column" indices for each block "column"
                    #   e.g. in 2d, indices[0:5] gives the *indices* of the (i=0, j) data
                    #   for the 4 nearest neighbors of i=0, plus the diagonal (0, 0) data.
    data = []       # stores block "column" (i, j)-data for each block "column", ordered by j.
                    #   e.g. in 2d, data[0] contains the (i, j)-data for the 1st 5-element block "column",
                    #   ordered by j.
    # construct data, indices, indptr; deal with boundary terms according to fermion_bc
    for i in range(V):
        x = index_to_coord(i, L)                    # x = n
        # ## "forward-backward" terms and spatial boundaries
        # ##
        # ## n = (Nx-1, nt) -> m = (0, mt)   U1(n) delta_(n+(1, 0), m)
        # ##
        fwd = list(x)
        fwd[0] = (fwd[0] + 1) % L[0]
        #fwd_sign = 1 # spatial PBC
        if fermion_bc[0] > 0:
            fwd_sign = 1                            # spatial PBC
        elif fermion_bc[0] < 0:
            fwd_sign = -1 if fwd[0]==0 else 1       # spatial APBC
        else:
            assert(fermion_bc[0] == 0)
            fwd_sign = 0 if fwd[0]==0 else 1        # spatial OBC
        # #
        # # m = (Nx-1, mt) <- n = (0, nt)   U1^d(n-(1, 0)) delta_(n-(1, 0), m)
        # #
        bwd = list(x)
        bwd[0] = (bwd[0] - 1) % L[0]
        # bwd_sign = 1 # spatial PBC
        if fermion_bc[0] > 0:
            bwd_sign = 1                            # spatial PBC
        elif fermion_bc[0] < 0:
            bwd_sign = -1 if bwd[0]==L[0]-1 else 1  # spatial APBC
        else:
            assert(fermion_bc[0] == 0)
            bwd_sign = 0 if bwd[0] == L[0]-1 else 1 # spatial OBC
        # "up-down" terms and temporal boundaries
        #
        #  m = (mx, 0)
        #   ^                               U2(n) delta_(n+(0, 1), m)
        #   |
        #  n = (nx, Nt-1)
        #
        up = list(x)
        up[1] = (up[1] + 1) % L[1]
        # up_sign = -1 if up[1] == 0 else 1 # temporal APBC
        if fermion_bc[1] > 0:
            up_sign = 1                             # temporal PBC
        elif fermion_bc[1] < 0:
            up_sign = -1 if up[1]==0 else 1         # temporal APBC
        else:
            assert(fermion_bc[1] == 0)
            print("ha 3")
            up_sign = 0 if up[1]==0 else 1          # temporal OBC
        ##
        ##  n = (nx, 0)
        ##   |                               U2^d(n-(0, 1)) delta_(n-(0, 1), m)
        ##   v
        ##  m = (mx, Nt-1)
        ##
        down = list(x)
        down[1] = (down[1] - 1) % L[1]
        # down_sign = -1 if down[1] == L[1]-1 else 1 # temporal APBC
        if fermion_bc[1] > 0:
            down_sign = 1                           # temporal PBC
        elif fermion_bc[1] < 0:
            down_sign = -1 if down[1]==L[1]-1 else 1# temporal APBC
        else:
            assert(fermion_bc[1] == 0)
            print("ha 4")
            down_sign = 0 if down[1]==L[1]-1 else 1 # temporal OBC
        #
        link_fwd = fwd_sign*cfg0[x]
        link_bwd = bwd_sign*np.conj(cfg0[tuple(bwd)])
        link_up = up_sign*cfg1[x]
        link_down = down_sign*np.conj(cfg1[tuple(down)])
        j_fwd = get_coord_index(fwd, L)             # j_fwd = m
        j_bwd = get_coord_index(bwd, L)             # j_bwd = m
        j_up = get_coord_index(up, L)               # j_up = m
        j_down = get_coord_index(down, L)           # j_down = m
        if NS == 2: # paulis for 2-spinors
            #
            # Basis choice:
            # g_1 (Euclidean) = -i g_1 (Minkowski) = pauli_x
            # g_2 (Euclidean) = g_0 (Minkowski)    = pauli_y
            #  => g_5 (Euclidean) = i g_1 g_2 (Euclidean) = - pauli_z
            # N.B the factor of i for chiral projector in Euclidean space in 2d
            #
            #
            # C * (1
            #   - kappa_s * H_s                                             spatial hopping term
            #   - kappa_t * H_t                                             temporal hopping term
            #    )
            #   = C * (1                                                    identity term
            #       - kappa_s *
            #           (spatial_bc_sgns depend on n, m *)
            #                   (
            #                           (r_s - g_1)   U_1(n)                fwd term
            #                           + (r_s + g_1) U1^d(n-1)             bwd term
            #                   )
            #       - kappa_t *
            #           (temporal_bc_sgns depend on n, m *)
            #                   (
            #                           (r_t - g_2)   U2(n)                 up term
            #                           + (r_t + g_2) U2^d(n-2)             down term
            #                   )
            #       )
            #
            # C = 1/(2 * kappa_t * a_t) = xi / (2 * kappa_t * a_s)
            #
            # Choose isotropic parameters:
            #
            # xi === a_s / a_t = 1
            # zeta === kappa_s / kappa_t = 1    => kappa_s = kappa_t = kappa
            #                                   => C = 1/(2 * kappa)
            # r_s = r_t = 1     (Wilson terms needed to deal with doublers)
            #
            # compute (i, j) spin-indexed (Ns x Ns - dimensional) data for Wilson D / Wilson D^d, depending on sign
            j_blocks = [(i, pauli(0)),                                                  # idnty delta_(n, m)
                        (j_fwd, -kappa * link_fwd * (pauli(0) - sign*pauli(1))),        # -kappa_s * (r_s -+ g_1) U1(n)          delta_(n + (1, 0), m) * bc_sgn(n, m)
                        (j_bwd, -kappa * link_bwd * (pauli(0) + sign*pauli(1))),        # -kappa_s * (r_s +- g_1) U1^d(n-(1, 0)) delta_(n - (1, 0), m) * bc_sgn(n, m)
                        (j_up, -kappa * link_up * (pauli(0) - sign*pauli(2))),          # -kappa_t * (r_s -+ g_2) U2(n)          delta_(n + (0, 1), m) * bc_sgn(n, m)
                        (j_down, -kappa * link_down * (pauli(0) + sign*pauli(2)))]      # -kappa_t * (r_s +- g_2) U2^d(n-(0, 1)) delta_(n - (0, 1), m) * bc_sgn(n, m)
        elif NS == 4: # gammas for 4-spinors
            j_blocks = [(i, gamma(-1)),
                        (j_fwd, -kappa * link_fwd * (gamma(-1) - sign*gamma(0))),
                        (j_bwd, -kappa * link_bwd * (gamma(-1) + sign*gamma(0))),
                        (j_up, -kappa * link_up * (gamma(-1) - sign*gamma(1))),
                        (j_down, -kappa * link_down * (gamma(-1) + sign*gamma(1)))]
        else: assert(False) # unknown NS
        j_blocks.sort(key=lambda x: x[0])       # sorts by j, index of coordinate m
        indptr.append(len(indices))
        for j,block in j_blocks:
            indices.append(j)                   # stores the indices where data resides
            data.append(block)
    indptr.append(len(indices))
    data = np.array(data, dtype=np.complex128)
    indptr = np.array(indptr)
    indices = np.array(indices)
    out = sp.sparse.bsr_matrix((data, indices, indptr), shape=(NS*V,NS*V))
    # print("TIME dirac op {:.2f}s".format(time.time() - start))
    rescale = 1/(2*kappa)                       # constant C
    return rescale*out

# outputs gamma_5 * dirac_op
def hermitize_dirac_op(dirac_op):
    Dw = sp.sparse.bsr_matrix(dirac_op)
    if NS==2:
        # Basis choice in 2D:
        # g_1 (Euclidean) = -i g_1 (Minkowski) = pauli_x
        # g_2 (Euclidean) = g_0 (Minkowski)    = pauli_y
        #  => g_5 (Euclidean) = i g_1 g_2 (Euclidean) = - pauli_z
        # N.B the factor of i for chiral projector in Euclidean space in 2d
        gamma5 = -1*pauli(3)
    elif NS==4:
        gamma5 = gamma(5)
    else: assert(False) # unknown NS
    Qw = sp.sparse.bsr_matrix((gamma5 * Dw.data, Dw.indices, Dw.indptr), shape=Dw.shape)
    return Qw
def _test_hermitize_dirac_op():
    Ndim = 2
    L = 2
    cfg = np.random.rand(Ndim, L, L)
    kappa = 1
    Dw = dirac_op(cfg, kappa)
    Qw = hermitize_dirac_op(Dw)
    # with sparse matrices, it's more efficient to check
    # how many elements are not equal to each other
    assert((Qw != Qw.conjugate().transpose()).nnz == 0)
    print('[PASSED test_hermitize_D]')
    return
if __name__ == '__main__': _test_hermitize_dirac_op()

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
    # shape of D = (V * Ns, V * Ns)
    # in 2d, Ns = 2
    #   => D[:2, :2] displays the part of D that's diagonal in coordinate space
    #   => D[:4, :4] will display also the nearest-neighbor terms in coordinate space
    print(Dmat[:4,:4])
    print(Dmat2[:4,:4])
    assert np.allclose(Dmat, Dmat2), 'Wilson Dirac ops do not match'
    print('[PASSED test_wilson_D]')
if __name__ == '__main__': _test_wilson_D()

# Derivative of Wilson D with respect to input gauge field, given gradients matrix
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

# Derivative of staggered D with respect to input gauge field, given gradients matrix
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

def gauge_force(cfg, verbose=False, gauge_bc = (1, 1)):
    F = np.zeros(cfg.shape, dtype=np.complex128)
    # specialized to U(1)
    # lattice complex propto w_G(n) U_mu(n) up to the factors of beta that are just
    # the same between the gauge action and the gauge force term.
    # Here w_G(n) takes care of signs from boundary conditions.
    plaqs = ensemble_plaqs(cfg, gauge_bc)
    F[0] = plaqs + np.conj(np.roll(plaqs, 1, axis=1))   # propto w_G(n) U_1(n) + w_G(n-(0, 1)) U^d_1(n-(0, 1))
    F[1] = np.conj(plaqs) + np.roll(plaqs, 1, axis=0)   # propto w_G(n) U_2^d(n) + w_G(n-(1, 0)) U(n-(1, 0))
    F = -1j * (F - np.conj(F)) / 2
    if verbose: print("gauge_force {:.8f}".format(np.mean(np.abs(F))))
    return F
# TEST:
def _test_gauge_force():
    print("test_gauge_force")
    L = [4,4]
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    beta = 2.0
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    old_S = -beta * np.sum(np.real(ensemble_plaqs(cfg)))

    # Random perturbation
    d = 0.000001
    dA = d*np.random.normal(size=shape)
    F = beta * gauge_force(cfg)
    dS_thy = np.sum(dA * F)

    new_cfg = cfg * np.exp(1j * dA)
    new_S = -beta * np.sum(np.real(ensemble_plaqs(new_cfg)))
    dS_emp = new_S - old_S
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    print("ratio = {:.8g}\n".format(dS_thy / dS_emp))
if __name__ == "__main__": _test_gauge_force()

# Sample momentum
def sample_pi(shape):
    return np.random.normal(size=shape)

# Sample phi ~ exp(-phi^dag (Mdag M)^(-1) phi)
def sample_pf(Mdag):
    chi = math.sqrt(0.5) * (np.random.normal(size=Mdag.shape[0]) + 1j * np.random.normal(size=Mdag.shape[0]))
    return Mdag @ chi

# given a0, a1, a2, ... a_(2n), returns R(y)  = a_0 Prod_(k=1)^n (y+a_(2k-1))/(y+a_2k)
def rational_func(coefficients):
    a = np.array(coefficients)
    return lambda y : a[0] * (np.prod(y + a[1::2]))/(np.prod(y + a[2::2]))

# Returns Zolotarev's optimal coefficients (a_0, a_1, ..., a_2n) and error delta
# for a rational approximation R(y) of y^(-1/2)
def zolotarev(smallest_eigenvalue, polynomial_degree, debug=debug_default):
    eps = smallest_eigenvalue
    n = int(polynomial_degree)
    assert(n > 0)
    k = math.sqrt(1-eps)
    #k = 1-eps
    assert(k > 0 and k < 1)
    # K(k) = int_0^pi/2 d theta / (sqrt(1 - k^2 * sin^2 theta))
    # K(m) = int_0^pi/2 d theta / (sqrt(1 - m   * sin^2 theta))
    m = k**2                                    # ellipk, ellipj take m as argument, not k
    v = ellipk(m)/(2.*n+1.)
    (sn, _, _, _) = ellipj(                     # returns sn, cn, dn, ph
            v * np.arange(1, 2*n+1, 1),         #
            m * np.ones(2*n)
            )
    c = np.square(sn)                           # c_r = sn^2(rv, k), r = 1, 2, ..., 2n
    d = k**(2*n+1)*np.square(np.prod(c[0::2]))  # d = k^(2n+1)*(c_1 * c_3 * ... * c_(2n-1))
    # R(y) approximates 1/sqrt(y) with error
    # For the optimal approximation, delta = d^2/(1+sqrt(1-d^2))^2
    delta = (d**2)/((1+math.sqrt(1-d**2))**2)
    # Zolotarev coefficients for R(y)  = a_0 Prod_(k=1)^n (y+a_(2k-1))/(y+a_2k)
    # a_0 = 2 * sqrt(delta) / d * (c_1 * c_3 * ... * c_(2n-1) / *(c_2 * c_4 * ... c_(2n))
    # a_r = (1-c_r)/c_r, r = 1, 2, ... 2n
    a0 = 2 * math.sqrt(delta)/d * (
            (np.prod(c[0::2]))/(np.prod(c[1::2]))
            )
    ar = (1-c)/c                                # r = 1, 2, ... 2n
    assert(np.all(ar > 0))                      # a_1 > a_2 > ... a_(2n)
    assert(np.all(ar[:-1] > ar[1:]))
    if debug:
        print("a0", a0)
        print("ar", ar)
        print("delta", delta)
    a = np.concatenate(([a0], ar))
    return a, delta

# Returns estimate of error for a rational approximation of y*(-alpha), R(y).
# Error is defined as delta = max_(y_min <= y <= y_max)|1- y^alpha R(y)|
# Estimate of error is made with npoints linearly spaced points on range (y_min, y_max),
def sample_delta(R, alpha, y_min, y_max=1, npoints=100):
    sampling_space = np.linspace(y_min, y_max, npoints)
    approximations = np.array(list(map(R, sampling_space)))
    delta = np.max(np.abs(1 - np.power(sampling_space, alpha) * approximations))
    return delta

# given coefficients of degree-n polynomials P0, Q, returns the residues
# of the partial fraction decomposition of P0/Q
def residues(numerator_coeffs, denominator_coeffs):
    a = np.array(numerator_coeffs)      # P0 = prod_(l=1)^n (y + a_l)
    b = np.array(denominator_coeffs)    # Q  = prod_(l=1)^n (y + b_l)
    assert(len(a)==len(b))
    n = len(a)                          # degree of polynomials
    # r_k = (prod_(l=1)^n -b_k + a_l) / (prod_(l neq k)^n -b_k + b_l)
    lambda k : (np.prod(-b[k-1]+a))/(np.prod(-b[k-1]+np.delete(b, k-1)))
    r = np.array(list(map(
        lambda k : (np.prod(-b[k]+a))/(np.prod(-b[k]+np.delete(b, k))),
        np.arange(0, n)
    )))
    return r

# given r, b, a0 returns R(y)  = a_0 (1 + sum_(k=1)^n r_k/(y + b_k))
def partial_frac_decomp(multiplicative_constant, residues, negative_the_poles):
    r = np.array(residues)
    b = np.array(negative_the_poles)
    assert(len(r)==len(b))
    a0 = multiplicative_constant
    return lambda y : a0 * (1 + np.sum(r/(y + b)))

# Rescales coefficients a0, a1, a2, ... a_(2n) in the rational approximation
# R(y) of y^(-1/2), R(y)  = a_0 Prod_(k=1)^n (y+a_(2k-1))/(y+a_2k)
# Optimal Zolotarev coefficients are for range of y in (epsilon, 1)
# If the range of y is not in (epsilon, 1) but in (epsilon, 1) * Lambdasq_m (Lambdasq_m > 0),
# Apply R to y/Lambdasq_m:
# R(y)                      ->      R(y/Lambdasq_m) * Lambda_m^(-1)
# a_0, a_1, a_2, ... a_(2n)         a_0/Lambda_m, a_1 * Lambda^2_m, a_2 * Lambda^2_m, ... a_(2n) * Lambda^2_m
def rescale_rational_func_coeffs(Lambdasq_m, coeffs):
    a = np.array(coeffs)
    assert(Lambdasq_m > 0)
    assert(len(a) % 2 == 1)
    a[0] /= math.sqrt(Lambdasq_m)
    a[1:] *= Lambdasq_m
    return a

# Rescales coefficients of the partial fraction decomposition for the rational approximation
# R(y) of y^(-1/2), R(y)  = a_0 (1 + sum_(k=1)^n r_k/(y + b_k))
# Optimal Zolotarev coefficients are for range of y in (epsilon, 1)
# If the range of y is not in (epsilon, 1) but in (epsilon, 1) * Lambdasq_m (Lambdasq_m > 0),
# Apply R to y/Lambdasq_m:
# R(y)                      ->      R(y/Lambdasq_m) * Lambda_m^(-1)
# a0                        ->      a_0 / Lambda_m^{-1}
# r_1, ..., r_n             ->      Lambda^2_m * r_1, ..., Lambda^2_m * r_n
# b_1, ..., b_n             ->      Lambda^2_m * b_1, ..., Lambda^2_m * b_n  = mu^2_1, ..., mu^2_n
def rescale_partial_frac_decomp_coeffs(Lambdasq_m, multiplicative_constant, residues, negative_the_poles):
    assert(Lambdasq_m > 0)
    a0 = multiplicative_constant
    r = np.array(residues)
    b = np.array(negative_the_poles)
    assert(len(r)==len(b))
    r *= Lambdasq_m
    musq = b * Lambdasq_m
    a0 /= math.sqrt(Lambdasq_m)
    return (a0, r, musq)

def _test_zolotarev(smallest_eigenvalue, largest_eigenvalue, polynomial_degree, verbose=verbose_default):
    alpha = 0.5                 # testing R(y) approximating y^(-alpha) = y^(-1/2)
    y_min = smallest_eigenvalue # sampling interval min
    y_max = largest_eigenvalue  # sampling interval max
    npoints = 1000              # sample delta error this many times
    coeffs, delta_theoretical = zolotarev(smallest_eigenvalue/largest_eigenvalue, polynomial_degree)    # rescaling [eps * C, C] -> [eps, 1]
    if largest_eigenvalue > 1:
        coeffs = rescale_rational_func_coeffs(largest_eigenvalue, coeffs)
    R = rational_func(coeffs)
    delta_sampled = sample_delta(R, alpha, y_min, y_max, npoints)
    assert(delta_theoretical > 0)
    relative_err = np.abs((delta_theoretical - delta_sampled))/delta_theoretical
    if verbose:
        print(f'epsilon={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}, delta={delta_theoretical:1.4e}, rerr={relative_err:1.4e}')
    assert(relative_err < 1/npoints)
    return
if __name__ == '__main__' and test_default:
    if verbose_default: print("Testing the implementation of optimal Zolotarev coefficients for 1/sqrt(x) approximation (w/o rescaling)")
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            _test_zolotarev(smallest_eigenvalue, 1, polynomial_degree)
    if verbose_default: print("Testing the implementation of optimal Zolotarev coefficients for 1/sqrt(x) approximation (w/ rescaling)")
    largest_eigenvalue = 100
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            # eigenvalue range is rescaled from (eps * C, C) to (eps, 1)
            _test_zolotarev(smallest_eigenvalue, largest_eigenvalue, polynomial_degree)

# compare the rational fraction form with its partial fraction decomposition
def _test_function_output(fun1, fun2, y_min, y_max, verbose=verbose_default):
    npoints = 1000
    sampling_space = np.linspace(y_min, y_max, npoints)
    funs1 = np.array(list(map(fun1, sampling_space)))
    funs2  = np.array(list(map(fun2, sampling_space)))
    assert(np.all(funs1 > 0))
    relative_errors = np.abs((funs1 - funs2))/funs1
    max_rel_err = np.max(relative_errors)
    if verbose:
        print(f'y_min={y_min:1.5f}, y_max={y_max:1.5f}, maximum relative error={max_rel_err:1.4e}')
    assert(np.all(np.isclose(funs1, funs2)))
    return
def _test_partial_frac_decomp(eps, n, max_evalue=1, verbose=verbose_default):
    assert(eps < max_evalue and eps > 0)
    a, delta = zolotarev(eps/max_evalue, n)     # rescaling [eps * C, C] -> [eps, 1]
    b = a[2::2]
    a0 = a[0]
    r = residues(a[1::2], b)
    if max_evalue > 1:
        a0, r, b = rescale_partial_frac_decomp_coeffs(max_evalue, a[0], r, a[2::2])
        a = rescale_rational_func_coeffs(max_evalue, a)
    R1 = rational_func(a)
    R2 = partial_frac_decomp(a0, r, b)
    _test_function_output(R1, R2, eps, max_evalue, verbose=verbose)
if __name__ == '__main__' and test_default:
    if verbose_default: print("Testing the partial fractions decomposition of the rational function (w/o rescaling)")
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            if verbose_default: print(f'eps={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}')
            _test_partial_frac_decomp(smallest_eigenvalue, polynomial_degree)
    if verbose_default: print("Testing the partial fractions decomposition of the rational function (w/ rescaling)")
    max_evalue = 100
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            if verbose_default: print(f'eps={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}')
            _test_partial_frac_decomp(smallest_eigenvalue, polynomial_degree, 100)

# returns multiplicative constant, residues, negative the poles, and the error
# for the partial fraction decomposition of the optimal rational approximation of 1/sqrt(x)
def negative_sqrt_coeffs(smallest_evalue, largest_evalue, polynomial_degree):
    assert(smallest_evalue > 0)
    assert(largest_evalue > smallest_evalue)
    eps = smallest_evalue / largest_evalue
    a, delta = zolotarev(eps, polynomial_degree)
    a0 = a[0]                                       # multiplicative constant
    b = a[2::2]                                     # negative the poles
    r = residues(a[1::2], b)
    a0, r, musq = rescale_partial_frac_decomp_coeffs(largest_evalue, a0, r, b)
    return a0, r, musq, delta
def _test_negative_sqrt_coeffs(smallest_evalue, largest_evalue, polynomial_degree, verbose=verbose_default):
    alpha = 0.5                 # testing R(y) approximating y^(-alpha) = y^(-1/2)
    y_min = smallest_eigenvalue # sampling interval min
    y_max = largest_eigenvalue  # sampling interval max
    npoints = 1000              # sample delta error this many times
    a0, r, musq, delta_theoretical = negative_sqrt_coeffs(smallest_evalue, largest_evalue, polynomial_degree)
    R = partial_frac_decomp(a0, r, musq)
    delta_sampled = sample_delta(R, alpha, y_min, y_max, npoints)
    assert(delta_theoretical > 0)
    relative_err = np.abs((delta_theoretical - delta_sampled))/delta_theoretical
    if verbose:
        print(f'epsilon={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}, delta={delta_theoretical:1.4e}, rerr={relative_err:1.4e}')
    assert(relative_err < 1/npoints)
    return
if __name__ == '__main__' and test_default:
    largest_eigenvalue = 100
    if verbose_default: print("Testing the interface for getting 1/sqrt(x) coefficients with automatic rescaling, using partial fractions decomposition")
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            _test_negative_sqrt_coeffs(smallest_eigenvalue, largest_eigenvalue, polynomial_degree)

def pf_zolotarev_delta(delta):
    # delta is not used anywhere; delta is for R, not S^\dagger S which gives R^{-1}
    # empirically, I see delta-s \approx delta^2... guessing that delta -> delta + delta^2 + delta^3 + delta^4 + ... gives a significantly better relative error
    # there should be a rigorous way to see this but it's not relevant for the accuracy of the approximation, so I'm not going to spend time on this
    return delta * 1/(1-delta)

# returns multiplicative constant, residues, negative the poles, and approximation error
# for the partial fraction decomposition of the rational approximation of S(y) s.t.
# S^\dag(y)S(y) approximates y^{+1/2}, which is used for pseudofermion initialization
def pf_init_coeffs_rational_form(smallest_evalue, largest_evalue, polynomial_degree):
    assert(smallest_evalue > 0)
    assert(largest_evalue > smallest_evalue)
    eps = smallest_evalue / largest_evalue
    a, delta = zolotarev(eps, polynomial_degree)
    delta = pf_zolotarev_delta(delta)
    a = np.array(rescale_rational_func_coeffs(largest_evalue, a), dtype=np.complex128)
    ainv = np.empty(a.size, dtype=np.complex128)     # a1, a3,... a_(2n-1) flipped with a2, a4, ... a_(2n)
    ainv[0] = 1/np.sqrt(a[0])
    ainv[1::2] = complex(0, 1) * np.sqrt(a[2::2])
    ainv[2::2] = complex(0, 1) * np.sqrt(a[1::2])
    return (ainv, delta)
def pf_init_coeffs(smallest_evalue, largest_evalue, polynomial_degree):
    assert(smallest_evalue > 0)
    assert(largest_evalue > smallest_evalue)
    eps = smallest_evalue / largest_evalue
    a, delta = zolotarev(eps, polynomial_degree)
    delta = pf_zolotarev_delta(delta)
    a = np.array(rescale_rational_func_coeffs(largest_evalue, a), dtype=np.complex128)
    ainv = np.empty(a.size, dtype=np.complex128)    # a1, a3,... a_(2n-1) flipped with a2, a4, ... a_(2n)
    ainv[0] = 1/a[0]
    ainv[1::2] = a[2::2]
    ainv[2::2] = a[1::2]
    a0 = np.sqrt(ainv[0])                           # multiplicative constant
    ir = complex(0, 1) * residues(np.sqrt(ainv[1::2]), np.sqrt(ainv[2::2]))
    imu = complex(0, 1) * np.sqrt(ainv[2::2])
    return (a0, ir, imu, delta)
def _test_pf_init_coeffs(smallest_evalue, largest_evalue, polynomial_degree, verbose=verbose_default):
    a0, ir, imu, delta_theoretical = pf_init_coeffs(smallest_evalue, largest_evalue, polynomial_degree)
    #coeffs, delta_theoretical = pf_init_coeffs_rational_form(smallest_evalue, largest_evalue, polynomial_degree)
    def _rational_func(coefficients):
        a = np.array(coefficients)
        # N.B. sqrt(y)
        return lambda y : a[0] * (np.prod(math.sqrt(y) + a[1::2]))/(np.prod(math.sqrt(y) + a[2::2]))
    def _partial_frac_decomp(multiplicative_constant, residues, negative_the_poles):
        ir = np.array(residues, dtype=np.complex128)
        imu = np.array(negative_the_poles, dtype=np.complex128)
        assert(len(ir)==len(imu))
        a0 = multiplicative_constant
        return lambda y : a0 * (1 + np.sum(ir/(math.sqrt(y) + imu)))
    #S = _rational_func(coeffs)
    S = _partial_frac_decomp(a0, ir, imu)
    alpha = -0.5                # testing S(y) s.t S^dag S approximates y^(1/2)
    y_min = smallest_eigenvalue # sampling interval min
    y_max = largest_eigenvalue  # sampling interval max
    npoints = 1000              # sample delta error this many times
    assert(delta_theoretical > 0)
    delta_sampled = sample_delta(lambda y : np.conj(S(y)) * S(y), alpha, y_min, y_max, npoints)
    relative_err = np.abs((delta_theoretical - delta_sampled))/delta_theoretical
    if verbose:
        print(f'epsilon={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}, delta (theoretical)={delta_theoretical:1.4e}, delta (sampled) = {delta_sampled:1.4e}, rerr={relative_err:1.4e}')
        # print(f'epsilon={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}, sampled delta={delta_sampled:1.4e}')
    return
if __name__ == '__main__' and test_default:
    largest_eigenvalue = 10000
    if verbose_default:
        print("Testing the interface for getting pf initialization coefficients with automatic rescaling, using partial fractions decomposition")
        print("N.B. the theoretical error for R(y) does not exactly equal the sampled one for S^\dagger S; cf. pf_zolotarev_delta:")
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6, 8):
            _test_pf_init_coeffs(smallest_eigenvalue, largest_eigenvalue, polynomial_degree)

# gets pf from chi (exposed for testing)
def _sample_pf_r(chi, M, Mdag,
                    ia0, ir, imu):  # coefficients for S(y) s.t. S^dag(y) S(y) = R^{-1})(y)
    K = Mdag @ M
    Q = hermitize_dirac_op(M)
    musq = np.conj(imu) * imu       # N.B. these are NOT the same shifts as the one used to compute the action or force, cf. notes.
    zeta, info = stupid_multishift_cg(K, musq, chi, tol=CG_TOL, atol=CG_TOL, maxiter=CG_MAXITER)
    phi = np.copy(chi)
    for k in range(len(musq)):
        if info[k] > 0:
            print(f'WARNING RHMC PF init (term {k}): CG failed to converge after {info[k]} iters')
            resid = np.linalg.norm(K @ zeta[k] + musq[k] * zeta[k] - chi)
            print('... residual (abs):', resid)
            print('... residual (rel):', resid / np.linalg.norm(chi))
        elif info[k] < 0:
            print(f'WARNING RHMC PF init (term {k}): CG illegal input or breakdown ({info[k]})')
        phi += ir[k] * (-imu[k]) * zeta[k]
        phi += Q * (ir[k] * zeta[k])
    phi *= ia0
    return phi
def sample_pf_r(M, Mdag,
                ia0, ir, imu):      # coefficients for S(y) s.t. S^dag(y) S(y) = R^{-1})(y)
    # print("initializing pf (RHMC)")
    chi = math.sqrt(0.5) * (np.random.normal(size=Mdag.shape[0]) + 1j * np.random.normal(size=Mdag.shape[0]))
    return _sample_pf_r(chi, M, Mdag, ia0, ir, imu)

def _test_sample_pf_r(rhmc_smallest, rhmc_largest, rhmc_poly_deg, fermion_bc=(1, -1)):
    print("test_sample_pf_r (RHMC)")
    L = [4,4]
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    kappa = 0.10
    M = dirac_op(cfg, kappa, sign=1, fermion_bc=fermion_bc)
    Mdag = dirac_op(cfg, kappa, sign=-1, fermion_bc=fermion_bc)
    # uses same chi field for all phi
    chi = math.sqrt(0.5) * (np.random.normal(size=Mdag.shape[0]) + 1j * np.random.normal(size=Mdag.shape[0]))
    chi_norm = np.sqrt( np.conj(chi) @ chi )
    print(f'norm of chi: {chi_norm:3.12f}')
    old_phi = np.zeros(shape=Mdag.shape[0])
    old_phi_norm = np.sqrt( np.conj(old_phi) @ old_phi )
    for i in range(len(rhmc_largest)):
        ia0, ir, imu, idelta = pf_init_coeffs(rhmc_smallest[i], rhmc_largest[i], rhmc_poly_deg[i])
        phi = _sample_pf_r(chi, M, Mdag, ia0, ir, imu)
        eps = rhmc_smallest[i]/rhmc_largest[i]
        phi_norm = np.sqrt( np.conj(phi) @ phi )
        phi_norm_change_by_percent = (phi_norm - old_phi_norm) / phi_norm * 100
        angle = np.arccos( (np.conj(old_phi) @ phi) / phi_norm )
        old_phi = phi
        old_phi_norm = phi_norm
        print(f'rhmc_smallest {rhmc_smallest[i]:1.4e}, rhmc_largest {rhmc_largest[i]:1.4e} (eps = {eps:1.4e}), n={rhmc_poly_deg[i]:2d}, phi_norm={phi_norm:3.12f}, change in norm by % = {phi_norm_change_by_percent:+2.5e}, angle = {angle:2.5e}')
if __name__ == "__main__":
    print("Test convergence of RHMC pf sampling with polynomial degree n")
    n = [5, 10, 15, 20, 25, 30, 35]
    rhmc_largest = 1000
    rhmc_smallest = 1e-3
    eps = rhmc_smallest / rhmc_largest
    _test_sample_pf_r(rhmc_smallest * np.ones(len(n)), rhmc_largest * np.ones(len(n)), n)
    print("Test rescaling of largest eigenvalue (same eps, same n)")
    n = 25                              # figure from previous test
    eps = 1e-6
    rhmc_largest = np.array([1., 10., 100., 1000., 1e4, 1e5])
    _test_sample_pf_r(eps * rhmc_largest, rhmc_largest, n * np.ones(len(rhmc_largest), dtype = np.int32))
    print("Test rescaling changing eps with reasonable largest eigenvalue (same eps, same n)")
    n = 20                              # figure from previous test
    rhmc_largest = 1e3                  # figure from previous test
    eps = np.array([1e-1, 1e-3, 1e-5, 1e-7])
    _test_sample_pf_r(eps * rhmc_largest, rhmc_largest * np.ones(len(eps)), n * np.ones(len(eps), dtype = np.int32))

# phi^dag (Mdag M)^(-1) phi
def pf_action(M, Mdag, phi):
    # print("computing pf action")
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

def pf_action_r(M, Mdag, phi, a0, r, musq):
    # print("computing pf RHMC action")
    K = Mdag @ M
    psi, info = stupid_multishift_cg(K, musq, phi, tol=CG_TOL, atol=CG_TOL, maxiter=CG_MAXITER)
    RK_phi = np.copy(phi)
    resid = np.zeros(musq.shape)
    for k in range(len(musq)):
        if info[k] > 0:
            print(f'WARNING RHMC PF action (term {k}): CG failed to converge after {info[k]} iters')
            resid[k] = np.linalg.norm(K @ psi[k] + musq[k] * psi[k]  - phi)
            print('... residual (abs):', resid[k])
            print('... residual (rel):', resid[k] / np.linalg.norm(phi))
        elif info[k] < 0:
            print(f'WARNING RHMC PF action (term {k}): CG illegal input or breakdown ({info[k]})')
        RK_phi += r[k] * psi[k]
    RK_phi *= a0
    return np.real(np.conj(phi) @ RK_phi)
def _test_pf_action_r(rhmc_smallest, rhmc_largest, rhmc_poly_deg, fermion_bc=(1, -1)):
    print("test_sample_pf_r (RHMC)")
    L = [4,4]
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    kappa = 0.10
    M = dirac_op(cfg, kappa, sign=1, fermion_bc=fermion_bc)
    Mdag = dirac_op(cfg, kappa, sign=-1, fermion_bc=fermion_bc)
    # use chi instead of phi to isolate test to pf action only
    chi = math.sqrt(0.5) * (np.random.normal(size=Mdag.shape[0]) + 1j * np.random.normal(size=Mdag.shape[0]))
    chidchi = np.conj(chi) @ chi
    old_S = 0
    for i in range(len(rhmc_largest)):
        a0, r, musq, delta = negative_sqrt_coeffs(rhmc_smallest[i], rhmc_largest[i], rhmc_poly_deg[i])
        S = pf_action_r(M, Mdag, chi, a0, r, musq)
        eps = rhmc_smallest[i]/rhmc_largest[i]
        change_by_percent = (S - old_S)/S * 100
        old_S = S
        print(f'rhmc_smallest {rhmc_smallest[i]:1.4e}, rhmc_largest {rhmc_largest[i]:1.4e} (eps = {eps:1.4e}), n={rhmc_poly_deg[i]:2d}, pf_action {S:3.12f}, change by % = {change_by_percent:+2.5e}')
if __name__ == "__main__":
    print("Test convergence of RHMC pf action with polynomial degree n")
    n = [5, 10, 15, 20, 25]
    rhmc_largest = 1000
    rhmc_smallest = 1e-3
    eps = rhmc_smallest / rhmc_largest
    _test_pf_action_r(rhmc_smallest * np.ones(len(n)), rhmc_largest * np.ones(len(n)), n)
    print("Test rescaling of largest eigenvalue (same eps, same n)")
    n = 20                              # figure from previous test
    eps = 1e-6
    rhmc_largest = np.array([1., 10., 100., 1000., 1e4, 1e5])
    _test_pf_action_r(eps * rhmc_largest, rhmc_largest, n * np.ones(len(rhmc_largest), dtype = np.int32))
    print("Test rescaling changing eps with reasonable largest eigenvalue (same eps, same n)")
    n = 20                              # figure from previous test
    rhmc_largest = 1e3                  # figure from previous test
    eps = np.array([1e-1, 1e-3, 1e-5, 1e-7])
    _test_pf_action_r(eps * rhmc_largest, rhmc_largest * np.ones(len(eps)), n * np.ones(len(eps), dtype = np.int32))


# deriv of dirac op w.r.t. U, conjugated by spinors zeta, psi.
# Returns Re[ zeta^dag dD / dA_mu psi ]
def deriv_dirac_op(zeta, psi, cfg, kappa, fermion_bc=(1, -1)):
    # cfg = (mu, ix, it) with shape (2, Lx, Lt)
    Nd = len(cfg.shape) - 1
    deriv = np.zeros(cfg.shape, dtype=np.complex128)            # force acts on each link
    zeta = zeta.reshape(tuple(list(cfg.shape[1:]) + [NS]))      # (Lx, Lt, Ns)
    psi = psi.reshape(tuple(list(cfg.shape[1:]) + [NS]))
    for mu in range(cfg.shape[0]):
        # deal with BCs: only need to take care of U, not U^d, explicitly (cf. notes)
        # so modify U_1((N_x-1, n_t)) and U_2((n_x, N_t - 1)) according to fermion_bc
        cfg_sign = np.copy(cfg[mu])
        # temporal BCs:
        if mu == Nd-1:
            if fermion_bc[1] > 0:
                cfg_sign[:,-1] *= 1         # PBC
            elif fermion_bc[1] < 0:
                cfg_sign[:,-1] *= -1        # APBC
            else:
                assert(fermion_bc[1] == 0)
                cfg_sign[:,-1] *= 0         # OBC
        # spatial BCs:
        else:
            if fermion_bc[0] > 0:
                cfg_sign[-1] *= 1           # PBC
            elif fermion_bc[0] < 0:
                cfg_sign[-1] *= -1          # APBC
            else:
                assert(fermion_bc[0] == 0)
                cfg_sign[-1] *= 0           # OBC
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
        # for anisotropic kappas:
        # - kappa_s / 2 for mu == 0 and -kappa_t / 2 for mu == 1
        deriv[mu] *= (-kappa/2)
    deriv = (deriv - np.conj(deriv)) / 2
    deriv *= 2j # mysterious factor copied from qlua
    # C = \xi/(2 a_s kappa_t)
    deriv *= 1/(2*kappa) # Chroma rescale
    return deriv

def pf_force(M, Mdag, cfg, phi, kappa, fermion_bc=(1, -1)):
    psi, info = sp.sparse.linalg.cg(Mdag @ M, phi, tol=CG_TOL, atol=CG_TOL, maxiter=CG_MAXITER)
    if info > 0:
        print(f'WARNING PF force: CG failed to converge after {info} iters')
        resid = np.linalg.norm(Mdag @ M @ psi - phi)
        print('... residual (abs):', resid)
        print('... residual (rel):', resid / np.linalg.norm(phi))
    elif info < 0:
        print(f'WARNING PF force: CG illegal input or breakdown ({info})')
    # psi = np.linalg.inv((Mdag @ M).toarray()) @ phi
    M_psi = M @ psi
    dD_dA = deriv_dirac_op(M_psi, psi, cfg, kappa, fermion_bc)
    # - 2* Re[ (D psi)^dag (dD / dA_mu psi) ]
    F = -2 * dD_dA
    print("pf_force {:.8f}".format(np.mean(np.abs(F))))
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

def pf_force_r(M, Mdag, cfg, phi, kappa, a0, r, musq, fermion_bc=(1,-1)):
    K = Mdag @ M
    psi, info = stupid_multishift_cg(K, musq, phi, tol=CG_TOL, atol=CG_TOL, maxiter=CG_MAXITER)
    F = 0
    M_psi = np.zeros(shape=psi.shape, dtype=np.complex128)
    for k in range(len(musq)):
        if info[k] > 0:
            print(f'WARNING RHMC PF force (term {k}): CG failed to converge after {info[k]} iters')
            resid = np.linalg.norm((K + musq[k] * sp.sparse.identity(K.shape[0])) *  psi[k] - phi)
            print('... residual (abs):', resid)
            print('... residual (rel):', resid / np.linalg.norm(phi))
        elif info[k] < 0:
            print(f'WARNING RHMC PF force (term {k}): CG illegal input or breakdown ({info[k]})')
        M_psi[k] = M @ psi[k]
        F += r[k] * deriv_dirac_op(M_psi[k], psi[k], cfg, kappa, fermion_bc)
    # F = - 2 * a0 * sum_k r_k Re[ (M psi)^dag (dM / dA_mu psi) ]
    F *= -2 * a0
    print("pf_force (RHMC) {:.8f}".format(np.mean(np.abs(F))))
    return F
def _test_pf_force_r(fermion_bc = (0, -1)):
    print("test_pf_force (RHMC)")
    L = [4,4]
    Nd = len(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.7*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    kappa = 0.10
    M = dirac_op(cfg, kappa, sign=1, fermion_bc=fermion_bc)
    Mdag = dirac_op(cfg, kappa, sign=-1, fermion_bc=fermion_bc)
    a0, r, musq, delta = negative_sqrt_coeffs(RHMC_SMALLEST, RHMC_LARGEST, RHMC_POLY_DEG)
    ia0, ir, imu, idelta = pf_init_coeffs(RHMC_SMALLEST, RHMC_LARGEST, RHMC_POLY_DEG)
    phi = sample_pf_r(M, Mdag, ia0, ir, imu)
    old_S = pf_action_r(M, Mdag, phi, a0, r, musq)
    d = 0.000001                                                                            # random perturbation
    dA = d*np.random.normal(size=shape)
    F = pf_force_r(M, Mdag, cfg, phi, kappa, a0, r, musq, fermion_bc=fermion_bc)
    dS_thy = np.sum(dA * F)
    new_cfg = cfg * np.exp(1j * dA)
    M = dirac_op(new_cfg, kappa, sign=1, fermion_bc=fermion_bc)
    Mdag = dirac_op(new_cfg, kappa, sign=-1, fermion_bc=fermion_bc)
    new_S = pf_action_r(M, Mdag, phi, a0, r, musq)
    dS_emp = new_S - old_S
    print("S_old = {:.5g}".format(old_S))
    print("S_new= {:.5g}".format(new_S))
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    print("ratio = {:.8g}\n".format(dS_thy / dS_emp))
if __name__ == "__main__": _test_pf_force_r()

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
    def __init__(self, beta, gauge_bc=(1, 1)):
        self.beta = beta
        self.gauge_bc = gauge_bc
    def compute_action(self, cfg):
        return -self.beta * np.sum(np.real(ensemble_plaqs(cfg, gauge_bc=self.gauge_bc)))
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        return self.beta * gauge_force(cfg, gauge_bc=self.gauge_bc)
    def make_tag(self):
        return 'w_b{:.2f}'.format(self.beta)

class TwoFlavorAction(Action):
    def __init__(self, beta, kappa, gauge_bc=(1, 1)):
        self.beta = beta
        self.kappa = kappa
        self.phi = None
        self.gauge_bc = gauge_bc
        self.fermion_bc = (gauge_bc[0],
                           -1 * gauge_bc[1])
    def compute_action(self, cfg, M=None, Mdag=None):
        if M is None: M = dirac_op(cfg, self.kappa,
                                    sign=1, fermion_bc=self.fermion_bc)
        if Mdag is None: Mdag = dirac_op(cfg, self.kappa,
                                        sign=-1, fermion_bc=self.fermion_bc)
        return (-self.beta * np.sum(np.real(ensemble_plaqs(cfg, gauge_bc=self.gauge_bc)))
                + pf_action(M, Mdag, self.phi))
    def init_traj(self, cfg):
        M = dirac_op(cfg, self.kappa,
                    sign=1, fermion_bc=self.fermion_bc)
        Mdag = dirac_op(cfg, self.kappa,
                        sign=-1, fermion_bc=self.fermion_bc)
        self.phi = sample_pf(Mdag)
        # print("Initializing trajectory, computing action")
        return self.compute_action(cfg, M, Mdag)
    # TODO: Naming following qlua, but this may actually be dS/dA. Should check.
    def force(self, cfg, t):
        M = dirac_op(cfg, self.kappa,
                    sign=1, fermion_bc=self.fermion_bc)
        Mdag = dirac_op(cfg, self.kappa,
                        sign=-1, fermion_bc=self.fermion_bc)
        F_g = self.beta * gauge_force(cfg, gauge_bc=self.gauge_bc)
        # print("Computing PF force")
        F_pf = pf_force(M, Mdag, cfg, self.phi, self.kappa, fermion_bc=self.fermion_bc)
        return F_g + F_pf
    def make_tag(self):
        return 'tf_b{:.2f}_k{:.3f}'.format(self.beta, self.kappa)

# TODO add different poly deg for accept reject vs. MD evolution
class OneFlavorAction(Action):
    def __init__(self, beta, kappa,
                    gauge_bc=(1, 1),
                    rhmc_poly_deg = RHMC_POLY_DEG,
                    rhmc_smallest=RHMC_SMALLEST,
                    rhmc_largest=RHMC_LARGEST,
                 ):
        self.beta = beta
        self.kappa = kappa
        self.phi = None
        self.gauge_bc = gauge_bc
        self.fermion_bc = (gauge_bc[0],
                           -1 * gauge_bc[1])
        self.set_optimal_coeffs(rhmc_poly_deg,
                                rhmc_smallest,
                                rhmc_largest)
    def set_optimal_coeffs(self,
                           rhmc_poly_deg,
                           rhmc_smallest,
                           rhmc_largest):
        self.rhmc_poly_deg = rhmc_poly_deg
        self.rhmc_smallest = rhmc_smallest
        self.rhmc_largest = rhmc_largest
        self.a0, self.r, self.musq, self.delta = negative_sqrt_coeffs(self.rhmc_smallest,
                                                  self.rhmc_largest,
                                                  self.rhmc_poly_deg)
        self.ia0, self.ir, self.imu, self.idelta = pf_init_coeffs(self.rhmc_smallest,
                                              self.rhmc_largest,
                                              self.rhmc_poly_deg)
    def compute_action(self, cfg, M=None, Mdag=None):
        if M is None: M = dirac_op(cfg, self.kappa,
                                    sign=1, fermion_bc=self.fermion_bc)
        if Mdag is None: Mdag = dirac_op(cfg, self.kappa,
                                        sign=-1, fermion_bc=self.fermion_bc)
        return (-self.beta * np.sum(np.real(ensemble_plaqs(cfg, gauge_bc=self.gauge_bc)))
                + pf_action_r(M, Mdag, self.phi, self.a0, self.r, self.musq))
    def init_traj(self, cfg):
        M = dirac_op(cfg, self.kappa,
                    sign=1, fermion_bc=self.fermion_bc)
        Mdag = dirac_op(cfg, self.kappa,
                        sign=-1, fermion_bc=self.fermion_bc)
        self.phi = sample_pf_r(M, Mdag,
                                self.ia0, self.ir, self.imu)
        # print("Initializing trajectory, computing action")
        return self.compute_action(cfg, M, Mdag)
    def force(self, cfg, t):
        M = dirac_op(cfg, self.kappa,
                    sign=1, fermion_bc=self.fermion_bc)
        Mdag = dirac_op(cfg, self.kappa,
                        sign=-1, fermion_bc=self.fermion_bc)
        F_g = self.beta * gauge_force(cfg, gauge_bc=self.gauge_bc)
        # print("Computing PF force")
        F_pf = pf_force_r(M, Mdag, cfg, self.phi, self.kappa,
                            self.a0, self.r, self.musq,
                            fermion_bc=self.fermion_bc)
        return F_g + F_pf
    def make_tag(self):
        return 'of_b{:.2f}_k{:.3f}'.format(self.beta, self.kappa)

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
        S_g = -self.beta * np.sum(np.real(ensemble_plaqs(cfg)))
        S_f = -self.Nf * np.log(np.linalg.det(Dmat))
        return S_f + S_g
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        F_g = self.beta * gauge_force(cfg)
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
        S_g = -self.beta * np.sum(np.real(ensemble_plaqs(cfg)))
        sign, logdet = np.linalg.slogdet(Dmat)
        S_f = -self.Nf * logdet  # np.log(np.abs(np.linalg.det(Dmat)))
        return S_f + S_g
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        F_g = self.beta * gauge_force(cfg)
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

def make_topo_cfg(L, Q):
    V = L**2
    coeff = 2*np.pi*Q / V
    A = np.zeros((2,L,L), dtype=np.complex128)
    A[0] = -np.arange(L) * coeff
    A[1,:,-1] = np.arange(L) * L * coeff
    U = np.exp(1j*A)
    Q_check = np.sum(compute_topo(U))
    assert np.isclose(Q_check, Q)
    return U

def _test_topo_hop_inv():
    # dQ = +1 should be the inverse of dQ = -1 for detailed balance of topo hops
    dU1 = make_topo_cfg(8, 1)
    dUm1 = make_topo_cfg(8, -1)
    assert np.allclose(dU1 * dUm1, 1.0)
    print('[PASSED test_topo_hop_inv]')

if __name__ == '__main__': _test_topo_hop_inv()
