from .gauge_theory import *

gamma = np.zeros((4,4,4), dtype = complex)

gamma[0,0,3] = 1j
gamma[0,1,2] = 1j
gamma[0,2,1] = -1j
gamma[0,3,0] = -1j

gamma[1,0,3] = -1
gamma[1,1,2] = 1
gamma[1,2,1] = 1
gamma[1,3,0] = -1

gamma[2,0,2] = 1j
gamma[2,1,3] = -1j
gamma[2,2,0] = -1j
gamma[2,3,1] = 1j

gamma[3,0,2] = 1
gamma[3,1,3] = 1
gamma[3,2,0] = 1
gamma[3,3,1] = 1

gamma5 = gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]

# transpose for efficient compute
gammaT = np.swapaxes(gamma, -1, -2)
Pplus_gammaT = np.identity(4) + gammaT
Pminus_gammaT = np.identity(4) - gammaT
gamma5T = np.swapaxes(gamma5, -1, -2)

def make_ferm_roll(bcs):
    Nd = len(bcs)
    def ferm_roll(psi, shift, axis):
        roll = np.roll(psi, shift, axis=axis)
        if bcs[axis] != 1:
            assert abs(shift) == 1
            if shift < 0:
                ind = tuple([slice(None)]*axis + [-1])
                roll[ind] *= bcs[axis]
            else:
                ind = tuple([slice(None)]*axis + [0])
                roll[ind] *= bcs[axis]
        return roll
    return ferm_roll

def make_dirac(kappa, U, bcs, sign):
    Nd = U.shape[0]
    Nc = U.shape[-1]
    assert len(bcs) == Nd
    f_roll = make_ferm_roll(bcs)
    Pfwd = Pplus_gammaT if sign == 1 else Pminus_gammaT
    Pbwd = Pminus_gammaT if sign == 1 else Pplus_gammaT
    def dirac(psi):
        Ns = psi.shape[-1]
        assert Ns == 4
        assert psi.shape[-2] == Nc
        out = np.copy(psi)
        for mu in range(Nd):
            out -= kappa * ((U[mu] @ f_roll(psi, -1, axis=mu)) @ Pfwd[mu])
            out -= kappa * ((f_roll(gauge_adj(U[mu]) @ psi, 1, axis=mu)) @ Pbwd[mu])
        return out
    return dirac

def CG(A, eps, max_iter, should_print=False):
    def solve(psi):
        psi_norm = np.linalg.norm(psi)
        start = time.time()
        x = np.zeros_like(psi)
        r = psi - A(x)
        p = np.copy(r)
        resid_sq = np.linalg.norm(r.flatten())**2
        for k in range(max_iter):
            old_resid_sq = resid_sq

            Ap = A(p)
            pAp = np.sum(np.real(np.dot(np.conj(p.flatten()), Ap.flatten())))
            alpha = resid_sq / pAp

            x = x + alpha*p
            r = r - alpha*Ap

            resid_sq = np.linalg.norm(r)**2
            if should_print:
                print('CG resid {:13.8e}'.format(np.sqrt(resid_sq)))

            if np.sqrt(resid_sq)/psi_norm < eps: break

            beta = resid_sq / old_resid_sq
            p = r + beta*p

        if should_print:
            print('CG TIME: {:.1f}s'.format(time.time() - start))
            print('CG SOLVE resid: {:13.8e}, iters: {:d}'.format(np.sqrt(resid_sq), k))

        return x
    return solve

# Solver for A^{dag} A x = A^{dag} psi
def CGNE(A, Adag, eps, max_iter, should_print=False):
    def AxA(x): return Adag(A(x))
    cg_solver = CG(AxA, eps, max_iter, should_print)
    def solve(psi):
        psi_prime = Adag(psi)
        return cg_solver(psi_prime)
    return solve

def multishift_CG(A, all_sigmas, eps, max_iter, should_print=False):
    assert(len(all_sigmas) >= 1)
    base_sigma = all_sigmas[0]
    sigmas = np.array(all_sigmas) - base_sigma
    def solve(psi):
        psi_norm = np.linalg.norm(psi)
        assert(psi_norm > 0)
        start = time.time()
        x = []
        p = []
        r = []
        for i in range(len(sigmas)):
            x.append(np.zeros_like(psi))
            p.append(np.copy(psi))
            r.append(np.copy(psi))
        zeta = np.ones(len(sigmas), dtype=np.float64)
        old_zeta = np.ones(len(sigmas), dtype=np.float64)
        beta = np.zeros(len(sigmas), dtype=np.float64)
        alpha = np.ones(len(sigmas), dtype=np.float64)
            
        resid_sq = np.linalg.norm(r[0])**2
        for k in range(max_iter):
            old_resid_sq = resid_sq

            Ap = A(p[0]) + base_sigma*p[0]
            pAp = np.sum(np.real(np.dot(np.conj(p[0].flatten()), Ap.flatten())))
            old_alpha = alpha[0]
            alpha[0] = resid_sq / pAp

            cur_zeta = np.copy(zeta)
            zeta[0] = 1.0
            for i in range(1,len(sigmas)):
                zeta[i] *= old_zeta[i] * old_alpha / (
                    old_alpha * old_zeta[i] * (1 + alpha[0]*sigmas[i])
                    + alpha[0]*beta[0]*(old_zeta[i] - zeta[i]))
                alpha[i] = alpha[0] * zeta[i] / cur_zeta[i]
            old_zeta = np.copy(cur_zeta)

            for i in range(len(sigmas)):
                if np.linalg.norm(r[i]) != 0:
                    x[i] += alpha[i]*p[i]
            r[0] -= alpha[0] * Ap
            for i in range(1,len(sigmas)):
                if np.linalg.norm(r[i]) != 0:
                    r[i] = zeta[i] * r[0]

            resid_sq = np.linalg.norm(r[0])**2
            all_resids = []
            for i in range(len(sigmas)):
                all_resids.append(np.linalg.norm(r[i]))
            max_resid = np.max(all_resids)
            if should_print:
                print('MULTISHIFT CG max resid {:13.8e}'.format(max_resid/psi_norm))

            if max_resid/psi_norm < eps: break

            beta[0] = resid_sq / old_resid_sq
            for i in range(1,len(sigmas)):
                beta[i] = beta[0] * zeta[i]**2 / old_zeta[i]**2
            for i in range(len(sigmas)):
                p[i] = r[i] + beta[i]*p[i]

        if should_print:
            print('TIME MULTISHIFT CG {:.1f}s'.format(time.time() - start))
            print('MULTISHIFT CG SOLVE max resid: {:13.8e}, iters: {:d}'.format(max_resid/psi_norm, k))

        return x
    return solve

def rational_op(alphas, betas, MxM, eps, max_iter, should_print):
    assert len(alphas) == len(betas) + 1
    solve = multishift_CG(MxM, betas, eps=eps, max_iter=max_iter, should_print=should_print)
    def apply_op(psi):
        xs = solve(psi)
        out = alphas[0] * psi
        for i,alpha in enumerate(alphas[1:]):
            out += xs[i] * alpha
        return out
    return apply_op
