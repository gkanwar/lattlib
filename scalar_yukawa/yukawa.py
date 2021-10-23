import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import tqdm.auto as tqdm

def make_D_1D(phi, *, M, g):
    assert len(phi.shape) == 1, '1D only'
    L = phi.shape[0]
    row, col, data = [], [], []
    for n in range(L):
        row.append(n)
        col.append((n+1)%L)
        sign = -1 if (n+1) >= L else 1
        data.append(-sign/2)
        row.append(n)
        col.append((n-1)%L)
        sign = -1 if (n-1) < 0 else 1
        data.append(sign/2)
        row.append(n)
        col.append(n)
        data.append(M + g*phi[n])
    D = sp.sparse.csr_matrix((data, (row,col)), shape=(L,L))
    return D

def make_D(phi, *, M, g, apbc):
    # assert len(phi.shape) == 1, '1D only'
    L = phi.shape[0]
    Nd = len(phi.shape)
    for mu, L_mu in enumerate(phi.shape):
        assert L == L_mu, 'lattice must be square'
    def to_ind(x):
        i = 0
        for mu, x_mu in enumerate(x):
            i *= L
            i += x_mu
        return i
    def eta_mu(x, mu):
        return (-1)**np.sum(x[:mu])
    row, col, data = [], [], []
    old_i = -1
    with np.nditer(phi, ['multi_index'], ['readonly']) as it:
        while not it.finished:
            x = it.multi_index
            i = to_ind(x)
            # TEST
            assert i == old_i+1
            old_i = i
            row.append(i)
            col.append(i)
            data.append(M + g*it[0])
            for mu in range(Nd):
                eta_mu_i = eta_mu(x, mu)
                x_fwd = list(x)
                x_fwd[mu] = (x_fwd[mu]+1) % L
                x_fwd = tuple(x_fwd)
                i_fwd = to_ind(x_fwd)
                row.append(i)
                col.append(i_fwd)
                data.append((-1/2) * eta_mu_i)
                if apbc and mu == Nd-1 and x_fwd[mu] == 0:
                    data[-1] *= -1
                x_bwd = list(x)
                x_bwd[mu] = (x_bwd[mu]-1) % L
                x_bwd = tuple(x_bwd)
                i_bwd = to_ind(x_bwd)
                row.append(i)
                col.append(i_bwd)
                data.append((1/2) * eta_mu_i)
                if apbc and mu == Nd-1 and x_bwd[mu] == L-1:
                    data[-1] *= -1
            it.iternext()
    D = sp.sparse.csr_matrix((data, (row,col)), shape=(L**Nd,L**Nd))
    return D

def _test_D():
    L = 4
    phi = np.random.normal(size=L)
    D = make_D(phi, M=0, g=0.1, apbc=True)
    D2 = make_D_1D(phi, M=0, g=0.1)
    assert np.allclose(D.todense(), D2.todense())
    print(np.linalg.det(D.todense()))
    phi = np.random.normal(size=(L,L))
    D = make_D(phi, M=0, g=0.1, apbc=True)
    print(np.linalg.det(D.todense()))
    print('[PASSED test_D]')
if __name__ == '__main__': _test_D()

def _make_sample_D():
    L = 3
    phi = np.zeros(shape=(L,L))
    print('PBC time')
    D = make_D(phi, M=0, g=0.1, apbc=False)
    print(D.toarray())
    print(D.toarray() + D.T.toarray())
    print('APBC time')
    D = make_D(phi, M=0, g=0.1, apbc=True)
    print(D.toarray())
    print(D.toarray() + D.T.toarray())
if __name__ == '__main__': _make_sample_D()
    

# sample varphi_{1,2} according to exp(-\varphi_i^T (D D^T)^{-1} \varphi_i)
N_pf = 2
def sample_pf(D, shape):
    assert len(D.shape) == 2, 'D must be a VxV matrix'
    V = D.shape[0]
    assert D.shape[1] == V
    assert np.prod(shape) == V
    eta = np.random.normal(size=(V,N_pf))
    varphi = D @ eta / np.sqrt(2)
    return varphi.reshape(shape + (N_pf,))

def DDTi(D, varphi, *, reg_eps=0, use_cg=False):
    DDT = D @ D.T + reg_eps * sp.sparse.identity(D.shape[0])
    DDTi_varphi = []
    assert varphi.shape[-1] == N_pf
    for i in range(N_pf):
        varphi_i = varphi[...,i]
        if use_cg:
            DDTi_varphi_i, info = sp.sparse.linalg.cg(DDT, varphi_i, tol=1e-8, atol=1e-8)
            # assert info == 0, info
            if info != 0:
                print(f'WARNING: tolerance not reached {info}')
        else:
            DDTi_varphi_i = np.linalg.inv(DDT.todense()) @ varphi_i
            DDTi_varphi_i = np.asarray(DDTi_varphi_i)[0]
        DDTi_varphi.append(DDTi_varphi_i)
    DDTi_varphi = np.transpose(DDTi_varphi)
    return DDTi_varphi

def S_pf(D, varphi, *, reg_eps=0):
    assert varphi.shape[-1] == N_pf
    varphi = np.reshape(varphi, (-1, varphi.shape[-1]))
    DDTi_varphi = DDTi(D, varphi, reg_eps=reg_eps)
    return np.sum(varphi * DDTi_varphi)

def d_S_pf(D, varphi, *, g):
    assert varphi.shape[-1] == N_pf
    shape = varphi.shape[:-1]
    varphi = np.reshape(varphi, (-1, varphi.shape[-1]))
    DDTi_varphi = DDTi(D, varphi)
    return -2*g * np.sum(DDTi_varphi * (D.T @ DDTi_varphi), axis=-1).reshape(shape)

def _test_d_S_pf():
    L = 3
    np.random.seed(1234)
    phi = np.random.normal(size=(L,L))
    print('phi', phi)
    g = 0.1
    M = 0.01
    D = make_D(phi, M=M, g=g, apbc=True)
    with np.printoptions(linewidth=1000):
        print('D', D.todense())
    varphi = sample_pf(D, phi.shape)
    print('varphi', varphi)
    old_S_pf = S_pf(D, varphi)
    deriv = d_S_pf(D, varphi, g=g)
    eps = 1e-6
    d_phi = eps * np.random.normal(size=phi.shape)
    phi += d_phi
    new_D = make_D(phi, M=M, g=g, apbc=True)
    new_S_pf = S_pf(new_D, varphi)
    dS_emp = (new_S_pf - old_S_pf)/eps
    dS_thy = np.sum(deriv * d_phi/eps)
    print(f'dS_emp = {dS_emp}')
    print(f'dS_thy = {dS_thy}')
    ratio = dS_emp / dS_thy
    print(f'ratio = {ratio}')
    assert np.isclose(ratio, 1.0, rtol=1e-4)
    print('[PASSED test_d_S_pf]')
if __name__ == '__main__': _test_d_S_pf()

def S_g(phi, *, m2, lam):
    Nd = len(phi.shape)
    S_K = sum([
        np.sum(-2 * phi * np.roll(phi, -1, axis=mu))
        for mu in range(len(phi.shape))])
    S_V = np.sum( (2*Nd+m2)*phi**2 + lam*phi**4 )
    return S_K + S_V
def d_S_g(phi, *, m2, lam):
    Nd = len(phi.shape)
    d_K = -2 * sum([
        np.roll(phi, -1, axis=mu) + np.roll(phi, 1, axis=mu)
        for mu in range(len(phi.shape))])
    d_V = 2*(2*Nd+m2)*phi + 4*lam*phi**3
    return d_K + d_V

def _test_d_S_g():
    L = 8
    np.random.seed(1234)
    phi = np.random.normal(size=(L,L))
    # kappa = 1.0
    m2 = 1.0
    lam = 4.0
    old_S_g = S_g(phi, m2=m2, lam=lam)
    deriv = d_S_g(phi, m2=m2, lam=lam)
    eps = 1e-8
    d_phi = eps * np.random.normal(size=phi.shape)
    phi += d_phi
    new_S_g = S_g(phi, m2=m2, lam=lam)
    dS_emp = (new_S_g - old_S_g)/eps
    dS_thy = np.sum(deriv * d_phi/eps)
    print(f'dS_emp = {dS_emp}')
    print(f'dS_thy = {dS_thy}')
    ratio = dS_emp / dS_thy
    print(f'ratio = {ratio}')
    assert np.isclose(ratio, 1.0)
    print('[PASSED test_d_S_g]')
if __name__ == '__main__': _test_d_S_g()

def _test_stochastic_convergence():
    L = 4
    np.random.seed(1234)
    phi = np.random.normal(size=L)
    D = make_D(phi, M=10, g=0.1, apbc=True)
    true_logZ = 2*np.log(np.abs(np.linalg.det(D.todense())))
    Zinvs = []
    for i in tqdm.tqdm(range(100*1024)):
        varphi = sample_pf(D, phi.shape)
        S_pf_i = S_pf(D, varphi)
        # tqdm.tqdm.write(f'S_pf_i = {S_pf_i}')
        # tqdm.tqdm.write(f'S_g = {np.sum(varphi**2)/2 + L*np.log(2*np.pi)}')
        Zinvs.append(
            np.exp(S_pf_i - np.sum(varphi**2)/2 - L*np.log(2*np.pi)) )
    Zinvs = np.array(Zinvs)
    print(Zinvs)
    print(al.bootstrap(Zinvs[:16], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:32], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:64], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:128], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:256], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:512], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:1024], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:5*1024], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:10*1024], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(al.bootstrap(Zinvs[:100*1024], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    # print(al.bootstrap(Zinvs[:1024*1024], Nboot=100, f=lambda Zis: np.log(1/np.mean(Zis))))
    print(f'vs true logZ = {true_logZ}')
# if __name__ == '__main__': _test_stochastic_convergence()


class YukawaPhi4Action:
    def __init__(self, m2, lam, M, g, apbc):
        self.m2 = m2
        self.lam = lam
        self.M = M
        self.g = g
        self.apbc = apbc
    def init_traj(self, phi):
        D = make_D(phi, M=self.M, g=self.g, apbc=self.apbc)
        self.varphi = sample_pf(D, phi.shape)
        return self.compute_action(phi)
    def compute_action(self, phi):
        D = make_D(phi, M=self.M, g=self.g, apbc=self.apbc)
        return S_g(phi, m2=self.m2, lam=self.lam) + S_pf(D, self.varphi)
    def force(self, phi, verbose=False):
        D = make_D(phi, M=self.M, g=self.g, apbc=self.apbc)
        F_g = -d_S_g(phi, m2=self.m2, lam=self.lam)
        F_pf = -d_S_pf(D, self.varphi, g=self.g)
        if verbose: print(f'|F_g| = {np.linalg.norm(F_g)}, |F_pf| = {np.linalg.norm(F_pf)}')
        return F_g + F_pf


# Mutates phi, pi according to leapfrog update
def leapfrog_update(phi, pi, action, tau, n_leap, verbose=False):
    dt = tau / n_leap
    phi += (dt/2)*pi
    for i in range(n_leap-1):
        pi += dt * action.force(phi, verbose=(verbose and i==0))
        phi += dt*pi
    pi += dt * action.force(phi)
    phi += (dt/2)*pi

def hmc_update(x, action, tau, n_leap, *, verbose):
    phi = x[0]
    old_phi = np.copy(phi)
    old_S = action.init_traj(old_phi)
    varphi = action.varphi
    old_pi = np.random.normal(size=phi.shape)
    old_K = np.sum(old_pi**2)/2
    old_H = old_S + old_K

    phi = np.copy(phi)
    new_pi = np.copy(old_pi)
    leapfrog_update(phi, new_pi, action, tau, n_leap, verbose=verbose)

    new_S = action.compute_action(phi)
    new_K = np.sum(new_pi**2)/2
    new_H = new_S + new_K

    delta_H = new_H - old_H
    if False:
        print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))
        print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
        print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))
        
    # metropolis step
    acc = 0
    if np.random.random() < np.exp(-delta_H):
        acc = 1
        S = new_S
    else:
        phi = old_phi
        S = old_S
    varphi = np.transpose(varphi, (len(varphi.shape)-1,) + tuple(range(len(varphi.shape)-1)))
    x = np.concatenate((phi[np.newaxis], varphi), axis=0)
    return x, S, acc


# We can compute det exactly/efficiently for 1D
def log_det_D2(phi, *, M, g):
    L = len(phi)
    # winding cases
    sign = -1 # APBCs
    det_D = sign * (
        (-1)**(L-1) * (1/2)**L +
        (-1)**(L-1) * (-1/2)**L )
    dimer = np.zeros_like(phi)
    no_dimer = np.zeros_like(phi)
    # no_dimer case for last link
    dimer[0] = 1/4
    no_dimer[0] = M + g*phi[0]
    for i in range(1, phi.shape[0]):
        dimer[i] = (1/4)*no_dimer[i-1]
        no_dimer[i] = (M + g*phi[i])*no_dimer[i-1] + dimer[i-1]
    det_D += no_dimer[-1] # no_dimer case
    # dimer case for last link
    dimer[0] = 0
    no_dimer[0] = 1
    for i in range(1, phi.shape[0]):
        dimer[i] = (1/4)*no_dimer[i-1]
        no_dimer[i] = (M + g*phi[i])*no_dimer[i-1] + dimer[i-1]
    det_D += dimer[-1] # dimer case
    return 2*np.log(np.abs(det_D))

def _test_log_det_D2():
    L = 8
    np.random.seed(1234)
    phi = np.random.normal(size=L)
    M = 0.5
    g = 0.1
    D = make_D(phi, M=M, g=g, apbc=True)
    true_log_det_D2 = 2*np.log(np.abs(np.linalg.det(D.todense())))
    thy_log_det_D2 = log_det_D2(phi, M=M, g=g)
    print(true_log_det_D2)
    print(thy_log_det_D2)
    assert np.isclose(true_log_det_D2, thy_log_det_D2)
    print('[PASSED test_log_det_D2]')
if __name__ == '__main__': _test_log_det_D2()


class YukawaPhi4ExactAction:
    def __init__(self, kappa, M, g):
        self.kappa = kappa
        self.M = M
        self.g = g
    def compute_action(self, phi):
        return S_g(phi, kappa=self.kappa) - log_det_D2(phi, M=self.M, g=self.g)
