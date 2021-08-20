"""
Library for lattice scalar field functionality.
"""

import itertools
import math
import numpy as np

### UTILITIES
def tup_add(t1, t2):
    assert len(t1) == len(t2)
    return tuple([t1[i] + t2[i] for i in range(len(t1))])
def tup_sub(t1, t2):
    assert len(t1) == len(t2)
    return tuple([t1[i] - t2[i] for i in range(len(t1))])

class ActionTerm(object):
    def local_action(self, x, phi):
        raise NotImplementedError
    def action(self, phi):
        raise NotImplementedError
    def force(self, phi):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
    def __repr__(self):
        raise NotImplementedError

class Action(object):
    def __init__(self, terms):
        self.terms = terms
    def local_action(self, x, phi):
        out = 0.0
        for term in self.terms:
            out += term.local_action(x, phi)
        return out
    def action(self, phi, verbose=False):
        out = np.zeros(phi.shape, dtype=phi.dtype)
        for term in self.terms:
            act = term.action(phi)
            if verbose: print('Action from {} = {}'.format(term, np.sum(act)))
            out += act
        return out
    def force(self, phi, verbose=False):
        out = np.zeros(phi.shape, dtype=phi.dtype)
        for term in self.terms:
            F = term.force(phi)
            if verbose: print('Force from {} = {:.8g}'.format(term, np.linalg.norm(F)))
            out += F
        return out
    def __str__(self):
        return str(self.terms)
    __repr__ = __str__

continuum_param = True

class ScalarKineticMassTerm(ActionTerm):
    def __init__(self, m2):
        self.m2 = m2
    def local_action(self, x, phi):
        dims = phi.shape
        Nd = len(dims)
        coords = [x]
        for mu in range(Nd):
            muhat = [0]*Nd
            muhat[mu] = 1
            coords.append(tup_add(x, muhat))
            coords.append(tup_sub(x, muhat))
        coords_arrs = tuple([[] for mu in range(Nd)])
        for tup in coords:
            for mu in range(Nd):
                coords_arrs[mu].append(tup[mu])
        lin_inds = np.ravel_multi_index(coords_arrs, dims, mode='wrap')
        SK = 0.0
        diag = phi[x]
        for i in range(Nd):
            fwd_coord = list(x)
            fwd_coord[i] = (fwd_coord[i] + 1) % dims[i]
            bwd_coord = list(x)
            bwd_coord[i] = (bwd_coord[i] - 1) % dims[i]
            fwd = phi[tuple(fwd_coord)]
            bwd = phi[tuple(bwd_coord)]
            SK += 2*np.conj(diag) * (diag - fwd - bwd) # factor of 2?
        SK = np.real(SK)
        SM = self.m2 * np.abs(diag)**2
        pre = 0.5 if continuum_param else 1.0
        return pre * (SK + SM)
    def action(self, phi):
        SK = np.zeros(phi.shape, dtype=np.float64)
        for mu in range(len(phi.shape)):
            fwd = np.roll(phi, -1, axis=mu)
            bwd = np.roll(phi, 1, axis=mu)
            SK += np.real(np.conj(phi) * (2*phi - fwd - bwd))
        SM = self.m2 * np.abs(phi)**2
        pre = 0.5 if continuum_param else 1.0
        return pre * (SK + SM)
    def force(self, phi):
        Nd = len(phi.shape)
        out = 2 * self.m2 * np.conj(phi)
        for mu in range(Nd):
            out += 2 * (
                2*np.conj(phi)
                - np.conj(np.roll(phi, 1, axis=mu))
                - np.conj(np.roll(phi, -1, axis=mu)))
        # F = -dS/dphi            
        pre = 0.5 if continuum_param else 1.0
        return -np.conj(out) * pre
    def __str__(self):
        return "KineticMassTerm(M2={:.4f})".format(self.m2)
    __repr__ = __str__
    
class ScalarPhi4Term(ActionTerm):
    def __init__(self, lam):
        self.lam = lam
    def local_action(self, x, phi):
        return self.lam * np.abs(phi[x])**4
    def action(self, phi):
        return self.lam * np.abs(phi)**4
    def force(self, phi):
        return -np.conj(4 * self.lam * np.conj(phi) * np.abs(phi)**2)
    def __str__(self):
        return "Phi4Term(lambda={:.8f})".format(self.lam)
    __repr__ = __str__

def wrap(p):
    return (p + math.pi) % (2*math.pi) - math.pi

def make_smear_corr(smear_x, smear_t):
    def smear_corr(corr):
        L = corr.shape
        Nd = len(L)
        corr_fft = np.fft.fftn(corr)
        ks = [ np.linspace(0, 2*math.pi, num=L_mu, endpoint=False)
               for L_mu in L ]
        weights = []
        for mu in range(Nd):
            assert(L[mu] % 2 == 0)
            ks[mu] = (ks[mu] + math.pi) % (2*math.pi) - math.pi
            radius = smear_t if mu == Nd-1 else smear_x
            weights.append(np.exp(-radius**2 * np.square(ks[mu]) / Nd))
        ein_inds = ",".join(map(chr, range(ord('a'), ord('a') + Nd)))
        weights = np.einsum(ein_inds, *weights) # big outer product
        assert(corr_fft.shape == weights.shape)
        corr_fft *= weights
        return np.fft.ifftn(corr_fft)
    return smear_corr

def all_corrs(phi, xspace, tspace):
    corrs = []
    coord_ranges = []
    for Lx in phi.shape[:-1]:
        assert(Lx % xspace == 0)
        coord_ranges.append(range(0, Lx, xspace))
    assert(phi.shape[-1] % tspace == 0)
    coord_ranges.append(range(0, phi.shape[-1], tspace))
    all_axes = tuple(range(len(phi.shape)))
    for src in itertools.product(*coord_ranges):
        shift = tuple(-np.array(src))
        corr = np.conj(phi[src]) * np.roll(phi, shift, axis=all_axes)
        corrs.append(corr)
    return np.array(corrs)
