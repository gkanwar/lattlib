"""
XY model lib.
"""

import numpy as np

class ActionTerm(object):
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
    def action(self, phi, verbose=False):
        out = np.zeros(phi.shape, dtype=np.float64)
        for term in self.terms:
            act = term.action(phi)
            if verbose: print('Action from {} = {}'.format(term, np.sum(act)))
            out += act
        return out
    def force(self, phi):
        out = np.zeros(phi.shape, dtype=np.float64)
        for term in self.terms:
            F = term.force(phi)
            print('Force from {} = {:.8g}'.format(term, np.linalg.norm(F)))
            out += F
        return out
    def __str__(self):
        return str(self.terms)
    __repr__ = __str__

class XYCosTerm(ActionTerm):
    def __init__(self, beta):
        self.beta = beta
    def action(self, theta):
        S = np.zeros(theta.shape, dtype=np.float64)
        for mu in range(len(theta.shape)):
            fwd = np.roll(theta, -1, axis=mu)
            S += -self.beta * np.cos(fwd - theta)
        return S
    def force(self, theta):
        out = np.zeros(theta.shape, dtype=np.float64)
        for mu in range(len(theta.shape)):
            fwd = np.roll(theta, -1, axis=mu)
            bwd = np.roll(theta, 1, axis=mu)
            out += -self.beta * (
                np.sin(fwd - theta) - np.sin(theta - bwd))
        # F = -dS/dtheta
        return -out
    def __str__(self):
        return "XYCosTherm(beta={:.4f})".format(self.beta)
    __repr__ = __str__
