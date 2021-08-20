"""
Test scalar field bits. Run using `python -m scalar_field.tests` from the root dir.
"""

from .scalar_field import *
from .scalar_field_hmc import *

import matplotlib.pyplot as plt
import numpy as np

def test_action_term_force(term):
    print('Testing force for {}'.format(term))
    np.random.seed(1234)
    L = [4,4,4]
    phi = np.random.normal(size=L) + 1j * np.random.normal(size=L)
    old_S = term.action(phi)
    F = term.force(phi).flatten()
    d = 0.00000001
    dphi = d * (np.random.normal(size=L) + 1j * np.random.normal(size=L))
    dS_thy = (
        -np.dot(np.real(F), np.real(dphi).flatten())
        -np.dot(np.imag(F), np.imag(dphi).flatten()))
    dS_emp = np.sum(term.action(phi+dphi) - old_S)
    print('dS (thy.) = {}'.format(dS_thy))
    print('dS (emp.) = {}'.format(dS_emp))
    ratio = dS_thy / dS_emp
    print('ratio = {}'.format(ratio))
    assert(np.isclose(ratio, 1.0))

def test_leapfrog(term):
    print('Testing leapfrog with term {}'.format(term))
    np.random.seed(1234)
    L = [4,4,4]
    phi = np.random.normal(size=L) + 1j * np.random.normal(size=L)
    pi = sample_pi(phi.shape, dtype=np.complex128)
    old_S = np.sum(term.action(phi))
    old_K = np.sum(np.abs(pi)**2) / 2
    old_H = old_S + old_K

    leapfrog_update(phi, pi, term, 0.01, 10)

    new_S = np.sum(term.action(phi))
    new_K = np.sum(np.abs(pi)**2) / 2
    new_H = new_S + new_K
    delta_H = new_H - old_H
    print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
    print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))
    print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))
    assert(np.isclose(new_S-old_S, old_K-new_K, rtol=1e-2))

def test_inf_smear():
    print('Testing inf smearing gives uniform')
    L = [4,4,4]
    phi = np.random.normal(size=L) + 1j * np.random.normal(size=L)
    smear_inf = make_smear_corr(10000000.0, 10000000.0)
    phi = smear_inf(phi)
    for mu in range(len(L)):
        assert(np.allclose(np.roll(phi, -1, axis=mu), phi))
    
if __name__ == "__main__":
    print('test kinetic mass term force')
    test_action_term_force(ScalarKineticMassTerm(0.1))
    print('OK\n')
    print('test phi4 term force')
    test_action_term_force(ScalarPhi4Term(1.0))
    print('OK\n')
    print('test kinetic mass term leapfrog')
    test_leapfrog(ScalarKineticMassTerm(0.1))
    print('OK\n')
    print('test inf smear')
    test_inf_smear()
    print('OK\n')
