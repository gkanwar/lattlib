### Draw samples of SU(N) matrices from the Haar measure.

import numpy as np

def generate_uN_haar(Nc, *, size):
    shape = (size, Nc, Nc)
    Z = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    Q, R = np.linalg.qr(Z)
    Lam = np.exp(1j * np.angle(np.einsum('...ii->...i', R)))
    U = np.einsum('...ab,...b -> ...ab', Q, Lam)
    return U

def generate_suN_haar(Nc, *, size):
    U = generate_uN_haar(Nc, size=size)
    theta = np.angle(np.linalg.det(U))
    z = 2*np.pi*np.random.randint(Nc, size=size)
    Up = U * np.exp(1j * (z - theta) / Nc)[...,np.newaxis,np.newaxis]
    return Up
