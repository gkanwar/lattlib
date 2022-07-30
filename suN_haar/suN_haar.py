### Draw samples of SU(N) matrices from the Haar measure.

import numpy as np

def generate_suN_haar(Nc, *, size):
    shape = (size, Nc, Nc)
    Z = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    Q, R = np.linalg.qr(Z)
    Lam = np.exp(1j * np.angle(np.einsum('...ii->...i', R)))
    U = np.einsum('...ab,...b -> ...ab', Q, Lam)
    return U
