### Heatbath for pure gauge Schwinger, i.e. U(1) gauge theory in 2D

import numpy as np
import tqdm
from .schwinger import *

class SchwingerHBAction(PureGaugeAction):
    def heatbath_update(self, cfg):
        Nd, Lx, Lt = cfg.shape # specialized for 2D
        for x in range(Lx):
            for t in range(Lt):
                # U = cfg[0, x, t]
                staple_up_1 = cfg[1, (x+1)%Lx, t]
                staple_up_2 = np.conj(cfg[0, x, (t+1)%Lt])
                staple_up_3 = np.conj(cfg[1, x, t])
                staple_down_1 = np.conj(cfg[1, (x+1)%Lx, (t-1)%Lt])
                staple_down_2 = np.conj(cfg[0, x, (t-1)%Lt])
                staple_down_3 = cfg[1, x, (t-1)%Lt]
                A = (staple_up_1 * staple_up_2 * staple_up_3 +
                     staple_down_1 * staple_down_2 * staple_down_3)
                R = np.abs(A)
                phi = np.angle(A)
                beta_eff = R*self.beta
                cfg[0, x, t] = np.exp(1j * np.random.vonmises(-phi, beta_eff))
                # U = cfg[1, x, t]
                staple_left_1 = np.conj(cfg[0, (x-1)%Lx, (t+1)%Lt])
                staple_left_2 = np.conj(cfg[1, (x-1)%Lx, t])
                staple_left_3 = cfg[0, (x-1)%Lx, t]
                staple_right_1 = cfg[0, x, (t+1)%Lt]
                staple_right_2 = np.conj(cfg[1, (x+1)%Lx, t])
                staple_right_3 = np.conj(cfg[0, x, t])
                A = (staple_left_1 * staple_left_2 * staple_left_3 +
                     staple_right_1 * staple_right_2 * staple_right_3)
                R = np.abs(A)
                phi = np.angle(A)
                beta_eff = R*self.beta
                cfg[1, x, t] = np.exp(1j * np.random.vonmises(-phi, beta_eff))

if __name__ == "__main__":
    action = SchwingerHBAction(1.0)
    shape = (2,10,10)
    init_cfg_A = 0.4*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    plaqs = []
    for i in tqdm.tqdm(range(-10, 100000)):
        action.heatbath_update(cfg)
        if i >= 0 and i % 10 == 0:
            plaqs.append(np.mean(ensemble_plaqs(cfg)))
    fname = 'test.plaq.dat'
    np.array(plaqs).tofile(fname)
    print('Saved plaqs to {}'.format(fname))
