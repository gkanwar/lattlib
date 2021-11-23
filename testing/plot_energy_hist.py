import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from schwinger.schwinger import PureGaugeAction

# NEEDs ens, beta, action function, nbins

ens = np.load("u1_exwils_Nf2_b2.00_k1.000_N1000_skip10_therm20_16_16_testing.npy")
beta = 2.0
nbins = 10

ncfg = len(ens)
latt = ens.shape[-2:]
S = np.zeros(shape=ncfg, dtype=np.complex128)
p = np.zeros(shape=ncfg, dtype=np.complex128)
pba = np.zeros(shape=((2,) + (ncfg, )))
z = np.sum(pba, axis=1)

for i in range(ncfg):
    S[i] += PureGaugeAction(beta).compute_action(ens[i])

pba = np.exp(-np.real(S))
z = np.sum(pba)

weights = np.ones(ncfg)/ncfg

plt.figure(figsize = (10, 5))
plt.title("Action distribution {X}x{T}".format(X=latt[0], T=latt[1]))
plt.xlabel('$S$')
plt.hist(-np.log(pba),
        weights=weights,
        density = True,
        bins = nbins,
        label='test')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
plt.legend()
plt.show()
