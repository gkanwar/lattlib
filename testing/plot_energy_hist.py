import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from schwinger.schwinger import PureGaugeAction

# This test qualitatively indicates whether there is anything seriously wrong with the sampled gauge field ensembles. For the sampled configurations ${U_i}$, compute:  exp(-S(U_i)) and Z = \sum \exp(-S(U_i))$  for S = S gauge. Plot the histograms of the negative of the log of the normalized distribution. The resulting histogram should look roughly normal; the exact location of the mean and the width is not important.

# NEEDs ens, beta, action function, nbins

action_tag = "u1_of_b2.50_k0.234_N100_skip10_therm40_16_24_gaugeBC01_match"
ens = np.load(f'./ens/_match/of_b2.50_k0.234/gaugeBC01/{action_tag}.npy')

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

ncfgplot=100
assert(ncfgplot <= ncfg)

weights = np.ones(ncfgplot)/ncfgplot

plt.figure(figsize = (10, 5))
plt.title(f'S for {action_tag}')
plt.xlabel('$S$')
plt.hist(np.real(S),
        weights=weights,
        density = True,
        bins = nbins,
        label='test')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
plt.legend()
plt.show()
