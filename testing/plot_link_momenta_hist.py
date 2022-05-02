import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from schwinger.schwinger import PureGaugeAction
from scipy.stats import chi2

# This test makes sure that random number generation for gauge-link momenta works correctly.
# The kinetic part of the Hamiltonian in HMC, K, is generated via 1/2 *P^2_l,
# with l summed over all Nd * latt_vol links in the configuration.
# Each momentum P_l is sampled from a standard/unit normal distribution N(0, 1 [1]).
# This is done at the beginning of each new MD trajectory to generate each new proposal.
# Therefore, 2K has to follow the chi-squared distribution with k=Nd*latt_vol degrees of freedom.
#
# [1] https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

# NEEDs moms, Nd, latt_vol, nbins

moms = np.load("u1_of_b2.00_k1.000_N100_skip10_therm20_16_16_gaugeBC11_testing_r.lmom.npy")
Nd = 2
latt_vol = 16**2
nbins = 10

ncfg = len(moms)

moms_pba = 2 * (np.real(moms))
weights = np.ones(len(moms_pba)) / len(moms_pba)

plt.figure(figsize = (10, 5))
plt.title("Distribution of link momenta")
plt.xlabel('$2 K = \\sum_i P^2_i$')
hist, bins, _ = plt.hist(moms_pba,
        density = True,
        bins = nbins,
        label='MCMC chain')
plt.plot(bins, chi2.pdf(bins, df=Nd*latt_vol))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
plt.legend()
plt.show()
