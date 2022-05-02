import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# This test qualitatively indicates whether there is anything seriously wrong with MC sampling. In particular, take a look at:
# - How the acceptance rate behaves during thermalization?
# - How does it behave during sampling?
# - What does this say about
#   - The required number of thermalization steps (burn-in),
#   - The required frequency of saving configurations (to prevent autocorrelation), and
#   - The step in the MD integrator during both the thermalization stage and the sampling stage?

# NEED acc_rates, npoints

acc_rates = np.load("u1_tf_b2.00_k1.000_N1000_skip10_therm20_16_16_gaugeBC01_testing.acc_rates.npy")

ncfg = len(acc_rates)
npoints = 100                                           # number of points along the chain to plot
assert(npoints <= ncfg)

plt.figure(figsize = (10, 5))
plt.title("Acceptance rate")

plt.ylabel('Acceptance rate')
plt.xlabel('MCMC chain')
plt.grid(True)

plt.plot(acc_rates[:npoints])

plt.show()
