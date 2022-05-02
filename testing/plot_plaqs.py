import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from schwinger.schwinger import ensemble_plaqs

# NEEDs ens, plaqs, gauge_bc

ens = np.load("u1_tf_b2.00_k1.000_N1000_skip10_therm20_16_16_gaugeBC01_testing.npy")
plaqs = np.load("u1_tf_b2.00_k1.000_N1000_skip10_therm20_16_16_gaugeBC01_testing.plaq.npy")
gauge_bc = (0, 1)

ncfg = len(plaqs)
plaq_avg = np.zeros(shape=(ncfg,), dtype=np.complex128)

for i in range(ncfg):
    plaq_avg[i] = np.mean(ensemble_plaqs((ens[:ncfg])[i], gauge_bc))

plt.ylabel('Re[Plaquette Average]')
plt.xlabel('MCMC chain')
plt.grid(True)

ncfgplot = 100
assert(ncfgplot <= ncfg)

plt.plot(np.real(plaq_avg[:ncfgplot]), label = "$Re(\\overline{U_{\\mu\\nu}})$ (recomputed)", linestyle = '-', alpha = 1.0)
plt.plot(np.real(plaqs[:ncfgplot]), label = "$Re(\\overline{U_{\\mu\\nu}})$ (recorded)", linestyle = ':', alpha = 0.8)
plt.legend()

plt.show()
