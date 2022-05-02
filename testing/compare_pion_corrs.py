# loads files to compare correlators & effective masses for pions between OBC & PBC

import matplotlib.pyplot as plt
import numpy as np
from pylib.analysis import mean, rmean, log_meff, acosh_meff, asinh_meff, bootstrap

#def asinh_meff(x):
#    corr = rmean(x)
#    return np.arcsinh((corr[:-2] - corr[2:])/(2*corr[1:-1]))

# needs
action_label = 'of' # 'of'
sim_label = 'testing'
Ncfg = 100
Lx = 4
Lt = 6
ntherm = 40

path = f'{sim_label}/{action_label}_b2.00_k1.000/gaugeBC01/'.format(args.tag, action.make_tag())

# average correlators over all configs
bcs = ["obc", "pbc"]
constructions = ['g2g5-g2g5']#, 'g2g5-g2g5', 'g2g5-g5', 'g5-g2g5']
meffs = {} # asinh_meff
meffs['g5-g5'] = (log_meff, 'log') # acosh_meff
meffs['g2g5-g2g5'] = (log_meff, 'log') # acosh_meff
meffs['g5-g2g5'] = (asinh_meff, 'asinh')
meffs['g2g5-g5'] = (asinh_meff, 'asinh')
corrs = {}
for construction in constructions:
    corrs[construction] = []
    corrs[construction].append(np.load(f'u1_{action_label}_b2.00_k1.000_N{Ncfg}_skip10_therm{ntherm}_{Lx}_{Lt}_gaugeBC01_{sim_label}.meson_Ct_{construction}.npy'))
    corrs[construction].append(np.load(f'u1_{action_label}_b2.00_k1.000_N{Ncfg}_skip10_therm{ntherm}_{Lx}_{Lt}_gaugeBC11_{sim_label}.meson_Ct_{construction}.npy'))

corrs_mean = {}
corrs_err = {}
logs_meff_mean = {}
logs_meff_err = {}
for construction in constructions:
    corrs_mean[construction] = []
    corrs_err[construction] = []
    logs_meff_mean[construction] = []
    logs_meff_err[construction] = []

for i in range(len(bcs)):
    for construction in constructions:
        corr_mean, corr_err = bootstrap(corrs[construction][i], Nboot = Ncfg, f = lambda x : mean(x))
        corrs_mean[construction].append(corr_mean)
        corrs_err[construction].append(corr_err)
        log_meff_mean, log_meff_err = bootstrap(corrs[construction][i],
                                                    Nboot = Ncfg,
                                                    f = meffs[construction][0])
        logs_meff_mean[construction].append(log_meff_mean)
        logs_meff_err[construction].append(log_meff_err)

print(corrs_mean[constructions[0]][0])
print(corrs_mean[constructions[0]][1])
print(logs_meff_mean[constructions[0]][0])
print(logs_meff_mean[constructions[0]][1])

tsteps = len(corrs_mean[constructions[0]][0])
# plot correlator
plt.title(f'$C(t)$ between OBC & PBC, Lx x Lt = {Lx} x {Lt}')
plt.xlabel("$t$")
plt.ylabel("$C(t)$")
for i in range(len(bcs)):
    bc_str = bcs[i]
    for construction in constructions:
        plt.errorbar(np.arange(tsteps)[2:-1], np.real(corrs_mean[construction][i])[2:-1],
                     yerr=corrs_err[construction][i][2:-1],
            linestyle='None', marker='o',
            label=bc_str + ', $\\Gamma_1$-$\\Gamma_2$ =' + construction)
plt.yscale('linear')
plt.legend()
plt.show()
# plot effective mass
plt.title("$M_{\\mathrm{eff}}(t)$ between OBC & PBC, Lx x Lt = %d x %d" % (Lx, Lt))
plt.xlabel("$t$")
plt.ylabel("$M_{\\mathrm{eff}}(t)$")
for i in range(len(bcs)):
    bc_str = bcs[i]
    for construction in constructions:
        plt.errorbar(
            np.arange(0.5, tsteps-1)[1:],
            np.real(logs_meff_mean[construction][i])[1:],
            yerr=logs_meff_err[construction][i][1:],
            linestyle='None', marker='o',
            label=bc_str + ', $\\Gamma_1-\\Gamma_2$ = ' + construction + ', $M_{\\mathrm{eff}}$=' + meffs[construction][1])
plt.yscale('linear')
plt.legend()
plt.show()
