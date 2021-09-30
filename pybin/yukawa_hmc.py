import argparse
import matplotlib.pyplot as plt
import os
import tqdm.auto as tqdm
from scalar_yukawa.yukawa import *

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, required=True)
parser.add_argument('--Nd', type=int, required=True)
parser.add_argument('--m2', type=float, required=True, help='Scalar mass squared')
parser.add_argument('--lam', type=float, required=True, help='Scalar phi^4 coupling')
parser.add_argument('--g', type=float, required=True, help='Yukawa coupling')
parser.add_argument('--M', type=float, required=True, help='Fermion mass')
globals().update(vars(parser.parse_args()))

apbc = True
logdir = 'hmc_toy_logs'

### TODO! Edit me.


prefix = os.path.join(logdir, f'ens_M{M}_g{g}_m{"m" if m2 < 0 else ""}{abs(m2)}_lam{lam}_Npf{N_pf}_Nd{Nd}_L{L}_a{apbc:d}')
action = ToyAction(m2, lam, M, g, apbc)
    phi = 0.1*np.random.normal(size=(1,) + (L,)*Nd)+0.1
    varphi = 0.1*np.random.normal(size=(2,) + (L,)*Nd)
    x = np.concatenate((phi, varphi), axis=0)
    n_therm = 100
    tau = 1.0
    n_leap = 10
    tot_acc = 0
    ens = []
    varphi_ens = []
    Ss = []
    for i in tqdm.tqdm(range(-n_therm, 10000)):
        x, S, acc = hmc_update(x, action, tau, n_leap, verbose=True)
        tot_acc += acc
        if i >= 0 and (i+1) % 10 == 0:
            ens.append(np.copy(x[0]))
            varphi_ens.append(np.copy(x[1:]))
            Ss.append(S)
        if i >= 0 and i % 10 == 0:
            tqdm.tqdm.write(f'Acc = {tot_acc/(i+1+n_therm)}')
            tqdm.tqdm.write(f'phi bar = {np.mean(x[0])}')
            # tqdm.tqdm.write(f'logp = {-exact_action.compute_action(x)}')

    ens = np.stack(ens, axis=0)
    ens_fname = f'{prefix}.npy'
    np.save(ens_fname, ens)
    varphi_ens = np.stack(varphi_ens, axis=0)
    varphi_ens_fname = f'{prefix}.varphi.npy'
    np.save(varphi_ens_fname, varphi_ens)
    S_fname = f'{prefix}.S.npy'
    np.save(S_fname, np.array(Ss))
    assert ens[0].shape[-1] == L
    axes = tuple(range(1, len(ens.shape)))
    mag = np.mean(np.abs(ens), axis=axes)
    assert len(mag.shape) == 1
    mag_fname = f'{prefix}.mag.npy'
    np.save(mag_fname, mag)

    phi0 = np.mean(ens, axis=axes)
    np.save(f'{prefix}.phi0.npy', phi0)
    rms_phi = np.sqrt(np.mean(ens**2, axis=axes))
    rms_phi_fname = f'{prefix}.rms_phi.npy'
    np.save(rms_phi_fname, rms_phi)
    assert len(phi0.shape) == 1
    assert len(rms_phi.shape) == 1
    print('<phi> =', al.bootstrap(phi0, Nboot=100, f=al.rmean))
    print('<|phi|> =', al.bootstrap(np.abs(phi0), Nboot=100, f=al.rmean))
    print('RMS phi =', al.bootstrap(rms_phi, Nboot=100, f=al.rmean))
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot(phi0)
    ax[1].hist(phi0, bins=20)
    plt.show()

    ens = np.load(f'{prefix}.npy')

    all_C = []
    all_Cphi = []
    all_cond = []
    for phi in ens:
        D = make_D(phi, M=M, g=g, apbc=apbc)
        Di = np.linalg.inv(D.todense())
        all_cond.append(np.trace(Di, axis1=-1, axis2=-2))
        C = np.zeros(L)
        for i in range(L**Nd):
            mx = []
            j = i
            for mu in range(Nd):
                mx.insert(0, -(j%L))
                j //= L
            mx = tuple(mx)
            Px = np.asarray(Di)[:,i].reshape((L,)*Nd)
            Px = np.roll(Px, mx, axis=tuple(range(len(Px.shape))))
            axes = tuple(range(0,Nd-1))
            C += np.sum(Px**2, axis=axes) / L**Nd
        # C = np.array([np.mean([Di[i,i - i%L + (i+t)%L]**2 for i in range(L**Nd)]) for t in range(L)])
        Cphi = np.array([np.mean(phi * np.roll(phi, -t, axis=-1)) for t in range(L)])
        all_C.append(C)
        all_Cphi.append(Cphi)
    all_cond = np.array(all_cond)
    cond_fname = f'{prefix}.cond.npy'
    np.save(cond_fname, all_cond)
    print('<bar{chi} chi> =', al.bootstrap(all_cond, Nboot=100, f=al.rmean))
    all_C = al.bootstrap(np.array(all_C), Nboot=100, f=al.rmean)
    all_Cphi = al.bootstrap(np.array(all_Cphi), Nboot=100, f=al.rmean)
    m = np.log(np.abs(all_C[0][1]/all_C[0][L//4])) / (L//4 - 1)
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    al.add_errorbar(all_C, ax=ax[0], marker='o')
    ax[0].set_yscale('log')
    ax[1].plot(ens[-1])
    al.add_errorbar(all_Cphi, ax=ax[2], marker='o')
    ax[2].set_yscale('log')
    return m

Ms = [0.0]
ms = [compute_mass(M) for M in Ms]
# fig = plt.figure()
# plt.plot(Ms, ms, marker='o', linestyle='-')
plt.show()
