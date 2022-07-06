import numpy as np

# Plot error traces
def add_errorbar(trace, *, ax, xs=None, off=0.0, flip=False, **kwargs):
    mean, err = trace
    if xs is None:
        xs = np.arange(len(mean), dtype=np.float64)
    else:
        xs = np.array(xs).astype(np.float64)
    xs += off
    if flip:
        return ax.errorbar(mean, xs, xerr=err, **kwargs)
    else:
        return ax.errorbar(xs, mean, yerr=err, **kwargs)
def add_errorbar_fill(trace, *, ax, xs=None, off=0.0, **kwargs):
    mean, err = trace
    if xs is None:
        xs = np.arange(len(mean), dtype=np.float64)
    else:
        xs = np.array(xs).astype(np.float64)
    xs += off
    kwargs_stripped = {}
    if 'color' in kwargs:
        kwargs_stripped['color'] = kwargs['color']
    ax.fill_between(xs, mean-err, mean+err, alpha=0.8, **kwargs_stripped)
    ax.plot(xs, mean, **kwargs)
# From SO: 37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# Standard bootstrapping fns
mean = lambda x: np.mean(x, axis=0)
rmean = lambda x: np.real(np.mean(x, axis=0))
imean = lambda x: np.imag(np.mean(x, axis=0))
amean = lambda x: np.abs(np.mean(x, axis=0))

def log_meff(x):
    corr = rmean(x)
    return np.log(corr[:-1] / corr[1:])
def acosh_meff(x):
    corr = rmean(x)
    return np.arccosh((corr[:-2] + corr[2:])/(2*corr[1:-1]))

def make_stn_f(*, N_inner_boot, f):
    def stn(x):
        mean, err = bootstrap(x, Nboot=N_inner_boot, f=f)
        stn = np.abs(mean) / np.abs(err)
        return stn
    return stn

# Bootstrapping framework
def bootstrap_gen(*samples, Nboot, seed=None):
    rng = np.random.default_rng(seed=seed)
    n = len(samples[0])
    for i in range(Nboot):
        inds = rng.integers(n, size=n)
        yield tuple(s[inds] for s in samples)

def bootstrap(*samples, Nboot, f, bias_correction=False, seed=None):
    boots = []
    for x in bootstrap_gen(*samples, Nboot=Nboot, seed=seed):
        boots.append(f(*x))
    boot_mean, boot_err = np.mean(boots, axis=0), np.std(boots, axis=0)
    if not bias_correction:
        return boot_mean, boot_err
    full_mean = f(*samples)
    corrected_mean = 2*full_mean - boot_mean
    return corrected_mean, boot_err

def covar_from_boots(boots):
    boots = np.array(boots)
    Nboot = boots.shape[0]
    means = np.mean(boots, axis=0, keepdims=True)
    deltas = boots - means
    return np.tensordot(deltas, deltas, axes=(0,0)) / (Nboot-1)

def shrink_covar(covar, *, lam):
    assert len(covar.shape) == 2 and covar.shape[0] == covar.shape[1]
    diag_covar = np.diag(covar) * np.identity(covar.shape[0])
    return (1-lam) * covar + lam * diag_covar

def bin_data(x, *, binsize, silent_trunc=True):
    x = np.array(x)
    if silent_trunc:
        x = x[:(x.shape[0] - x.shape[0]%binsize)]
    else:
        assert x.shape[0] % binsize == 0
    ts = np.arange(0, x.shape[0], binsize) # left endpoints of bins
    x = np.reshape(x, (-1, binsize) + x.shape[1:])
    return ts, np.mean(x, axis=1)

# Autocorrelations
def compute_autocorr(Os, *, tmax, vacsub=True):
    if vacsub:
        dOs = Os - np.mean(Os)
    else:
        dOs = Os
    Gamma = np.array([np.mean(dOs[t:] - dOs[:-t]) for t in range(1,tmax)])
    Gamma = np.insert(Gamma, 0, np.mean(dOs**2))
    rho = Gamma / Gamma[0]
    return rho
def compute_tint(Os, *, tmax, vacsub=True):
    rho = compute_autocorr(Os, tmax=tmax, vacsub=vacsub)
    tint = 0.5 + np.cumsum(rho[1:])
    return tint
def self_consistent_tint(tints, *, W=4):
    after_W_tint = tints < np.arange(len(tints)) / W
    if not np.any(after_W_tint):
        # print('WARNING: self-consistent tint condition never satisfied, returning last tint')
        return tints[-1]
    i = np.argmax(after_W_tint)
    return tints[i]
