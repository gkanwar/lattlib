### For SU(N=2) this is pascal's triangle, for N>2 it's a higher-dim
### generalization. We should accumulate a factor of (1/2) at each stage.

import scipy as sp
import scipy.special

def get_fund_dirs(N):
    out = [(1,)+(0,)*(N-2), (0,)*(N-2)+(-1,)]
    for i in range(1,N-1):
        d = [0]*(N-1)
        d[i-1] = -1
        d[i] = +1
        out.append(tuple(d))
    return out

def get_antifund_dirs(N):
    return [tuple(-x for x in d) for d in get_fund_dirs(N)]

def tup_add(t1, t2):
    return tuple(x1+x2 for x1,x2 in zip(t1, t2))

def update_weights(weights, dirs, coeff):
    new_weights = {}
    for pt in weights:
        val = weights[pt]
        for d in dirs:
            new_pt = tup_add(pt, d)
            if any(x < 0 for x in new_pt): continue
            new_weights[new_pt] = new_weights.get(new_pt, 0.0) + coeff * val
    return new_weights

def compute_raw(beta, N, start, *, k_max):
    origin = (0,)*(N-1)
    weights = {
        start: 1
    }
    all_dirs = get_fund_dirs(N) + get_antifund_dirs(N)
    # coeff = 1.0
    total = 0.0
    est_err = 0.0
    for k in range(k_max):
        total += weights.get(origin, 0.0)
        if k == k_max-1:
            est_err += weights.get(origin, 0.0)
        # NOTE: converges very quickly
        # print(total)
        # coeff *= (beta/(2*N)) / (k+1)
        weights = update_weights(weights, all_dirs, (beta/(2*N)) / (k+1))
    return total, est_err

def compute_z(beta, N, *, k_max=30):
    return compute_raw(beta, N, start=(0,)*(N-1), k_max=k_max)
def compute_w(beta, N, *, k_max=30):
    start = (1,) + (0,)*(N-2)
    numer, numer_err = compute_raw(beta, N, start=start, k_max=k_max)
    denom, denom_err = compute_z(beta, N, k_max=k_max)
    out = numer / (N * denom)
    return out, out*np.sqrt((numer_err/numer)**2 + (denom_err/denom)**2)

# NOTE: SU(2) results seem to match
# print('z', compute_z(beta=2.0, N=2))
# w = compute_w(beta=2.0, N=2)
# print('w', w)
# print('w true', sp.special.iv(2, 2.0)/sp.special.iv(1, 2.0))
# print('z', compute_z(beta=3.0, N=3))
# print('w', compute_w(beta=3.0, N=3))
