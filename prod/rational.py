### Zolotarev rational approximation

from scipy.special import ellipk, ellipj
import numpy as np
import math

debug_default = False
verbose_default = True
test_default=True

# given a0, a1, a2, ... a_(2n), returns R(y)  = a_0 Prod_(k=1)^n (y+a_(2k-1))/(y+a_2k)
def rational_func(coefficients):
    a = np.array(coefficients)
    return lambda y : a[0] * (np.prod(y + a[1::2]))/(np.prod(y + a[2::2]))

# Returns Zolotarev's optimal coefficients (a_0, a_1, ..., a_2n) and error delta
# for a rational approximation R(y) of y^(-1/2)
def zolotarev(smallest_eigenvalue, polynomial_degree, debug=debug_default):
    eps = smallest_eigenvalue
    n = int(polynomial_degree)
    assert(n > 0)
    k = math.sqrt(1-eps)
    #k = 1-eps
    assert(k > 0 and k < 1)
    # K(k) = int_0^pi/2 d theta / (sqrt(1 - k^2 * sin^2 theta))
    # K(m) = int_0^pi/2 d theta / (sqrt(1 - m   * sin^2 theta))
    m = k**2                                    # ellipk, ellipj take m as argument, not k
    v = ellipk(m)/(2.*n+1.)
    (sn, _, _, _) = ellipj(                     # returns sn, cn, dn, ph
            v * np.arange(1, 2*n+1, 1),         #
            m * np.ones(2*n)
            )
    c = np.square(sn)                           # c_r = sn^2(rv, k), r = 1, 2, ..., 2n
    d = k**(2*n+1)*np.square(np.prod(c[0::2]))  # d = k^(2n+1)*(c_1 * c_3 * ... * c_(2n-1))
    # R(y) approximates 1/sqrt(y) with error
    # For the optimal approximation, delta = d^2/(1+sqrt(1-d^2))^2
    delta = (d**2)/((1+math.sqrt(1-d**2))**2)
    # Zolotarev coefficients for R(y)  = a_0 Prod_(k=1)^n (y+a_(2k-1))/(y+a_2k)
    # a_0 = 2 * sqrt(delta) / d * (c_1 * c_3 * ... * c_(2n-1) / *(c_2 * c_4 * ... c_(2n))
    # a_r = (1-c_r)/c_r, r = 1, 2, ... 2n
    a0 = 2 * math.sqrt(delta)/d * (
            (np.prod(c[0::2]))/(np.prod(c[1::2]))
            )
    ar = (1-c)/c                                # r = 1, 2, ... 2n
    assert(np.all(ar > 0))                      # a_1 > a_2 > ... a_(2n)
    assert(np.all(ar[:-1] > ar[1:]))
    if debug:
        print("a0", a0)
        print("ar", ar)
        print("delta", delta)
    a = np.concatenate(([a0], ar))
    return a, delta

# Returns estimate of error for a rational approximation of y*(-alpha), R(y).
# Error is defined as delta = max_(y_min <= y <= y_max)|1- y^alpha R(y)|
# Estimate of error is made with npoints linearly spaced points on range (y_min, y_max),
def sample_delta(R, alpha, y_min, y_max=1, npoints=100):
    sampling_space = np.linspace(y_min, y_max, npoints)
    approximations = np.array(list(map(R, sampling_space)))
    delta = np.max(np.abs(1 - np.power(sampling_space, alpha) * approximations))
    return delta

# given coefficients of degree-n polynomials P0, Q, returns the residues
# of the partial fraction decomposition of P0/Q
def residues(numerator_coeffs, denominator_coeffs):
    a = np.array(numerator_coeffs)      # P0 = prod_(l=1)^n (y + a_l)
    b = np.array(denominator_coeffs)    # Q  = prod_(l=1)^n (y + b_l)
    assert(len(a)==len(b))
    n = len(a)                          # degree of polynomials
    # r_k = (prod_(l=1)^n -b_k + a_l) / (prod_(l neq k)^n -b_k + b_l)
    lambda k : (np.prod(-b[k-1]+a))/(np.prod(-b[k-1]+np.delete(b, k-1)))
    r = np.array(list(map(
        lambda k : (np.prod(-b[k]+a))/(np.prod(-b[k]+np.delete(b, k))),
        np.arange(0, n)
    )))
    return r

# given r, b, a0 returns R(y)  = a_0 (1 + sum_(k=1)^n r_k/(y + b_k))
def partial_frac_decomp(multiplicative_constant, residues, negative_the_poles):
    r = np.array(residues)
    b = np.array(negative_the_poles)
    assert(len(r)==len(b))
    a0 = multiplicative_constant
    return lambda y : a0 * (1 + np.sum(r/(y + b)))

# Rescales coefficients a0, a1, a2, ... a_(2n) in the rational approximation
# R(y) of y^(-1/2), R(y)  = a_0 Prod_(k=1)^n (y+a_(2k-1))/(y+a_2k)
# Optimal Zolotarev coefficients are for range of y in (epsilon, 1)
# If the range of y is not in (epsilon, 1) but in (epsilon, 1) * Lambdasq_m (Lambdasq_m > 0),
# Apply R to y/Lambdasq_m:
# R(y)                      ->      R(y/Lambdasq_m) * Lambda_m^(-1)
# a_0, a_1, a_2, ... a_(2n)         a_0/Lambda_m, a_1 * Lambda^2_m, a_2 * Lambda^2_m, ... a_(2n) * Lambda^2_m
def rescale_rational_func_coeffs(Lambdasq_m, coeffs):
    a = np.array(coeffs)
    assert(Lambdasq_m > 0)
    assert(len(a) % 2 == 1)
    a[0] /= math.sqrt(Lambdasq_m)
    a[1:] *= Lambdasq_m
    return a

# Rescales coefficients of the partial fraction decomposition for the rational approximation
# R(y) of y^(-1/2), R(y)  = a_0 (1 + sum_(k=1)^n r_k/(y + b_k))
# Optimal Zolotarev coefficients are for range of y in (epsilon, 1)
# If the range of y is not in (epsilon, 1) but in (epsilon, 1) * Lambdasq_m (Lambdasq_m > 0),
# Apply R to y/Lambdasq_m:
# R(y)                      ->      R(y/Lambdasq_m) * Lambda_m^(-1)
# a0                        ->      a_0 / Lambda_m^{-1}
# r_1, ..., r_n             ->      Lambda^2_m * r_1, ..., Lambda^2_m * r_n
# b_1, ..., b_n             ->      Lambda^2_m * b_1, ..., Lambda^2_m * b_n  = mu^2_1, ..., mu^2_n
def rescale_partial_frac_decomp_coeffs(Lambdasq_m, multiplicative_constant, residues, negative_the_poles):
    assert(Lambdasq_m > 0)
    a0 = multiplicative_constant
    r = np.array(residues)
    b = np.array(negative_the_poles)
    assert(len(r)==len(b))
    r *= Lambdasq_m
    musq = b * Lambdasq_m
    a0 /= math.sqrt(Lambdasq_m)
    return (a0, r, musq)

def _test_zolotarev(smallest_eigenvalue, largest_eigenvalue, polynomial_degree, verbose=verbose_default):
    alpha = 0.5                 # testing R(y) approximating y^(-alpha) = y^(-1/2)
    y_min = smallest_eigenvalue # sampling interval min
    y_max = largest_eigenvalue  # sampling interval max
    npoints = 1000              # sample delta error this many times
    coeffs, delta_theoretical = zolotarev(smallest_eigenvalue/largest_eigenvalue, polynomial_degree)    # rescaling [eps * C, C] -> [eps, 1]
    if largest_eigenvalue > 1:
        coeffs = rescale_rational_func_coeffs(largest_eigenvalue, coeffs)
    R = rational_func(coeffs)
    delta_sampled = sample_delta(R, alpha, y_min, y_max, npoints)
    assert(delta_theoretical > 0)
    relative_err = np.abs((delta_theoretical - delta_sampled))/delta_theoretical
    if verbose:
        print(f'epsilon={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}, delta={delta_theoretical:1.4e}, rerr={relative_err:1.4e}')
    assert(relative_err < 1/npoints)
    return
if __name__ == '__main__' and test_default:
    if verbose_default: print("Testing the implementation of optimal Zolotarev coefficients for 1/sqrt(x) approximation (w/o rescaling)")
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            _test_zolotarev(smallest_eigenvalue, 1, polynomial_degree)
    if verbose_default: print("Testing the implementation of optimal Zolotarev coefficients for 1/sqrt(x) approximation (w/ rescaling)")
    largest_eigenvalue = 100
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            # eigenvalue range is rescaled from (eps * C, C) to (eps, 1)
            _test_zolotarev(smallest_eigenvalue, largest_eigenvalue, polynomial_degree)

# compare the rational fraction form with its partial fraction decomposition
def _test_function_output(fun1, fun2, y_min, y_max, verbose=verbose_default):
    npoints = 1000
    sampling_space = np.linspace(y_min, y_max, npoints)
    funs1 = np.array(list(map(fun1, sampling_space)))
    funs2  = np.array(list(map(fun2, sampling_space)))
    assert(np.all(funs1 > 0))
    relative_errors = np.abs((funs1 - funs2))/funs1
    max_rel_err = np.max(relative_errors)
    if verbose:
        print(f'y_min={y_min:1.5f}, y_max={y_max:1.5f}, maximum relative error={max_rel_err:1.4e}')
    assert(np.all(np.isclose(funs1, funs2)))
    return
def _test_partial_frac_decomp(eps, n, max_evalue=1, verbose=verbose_default):
    assert(eps < max_evalue and eps > 0)
    a, delta = zolotarev(eps/max_evalue, n)     # rescaling [eps * C, C] -> [eps, 1]
    b = a[2::2]
    a0 = a[0]
    r = residues(a[1::2], b)
    if max_evalue > 1:
        a0, r, b = rescale_partial_frac_decomp_coeffs(max_evalue, a[0], r, a[2::2])
        a = rescale_rational_func_coeffs(max_evalue, a)
    R1 = rational_func(a)
    R2 = partial_frac_decomp(a0, r, b)
    _test_function_output(R1, R2, eps, max_evalue, verbose=verbose)
if __name__ == '__main__' and test_default:
    if verbose_default: print("Testing the partial fractions decomposition of the rational function (w/o rescaling)")
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            if verbose_default: print(f'eps={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}')
            _test_partial_frac_decomp(smallest_eigenvalue, polynomial_degree)
    if verbose_default: print("Testing the partial fractions decomposition of the rational function (w/ rescaling)")
    max_evalue = 100
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            if verbose_default: print(f'eps={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}')
            _test_partial_frac_decomp(smallest_eigenvalue, polynomial_degree, 100)

# returns multiplicative constant, residues, negative the poles, and the error
# for the partial fraction decomposition of the optimal rational approximation of 1/sqrt(x)
def negative_sqrt_coeffs(smallest_evalue, largest_evalue, polynomial_degree):
    assert(smallest_evalue > 0)
    assert(largest_evalue > smallest_evalue)
    eps = smallest_evalue / largest_evalue
    a, delta = zolotarev(eps, polynomial_degree)
    a0 = a[0]                                       # multiplicative constant
    b = a[2::2]                                     # negative the poles
    r = residues(a[1::2], b)
    a0, r, musq = rescale_partial_frac_decomp_coeffs(largest_evalue, a0, r, b)
    return a0, r, musq, delta
def _test_negative_sqrt_coeffs(smallest_evalue, largest_evalue, polynomial_degree, verbose=verbose_default):
    alpha = 0.5                 # testing R(y) approximating y^(-alpha) = y^(-1/2)
    y_min = smallest_eigenvalue # sampling interval min
    y_max = largest_eigenvalue  # sampling interval max
    npoints = 1000              # sample delta error this many times
    a0, r, musq, delta_theoretical = negative_sqrt_coeffs(smallest_evalue, largest_evalue, polynomial_degree)
    R = partial_frac_decomp(a0, r, musq)
    delta_sampled = sample_delta(R, alpha, y_min, y_max, npoints)
    assert(delta_theoretical > 0)
    relative_err = np.abs((delta_theoretical - delta_sampled))/delta_theoretical
    if verbose:
        print(f'epsilon={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}, delta={delta_theoretical:1.4e}, rerr={relative_err:1.4e}')
    assert(relative_err < 1/npoints)
    return
if __name__ == '__main__' and test_default:
    largest_eigenvalue = 100
    if verbose_default: print("Testing the interface for getting 1/sqrt(x) coefficients with automatic rescaling, using partial fractions decomposition")
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6):
            _test_negative_sqrt_coeffs(smallest_eigenvalue, largest_eigenvalue, polynomial_degree)

def pf_zolotarev_delta(delta):
    # delta is not used anywhere; delta is for R, not S^\dagger S which gives R^{-1}
    # empirically, I see delta-s \approx delta^2... guessing that delta -> delta + delta^2 + delta^3 + delta^4 + ... gives a significantly better relative error
    # there should be a rigorous way to see this but it's not relevant for the accuracy of the approximation, so I'm not going to spend time on this
    return delta * 1/(1-delta)

# returns multiplicative constant, residues, negative the poles, and approximation error
# for the partial fraction decomposition of the rational approximation of S(y) s.t.
# S^\dag(y)S(y) approximates y^{+1/2}, which is used for pseudofermion initialization
def pf_init_coeffs_rational_form(smallest_evalue, largest_evalue, polynomial_degree):
    assert(smallest_evalue > 0)
    assert(largest_evalue > smallest_evalue)
    eps = smallest_evalue / largest_evalue
    a, delta = zolotarev(eps, polynomial_degree)
    delta = pf_zolotarev_delta(delta)
    a = np.array(rescale_rational_func_coeffs(largest_evalue, a), dtype=np.complex128)
    ainv = np.empty(a.size, dtype=np.complex128)     # a1, a3,... a_(2n-1) flipped with a2, a4, ... a_(2n)
    ainv[0] = 1/np.sqrt(a[0])
    ainv[1::2] = complex(0, 1) * np.sqrt(a[2::2])
    ainv[2::2] = complex(0, 1) * np.sqrt(a[1::2])
    return (ainv, delta)
def pf_init_coeffs(smallest_evalue, largest_evalue, polynomial_degree):
    assert(smallest_evalue > 0)
    assert(largest_evalue > smallest_evalue)
    eps = smallest_evalue / largest_evalue
    a, delta = zolotarev(eps, polynomial_degree)
    delta = pf_zolotarev_delta(delta)
    a = np.array(rescale_rational_func_coeffs(largest_evalue, a), dtype=np.complex128)
    ainv = np.empty(a.size, dtype=np.complex128)    # a1, a3,... a_(2n-1) flipped with a2, a4, ... a_(2n)
    ainv[0] = 1/a[0]
    ainv[1::2] = a[2::2]
    ainv[2::2] = a[1::2]
    a0 = np.sqrt(ainv[0])                           # multiplicative constant
    ir = complex(0, 1) * residues(np.sqrt(ainv[1::2]), np.sqrt(ainv[2::2]))
    imu = complex(0, 1) * np.sqrt(ainv[2::2])
    return (a0, ir, imu, delta)
def _test_pf_init_coeffs(smallest_evalue, largest_evalue, polynomial_degree, verbose=verbose_default):
    a0, ir, imu, delta_theoretical = pf_init_coeffs(smallest_evalue, largest_evalue, polynomial_degree)
    #coeffs, delta_theoretical = pf_init_coeffs_rational_form(smallest_evalue, largest_evalue, polynomial_degree)
    def _rational_func(coefficients):
        a = np.array(coefficients)
        # N.B. sqrt(y)
        return lambda y : a[0] * (np.prod(math.sqrt(y) + a[1::2]))/(np.prod(math.sqrt(y) + a[2::2]))
    def _partial_frac_decomp(multiplicative_constant, residues, negative_the_poles):
        ir = np.array(residues, dtype=np.complex128)
        imu = np.array(negative_the_poles, dtype=np.complex128)
        assert(len(ir)==len(imu))
        a0 = multiplicative_constant
        return lambda y : a0 * (1 + np.sum(ir/(math.sqrt(y) + imu)))
    #S = _rational_func(coeffs)
    S = _partial_frac_decomp(a0, ir, imu)
    alpha = -0.5                # testing S(y) s.t S^dag S approximates y^(1/2)
    y_min = smallest_eigenvalue # sampling interval min
    y_max = largest_eigenvalue  # sampling interval max
    npoints = 1000              # sample delta error this many times
    assert(delta_theoretical > 0)
    delta_sampled = sample_delta(lambda y : np.conj(S(y)) * S(y), alpha, y_min, y_max, npoints)
    relative_err = np.abs((delta_theoretical - delta_sampled))/delta_theoretical
    if verbose:
        print(f'epsilon={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}, delta (theoretical)={delta_theoretical:1.4e}, delta (sampled) = {delta_sampled:1.4e}, rerr={relative_err:1.4e}')
        # print(f'epsilon={smallest_eigenvalue:1.5f}, n={polynomial_degree:2d}, sampled delta={delta_sampled:1.4e}')
    return
if __name__ == '__main__' and test_default:
    largest_eigenvalue = 100
    if verbose_default:
        print("Testing the interface for getting pf initialization coefficients with automatic rescaling, using partial fractions decomposition")
        print("N.B. the theoretical error for R(y) does not exactly equal the sampled one for S^\dagger S; cf. pf_zolotarev_delta:")
    for smallest_eigenvalue in (0.1, 0.01, 0.001):
        for polynomial_degree in (2, 4, 6, 8):
            _test_pf_init_coeffs(smallest_eigenvalue, largest_eigenvalue, polynomial_degree)

