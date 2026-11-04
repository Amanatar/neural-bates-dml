import numpy as np
import math

def black_scholes_call(S0, K, r, q, T, sigma):
    """
    Exact closed-form Black-Scholes call price with continuous dividend yield.
    Uses math.erf for high numerical precision and zero third-party dependencies.
    """
    if T <= 0:
        return max(float(S0 - K), 0.0)
    if sigma <= 0:
        return max(float(S0 * math.exp(-q * T) - K * math.exp(-r * T)), 0.0)
    
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    
    price = S0 * math.exp(-q * T) * nd1 - K * math.exp(-r * T) * nd2
    return max(float(price), 0.0)


def black_scholes_put(S0, K, r, q, T, sigma):
    """
    Exact closed-form Black-Scholes put price with continuous dividend yield.
    Uses math.erf for high numerical precision.
    """
    if T <= 0:
        return max(float(K - S0), 0.0)
    if sigma <= 0:
        return max(float(K * math.exp(-r * T) - S0 * math.exp(-q * T)), 0.0)
    
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    n_minus_d1 = 0.5 * (1.0 - math.erf(d1 / math.sqrt(2.0)))
    n_minus_d2 = 0.5 * (1.0 - math.erf(d2 / math.sqrt(2.0)))
    
    price = K * math.exp(-r * T) * n_minus_d2 - S0 * math.exp(-q * T) * n_minus_d1
    return max(float(price), 0.0)


def black_scholes_vega(S0, K, r, q, T, sigma):
    """
    Exact analytical Black-Scholes Vega (dPrice / dSigma).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    return float(S0 * math.exp(-q * T) * math.sqrt(T) * pdf)


def invert_black_scholes_iv(price, S0, K, r, q, T, is_call=True, iv_min=1e-4, iv_max=3.0):
    """
    Invert Black-Scholes implied volatility using Brent's root-finding method.
    Returns float IV or np.nan if price violates arbitrage bounds.
    """
    from scipy.optimize import brentq
    if T <= 0 or price <= 0:
        return float(np.nan)
        
    intrinsic = max(S0 * math.exp(-q * T) - K * math.exp(-r * T), 0.0) if is_call else max(K * math.exp(-r * T) - S0 * math.exp(-q * T), 0.0)
    upper_bound = S0 * math.exp(-q * T) if is_call else K * math.exp(-r * T)
    
    if price <= intrinsic + 1e-6 or price >= upper_bound:
        return float(np.nan)
        
    def objective(sig):
        if is_call:
            return black_scholes_call(S0, K, r, q, T, sig) - price
        else:
            return black_scholes_put(S0, K, r, q, T, sig) - price
            
    try:
        iv = brentq(objective, iv_min, iv_max, xtol=1e-6, maxiter=100)
        return float(iv)
    except Exception:
        return float(np.nan)


def bates_cf(u, S0, r, q, T, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ):
    """
    Characteristic function of the Bates SVJD (Heston + Merton jumps) model.
    Numerically stable formulation (Albrecher et al. 2007 / Little Heston Trap free)
    with branch-cut continuous complex logarithm and non-exploding exponent.
    """
    i = 1j
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    sigma2 = sigma**2
    
    # Heston component (Albrecher et al. 2007)
    beta = kappa - i * rho * sigma * u
    d = np.sqrt(beta**2 + (u**2 + i * u) * sigma2)
    # Enforce Re(d) >= 0 to ensure exp(-d*T) <= 1
    d = np.where(np.real(d) < 0, -d, d)
    
    g = (beta - d) / (beta + d)
    exp_minus_dT = np.exp(-d * T)
    
    # Continuous log formulation: log((1 - g * exp(-d*T)) / (1 - g))
    log_term = np.log((1.0 - g * exp_minus_dT) / (1.0 - g))
    C = (r - q) * i * u * T + (kappa * theta / sigma2) * ((beta - d) * T - 2.0 * log_term)
    D = ((beta - d) / sigma2) * ((1.0 - exp_minus_dT) / (1.0 - g * exp_minus_dT))
    
    cf_heston = np.exp(i * u * np.log(S0) + C + D * v0)
    
    # Merton jump component
    omega = -lambda_ * (np.exp(muJ + 0.5 * vJ**2) - 1.0)
    cf_jump = np.exp(lambda_ * T * (np.exp(i * u * muJ - 0.5 * (u**2) * (vJ**2)) - 1.0) + i * u * omega * T)
    
    return cf_heston * cf_jump


def carr_madan_fft(S0, r, q, T, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ, N=4096, eta=0.1, alpha=1.5):
    """
    Carr-Madan (1999) FFT option pricing for Bates SVJD model.
    Uses damping factor alpha = 1.5 and composite Simpson's 1/3 rule weights.
    Returns (strikes, call_prices).
    """
    i = 1j
    v = np.arange(N) * eta
    
    u = v - (alpha + 1.0) * i
    cf = bates_cf(u, S0, r, q, T, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ)
    
    denom = (alpha**2 + alpha - v**2) + i * (2.0 * alpha + 1.0) * v
    psi_v = np.exp(-r * T) * cf / denom
    
    # Composite Simpson's 1/3 rule integration weights: [1, 4, 2, 4, ..., 4, 1] / 3
    weights = np.ones(N)
    weights[1::2] = 4.0
    weights[2::2] = 2.0
    weights[0] = 1.0
    weights[-1] = 1.0
    weights = weights / 3.0
    
    lambda_grid = 2.0 * np.pi / (N * eta)
    b = (N * lambda_grid) / 2.0
    
    ku = -b + lambda_grid * np.arange(N)
    strikes = np.exp(ku)
    
    # FFT input
    x = np.exp(i * b * v) * psi_v * eta * weights
    y = np.fft.fft(x)
    
    # Option prices
    call_prices = (np.exp(-alpha * ku) / np.pi) * np.real(y)
    
    # No-arbitrage lower bound: non-negative
    call_prices = np.maximum(call_prices, 0.0)
    return strikes, call_prices


def price_bates_surface(S0, r, q, maturities, moneyness_grid, params):
    """
    Generate prices for a grid of maturities and moneyness using FFT + interpolation.
    """
    v0, kappa, theta, sigma, rho, lambda_, muJ, vJ = params
    
    prices = np.zeros((len(maturities), len(moneyness_grid)))
    
    for i, T in enumerate(maturities):
        strikes, calls = carr_madan_fft(S0, r, q, T, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ)
        target_strikes = moneyness_grid * S0
        prices[i, :] = np.interp(target_strikes, strikes, calls)
    
    return np.maximum(prices, 0.0)


def verify_fft_against_black_scholes(moneyness_range=(0.80, 1.20), n_strikes=11, tol=1e-3, verbose=True):
    """
    Automated unit verification:
    When sigma_v -> 1e-4, kappa -> 1.0, v0 = theta = sigma_bs^2, and lambda_j = 0.0,
    asserts that Carr-Madan FFT option prices match exact closed-form Black-Scholes
    prices to within < 1e-3 across moneyness K/S in [0.80, 1.20].
    """
    S0 = 1.0
    r = 0.05
    q = 0.0
    T = 0.25
    sigma_bs = 0.20
    v0 = sigma_bs**2
    theta = sigma_bs**2
    kappa = 1.0
    sigma_v = 1e-4
    rho = 0.0
    lambda_j = 0.0
    mu_j = 0.0
    sigma_j = 0.1
    
    strikes_fft, calls_fft = carr_madan_fft(
        S0, r, q, T, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j,
        N=4096, eta=0.1, alpha=1.5
    )
    
    moneyness_grid = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
    target_strikes = moneyness_grid * S0
    interp_fft = np.interp(target_strikes, strikes_fft, calls_fft)
    
    bs_prices = np.array([black_scholes_call(S0, k, r, q, T, sigma_bs) for k in target_strikes])
    errors = np.abs(interp_fft - bs_prices)
    max_err = float(np.max(errors))
    
    if verbose:
        print("=" * 70)
        print("ANALYTICAL UNIT VERIFICATION: CARR-MADAN FFT VS BLACK-SCHOLES")
        print("=" * 70)
        print(f"Parameters: S0={S0}, T={T}y, r={r:.2f}, q={q:.2f}, sigma_bs={sigma_bs:.2f}")
        print(f"Bates Limit: sigma_v={sigma_v:.1e}, kappa={kappa:.1f}, v0=theta={v0:.4f}, lambda_j={lambda_j:.1f}")
        print(f"Max Absolute Error across [{moneyness_range[0]:.2f}, {moneyness_range[1]:.2f}]: {max_err:.6e} (Threshold: {tol:.6e})")
        print("-" * 70)
        for m, c_fft, c_bs, err in zip(moneyness_grid, interp_fft, bs_prices, errors):
            print(f"  Moneyness K/S = {m:.2f} | FFT Price = {c_fft:.6f} | BS Price = {c_bs:.6f} | Abs Error = {err:.6e}")
        print("-" * 70)
        print(f"Assertion Status: {'PASSED (max_err < 1e-3)' if max_err < tol else 'FAILED'}")
        print("=" * 70)
        
    assert max_err < tol, f"Verification failed: max_err {max_err:.6e} >= {tol:.6e}"
    return max_err, errors


if __name__ == '__main__':
    verify_fft_against_black_scholes()
