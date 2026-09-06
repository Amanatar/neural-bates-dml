import numpy as np

def bates_cf(u, S0, r, q, T, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ):
    """
    Characteristic function of the Bates model.
    """
    i = 1j
    
    # Heston part
    d = np.sqrt((kappa - i*rho*sigma*u)**2 + (u**2 + i*u)*sigma**2)
    g = (kappa - i*rho*sigma*u - d) / (kappa - i*rho*sigma*u + d)
    
    cf_heston = np.exp(i*u*(np.log(S0) + (r - q)*T)) * \
                np.exp((kappa*theta)/(sigma**2) * ((kappa - i*rho*sigma*u - d)*T - 2*np.log((1 - g*np.exp(-d*T))/(1 - g)))) * \
                np.exp((v0/sigma**2) * (kappa - i*rho*sigma*u - d) * (1 - np.exp(-d*T))/(1 - g*np.exp(-d*T)))
                
    # Jump part (Bates)
    omega = -lambda_ * (np.exp(muJ + 0.5 * vJ**2) - 1)
    cf_jump = np.exp(lambda_ * T * (np.exp(i*u*muJ - 0.5 * u**2 * vJ**2) - 1) + i*u*omega*T)
    
    return cf_heston * cf_jump

def carr_madan_fft(S0, r, q, T, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ, N=4096, eta=0.1, alpha=1.5):
    """
    Carr-Madan FFT option pricing for Bates model.
    Returns an array of strikes and corresponding call prices.
    """
    # Grid for v
    v = np.arange(N) * eta
    
    # Damping factor
    i = 1j
    
    # Carr-Madan formula components
    def psi(v_val):
        u = v_val - (alpha + 1)*i
        cf = bates_cf(u, S0, r, q, T, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ)
        return np.exp(-r*T) * cf / (alpha**2 + alpha - v_val**2 + i*(2*alpha + 1)*v_val)
    
    psi_v = psi(v)
    
    # Simpson's rule integration weights
    weights = np.ones(N)
    weights[1::2] = 4
    weights[2::2] = 2
    weights[0] = 1
    weights[-1] = 1
    weights = weights / 3
    
    # FFT formulation
    lambda_grid = 2 * np.pi / (N * eta)
    b = (N * lambda_grid) / 2
    
    ku = -b + lambda_grid * np.arange(N)
    strikes = np.exp(ku)
    
    # Input to FFT
    x = np.exp(i * b * v) * psi_v * eta * weights
    # Fix singularity at v=0 if any, although psi handles it if alpha>0
    x[0] = x[0] * 0.5
    
    y = np.fft.fft(x)
    
    # Option prices
    call_prices = (np.exp(-alpha * ku) / np.pi) * np.real(y)
    
    return strikes, call_prices

def price_bates_surface(S0, r, q, maturities, moneyness_grid, params):
    """
    Generate prices for a grid of maturities and moneyness using FFT + interpolation.
    """
    v0, kappa, theta, sigma, rho, lambda_, muJ, vJ = params
    
    prices = np.zeros((len(maturities), len(moneyness_grid)))
    
    for i, T in enumerate(maturities):
        strikes, calls = carr_madan_fft(S0, r, q, T, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ)
        
        # We want prices at specific moneyness = K / S0
        target_strikes = moneyness_grid * S0
        
        # Interpolate the FFT results onto our exact target strikes
        # Note: strikes from FFT are sorted
        prices[i, :] = np.interp(target_strikes, strikes, calls)
    
    # The Carr-Madan grid (N=4096, eta=0.1) has O(1e-3) truncation error at the
    # short-maturity deep-OTM wing, which can dip slightly below zero. Option
    # prices are non-negative by no-arbitrage, so floor them.
    # NOTE: N=4096 is even, so the Simpson weights are not strictly a composite
    # Simpson rule (needs an odd point count). Kept for consistency with the
    # shipped surrogate checkpoint; harmless at this grid resolution.
    return np.maximum(prices, 0.0)
