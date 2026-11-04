import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bates_fft import verify_fft_against_black_scholes

def test_carr_madan_fft_vs_black_scholes_limit():
    """
    Unit verification:
    When sigma_v -> 1e-4, kappa -> 1.0, v0 = theta = sigma_bs^2, lambda_j = 0.0,
    asserts Carr-Madan FFT option prices match exact Black-Scholes prices
    to within < 1e-3 across moneyness K/S in [0.80, 1.20].
    """
    max_err, errors = verify_fft_against_black_scholes(moneyness_range=(0.80, 1.20), n_strikes=11, tol=1e-3, verbose=True)
    assert max_err < 1e-3, f"FFT error {max_err:.6e} exceeded 1e-3 threshold"
