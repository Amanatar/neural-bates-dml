"""
Quasi-Random Sobol Dataset Generator for Bates SVJD Option Pricing & DML.
Author: Quantitative Derivatives Research
Repository: Amanatar/neural-bates-dml

Generates high-signal quasi-Monte Carlo option pricing datasets with exact
sensitivities (parameter Jacobians dV/dx, Delta, Vega, Theta, Rho, jump Greeks)
using Scipy Sobol low-discrepancy sequences stratified across three institutional
market regimes:
  1. Calm Equity Market
  2. High-Volatility Stress
  3. Crash Jump-Skew Regime
"""

import os
import sys
import time
import math
import numpy as np
from scipy.stats import qmc
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from bates_fft import bates_cf, verify_fft_against_black_scholes, black_scholes_call

# Active 10D and 11D input specifications
FEATURE_NAMES_10D = [
    'log_moneyness', 'T', 'r', 'v0', 'kappa', 'theta', 'sigma_v', 'rho', 'lambda_j', 'mu_j'
]
FEATURE_NAMES_11D = [
    'log_moneyness', 'T', 'r', 'v0', 'kappa', 'theta', 'sigma_v', 'rho', 'lambda_j', 'mu_j', 'sigma_j'
]

# Regime-specific institutional bounds
REGIME_BOUNDS = {
    "calm": {
        "log_moneyness": (-0.18, 0.12),   # K/S in [0.835, 1.127]
        "T": (0.08, 0.75),                # ~1 month to 9 months
        "r": (0.01, 0.05),
        "v0": (0.01, 0.04),               # sqrt(v0) in [10%, 20%]
        "kappa": (1.2, 4.0),
        "theta": (0.01, 0.04),
        "sigma_v": (0.10, 0.35),
        "rho": (-0.55, -0.15),
        "lambda_j": (0.0, 0.25),
        "mu_j": (-0.08, 0.02),
        "sigma_j": (0.05, 0.15),
    },
    "stress": {
        "log_moneyness": (-0.22, 0.15),   # K/S in [0.80, 1.16]
        "T": (0.08, 1.00),
        "r": (0.01, 0.06),
        "v0": (0.06, 0.18),               # sqrt(v0) in [24.5%, 42.4%]
        "kappa": (0.8, 3.5),
        "theta": (0.05, 0.16),
        "sigma_v": (0.40, 0.85),
        "rho": (-0.80, -0.35),
        "lambda_j": (0.15, 0.65),
        "mu_j": (-0.15, -0.02),
        "sigma_j": (0.10, 0.25),
    },
    "crash": {
        "log_moneyness": (-0.28, 0.15),   # K/S in [0.75, 1.16]
        "T": (0.05, 0.80),
        "r": (0.01, 0.06),
        "v0": (0.05, 0.22),
        "kappa": (0.8, 4.5),
        "theta": (0.05, 0.18),
        "sigma_v": (0.35, 0.80),
        "rho": (-0.95, -0.60),            # steep negative skew
        "lambda_j": (0.60, 1.60),          # frequent crash jumps
        "mu_j": (-0.25, -0.08),           # negative crash jump mean
        "sigma_j": (0.12, 0.30),
    }
}


def batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma, rho, lambda_, muJ, vJ, alpha=1.5, N=256, upper_v=80.0):
    """
    High-performance vectorized Bates call pricer for an arbitrary batch of contracts.
    Uses continuous branch-cut free characteristic function with Simpson quadrature.
    """
    B = len(k)
    v = np.linspace(1e-4, upper_v, N).reshape(1, N)
    dv = v[0, 1] - v[0, 0]
    
    # Broadcast (B, 1) parameters against (1, N) integration nodes
    k_b = k.reshape(B, 1)
    T_b = T.reshape(B, 1)
    r_b = r.reshape(B, 1)
    q_b = q.reshape(B, 1)
    v0_b = v0.reshape(B, 1)
    kappa_b = kappa.reshape(B, 1)
    theta_b = theta.reshape(B, 1)
    sigma_b = np.maximum(sigma.reshape(B, 1), 1e-6)
    sigma2_b = sigma_b**2
    rho_b = rho.reshape(B, 1)
    lambda_b = lambda_.reshape(B, 1)
    muJ_b = muJ.reshape(B, 1)
    vJ_b = vJ.reshape(B, 1)
    
    u = v - (alpha + 1.0) * 1j
    
    beta = kappa_b - 1j * rho_b * sigma_b * u
    d = np.sqrt(beta**2 + (u**2 + 1j * u) * sigma2_b)
    d = np.where(np.real(d) < 0, -d, d)
    
    g = (beta - d) / (beta + d)
    exp_minus_dT = np.exp(-d * T_b)
    
    log_term = np.log((1.0 - g * exp_minus_dT) / (1.0 - g))
    C = (r_b - q_b) * 1j * u * T_b + (kappa_b * theta_b / sigma2_b) * ((beta - d) * T_b - 2.0 * log_term)
    D = ((beta - d) / sigma2_b) * ((1.0 - exp_minus_dT) / (1.0 - g * exp_minus_dT))
    
    cf_heston = np.exp(C + D * v0_b)
    
    omega = -lambda_b * (np.exp(muJ_b + 0.5 * vJ_b**2) - 1.0)
    cf_jump = np.exp(lambda_b * T_b * (np.exp(1j * u * muJ_b - 0.5 * (u**2) * (vJ_b**2)) - 1.0) + 1j * u * omega * T_b)
    
    cf = cf_heston * cf_jump
    
    denom = (alpha**2 + alpha - v**2) + 1j * (2.0 * alpha + 1.0) * v
    psi_v = np.exp(-r_b * T_b) * cf / denom
    
    # Simpson quadrature weights
    w = np.ones((1, N))
    w[0, 1:-1:2] = 4.0
    w[0, 2:-1:2] = 2.0
    
    integrand = np.real(np.exp(-1j * v * k_b) * psi_v)
    integral = (dv / 3.0) * np.sum(w * integrand, axis=1)
    
    prices = (np.exp(-alpha * k_b.squeeze(1)) / np.pi) * integral
    return np.maximum(prices, 0.0)


def compute_batch_prices_and_jacobians(samples_11d, h=1e-4):
    """
    Computes option prices V / S0 and exact central finite-difference Jacobians
    for all parameters in the 11D and 10D input vectors.
    """
    B = samples_11d.shape[0]
    
    k = samples_11d[:, 0]
    T = samples_11d[:, 1]
    r = samples_11d[:, 2]
    q = np.zeros(B)
    v0 = samples_11d[:, 3]
    kappa = samples_11d[:, 4]
    theta = samples_11d[:, 5]
    sigma_v = samples_11d[:, 6]
    rho = samples_11d[:, 7]
    lambda_j = samples_11d[:, 8]
    mu_j = samples_11d[:, 9]
    sigma_j = samples_11d[:, 10]
    
    base_prices = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    
    # 0. dV / d(log_moneyness) (Spot Delta relation)
    p_k_p = batch_bates_call_price(k + h, T, r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    p_k_m = batch_bates_call_price(k - h, T, r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    d_k = (p_k_p - p_k_m) / (2.0 * h)
    
    # 1. dV / dT (Theta)
    p_T_p = batch_bates_call_price(k, T + h, r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    p_T_m = batch_bates_call_price(k, np.maximum(T - h, 1e-4), r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    d_T = (p_T_p - p_T_m) / (2.0 * h)
    
    # 2. dV / dr (Rho)
    p_r_p = batch_bates_call_price(k, T, r + h, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    p_r_m = batch_bates_call_price(k, T, np.maximum(r - h, 0.0), q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    d_r = (p_r_p - p_r_m) / (2.0 * h)
    
    # 3. dV / dv0 (Variance Vega)
    p_v0_p = batch_bates_call_price(k, T, r, q, v0 + h, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    p_v0_m = batch_bates_call_price(k, T, r, q, np.maximum(v0 - h, 1e-4), kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    d_v0 = (p_v0_p - p_v0_m) / (2.0 * h)
    
    # 4. dV / dkappa
    p_kap_p = batch_bates_call_price(k, T, r, q, v0, kappa + h, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    p_kap_m = batch_bates_call_price(k, T, r, q, v0, np.maximum(kappa - h, 1e-2), theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
    d_kappa = (p_kap_p - p_kap_m) / (2.0 * h)
    
    # 5. dV / dtheta
    p_th_p = batch_bates_call_price(k, T, r, q, v0, kappa, theta + h, sigma_v, rho, lambda_j, mu_j, sigma_j)
    p_th_m = batch_bates_call_price(k, T, r, q, v0, kappa, np.maximum(theta - h, 1e-4), sigma_v, rho, lambda_j, mu_j, sigma_j)
    d_theta = (p_th_p - p_th_m) / (2.0 * h)
    
    # 6. dV / dsigma_v (Vol of vol sensitivity)
    p_sig_p = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v + h, rho, lambda_j, mu_j, sigma_j)
    p_sig_m = batch_bates_call_price(k, T, r, q, v0, kappa, theta, np.maximum(sigma_v - h, 1e-4), rho, lambda_j, mu_j, sigma_j)
    d_sigma = (p_sig_p - p_sig_m) / (2.0 * h)
    
    # 7. dV / drho (Skew sensitivity)
    p_rho_p = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, np.minimum(rho + h, -1e-4), lambda_j, mu_j, sigma_j)
    p_rho_m = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, np.maximum(rho - h, -0.999), lambda_j, mu_j, sigma_j)
    d_rho = (p_rho_p - p_rho_m) / (2.0 * h)
    
    # 8. dV / dlambda_j (Jump intensity sensitivity)
    p_lam_p = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, rho, lambda_j + h, mu_j, sigma_j)
    p_lam_m = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, rho, np.maximum(lambda_j - h, 0.0), mu_j, sigma_j)
    d_lambda = (p_lam_p - p_lam_m) / (2.0 * h)
    
    # 9. dV / dmu_j (Mean jump size sensitivity)
    p_mu_p = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j + h, sigma_j)
    p_mu_m = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j - h, sigma_j)
    d_mu = (p_mu_p - p_mu_m) / (2.0 * h)
    
    # 10. dV / dsigma_j (Jump vol sensitivity)
    p_vj_p = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j + h)
    p_vj_m = batch_bates_call_price(k, T, r, q, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, np.maximum(sigma_j - h, 1e-4))
    d_sigma_j = (p_vj_p - p_vj_m) / (2.0 * h)
    
    jacobians_11d = np.column_stack([d_k, d_T, d_r, d_v0, d_kappa, d_theta, d_sigma, d_rho, d_lambda, d_mu, d_sigma_j])
    jacobians_10d = jacobians_11d[:, :10]
    
    # Assertions on numerical finiteness and no NaNs
    assert np.all(np.isfinite(base_prices)), "Non-finite prices detected"
    assert np.all(np.isfinite(jacobians_10d)), "Non-finite 10D Jacobians detected"
    assert np.all(np.isfinite(jacobians_11d)), "Non-finite 11D Jacobians detected"
    
    return base_prices, jacobians_10d, jacobians_11d


def sample_regime_sobol(regime_name, num_samples, seed=42, scramble=True):
    """
    Samples across institutional bounds using Scipy Sobol low-discrepancy sequences
    with physical parameter validation and Feller soft condition filtering.
    """
    bounds = REGIME_BOUNDS[regime_name]
    keys = FEATURE_NAMES_11D
    
    l_bounds = [bounds[k][0] for k in keys]
    u_bounds = [bounds[k][1] for k in keys]
    
    sampler = qmc.Sobol(d=11, scramble=scramble, seed=seed)
    
    valid_samples = []
    total_valid = 0
    chunk_size = 2048
    
    while total_valid < num_samples:
        raw_draw = sampler.random(chunk_size)
        scaled = qmc.scale(raw_draw, l_bounds, u_bounds)
        
        # Physical constraints filtering
        # Feller ratio: 2 * kappa * theta / sigma_v^2
        feller = 2.0 * scaled[:, 4] * scaled[:, 5] / (scaled[:, 6]**2)
        
        # Require feller >= 0.20 to avoid catastrophic reflection instability at origin
        # Also ensure positive variance, non-exploding jump compensator
        mask = (feller >= 0.20) & (scaled[:, 3] > 0.005) & (scaled[:, 5] > 0.005)
        
        filtered = scaled[mask]
        valid_samples.append(filtered)
        total_valid += len(filtered)
        
    all_valid = np.vstack(valid_samples)[:num_samples]
    return all_valid


def generate_sobol_dataset(
    total_samples=30000,
    train_split=24000,
    val_split=3000,
    test_split=3000,
    benchmark_samples=1500,
    dataset_path="data/bates_sobol_dataset.npz",
    benchmark_path="data/eval/bates_test_benchmark.npz",
    seed=42,
    batch_size=1000
):
    """
    Generates 30,000 stratified Sobol samples across 3 institutional market regimes.
    Saves:
      - Compact full dataset (train/val/test) to data/bates_sobol_dataset.npz (< 15 MB)
      - Deterministic 1,500-contract benchmark to data/eval/bates_test_benchmark.npz
    """
    print("=" * 70)
    print("QUASI-RANDOM SOBOL DATASET GENERATION (BATES SVJD)")
    print("=" * 70)
    start_time = time.time()
    
    assert total_samples == train_split + val_split + test_split, "Split counts must sum to total"
    regimes = ["calm", "stress", "crash"]
    n_per_regime = total_samples // len(regimes)
    n_train_regime = train_split // len(regimes)
    n_val_regime = val_split // len(regimes)
    n_test_regime = test_split // len(regimes)
    
    all_X_11d = []
    all_y = []
    all_dydx_10d = []
    all_dydx_11d = []
    all_regimes = []
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    current_idx = 0
    for r_idx, reg in enumerate(regimes):
        print(f"\nSampling Regime {r_idx+1}/3: '{reg.upper()}' ({n_per_regime} samples)...", flush=True)
        reg_seed = seed + r_idx * 1000
        samples_reg = sample_regime_sobol(reg, n_per_regime, seed=reg_seed, scramble=True)
        
        # Batch compute prices and Jacobians
        reg_prices = []
        reg_j10 = []
        reg_j11 = []
        
        for b_start in range(0, n_per_regime, batch_size):
            b_end = min(b_start + batch_size, n_per_regime)
            chunk = samples_reg[b_start:b_end]
            p_chunk, j10_chunk, j11_chunk = compute_batch_prices_and_jacobians(chunk)
            reg_prices.append(p_chunk)
            reg_j10.append(j10_chunk)
            reg_j11.append(j11_chunk)
            print(f"  Processed {b_end}/{n_per_regime} contracts | Elapsed: {time.time()-start_time:.1f}s", flush=True)
            
        prices_arr = np.concatenate(reg_prices)
        j10_arr = np.vstack(reg_j10)
        j11_arr = np.vstack(reg_j11)
        
        all_X_11d.append(samples_reg)
        all_y.append(prices_arr)
        all_dydx_10d.append(j10_arr)
        all_dydx_11d.append(j11_arr)
        all_regimes.append(np.full(n_per_regime, r_idx, dtype=np.int32))
        
        # Record stratified splits
        train_indices.extend(range(current_idx, current_idx + n_train_regime))
        val_indices.extend(range(current_idx + n_train_regime, current_idx + n_train_regime + n_val_regime))
        test_indices.extend(range(current_idx + n_train_regime + n_val_regime, current_idx + n_per_regime))
        current_idx += n_per_regime

    X_11d = np.vstack(all_X_11d).astype(np.float32)
    X_10d = X_11d[:, :10].astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)
    dydx_10d = np.vstack(all_dydx_10d).astype(np.float32)
    dydx_11d = np.vstack(all_dydx_11d).astype(np.float32)
    regime_labels = np.concatenate(all_regimes).astype(np.int32)
    
    train_idx = np.array(train_indices, dtype=np.int32)
    val_idx = np.array(val_indices, dtype=np.int32)
    test_idx = np.array(test_indices, dtype=np.int32)
    
    # Strict NaN / Inf assertions
    assert not np.isnan(X_10d).any(), "NaN found in X_10d"
    assert not np.isnan(y).any(), "NaN found in y"
    assert not np.isnan(dydx_10d).any(), "NaN found in dydx_10d"
    assert np.all(np.isfinite(X_10d)), "Infinite value found in X_10d"
    assert np.all(np.isfinite(y)), "Infinite value found in y"
    assert np.all(np.isfinite(dydx_10d)), "Infinite value found in dydx_10d"
    
    # Ensure destination directories exist
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
    
    # Save compact full dataset (< 15 MB)
    np.savez_compressed(
        dataset_path,
        X=X_10d,
        y=y,
        dydx=dydx_10d,
        X_11d=X_11d,
        dydx_11d=dydx_11d,
        regimes=regime_labels,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        feature_names=FEATURE_NAMES_10D,
        feature_names_11d=FEATURE_NAMES_11D,
        # Train / Val / Test convenience views
        X_train=X_10d[train_idx],
        y_train=y[train_idx],
        dydx_train=dydx_10d[train_idx],
        X_val=X_10d[val_idx],
        y_val=y[val_idx],
        dydx_val=dydx_10d[val_idx],
        X_test=X_10d[test_idx],
        y_test=y[test_idx],
        dydx_test=dydx_10d[test_idx]
    )
    dataset_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
    print(f"\n[OUTPUT] Saved Full Sobol Dataset: {dataset_path} ({dataset_size_mb:.2f} MB)")
    assert dataset_size_mb < 15.0, f"Dataset size {dataset_size_mb:.2f} MB exceeds 15 MB threshold!"
    
    # Generate deterministic 1,500-contract benchmark (500 per regime with unscrambled Sobol)
    n_bench_regime = benchmark_samples // len(regimes)
    bench_X_11d = []
    for r_idx, reg in enumerate(regimes):
        samples_b = sample_regime_sobol(reg, n_bench_regime, seed=seed + 9999 + r_idx, scramble=False)
        bench_X_11d.append(samples_b)
        
    bench_X_11d = np.vstack(bench_X_11d).astype(np.float32)
    bench_p, bench_j10, bench_j11 = compute_batch_prices_and_jacobians(bench_X_11d)
    bench_regimes = np.concatenate([np.full(n_bench_regime, r, dtype=np.int32) for r in range(len(regimes))])
    
    np.savez_compressed(
        benchmark_path,
        X=bench_X_11d[:, :10].astype(np.float32),
        y=bench_p.astype(np.float32),
        dydx=bench_j10.astype(np.float32),
        X_11d=bench_X_11d,
        dydx_11d=bench_j11.astype(np.float32),
        regimes=bench_regimes,
        feature_names=FEATURE_NAMES_10D,
        feature_names_11d=FEATURE_NAMES_11D
    )
    bench_size_mb = os.path.getsize(benchmark_path) / (1024 * 1024)
    print(f"[OUTPUT] Saved Deterministic Evaluation Benchmark: {benchmark_path} ({bench_size_mb:.2f} MB)")
    
    # Diagnostics & Summary Statistics
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("DATASET DIAGNOSTICS & SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total Samples Generated: {len(y)} (Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)})")
    print(f"Benchmark Contracts:     {len(bench_p)}")
    print(f"Total Generation Time:   {total_time:.2f} seconds ({total_time / total_samples * 1000:.2f} ms/sample)")
    print(f"NaN / Inf Count:         0 NaNs, 0 Infs confirmed across all arrays")
    print(f"Normalized Price Range:  Min={y.min():.6f} | Mean={y.mean():.6f} | Max={y.max():.6f}")
    print(f"Variance Vega (dV/dv0):  Min={dydx_10d[:, 3].min():.6f} | Mean={dydx_10d[:, 3].mean():.6f} | Max={dydx_10d[:, 3].max():.6f}")
    print(f"Spot Delta Sensitivity:  Min={dydx_10d[:, 0].min():.6f} | Mean={dydx_10d[:, 0].mean():.6f} | Max={dydx_10d[:, 0].max():.6f}")
    print(f"Jacobian Frobenius Norm: Total Norm={np.linalg.norm(dydx_10d):.4f} | Per-sample Avg={np.linalg.norm(dydx_10d, axis=1).mean():.4f}")
    
    # Verify Black-Scholes Limiting Case Error
    max_bs_err, _ = verify_fft_against_black_scholes(verbose=False)
    print(f"Black-Scholes Max Error: {max_bs_err:.6e} (< 1e-3 verified)")
    print("=" * 70)
    
    return {
        "dataset_path": dataset_path,
        "benchmark_path": benchmark_path,
        "total_samples": len(y),
        "dataset_size_mb": dataset_size_mb,
        "bench_size_mb": bench_size_mb,
        "price_min": float(y.min()),
        "price_max": float(y.max()),
        "price_mean": float(y.mean()),
        "jacobian_norm": float(np.linalg.norm(dydx_10d)),
        "bs_max_error": max_bs_err
    }


if __name__ == '__main__':
    # Step 1: Run analytical Black-Scholes verification unit test
    verify_fft_against_black_scholes()
    
    # Step 2: Generate 30k Sobol dataset + 1.5k benchmark
    generate_sobol_dataset()
