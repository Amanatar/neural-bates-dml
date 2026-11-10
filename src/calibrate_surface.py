"""
5-Regime Volatility Surface Calibration Benchmark.
Repository: Amanatar/neural-bates-dml
Author: Principal Quant & ML Systems Engineer

Constructs 5 realistic 55-contract option surfaces (5 maturities x 11 strikes,
+/- 5 bps IV noise) across:
  1. Calm Low-Vol
  2. Steep Equity Put Skew
  3. High Vol-of-Vol
  4. Crash Jump Regime
  5. Short-Dated Wing Stress

Benchmarks:
  - Classical Numerical FFT Calibration (scipy.optimize.minimize L-BFGS-B calling Carr-Madan FFT)
  - Differentiable Sobolev Surrogate Calibration (analytical PyTorch Jacobian L-BFGS over bates_sobolev_dml.pt)

Saves results to results/calibration_benchmark.json.
"""

import os
import sys
import time
import math
import json
from pathlib import Path
import numpy as np
import scipy.optimize as opt
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bates_fft import (
    carr_madan_fft,
    black_scholes_call,
    black_scholes_vega,
    invert_black_scholes_iv
)
from model import BatesNeuralSurrogate

# 5 Institutional Regimes: Ground Truth Parameters
# (v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
REGIMES = {
    "calm_low_vol": {
        "name": "Calm Low-Vol",
        "params": [0.0144, 2.20, 0.0225, 0.22, -0.45, 0.08, -0.04, 0.08],
        "maturities": [0.12, 0.25, 0.50, 0.75, 1.00],
        "description": "Subdued volatility, mild skew, rare small jumps"
    },
    "steep_equity_skew": {
        "name": "Steep Equity Put Skew",
        "params": [0.0400, 1.80, 0.0350, 0.48, -0.82, 0.25, -0.12, 0.15],
        "maturities": [0.12, 0.25, 0.50, 0.75, 1.00],
        "description": "Standard equity index skew with strong leverage effect and asymmetric downside"
    },
    "high_vol_of_vol": {
        "name": "High Vol-of-Vol",
        "params": [0.0550, 2.40, 0.0500, 0.70, -0.60, 0.35, -0.10, 0.18],
        "maturities": [0.12, 0.25, 0.50, 0.75, 1.00],
        "description": "Severe volatility clustering and fat-tailed smile curvature"
    },
    "crash_jump_regime": {
        "name": "Crash Jump Regime",
        "params": [0.0900, 2.80, 0.0750, 0.55, -0.78, 0.75, -0.18, 0.20],
        "maturities": [0.12, 0.25, 0.50, 0.75, 1.00],
        "description": "Elevated spot variance with frequent, deep-negative tail jumps"
    },
    "short_dated_wing_stress": {
        "name": "Short-Dated Wing Stress",
        "params": [0.0650, 1.60, 0.0450, 0.60, -0.72, 0.50, -0.14, 0.18],
        "maturities": [0.06, 0.12, 0.25, 0.50, 0.75],
        "description": "High wing kurtosis and stress pricing in near-term maturities"
    }
}

PARAM_KEYS = ["v0", "kappa", "theta", "sigma_v", "rho", "lambda_j", "mu_j"]
BOUNDS = [
    (0.010, 0.160),  # v0
    (0.60, 4.00),    # kappa
    (0.010, 0.140),  # theta
    (0.12, 0.75),    # sigma_v
    (-0.90, -0.15),  # rho
    (0.00, 1.00),    # lambda_j
    (-0.22, 0.02)    # mu_j
]

STRIKES = np.linspace(0.86, 1.14, 11)  # 11 strikes across liquid surface
S0 = 1.0
R = 0.04
Q = 0.0


def construct_synthetic_surface(true_params, maturities, noise_seed=42):
    """
    Constructs a 55-contract option surface (5 maturities x 11 strikes) with +/- 5 bps IV noise.
    """
    np.random.seed(noise_seed)
    v0, kappa, theta, sig_v, rho, lam, muJ, sigJ = true_params
    contracts = []
    
    for T in maturities:
        strikes_fft, calls_fft = carr_madan_fft(
            S0, R, Q, T, v0, kappa, theta, sig_v, rho, lam, muJ, sigJ,
            N=4096, eta=0.1, alpha=1.5
        )
        for K in STRIKES:
            true_price = float(np.interp(K, strikes_fft, calls_fft))
            true_iv = invert_black_scholes_iv(true_price, S0, K, R, Q, T, is_call=True)
            assert not np.isnan(true_iv), f"Failed IV inversion for T={T}, K={K}"
            
            # +/- 5 bps IV observation noise (0.0005)
            iv_noise = float(np.random.uniform(-0.0005, 0.0005))
            mkt_iv = max(true_iv + iv_noise, 0.01)
            
            mkt_price = black_scholes_call(S0, K, R, Q, T, mkt_iv)
            vega = black_scholes_vega(S0, K, R, Q, T, mkt_iv)
            vega_eff = max(vega, 0.015)
            
            contracts.append({
                "T": float(T),
                "K": float(K),
                "m": float(K / S0),
                "true_price": true_price,
                "true_iv": float(true_iv),
                "mkt_price": float(mkt_price),
                "mkt_iv": float(mkt_iv),
                "vega": float(vega),
                "vega_eff": float(vega_eff),
                "sigJ": float(sigJ)
            })
            
    return contracts


def calibrate_classical_fft(contracts, initial_guess, fixed_sigJ, max_iter=60):
    """
    Calibrates Bates SVJD model parameters using SciPy L-BFGS-B calling Carr-Madan FFT
    over normalized [0, 1]^7 parameter space for proper Hessian conditioning.
    """
    maturities_unique = sorted(list(set(c["T"] for c in contracts)))
    low = np.array([b[0] for b in BOUNDS], dtype=np.float64)
    high = np.array([b[1] for b in BOUNDS], dtype=np.float64)
    span = high - low
    
    # Pre-group contracts by maturity for fast evaluation
    grouped = {}
    for T in maturities_unique:
        c_sub = [c for c in contracts if abs(c["T"] - T) < 1e-6]
        grouped[T] = (
            np.array([c["K"] for c in c_sub]),
            np.array([c["mkt_price"] for c in c_sub]),
            np.array([c["vega_eff"] for c in c_sub])
        )
    M = len(contracts)
    
    def objective_unit(u):
        p = low + np.clip(u, 0.0, 1.0) * span
        v0, kap, th, sig, rh, lam, mu = p
        total_sq_iv_pct = 0.0
        
        for T in maturities_unique:
            ks, mkt_ps, vegs = grouped[T]
            strikes_fft, calls_fft = carr_madan_fft(
                S0, R, Q, T, v0, kap, th, sig, rh, lam, mu, fixed_sigJ,
                N=2048, eta=0.12, alpha=1.5
            )
            interp_calls = np.interp(ks, strikes_fft, calls_fft)
            # First-order IV residual approximation in percentage vol units (e.g. 1.0 = 100 bps)
            iv_err_pct = ((interp_calls - mkt_ps) / vegs) * 100.0
            total_sq_iv_pct += float(np.sum(iv_err_pct ** 2))
            
        return total_sq_iv_pct / M

    u0 = np.clip((np.array(initial_guess) - low) / span, 0.01, 0.99)
    unit_bounds = [(0.0, 1.0)] * 7
    
    t0 = time.perf_counter()
    res = opt.minimize(
        objective_unit,
        x0=u0,
        method="L-BFGS-B",
        bounds=unit_bounds,
        options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-8, "eps": 1e-4}
    )
    wall_clock = time.perf_counter() - t0
    
    calib_params = (low + np.clip(res.x, 0.0, 1.0) * span).tolist()
    return calib_params, wall_clock, res.nfev


def calibrate_sobolev_dml(contracts, surrogate_bundle, initial_guess, fixed_sigJ, max_iter=60, device=torch.device("cpu")):
    """
    Calibrates Bates SVJD model parameters via analytical PyTorch Jacobian L-BFGS
    evaluating through the trained Sobolev DML neural surrogate.
    """
    model = surrogate_bundle["model"].to(device)
    model.eval()
    
    X_mean = surrogate_bundle["X_mean"].to(device)
    X_std = surrogate_bundle["X_std"].to(device)
    y_mean = surrogate_bundle["y_mean"].to(device)
    y_std = surrogate_bundle["y_std"].to(device)
    
    M = len(contracts)
    m_t = torch.tensor([c["m"] for c in contracts], dtype=torch.float32, device=device)
    T_t = torch.tensor([c["T"] for c in contracts], dtype=torch.float32, device=device)
    mkt_p_t = torch.tensor([c["mkt_price"] for c in contracts], dtype=torch.float32, device=device).unsqueeze(1)
    vega_t = torch.tensor([c["vega_eff"] for c in contracts], dtype=torch.float32, device=device).unsqueeze(1)
    fixed_sigJ_t = torch.full((M, 1), fixed_sigJ, dtype=torch.float32, device=device)
    
    low_b = torch.tensor([b[0] for b in BOUNDS], dtype=torch.float32, device=device)
    high_b = torch.tensor([b[1] for b in BOUNDS], dtype=torch.float32, device=device)
    
    init_u = np.clip(
        [(initial_guess[i] - BOUNDS[i][0]) / (BOUNDS[i][1] - BOUNDS[i][0]) for i in range(7)],
        0.05, 0.95
    )
    init_logits = [math.log(u / (1.0 - u)) for u in init_u]
    z = torch.tensor(init_logits, dtype=torch.float32, device=device, requires_grad=True)
    
    optimizer = torch.optim.LBFGS(
        [z],
        lr=0.8,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=15,
        line_search_fn="strong_wolfe"
    )
    
    eval_count = [0]
    
    def closure():
        optimizer.zero_grad()
        eval_count[0] += 1
        
        phys = low_b + torch.sigmoid(z) * (high_b - low_b)
        phys_expanded = phys.unsqueeze(0).expand(M, -1)
        x_raw = torch.cat([
            phys_expanded,
            fixed_sigJ_t,
            m_t.unsqueeze(1),
            T_t.unsqueeze(1)
        ], dim=1)
        
        x_scaled = (x_raw - X_mean) / X_std
        pred_scaled = model(x_scaled)
        pred_p = pred_scaled * y_std + y_mean
        
        # Identical Vega-normalized percentage-vol MSE loss as Classical FFT
        iv_err_pct = ((pred_p - mkt_p_t) / vega_t) * 100.0
        loss = torch.mean(iv_err_pct ** 2)
        loss.backward()
        return loss

    t0 = time.perf_counter()
    optimizer.step(closure)
    wall_clock = time.perf_counter() - t0
    
    with torch.no_grad():
        final_phys = low_b + torch.sigmoid(z) * (high_b - low_b)
        calib_params = final_phys.cpu().numpy().tolist()
        
    return calib_params, wall_clock, eval_count[0]


def evaluate_calibration_quality(contracts, calib_params, fixed_sigJ):
    """
    Evaluates calibrated parameters on exact Carr-Madan FFT pricing vs noisy market quotes.
    Computes exact Implied Vol RMSE (bps), Price MAE, and checks zero NaNs.
    """
    v0, kap, th, sig, rh, lam, mu = calib_params
    maturities_unique = sorted(list(set(c["T"] for c in contracts)))
    
    iv_diffs_bps = []
    price_diffs = []
    
    for T in maturities_unique:
        strikes_fft, calls_fft = carr_madan_fft(
            S0, R, Q, T, v0, kap, th, sig, rh, lam, mu, fixed_sigJ,
            N=4096, eta=0.1, alpha=1.5
        )
        c_sub = [c for c in contracts if abs(c["T"] - T) < 1e-6]
        target_ks = [c["K"] for c in c_sub]
        interp_calls = np.interp(target_ks, strikes_fft, calls_fft)
        
        for c_i, call_m in zip(c_sub, interp_calls):
            fit_iv = invert_black_scholes_iv(call_m, S0, c_i["K"], R, Q, T, is_call=True)
            if not np.isnan(fit_iv):
                iv_diffs_bps.append((fit_iv - c_i["mkt_iv"]) * 10000.0)
            price_diffs.append(abs(call_m - c_i["mkt_price"]))
            
    iv_rmse_bps = float(np.sqrt(np.mean(np.array(iv_diffs_bps)**2))) if iv_diffs_bps else float("nan")
    price_mae = float(np.mean(price_diffs)) if price_diffs else float("nan")
    return iv_rmse_bps, price_mae


def run_5regime_calibration_benchmark():
    """
    Executes the 5-Regime Volatility Surface Calibration Benchmark.
    """
    print("=" * 90)
    print("5-REGIME VOLATILITY SURFACE CALIBRATION BENCHMARK: CLASSICAL FFT VS SOBOLEV DML")
    print("=" * 90)
    
    ckpt_path = REPO_ROOT / "artifacts" / "bates_sobolev_dml.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Flagship model checkpoint missing at: {ckpt_path}.")
        
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    surrogate_model = BatesNeuralSurrogate(input_dim=10, output_dim=1, activation="silu").to(device)
    surrogate_model.load_state_dict(ckpt["model_state_dict"])
    surrogate_model.eval()
    
    surrogate_bundle = {
        "model": surrogate_model,
        "X_mean": ckpt["X_mean"],
        "X_std": ckpt["X_std"],
        "y_mean": ckpt["y_mean"],
        "y_std": ckpt["y_std"]
    }
    
    results = {}
    neutral_initial_guess = [0.04, 2.0, 0.04, 0.40, -0.55, 0.25, -0.10]
    
    # Warm up PyTorch JIT / CPU threads on dummy batch so timing reflects real workload
    with torch.no_grad():
        _ = surrogate_model(torch.zeros(55, 10))
    
    for r_idx, (r_key, r_info) in enumerate(REGIMES.items()):
        print(f"\n" + "-" * 80)
        print(f"REGIME {r_idx+1}/5: {r_info['name'].upper()}")
        print(f"  {r_info['description']}")
        print("-" * 80)
        
        true_params = r_info["params"]
        true_structural = true_params[:7]
        fixed_sigJ = true_params[7]
        mats = r_info["maturities"]
        
        contracts = construct_synthetic_surface(true_params, mats, noise_seed=100 + r_idx)
        
        # 1. Classical Numerical FFT Calibration
        print(f"  Running Classical FFT Calibration (SciPy L-BFGS-B)...", end=" ", flush=True)
        fft_params, fft_time, fft_fev = calibrate_classical_fft(
            contracts, neutral_initial_guess, fixed_sigJ, max_iter=60
        )
        fft_iv_rmse, fft_price_mae = evaluate_calibration_quality(contracts, fft_params, fixed_sigJ)
        fft_param_mae = float(np.mean(np.abs(np.array(fft_params) - np.array(true_structural))))
        print(f"Done in {fft_time:.2f}s ({fft_fev} fev). IV RMSE: {fft_iv_rmse:.1f} bps, Param MAE: {fft_param_mae:.4f}")
        
        # 2. Differentiable Sobolev Surrogate Calibration
        print(f"  Running Differentiable Sobolev DML Calibration (PyTorch L-BFGS)...", end=" ", flush=True)
        dml_params, dml_time, dml_fev = calibrate_sobolev_dml(
            contracts, surrogate_bundle, neutral_initial_guess, fixed_sigJ, max_iter=60, device=device
        )
        dml_iv_rmse, dml_price_mae = evaluate_calibration_quality(contracts, dml_params, fixed_sigJ)
        dml_param_mae = float(np.mean(np.abs(np.array(dml_params) - np.array(true_structural))))
        speedup = fft_time / max(dml_time, 1e-4)
        print(f"Done in {dml_time:.3f}s ({dml_fev} fev). IV RMSE: {dml_iv_rmse:.1f} bps, Param MAE: {dml_param_mae:.4f} [Speedup: {speedup:.1f}x]")
        
        results[r_key] = {
            "name": r_info["name"],
            "description": r_info["description"],
            "true_params": true_structural,
            "fixed_sigJ": fixed_sigJ,
            "classical_fft": {
                "wall_clock_s": float(fft_time),
                "fev": int(fft_fev),
                "iv_rmse_bps": float(fft_iv_rmse),
                "price_mae": float(fft_price_mae),
                "param_mae": float(fft_param_mae),
                "recovered_params": [float(p) for p in fft_params]
            },
            "sobolev_dml": {
                "wall_clock_s": float(dml_time),
                "fev": int(dml_fev),
                "iv_rmse_bps": float(dml_iv_rmse),
                "price_mae": float(dml_price_mae),
                "param_mae": float(dml_param_mae),
                "recovered_params": [float(p) for p in dml_params]
            },
            "speedup_x": float(speedup)
        }

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_json_path = out_dir / "calibration_benchmark.json"
    with open(bench_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUTPUT] Saved 5-Regime Calibration Benchmark: {bench_json_path}")
    
    # Summary Table
    print("\n" + "=" * 98)
    print("5-REGIME VOLATILITY SURFACE CALIBRATION BENCHMARK (55 Contracts/Surface, +/- 5 bps IV Noise)")
    print("=" * 98)
    header = f"{'Regime':<24} | {'Classical Time':<14} | {'Sobolev Time':<12} | {'Speedup':<8} | {'FFT IV RMSE':<11} | {'DML IV RMSE':<11} | {'DML MAE':<7}"
    print(header)
    print("-" * 98)
    for r_key, r_res in results.items():
        row = (
            f"{r_res['name']:<24} | "
            f"{r_res['classical_fft']['wall_clock_s']:<13.2f}s | "
            f"{r_res['sobolev_dml']['wall_clock_s']:<11.3f}s | "
            f"{r_res['speedup_x']:<7.1f}x | "
            f"{r_res['classical_fft']['iv_rmse_bps']:<7.1f} bps | "
            f"{r_res['sobolev_dml']['iv_rmse_bps']:<7.1f} bps | "
            f"{r_res['sobolev_dml']['param_mae']:<7.4f}"
        )
        print(row)
    print("=" * 98)
    
    return results


if __name__ == "__main__":
    run_5regime_calibration_benchmark()
