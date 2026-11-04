"""
Head-to-Head Calibration Benchmark: Classical Carr-Madan FFT vs. Vanilla MLP vs. Sobolev DML.
Author: Quantitative Volatility Surface Calibration Engineer
Repository: Amanatar/neural-bates-dml
"""

import os
import sys
import time
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.optimize import minimize

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from bates_fft import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_vega,
    invert_black_scholes_iv,
)
from generate_dataset import batch_bates_call_price
from model import BatesNeuralSurrogate
from live_calibration import fetch_and_snapshot_spy_chain

SNAPSHOT_PATH = os.path.join(REPO_ROOT, "data", "eval", "spy_options_snapshot.csv")
DML_CKPT_PATH = os.path.join(REPO_ROOT, "artifacts", "bates_dml_v1.pt")
BASELINE_CKPT_PATH = os.path.join(REPO_ROOT, "artifacts", "price_only_baseline.pt")
BENCHMARK_RESULTS_PATH = os.path.join(REPO_ROOT, "results", "spy_calibration_benchmark.json")

FIXED_SIGMA_J = 0.15  # Institutional baseline Merton jump vol


def compute_surface_metrics(df, pred_norm_prices, S0):
    """
    Computes rigorous financial metrics across the calibrated SPY option surface:
    - Dollar Price MAE & RMSE ($)
    - Implied Volatility RMSE & MAE (bps)
    - Static Arbitrage-Free Pass Rate (%) (Butterfly spread convexity)
    """
    market_norm_prices = df["normalized_call_price"].values
    market_ivs = df["market_iv"].values
    strikes = df["strike"].values
    Ts = df["T"].values
    rs = df["r"].values
    qs = df["q"].values
    
    # Dollar Pricing Errors
    model_dollar = pred_norm_prices * S0
    market_dollar = market_norm_prices * S0
    diff_dollar = np.abs(model_dollar - market_dollar)
    price_mae = float(np.mean(diff_dollar))
    price_rmse = float(np.sqrt(np.mean(diff_dollar**2)))
    
    # Implied Volatility Inversion & Error (basis points)
    iv_diffs = []
    inverted_count = 0
    for pn, k, T, r, q, miv in zip(pred_norm_prices, strikes, Ts, rs, qs, market_ivs):
        p_doll = pn * S0
        iv = invert_black_scholes_iv(p_doll, S0, k, r, q, T, is_call=True)
        if not np.isnan(iv):
            iv_diffs.append((iv - miv) * 10000.0)  # 1 bp = 0.0001
            inverted_count += 1
            
    if iv_diffs:
        iv_arr = np.array(iv_diffs)
        iv_rmse_bps = float(np.sqrt(np.mean(iv_arr**2)))
        iv_mae_bps = float(np.mean(np.abs(iv_arr)))
    else:
        iv_rmse_bps = 9999.0
        iv_mae_bps = 9999.0
        
    # Butterfly spread convexity check: C(K - dK) - 2*C(K) + C(K + dK) >= -1e-4
    total_butterfly = 0
    butterfly_violations = 0
    for exp, grp in df.groupby("expiry"):
        sorted_grp = grp.sort_values(by="strike")
        grp_prices = model_dollar[sorted_grp.index]
        if len(grp_prices) >= 3:
            for i in range(1, len(grp_prices) - 1):
                convexity = grp_prices[i-1] - 2.0 * grp_prices[i] + grp_prices[i+1]
                total_butterfly += 1
                if convexity < -1e-4:
                    butterfly_violations += 1
                    
    pass_rate_pct = 100.0 * (1.0 - butterfly_violations / max(total_butterfly, 1))
    
    return {
        "price_mae_dollars": price_mae,
        "price_rmse_dollars": price_rmse,
        "iv_rmse_bps": iv_rmse_bps,
        "iv_mae_bps": iv_mae_bps,
        "arbitrage_pass_rate_pct": pass_rate_pct,
        "inverted_contracts_count": inverted_count,
        "total_contracts_count": len(df)
    }


def calibrate_classical_fft(df, bounds, x0, weights):
    """
    Method 1: Classical Numerical Carr-Madan FFT calibration using SciPy L-BFGS-B.
    Evaluates analytical FFT integral across all contracts at every optimizer step.
    """
    S0 = float(df["spot"].iloc[0])
    k = df["log_moneyness"].values
    T = df["T"].values
    r = df["r"].values
    q = df["q"].values
    y_mkt = df["normalized_call_price"].values
    B = len(k)
    
    n_evals = 0
    
    def objective_fft(params):
        nonlocal n_evals
        n_evals += 1
        v0, kap, th, sig, rho, lam, mu = params
        try:
            preds = batch_bates_call_price(
                k, T, r, q,
                np.full(B, v0), np.full(B, kap), np.full(B, th),
                np.full(B, sig), np.full(B, rho), np.full(B, lam),
                np.full(B, mu), np.full(B, FIXED_SIGMA_J)
            )
            loss = np.mean(weights * (preds - y_mkt)**2)
            return float(loss)
        except Exception:
            return 1e6

    print("\n--- Method 1: Classical Numerical Carr-Madan FFT (SciPy L-BFGS-B) ---")
    start_time = time.time()
    res = minimize(
        objective_fft,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 60, "ftol": 1e-12, "gtol": 1e-8, "eps": 1e-4}
    )
    elapsed_time = time.time() - start_time
    calib_ms = elapsed_time * 1000.0
    
    p_opt = res.x
    final_preds = batch_bates_call_price(
        k, T, r, q,
        np.full(B, p_opt[0]), np.full(B, p_opt[1]), np.full(B, p_opt[2]),
        np.full(B, p_opt[3]), np.full(B, p_opt[4]), np.full(B, p_opt[5]),
        np.full(B, p_opt[6]), np.full(B, FIXED_SIGMA_J)
    )
    
    metrics = compute_surface_metrics(df, final_preds, S0)
    
    return {
        "method": "Classical Carr-Madan FFT (SciPy L-BFGS-B)",
        "calibration_time_s": float(elapsed_time),
        "calibration_time_ms": float(calib_ms),
        "iterations": int(res.nit),
        "function_evaluations": int(n_evals),
        "parameters": {
            "v0": float(p_opt[0]),
            "kappa": float(p_opt[1]),
            "theta": float(p_opt[2]),
            "sigma_v": float(p_opt[3]),
            "rho": float(p_opt[4]),
            "lambda_j": float(p_opt[5]),
            "mu_j": float(p_opt[6]),
            "sigma_j": FIXED_SIGMA_J
        },
        **metrics
    }


def calibrate_neural_surrogate(df, ckpt_path, method_label, bounds_low, bounds_high, weights_torch, n_restarts=3, max_iters=150):
    """
    Calibrates using a Neural Surrogate (Method 2: Vanilla MLP or Method 3: Sobolev DML)
    via multi-start PyTorch Adam optimization.
    """
    device = torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = BatesNeuralSurrogate(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        activation=checkpoint.get("activation", "silu")
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    X_mean = checkpoint["X_mean"].to(device)
    X_std = checkpoint["X_std"].to(device)
    y_mean = checkpoint["y_mean"].to(device)
    y_std = checkpoint["y_std"].to(device)
    
    M = len(df)
    S0 = float(df["spot"].iloc[0])
    k_arr = torch.tensor(df["log_moneyness"].values, dtype=torch.float32, device=device)
    T_arr = torch.tensor(df["T"].values, dtype=torch.float32, device=device)
    r_arr = torch.tensor(df["r"].values, dtype=torch.float32, device=device)
    y_target = torch.tensor(df["normalized_call_price"].values, dtype=torch.float32, device=device)
    
    print(f"\n--- {method_label} ---")
    start_time = time.time()
    best_loss = float("inf")
    best_params = None
    total_steps = 0
    
    for restart in range(n_restarts):
        if restart == 0:
            raw_params = torch.zeros(7, device=device, requires_grad=True)
        else:
            raw_params = (torch.rand(7, device=device) * 4.0 - 2.0).requires_grad_(True)
            
        optimizer = optim.Adam([raw_params], lr=0.08)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-3)
        
        for it in range(max_iters):
            total_steps += 1
            optimizer.zero_grad()
            phys_theta = bounds_low + torch.sigmoid(raw_params) * (bounds_high - bounds_low)
            
            param_block = phys_theta.unsqueeze(0).expand(M, -1)
            fixed_block = torch.stack([k_arr, T_arr, r_arr], dim=1)
            X_batch = torch.cat([fixed_block, param_block], dim=1)
            
            pred_scaled = model((X_batch - X_mean) / X_std).squeeze(-1)
            pred_unit = pred_scaled * y_std + y_mean
            
            loss = torch.mean(weights_torch * (pred_unit - y_target)**2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_val = float(loss.item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = phys_theta.detach().clone()
                
    elapsed_time = time.time() - start_time
    calib_ms = elapsed_time * 1000.0
    
    # Model surface evaluation
    with torch.no_grad():
        param_block = best_params.unsqueeze(0).expand(M, -1)
        fixed_block = torch.stack([k_arr, T_arr, r_arr], dim=1)
        X_batch = torch.cat([fixed_block, param_block], dim=1)
        pred_scaled = model((X_batch - X_mean) / X_std).squeeze(-1)
        pred_norm_prices = (pred_scaled * y_std + y_mean).cpu().numpy()
        
    p_opt = best_params.cpu().numpy()
    metrics = compute_surface_metrics(df, pred_norm_prices, S0)
    
    return {
        "method": method_label,
        "calibration_time_s": float(elapsed_time),
        "calibration_time_ms": float(calib_ms),
        "iterations": total_steps,
        "function_evaluations": total_steps,
        "parameters": {
            "v0": float(p_opt[0]),
            "kappa": float(p_opt[1]),
            "theta": float(p_opt[2]),
            "sigma_v": float(p_opt[3]),
            "rho": float(p_opt[4]),
            "lambda_j": float(p_opt[5]),
            "mu_j": float(p_opt[6]),
            "sigma_j": FIXED_SIGMA_J
        },
        **metrics
    }


def run_micro_benchmark(df, ckpt_path):
    """
    Micro-benchmarks the raw per-evaluation latency of pricing the entire 151-contract
    surface: Classical Carr-Madan FFT vs Neural Surrogate.
    """
    k = df["log_moneyness"].values
    T = df["T"].values
    r = df["r"].values
    q = df["q"].values
    B = len(k)
    p = [0.035, 2.0, 0.035, 0.40, -0.60, 0.20, -0.10]
    
    # 1. FFT Surface Latency
    t0 = time.time()
    n_fft = 30
    for _ in range(n_fft):
        batch_bates_call_price(
            k, T, r, q,
            np.full(B, p[0]), np.full(B, p[1]), np.full(B, p[2]),
            np.full(B, p[3]), np.full(B, p[4]), np.full(B, p[5]),
            np.full(B, p[6]), np.full(B, FIXED_SIGMA_J)
        )
    ms_fft_surface = (time.time() - t0) / n_fft * 1000.0
    
    # 2. Neural Surrogate Latency
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BatesNeuralSurrogate(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        activation=checkpoint.get("activation", "silu")
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    dummy_x = torch.zeros(B, 10)
    t0 = time.time()
    n_net = 100
    with torch.no_grad():
        for _ in range(n_net):
            model(dummy_x)
    ms_net_surface = (time.time() - t0) / n_net * 1000.0
    
    speedup = ms_fft_surface / max(ms_net_surface, 1e-6)
    
    return {
        "fft_surface_eval_ms": ms_fft_surface,
        "neural_surface_eval_ms": ms_net_surface,
        "per_evaluation_speedup_factor": speedup
    }


def main():
    print("=" * 80)
    print("HEAD-TO-HEAD CALIBRATION BENCHMARK: CARR-MADAN FFT VS PRICE-ONLY MLP VS SOBOLEV DML")
    print("=" * 80)
    
    # 1. Load real multi-expiry SPY option surface
    df = fetch_and_snapshot_spy_chain(csv_path=SNAPSHOT_PATH, force_live=False)
    M = len(df)
    S0 = float(df["spot"].iloc[0])
    print(f"Surface Snapshot: {M} contracts across {df['expiry'].nunique()} maturities | S0 = ${S0:.2f}")
    
    # Parameter bounds: [v0, kappa, theta, sigma_v, rho, lambda_j, mu_j]
    bounds = [
        (0.01, 0.22),   # v0: spot vol in [10%, 46.9%]
        (0.50, 4.50),   # kappa: mean reversion rate
        (0.01, 0.20),   # theta: long-term variance
        (0.10, 0.85),   # sigma_v: vol of vol
        (-0.95, -0.10), # rho: leverage skew correlation
        (0.00, 1.50),   # lambda_j: jump arrival rate
        (-0.25, 0.05)   # mu_j: mean jump size
    ]
    bounds_low_t = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
    bounds_high_t = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
    x0 = [0.035, 2.0, 0.035, 0.40, -0.60, 0.20, -0.10]
    
    # Vega weights for weighted pricing error
    vegas = df["bs_vega_normalized"].values
    weights_np = 1.0 / np.clip(vegas, 1e-3, None)**2
    weights_np = weights_np / weights_np.mean()
    weights_torch = torch.tensor(weights_np, dtype=torch.float32)
    
    # 2. Method 1: Classical Carr-Madan FFT (SciPy L-BFGS-B)
    fft_results = calibrate_classical_fft(df, bounds, x0, weights_np)
    
    # 3. Method 2: Vanilla Price-Only MLP Surrogate
    baseline_results = calibrate_neural_surrogate(
        df, BASELINE_CKPT_PATH, "Vanilla Price-Only MLP (lambda=0.0)",
        bounds_low_t, bounds_high_t, weights_torch, n_restarts=3, max_iters=150
    )
    
    # 4. Method 3: Sobolev DML Surrogate
    dml_results = calibrate_neural_surrogate(
        df, DML_CKPT_PATH, "Sobolev DML Surrogate (lambda=1.0)",
        bounds_low_t, bounds_high_t, weights_torch, n_restarts=3, max_iters=150
    )
    
    # 5. Raw per-evaluation micro-benchmark
    micro = run_micro_benchmark(df, DML_CKPT_PATH)
    
    # Compute relative speedups
    fft_time = fft_results["calibration_time_s"]
    base_speedup = fft_time / max(baseline_results["calibration_time_s"], 1e-6)
    dml_speedup = fft_time / max(dml_results["calibration_time_s"], 1e-6)
    
    baseline_results["speedup_vs_fft"] = base_speedup
    dml_results["speedup_vs_fft"] = dml_speedup
    
    # Save benchmark JSON
    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spy_spot_price": S0,
        "total_contracts": M,
        "maturities_count": int(df["expiry"].nunique()),
        "micro_benchmark": micro,
        "methods": {
            "classical_carr_madan_fft": fft_results,
            "vanilla_price_only_mlp": baseline_results,
            "sobolev_dml_surrogate": dml_results
        }
    }
    
    os.makedirs(os.path.dirname(BENCHMARK_RESULTS_PATH), exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OUTPUT] Calibration Benchmark Matrix saved to: {BENCHMARK_RESULTS_PATH}")
    
    # Print formatted calibration comparison table
    print("\n" + "=" * 105)
    print("HEAD-TO-HEAD CALIBRATION BENCHMARK RESULTS TABLE (Real SPY Surface: 151 Contracts)")
    print("=" * 105)
    header = f"{'Method':<36} | {'Calib Time':<12} | {'Speedup':<8} | {'Price MAE':<10} | {'IV RMSE':<10} | {'Arbitrage Pass %':<16}"
    print(header)
    print("-" * 105)
    
    row_fft = f"{'Classical FFT (L-BFGS-B)':<36} | {fft_results['calibration_time_s']:>8.2f} s    | {'1.0x':<8} | ${fft_results['price_mae_dollars']:<9.4f} | {fft_results['iv_rmse_bps']:>7.1f} bps | {fft_results['arbitrage_pass_rate_pct']:>13.1f}%"
    row_base = f"{'Vanilla Price-Only MLP (lambda=0)':<36} | {baseline_results['calibration_time_s']:>8.2f} s    | {base_speedup:>6.1f}x  | ${baseline_results['price_mae_dollars']:<9.4f} | {baseline_results['iv_rmse_bps']:>7.1f} bps | {baseline_results['arbitrage_pass_rate_pct']:>13.1f}%"
    row_dml = f"{'Sobolev DML Surrogate (lambda=1.0)':<36} | {dml_results['calibration_time_s']:>8.2f} s    | {dml_speedup:>6.1f}x  | ${dml_results['price_mae_dollars']:<9.4f} | {dml_results['iv_rmse_bps']:>7.1f} bps | {dml_results['arbitrage_pass_rate_pct']:>13.1f}%"
    
    print(row_fft)
    print(row_base)
    print(row_dml)
    print("=" * 105)
    
    # Print Recovered Parameters Table
    print("\n" + "=" * 105)
    print("RECOVERED BATES SVJD PARAMETERS ON SPY (S0 = $754.35)")
    print("=" * 105)
    p_header = f"{'Parameter':<10} | {'Meaning / Institutional Role':<38} | {'Classical FFT':<14} | {'Vanilla MLP':<14} | {'Sobolev DML':<14}"
    print(p_header)
    print("-" * 105)
    
    keys = [
        ("v0", "Initial Variance (Spot Vol %)", True),
        ("kappa", "Mean-Reversion Speed", False),
        ("theta", "Long-Run Variance (Long Vol %)", True),
        ("sigma_v", "Volatility of Variance", False),
        ("rho", "Spot-Vol Correlation (Skew)", False),
        ("lambda_j", "Jump Arrival Rate (Freq/Year)", False),
        ("mu_j", "Mean Jump Size (Crash Drift)", False),
        ("sigma_j", "Jump Volatility (Fixed Merton)", False),
    ]
    
    for k, desc, is_vol in keys:
        v_fft = fft_results["parameters"][k]
        v_base = baseline_results["parameters"][k]
        v_dml = dml_results["parameters"][k]
        
        if is_vol:
            s_fft = f"{v_fft:.4f} ({math.sqrt(v_fft)*100:.1f}%)"
            s_base = f"{v_base:.4f} ({math.sqrt(v_base)*100:.1f}%)"
            s_dml = f"{v_dml:.4f} ({math.sqrt(v_dml)*100:.1f}%)"
        else:
            s_fft = f"{v_fft:.4f}"
            s_base = f"{v_base:.4f}"
            s_dml = f"{v_dml:.4f}"
            
        print(f"{k:<10} | {desc:<38} | {s_fft:<14} | {s_base:<14} | {s_dml:<14}")
        
    print("=" * 105)
    print(f"Per-Surface Forward Evaluation: FFT {micro['fft_surface_eval_ms']:.2f} ms vs Neural {micro['neural_surface_eval_ms']:.3f} ms -> {micro['per_evaluation_speedup_factor']:.0f}x Speedup per Surface")
    print("=" * 105)


if __name__ == "__main__":
    main()
