"""
Benchmark Ablation: Sobolev Differential Machine Learning vs. Price-Only MLP.
Repository: Amanatar/neural-bates-dml
Author: Principal Quant & ML Systems Engineer

Generates deterministic Bates SVJD Carr-Madan option pricing and analytical
sensitivity tuples across the 10-D parameter space:
  (v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j, K/S, T)
for training sizes N in {512, 2048, 8192} and held-out test size N=2048.

Trains matched-capacity networks:
  1. Baseline: Price-Only MLP (lambda_greeks = 0.0)
  2. Flagship: Twin-Net Sobolev DML (lambda_greeks = 1.0)
to full convergence with CosineAnnealingLR and gradient clipping.

Saves:
  - artifacts/bates_sobolev_dml.pt (< 2 MB)
  - results/ablation_metrics.json
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import scipy.stats.qmc as qmc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import BatesNeuralSurrogate

FEATURE_NAMES = [
    "v0", "kappa", "theta", "sigma_v", "rho", "lambda_j", "mu_j", "sigma_j", "moneyness", "T"
]

PARAM_BOUNDS = {
    "v0": (0.01, 0.16),         # Spot vol ~10% to 40%
    "kappa": (0.6, 4.0),        # Mean reversion speed
    "theta": (0.01, 0.14),       # Long-run vol ~10% to 37.4%
    "sigma_v": (0.12, 0.75),    # Vol-of-vol
    "rho": (-0.90, -0.15),      # Leverage correlation
    "lambda_j": (0.00, 1.00),   # Jump intensity
    "mu_j": (-0.22, 0.02),      # Mean jump size
    "sigma_j": (0.05, 0.25),    # Jump volatility
    "moneyness": (0.78, 1.22),  # K / S0
    "T": (0.05, 1.20)           # Maturity in years
}


def bates_call_and_jacobians_batch(x_tensor, r=0.04, q=0.0, alpha=1.5, N=256, upper_v=100.0):
    """
    Computes Bates SVJD call option prices and exact 10-D analytical Jacobians
    via PyTorch autograd through complex Simpson quadrature.
    """
    x_in = x_tensor.clone().detach().requires_grad_(True)
    v0, kappa, theta, sig_v, rho, lam, muJ, sigJ, m, T = [x_in[:, i:i+1] for i in range(10)]
    
    v = torch.linspace(1e-4, upper_v, N, dtype=torch.float64).unsqueeze(0)
    dv = v[0, 1] - v[0, 0]
    u = torch.complex(v, torch.full_like(v, -(alpha + 1.0)))
    i1 = torch.complex(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    
    beta = kappa - i1 * rho * sig_v * u
    sig2 = sig_v ** 2
    d = torch.sqrt(beta**2 + (u**2 + i1 * u) * sig2)
    d = torch.where(d.real < 0, -d, d)
    
    g = (beta - d) / (beta + d)
    exp_m_dT = torch.exp(-d * T)
    log_term = torch.log((1.0 - g * exp_m_dT) / (1.0 - g))
    C_H = (r - q) * i1 * u * T + (kappa * theta / sig2) * ((beta - d) * T - 2.0 * log_term)
    D_H = ((beta - d) / sig2) * ((1.0 - exp_m_dT) / (1.0 - g * exp_m_dT))
    cf_h = torch.exp(C_H + D_H * v0)
    
    omega = -lam * (torch.exp(muJ + 0.5 * sigJ**2) - 1.0)
    cf_j = torch.exp(lam * T * (torch.exp(i1 * u * muJ - 0.5 * (u**2) * (sigJ**2)) - 1.0) + i1 * u * omega * T)
    cf = cf_h * cf_j
    
    denom = torch.complex(alpha**2 + alpha - v**2, (2.0 * alpha + 1.0) * v)
    psi = torch.exp(-r * T) * cf / denom
    
    k = torch.log(m)
    w = torch.ones((1, N), dtype=torch.float64)
    w[0, 1:-1:2] = 4.0
    w[0, 2:-1:2] = 2.0
    
    integrand = (torch.exp(-i1 * v * k) * psi).real
    integral = (dv / 3.0) * torch.sum(w * integrand, dim=1, keepdim=True)
    prices = torch.clamp((torch.exp(-alpha * k) / torch.pi) * integral, min=0.0)
    
    prices.sum().backward()
    jacobians = x_in.grad.detach()
    
    return prices.detach(), jacobians


def generate_sobol_10d_dataset(total_samples=10240, seed=42):
    """
    Generates deterministic quasi-random Sobol 10-D parameter tuples and
    computes exact prices, sensitivities, Delta, and Vega.
    """
    print(f"\n[DATA] Generating {total_samples} deterministic 10-D Sobol samples (seed={seed})...")
    l_bounds = [PARAM_BOUNDS[name][0] for name in FEATURE_NAMES]
    u_bounds = [PARAM_BOUNDS[name][1] for name in FEATURE_NAMES]
    
    sampler = qmc.Sobol(d=10, scramble=True, seed=seed)
    raw = sampler.random(total_samples)
    scaled_x = qmc.scale(raw, l_bounds, u_bounds)
    
    # Filter soft Feller condition to avoid numerical instability
    feller = 2.0 * scaled_x[:, 1] * scaled_x[:, 2] / (scaled_x[:, 3]**2)
    valid_mask = feller >= 0.15
    valid_x = scaled_x[valid_mask]
    
    # If filtered points fell short, sample additional to reach total_samples
    if len(valid_x) < total_samples:
        extra = sampler.random(int((total_samples - len(valid_x)) * 1.5) + 100)
        extra_scaled = qmc.scale(extra, l_bounds, u_bounds)
        extra_feller = 2.0 * extra_scaled[:, 1] * extra_scaled[:, 2] / (extra_scaled[:, 3]**2)
        valid_x = np.vstack([valid_x, extra_scaled[extra_feller >= 0.15]])
        
    X_data = valid_x[:total_samples].astype(np.float64)
    
    # Vectorized batch evaluation
    chunk_size = 2048
    all_prices = []
    all_jacs = []
    
    t0 = time.time()
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        x_chunk = torch.tensor(X_data[start:end], dtype=torch.float64)
        p_chunk, j_chunk = bates_call_and_jacobians_batch(x_chunk)
        all_prices.append(p_chunk.numpy())
        all_jacs.append(j_chunk.numpy())
        print(f"  Processed {end}/{total_samples} contracts in {time.time()-t0:.2f}s")
        
    prices = np.vstack(all_prices).astype(np.float32)
    jacobians = np.vstack(all_jacs).astype(np.float32)
    X_float32 = X_data.astype(np.float32)
    
    # Spot Delta: Delta = C - m * (dC/dm)
    # where m is FEATURE_NAMES index 8
    m_vals = X_float32[:, 8:9]
    dC_dm = jacobians[:, 8:9]
    deltas = prices - m_vals * dC_dm
    
    # Variance Vega: dC / dv0 (index 0)
    vegas = jacobians[:, 0:1]
    
    assert np.all(np.isfinite(prices)), "Non-finite values in prices"
    assert np.all(np.isfinite(jacobians)), "Non-finite values in jacobians"
    assert np.all(np.isfinite(deltas)), "Non-finite values in deltas"
    assert np.all(np.isfinite(vegas)), "Non-finite values in vegas"
    
    print(f"[DATA] Successfully computed {total_samples} samples. Zero NaNs verified.")
    return X_float32, prices, jacobians, deltas, vegas


def train_model(
    X_tr, y_tr, j_tr,
    X_mean, X_std, y_mean, y_std,
    lambda_greeks=1.0,
    epochs=80,
    batch_size=256,
    lr=2e-3,
    device=torch.device("cpu"),
    seed=42
):
    """
    Trains a BatesNeuralSurrogate model with matched capacity.
    If lambda_greeks == 0.0, trains standard Price-Only MLP.
    If lambda_greeks == 1.0, trains Twin-Net Sobolev DML.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Standardize data
    X_tr_s = (X_tr - X_mean) / X_std
    y_tr_s = (y_tr - y_mean) / y_std
    j_tr_s = j_tr * (X_std / y_std)
    
    dataset = TensorDataset(X_tr_s, y_tr_s, j_tr_s)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(X_tr_s)), shuffle=True)
    
    model = BatesNeuralSurrogate(input_dim=10, output_dim=1, activation="silu").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for b_X, b_y, b_j in loader:
            optimizer.zero_grad()
            b_X_in = b_X.clone().requires_grad_(True)
            pred_y = model(b_X_in)
            loss_price = nn.functional.mse_loss(pred_y, b_y)
            
            if lambda_greeks > 0.0:
                pred_grad = torch.autograd.grad(
                    outputs=pred_y.sum(),
                    inputs=b_X_in,
                    create_graph=True,
                    retain_graph=True
                )[0]
                loss_greek = nn.functional.mse_loss(pred_grad, b_j)
                total_loss = loss_price + lambda_greeks * loss_greek
            else:
                total_loss = loss_price
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        
    train_time = time.time() - t0
    model.eval()
    return model, train_time


def evaluate_model(model, X_test, y_test, j_test, delta_test, vega_test, X_mean, X_std, y_mean, y_std, device=torch.device("cpu")):
    """
    Evaluates Price, Delta, and Vega RMSE/MAE on held-out test set,
    and measures CPU batched inference latency.
    """
    model.eval()
    X_t = X_test.to(device)
    y_t = y_test.to(device)
    
    X_s = ((X_t - X_mean) / X_std).requires_grad_(True)
    pred_y_s = model(X_s)
    
    # Exact autograd sensitivities
    pred_grad_s = torch.autograd.grad(pred_y_s.sum(), X_s, create_graph=False)[0]
    
    pred_y = (pred_y_s * y_std + y_mean).detach()
    pred_j = (pred_grad_s * (y_std / X_std)).detach()
    
    # Unpack Spot Delta: pred_delta = pred_y - m * (dC/dm)
    m_test = X_t[:, 8:9]
    pred_delta = pred_y - m_test * pred_j[:, 8:9]
    pred_vega = pred_j[:, 0:1] # dC / dv0
    
    # Compute error metrics
    price_err = (pred_y - y_t).numpy()
    delta_err = (pred_delta - delta_test).numpy()
    vega_err = (pred_vega - vega_test).numpy()
    
    price_rmse = float(np.sqrt(np.mean(price_err**2)))
    price_mae = float(np.mean(np.abs(price_err)))
    delta_rmse = float(np.sqrt(np.mean(delta_err**2)))
    delta_mae = float(np.mean(np.abs(delta_err)))
    vega_rmse = float(np.sqrt(np.mean(vega_err**2)))
    vega_mae = float(np.mean(np.abs(vega_err)))
    
    zero_nans = bool(
        torch.isfinite(pred_y).all().item() and
        torch.isfinite(pred_delta).all().item() and
        torch.isfinite(pred_vega).all().item()
    )
    
    # CPU Batched Inference Latency
    latency_128_ms = 0.0
    latency_1024_ms = 0.0
    with torch.no_grad():
        for b_size, target_key in [(128, "lat128"), (1024, "lat1024")]:
            dummy = torch.randn(b_size, 10, device=device)
            for _ in range(10): # warmup
                _ = model(dummy)
            t0 = time.perf_counter()
            for _ in range(50):
                _ = model(dummy)
            t1 = time.perf_counter()
            ms = ((t1 - t0) / 50.0) * 1000.0
            if b_size == 128:
                latency_128_ms = float(ms)
            else:
                latency_1024_ms = float(ms)
                
    return {
        "price_rmse": price_rmse,
        "price_mae": price_mae,
        "delta_rmse": delta_rmse,
        "delta_mae": delta_mae,
        "vega_rmse": vega_rmse,
        "vega_mae": vega_mae,
        "batch_latency_128_ms": latency_128_ms,
        "batch_latency_1024_ms": latency_1024_ms,
        "zero_nans": zero_nans
    }


def run_benchmark_ablation():
    """
    Executes the full Sample-Efficiency and Greek Sensitivity Ablation Study.
    """
    print("=" * 80)
    print("BATES SVJD NEURAL SURROGATE: SAMPLE-EFFICIENCY & GREEK SENSITIVITY ABLATION")
    print("=" * 80)
    
    device = torch.device("cpu")
    N_test = 2048
    train_sizes = [512, 2048, 8192]
    max_train = max(train_sizes)
    total_samples = max_train + N_test
    
    # Generate full dataset
    X_all, y_all, j_all, deltas_all, vegas_all = generate_sobol_10d_dataset(total_samples=total_samples, seed=42)
    
    # Split into train pool (8192) and held-out test set (2048)
    X_train_pool = torch.tensor(X_all[:max_train], dtype=torch.float32)
    y_train_pool = torch.tensor(y_all[:max_train], dtype=torch.float32)
    j_train_pool = torch.tensor(j_all[:max_train], dtype=torch.float32)
    
    X_test = torch.tensor(X_all[max_train:], dtype=torch.float32)
    y_test = torch.tensor(y_all[max_train:], dtype=torch.float32)
    j_test = torch.tensor(j_all[max_train:], dtype=torch.float32)
    delta_test = torch.tensor(deltas_all[max_train:], dtype=torch.float32)
    vega_test = torch.tensor(vegas_all[max_train:], dtype=torch.float32)
    
    results = {}
    flagship_model_8192 = None
    stats_8192 = None
    
    for n in train_sizes:
        print(f"\n" + "-" * 70)
        print(f"EVALUATING TRAINING SIZE N = {n}")
        print("-" * 70)
        
        X_tr = X_train_pool[:n]
        y_tr = y_train_pool[:n]
        j_tr = j_train_pool[:n]
        
        # Compute standardization stats for this subset
        X_mean = X_tr.mean(dim=0, keepdim=True)
        X_std = torch.clamp(X_tr.std(dim=0, keepdim=True), min=1e-5)
        y_mean = y_tr.mean(dim=0, keepdim=True)
        y_std = torch.clamp(y_tr.std(dim=0, keepdim=True), min=1e-5)
        
        # 1. Baseline: Price-Only MLP
        print(f"  Training Baseline (Price-Only MLP, lambda=0.0)...", end=" ", flush=True)
        base_model, base_time = train_model(
            X_tr, y_tr, j_tr, X_mean, X_std, y_mean, y_std,
            lambda_greeks=0.0, epochs=80, batch_size=256, device=device, seed=42
        )
        base_metrics = evaluate_model(base_model, X_test, y_test, j_test, delta_test, vega_test, X_mean, X_std, y_mean, y_std, device=device)
        base_metrics["training_time_s"] = float(base_time)
        print(f"Done ({base_time:.2f}s). Price RMSE={base_metrics['price_rmse']:.6f}, Delta RMSE={base_metrics['delta_rmse']:.6f}")
        
        # 2. Flagship: Twin-Net Sobolev DML
        print(f"  Training Flagship (Twin-Net Sobolev DML, lambda=1.0)...", end=" ", flush=True)
        dml_model, dml_time = train_model(
            X_tr, y_tr, j_tr, X_mean, X_std, y_mean, y_std,
            lambda_greeks=1.0, epochs=80, batch_size=256, device=device, seed=42
        )
        dml_metrics = evaluate_model(dml_model, X_test, y_test, j_test, delta_test, vega_test, X_mean, X_std, y_mean, y_std, device=device)
        dml_metrics["training_time_s"] = float(dml_time)
        print(f"Done ({dml_time:.2f}s). Price RMSE={dml_metrics['price_rmse']:.6f}, Delta RMSE={dml_metrics['delta_rmse']:.6f}")
        
        delta_gain_pct = (1.0 - dml_metrics['delta_rmse'] / base_metrics['delta_rmse']) * 100.0
        vega_gain_pct = (1.0 - dml_metrics['vega_rmse'] / base_metrics['vega_rmse']) * 100.0
        print(f"  --> Greek Error Reduction: Delta RMSE {delta_gain_pct:+.1f}%, Vega RMSE {vega_gain_pct:+.1f}%")
        
        results[str(n)] = {
            "price_only": base_metrics,
            "sobolev_dml": dml_metrics
        }
        
        if n == 8192:
            flagship_model_8192 = dml_model
            stats_8192 = {
                "X_mean": X_mean,
                "X_std": X_std,
                "y_mean": y_mean,
                "y_std": y_std
            }

    # Save ablation metrics JSON
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ablation_json_path = results_dir / "ablation_metrics.json"
    with open(ablation_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUTPUT] Saved ablation metrics: {ablation_json_path}")
    
    # Save compact flagship model checkpoint (< 2 MB)
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / "bates_sobolev_dml.pt"
    
    torch.save({
        "model_state_dict": flagship_model_8192.state_dict(),
        "feature_names": FEATURE_NAMES,
        "param_bounds": PARAM_BOUNDS,
        "X_mean": stats_8192["X_mean"],
        "X_std": stats_8192["X_std"],
        "y_mean": stats_8192["y_mean"],
        "y_std": stats_8192["y_std"],
        "input_dim": 10,
        "output_dim": 1,
        "activation": "silu",
        "lambda_greeks": 1.0,
        "n_train": 8192,
        "metrics": results["8192"]["sobolev_dml"]
    }, ckpt_path)
    
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"[OUTPUT] Saved Flagship Checkpoint: {ckpt_path} ({ckpt_size_mb:.2f} MB)")
    assert ckpt_size_mb < 2.0, f"Checkpoint size {ckpt_size_mb:.2f} MB exceeds 2.0 MB threshold!"
    
    # Print formatted Summary Table
    print("\n" + "=" * 90)
    print("SAMPLE-EFFICIENCY & GREEK SENSITIVITY ABLATION MATRIX (N_test = 2048)")
    print("=" * 90)
    print(f"{'N_train':<8} | {'Model':<16} | {'Price RMSE':<12} | {'Delta RMSE':<12} | {'Vega RMSE':<12} | {'128-b Lat (ms)':<14} | {'Train (s)':<9}")
    print("-" * 90)
    for n in train_sizes:
        n_str = str(n)
        for m_key, m_name in [("price_only", "Price-Only MLP"), ("sobolev_dml", "Sobolev DML")]:
            m = results[n_str][m_key]
            print(f"{n:<8} | {m_name:<16} | {m['price_rmse']:<12.6f} | {m['delta_rmse']:<12.6f} | {m['vega_rmse']:<12.6f} | {m['batch_latency_128_ms']:<14.3f} | {m['training_time_s']:<9.2f}")
        print("-" * 90)
    print("=" * 90)
    
    return results


if __name__ == "__main__":
    run_benchmark_ablation()
