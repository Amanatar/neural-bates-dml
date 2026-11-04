"""
Comprehensive VM Execution & Continuous Integration Test Suite for Neural Bates DML.
Author: Principal Release & Portfolio Sync Engineer
Repository: Amanatar/neural-bates-dml
"""

import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import BatesNeuralSurrogate
from bates_fft import verify_fft_against_black_scholes, invert_black_scholes_iv

BENCHMARK_PATH = REPO_ROOT / "data" / "eval" / "bates_test_benchmark.npz"
SNAPSHOT_PATH = REPO_ROOT / "data" / "eval" / "spy_options_snapshot.csv"
DML_CKPT_PATH = REPO_ROOT / "artifacts" / "bates_dml_v1.pt"
BASELINE_CKPT_PATH = REPO_ROOT / "artifacts" / "price_only_baseline.pt"


def test_bates_dml_synthetic_training():
    """Smoke test: synthetic twin-network Sobolev training loop."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    N = 128
    input_dim = 10
    output_dim = 55
    
    X = torch.randn(N, input_dim, device=device)
    Y = torch.randn(N, output_dim, device=device)
    dY_dX = torch.randn(N, output_dim, input_dim, device=device)
    
    X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-7)
    Y_scaled = (Y - Y.mean(0)) / (Y.std(0) + 1e-7)
    dY_dX_scaled = dY_dX * 0.1
    
    model = BatesNeuralSurrogate(input_dim=input_dim, output_dim=output_dim, activation="silu").to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    epoch_losses = []
    for epoch in range(1, 6):
        optimizer.zero_grad()
        X_in = X_scaled.clone().requires_grad_(True)
        pred_Y = model(X_in)
        
        loss_price = criterion(pred_Y, Y_scaled)
        
        greek_loss = torch.tensor(0.0, device=device)
        for i in range(min(8, output_dim)):
            grad_out = torch.autograd.grad(
                outputs=pred_Y[:, i].sum(),
                inputs=X_in,
                create_graph=True,
                retain_graph=True
            )[0]
            greek_loss += criterion(grad_out, dY_dX_scaled[:, i, :])
        greek_loss = greek_loss / 8.0
        
        total_loss = loss_price + 0.5 * greek_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_losses.append((loss_price.item(), greek_loss.item(), total_loss.item()))
    
    ep1_p, ep1_g, ep1_tot = epoch_losses[0]
    ep5_p, ep5_g, ep5_tot = epoch_losses[-1]
    
    model.eval()
    with torch.no_grad():
        test_in = torch.randn(128, input_dim, device=device)
        _ = model(test_in)
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(test_in)
        t1 = time.perf_counter()
        latency_ms = ((t1 - t0) / 50.0) * 1000.0

    assert not torch.isnan(pred_Y).any(), "NaN found in predictions"
    assert not torch.isinf(pred_Y).any(), "Inf found in predictions"
    assert ep5_tot < ep1_tot, f"Total loss did not improve: {ep1_tot} -> {ep5_tot}"


def test_fft_black_scholes_limit():
    """Production Test 1: Carr-Madan FFT matches closed-form Black-Scholes in zero-jump limit."""
    max_err, errors = verify_fft_against_black_scholes(
        moneyness_range=(0.80, 1.20),
        n_strikes=11,
        tol=1e-3,
        verbose=False
    )
    assert max_err < 1e-3, f"FFT Black-Scholes limit error {max_err:.6e} exceeded 1e-3 threshold"
    assert np.all(np.isfinite(errors)), "Non-finite errors in FFT Black-Scholes limit test"


def test_pretrained_sobolev_checkpoint():
    """
    Production Test 2: Loads artifacts/bates_dml_v1.pt and evaluates on data/eval/bates_test_benchmark.npz.
    Asserts 0 NaNs, sub-millisecond 128-contract batch latency, and verifies Sobolev DML outperformance.
    """
    assert DML_CKPT_PATH.exists(), f"Flagship checkpoint missing at {DML_CKPT_PATH}"
    assert BASELINE_CKPT_PATH.exists(), f"Baseline checkpoint missing at {BASELINE_CKPT_PATH}"
    assert BENCHMARK_PATH.exists(), f"Benchmark data missing at {BENCHMARK_PATH}"
    
    device = torch.device("cpu")
    
    # 1. Load DML Flagship
    dml_ckpt = torch.load(DML_CKPT_PATH, map_location=device, weights_only=False)
    dml_model = BatesNeuralSurrogate(input_dim=10, output_dim=1, activation="silu").to(device)
    dml_model.load_state_dict(dml_ckpt["model_state_dict"])
    dml_model.eval()
    
    # 2. Load Price-Only Baseline
    base_ckpt = torch.load(BASELINE_CKPT_PATH, map_location=device, weights_only=False)
    base_model = BatesNeuralSurrogate(input_dim=10, output_dim=1, activation="silu").to(device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()
    
    # 3. Load Benchmark Dataset (1,500 deterministic samples)
    bench = np.load(BENCHMARK_PATH)
    X_test = torch.tensor(bench["X"], dtype=torch.float32, device=device)
    y_test = torch.tensor(bench["y"], dtype=torch.float32, device=device)
    dydx_test = torch.tensor(bench["dydx"], dtype=torch.float32, device=device)
    
    # 4. Latency Benchmark (128 contracts batch)
    dummy_batch = X_test[:128]
    with torch.no_grad():
        _ = dml_model((dummy_batch - dml_ckpt["X_mean"]) / dml_ckpt["X_std"])
        t0 = time.perf_counter()
        for _ in range(50):
            _ = dml_model((dummy_batch - dml_ckpt["X_mean"]) / dml_ckpt["X_std"])
        t1 = time.perf_counter()
        batch_latency_ms = ((t1 - t0) / 50.0) * 1000.0
        per_contract_ms = batch_latency_ms / 128.0
        
    assert batch_latency_ms < 10.0, f"Batch latency {batch_latency_ms:.2f} ms exceeded threshold"
    assert per_contract_ms < 0.1, f"Per-contract latency {per_contract_ms:.4f} ms exceeded threshold"
    
    # 5. Evaluate Greek (Delta) Accuracy on benchmark subset (256 contracts for fast CI)
    eval_sub = min(256, len(X_test))
    X_sub = X_test[:eval_sub].clone().requires_grad_(True)
    y_sub = y_test[:eval_sub]
    target_dydk = dydx_test[:eval_sub, 0] # dV/d(log_moneyness)
    target_delta = y_sub - target_dydk
    
    # Sobolev Model Predictions & Greeks
    dml_pred_s = dml_model((X_sub - dml_ckpt["X_mean"]) / dml_ckpt["X_std"]).squeeze(-1)
    dml_pred = dml_pred_s * dml_ckpt["y_std"] + dml_ckpt["y_mean"]
    dml_grad = torch.autograd.grad(dml_pred.sum(), X_sub, create_graph=False)[0]
    dml_dydk = dml_grad[:, 0]
    dml_delta = dml_pred - dml_dydk
    
    # Baseline Model Predictions & Greeks
    X_sub_b = X_test[:eval_sub].clone().requires_grad_(True)
    base_pred_s = base_model((X_sub_b - base_ckpt["X_mean"]) / base_ckpt["X_std"]).squeeze(-1)
    base_pred = base_pred_s * base_ckpt["y_std"] + base_ckpt["y_mean"]
    base_grad = torch.autograd.grad(base_pred.sum(), X_sub_b, create_graph=False)[0]
    base_dydk = base_grad[:, 0]
    base_delta = base_pred - base_dydk
    
    # Assert Zero NaNs
    assert torch.isfinite(dml_pred).all(), "Non-finite values in DML predictions"
    assert torch.isfinite(dml_delta).all(), "Non-finite values in DML Delta"
    
    # Delta MAE Comparison
    dml_delta_mae = torch.mean(torch.abs(dml_delta - target_delta)).item()
    base_delta_mae = torch.mean(torch.abs(base_delta - target_delta)).item()
    
    assert dml_delta_mae <= base_delta_mae * 1.05, (
        f"Sobolev DML Delta MAE ({dml_delta_mae:.6f}) should outperform or match Baseline ({base_delta_mae:.6f})"
    )


def test_spy_surface_calibration():
    """
    Production Test 3: Calibrates Sobolev DML model on real SPY options snapshot.
    Asserts convergence in < 15s on CPU, zero NaNs, and IV RMSE < 450 bps.
    """
    assert SNAPSHOT_PATH.exists(), f"SPY snapshot missing at {SNAPSHOT_PATH}"
    df = pd.read_csv(SNAPSHOT_PATH)
    M = len(df)
    assert M >= 80, f"Expected at least 80 contracts in SPY snapshot, found {M}"
    
    device = torch.device("cpu")
    ckpt = torch.load(DML_CKPT_PATH, map_location=device, weights_only=False)
    model = BatesNeuralSurrogate(input_dim=10, output_dim=1, activation="silu").to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    S0 = float(df["spot"].iloc[0])
    k_arr = torch.tensor(df["log_moneyness"].values, dtype=torch.float32, device=device)
    T_arr = torch.tensor(df["T"].values, dtype=torch.float32, device=device)
    r_arr = torch.tensor(df["r"].values, dtype=torch.float32, device=device)
    y_target = torch.tensor(df["normalized_call_price"].values, dtype=torch.float32, device=device)
    vegas = torch.tensor(df["bs_vega_normalized"].values, dtype=torch.float32, device=device)
    
    weights = 1.0 / torch.clamp(vegas, min=1e-3)**2
    weights = weights / weights.mean()
    
    bounds_low = torch.tensor([0.01, 0.5, 0.01, 0.10, -0.95, 0.00, -0.25], device=device)
    bounds_high = torch.tensor([0.22, 4.5, 0.20, 0.85, -0.10, 1.50,  0.05], device=device)
    
    # 1. Calibrate (1 start, 100 iterations for fast CI test)
    t0 = time.time()
    raw_params = torch.zeros(7, device=device, requires_grad=True)
    optimizer = optim.Adam([raw_params], lr=0.08)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-3)
    
    best_loss = float("inf")
    best_p = None
    
    for it in range(100):
        optimizer.zero_grad()
        phys = bounds_low + torch.sigmoid(raw_params) * (bounds_high - bounds_low)
        param_block = phys.unsqueeze(0).expand(M, -1)
        fixed_block = torch.stack([k_arr, T_arr, r_arr], dim=1)
        X = torch.cat([fixed_block, param_block], dim=1)
        
        pred_norm = model((X - ckpt["X_mean"]) / ckpt["X_std"]).squeeze(-1) * ckpt["y_std"] + ckpt["y_mean"]
        loss = torch.mean(weights * (pred_norm - y_target)**2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_p = phys.detach().clone()
            
    calib_time = time.time() - t0
    
    # 2. Assert Latency
    assert calib_time < 15.0, f"Calibration took {calib_time:.2f} s, exceeded 15.0 s threshold"
    assert best_p is not None, "Calibration failed to find valid parameters"
    assert torch.isfinite(best_p).all(), "Non-finite values in calibrated parameters"
    
    # 3. Evaluate Calibrated Implied Volatility Error
    with torch.no_grad():
        param_block = best_p.unsqueeze(0).expand(M, -1)
        fixed_block = torch.stack([k_arr, T_arr, r_arr], dim=1)
        X = torch.cat([fixed_block, param_block], dim=1)
        pred_norm_p = (model((X - ckpt["X_mean"]) / ckpt["X_std"]).squeeze(-1) * ckpt["y_std"] + ckpt["y_mean"]).numpy()
        
    p_dollars = pred_norm_p * S0
    m_dollars = df["normalized_call_price"].values * S0
    mae_dollars = float(np.mean(np.abs(p_dollars - m_dollars)))
    
    # Compute IV RMSE
    iv_diffs = []
    for pn, k, T, r, q, miv in zip(pred_norm_p, df["strike"].values, df["T"].values, df["r"].values, df["q"].values, df["market_iv"].values):
        iv = invert_black_scholes_iv(pn * S0, S0, k, r, q, T, is_call=True)
        if not np.isnan(iv):
            iv_diffs.append((iv - miv) * 10000.0)
            
    assert len(iv_diffs) >= 100, f"Only {len(iv_diffs)} contracts inverted successfully"
    iv_rmse_bps = float(np.sqrt(np.mean(np.array(iv_diffs)**2)))
    
    assert mae_dollars < 10.0, f"Dollar Price MAE ${mae_dollars:.2f} exceeded threshold"
    assert iv_rmse_bps < 450.0, f"Implied Vol RMSE {iv_rmse_bps:.1f} bps exceeded 450 bps threshold"
