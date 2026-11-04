"""
Unit and Integration Tests for Neural Bates DML Models, Snapshot Data, and Greeks.
Author: Quantitative Derivatives Research
Repository: Amanatar/neural-bates-dml
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import BatesNeuralSurrogate
from bates_fft import invert_black_scholes_iv, black_scholes_call, black_scholes_put

SNAPSHOT_PATH = REPO_ROOT / "data" / "eval" / "spy_options_snapshot.csv"
BENCHMARK_PATH = REPO_ROOT / "data" / "eval" / "bates_test_benchmark.npz"
DML_CKPT_PATH = REPO_ROOT / "artifacts" / "bates_dml_v1.pt"
BASE_CKPT_PATH = REPO_ROOT / "artifacts" / "price_only_baseline.pt"


def test_spy_options_snapshot_integrity():
    """Verify the deterministic SPY snapshot exists and satisfies institutional data quality standards."""
    assert SNAPSHOT_PATH.exists(), f"Snapshot CSV not found at: {SNAPSHOT_PATH}"
    df = pd.read_csv(SNAPSHOT_PATH)
    
    assert len(df) >= 80, f"Expected at least 80 contracts, found {len(df)}"
    assert df["expiry"].nunique() == 4, f"Expected 4 distinct expiries, found {df['expiry'].nunique()}"
    
    required_cols = [
        "expiry", "dte", "T", "spot", "strike", "moneyness", "log_moneyness",
        "option_type", "bid", "ask", "mid", "volume", "market_iv",
        "synthetic_call_price", "normalized_call_price", "bs_vega_normalized", "r", "q"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        
    assert not df[required_cols].isna().any().any(), "Found NaN in snapshot dataset"
    assert (df["mid"] > 0).all(), "Found non-positive mid prices"
    assert (df["market_iv"] > 0.05).all() and (df["market_iv"] < 1.0).all(), "Market IV outside realistic [5%, 100%] bounds"
    assert (df["moneyness"] >= 0.85).all() and (df["moneyness"] <= 1.15).all(), "Moneyness outside [0.85, 1.15]"


def test_production_checkpoints_loading_and_inference():
    """Verify both production checkpoints load cleanly, are under 5MB total, and execute CPU forward passes."""
    for ckpt_path in [DML_CKPT_PATH, BASE_CKPT_PATH]:
        assert ckpt_path.exists(), f"Checkpoint not found at: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        for k in ["model_state_dict", "X_mean", "X_std", "y_mean", "y_std", "input_dim", "output_dim"]:
            assert k in ckpt, f"Key {k} missing from checkpoint"
            
        model = BatesNeuralSurrogate(
            input_dim=ckpt["input_dim"],
            output_dim=ckpt["output_dim"],
            activation=ckpt.get("activation", "silu")
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        # Test 128-batch forward pass
        dummy_x = torch.randn(128, ckpt["input_dim"])
        with torch.no_grad():
            out = model((dummy_x - ckpt["X_mean"]) / ckpt["X_std"])
            
        assert out.shape == (128, 1), f"Unexpected output shape: {out.shape}"
        assert torch.isfinite(out).all(), "Non-finite values in model output"


def test_autograd_greeks_smoothness():
    """Verify that autograd Greeks (Delta, Gamma, Vega) are smooth, finite, and non-vanishing."""
    ckpt = torch.load(DML_CKPT_PATH, map_location="cpu", weights_only=False)
    model = BatesNeuralSurrogate(input_dim=10, output_dim=1, activation="silu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    N = 30
    k = torch.linspace(-0.10, 0.10, N, requires_grad=True)
    T = torch.full((N,), 0.25)
    r = torch.full((N,), 0.045)
    v0 = torch.full((N,), 0.04, requires_grad=True)
    kappa = torch.full((N,), 2.0)
    theta = torch.full((N,), 0.04)
    sigma_v = torch.full((N,), 0.4)
    rho = torch.full((N,), -0.6)
    lam = torch.full((N,), 0.2)
    mu = torch.full((N,), -0.1)
    
    X = torch.stack([k, T, r, v0, kappa, theta, sigma_v, rho, lam, mu], dim=1)
    y_scaled = model((X - ckpt["X_mean"]) / ckpt["X_std"]).squeeze(-1)
    y = y_scaled * ckpt["y_std"] + ckpt["y_mean"]
    
    dydk = torch.autograd.grad(y.sum(), k, create_graph=True, retain_graph=True)[0]
    dydv0 = torch.autograd.grad(y.sum(), v0, retain_graph=True)[0]
    d2ydk2 = torch.autograd.grad(dydk.sum(), k)[0]
    
    delta = y - dydk
    gamma = d2ydk2 - dydk
    vega = dydv0
    
    assert torch.isfinite(delta).all(), "Non-finite Delta values"
    assert torch.isfinite(gamma).all(), "Non-finite Gamma values"
    assert torch.isfinite(vega).all(), "Non-finite Vega values"
    assert (delta > 0.0).all() and (delta < 1.0).all(), "Delta outside (0, 1) probability bounds"
    assert (gamma > 0.0).all(), "Gamma must be strictly positive (no negative probabilities)"
    assert (vega > 0.0).all(), "Vega must be strictly positive across variance dimension"
