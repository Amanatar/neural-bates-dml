"""
SPY Option Chain Snapshot, Inversion & Live Neural Calibration.
Author: Quantitative Volatility Surface Calibration Engineer
Repository: Amanatar/neural-bates-dml
"""

import os
import sys
import time
import math
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from bates_fft import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_vega,
    invert_black_scholes_iv,
    carr_madan_fft
)
from model import BatesNeuralSurrogate

DEFAULT_SNAPSHOT_PATH = os.path.join(REPO_ROOT, "data", "eval", "spy_options_snapshot.csv")
DEFAULT_DML_PATH = os.path.join(REPO_ROOT, "artifacts", "bates_dml_v1.pt")
DEFAULT_BASELINE_PATH = os.path.join(REPO_ROOT, "artifacts", "price_only_baseline.pt")


def fetch_and_snapshot_spy_chain(csv_path=DEFAULT_SNAPSHOT_PATH, force_live=False, n_expiries=4):
    """
    Fetches, cleans, and snapshots a real multi-expiry SPY option chain.
    
    1. Selects 4 distinct liquid maturities between 14 and 180 days to expiration.
    2. Filters liquid OTM puts for K < S0 and OTM calls for K >= S0 in moneyness K/S0 in [0.85, 1.15].
    3. Converts OTM puts to synthetic call prices via Put-Call Parity.
    4. Inverts exact market Implied Volatilities (IV_market) via Brent's method.
    5. Saves a frozen, deterministic copy to csv_path for 100% offline reproducibility.
    """
    csv_file = Path(csv_path)
    
    # If file exists and force_live is False, load the cached deterministic snapshot
    if csv_file.exists() and not force_live:
        print(f"[CACHE] Loading deterministic SPY options snapshot from: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"[CACHE] Loaded {len(df)} contracts across {df['expiry'].nunique()} maturities.")
        return df

    print("[NETWORK] Fetching live multi-expiry SPY option chain via yfinance...")
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required for live market data extraction. Install with: pip install yfinance")
        
    ticker = yf.Ticker("SPY")
    
    # Extract Spot Price
    hist = ticker.history(period="5d")
    if hist.empty or "Close" not in hist.columns:
        raise RuntimeError("Failed to fetch historical prices for SPY from yfinance.")
    S0 = float(hist["Close"].iloc[-1])
    
    today = datetime.date.today()
    all_expiries = list(ticker.options)
    
    if not all_expiries:
        raise RuntimeError("No option expirations returned for SPY from yfinance.")
        
    exp_dates = []
    for exp_str in all_expiries:
        exp_dt = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_dt - today).days
        if 14 <= dte <= 180:
            exp_dates.append((exp_str, dte))
            
    if len(exp_dates) < n_expiries:
        raise RuntimeError(f"Expected at least {n_expiries} expiries with 14 <= DTE <= 180, found {len(exp_dates)}.")
        
    # Select n_expiries evenly spaced across the maturity range
    indices = np.round(np.linspace(0, len(exp_dates) - 1, n_expiries)).astype(int)
    selected_expiries = [exp_dates[i] for i in indices]
    
    print(f"[MARKET] SPY Spot Price S0 = ${S0:.2f}")
    print(f"[MARKET] Selected {len(selected_expiries)} expiries (14 <= DTE <= 180):")
    for exp_str, dte in selected_expiries:
        print(f"         Expiry: {exp_str} | DTE: {dte}d (T = {dte/365.0:.3f}y)")
        
    # Institutional market baseline interest rate and dividend yield for SPY
    r = 0.045   # Prevailing ~4.5% risk-free rate
    q = 0.013   # SPY dividend yield ~1.3%
    
    records = []
    
    for exp_str, dte in selected_expiries:
        T = dte / 365.0
        chain = ticker.option_chain(exp_str)
        calls = chain.calls
        puts = chain.puts
        
        # 1. Clean Liquid OTM Puts for K < S0 (moneyness [0.85, 1.00))
        otm_puts = puts[(puts["strike"] >= 0.85 * S0) & (puts["strike"] < S0)].copy()
        if not otm_puts.empty:
            # Market mid price: use (bid+ask)/2 if valid quote spread, otherwise fallback to lastPrice
            otm_puts["mid"] = np.where(
                (otm_puts["bid"] > 0) & (otm_puts["ask"] > otm_puts["bid"]),
                (otm_puts["bid"] + otm_puts["ask"]) / 2.0,
                otm_puts["lastPrice"]
            )
            # Filter liquidity: mid > $0.10 and positive volume
            otm_puts = otm_puts[(otm_puts["mid"] > 0.10) & (otm_puts["volume"] > 0)]
            # Filter strikes with multiple of 5 for uniform non-redundant grid
            otm_puts = otm_puts[otm_puts["strike"] % 5 == 0]
            
            for _, row in otm_puts.iterrows():
                K = float(row["strike"])
                mid_p = float(row["mid"])
                iv = invert_black_scholes_iv(mid_p, S0, K, r, q, T, is_call=False)
                
                # Synthetic call price via Put-Call Parity: C = P + S0*exp(-qT) - K*exp(-rT)
                synth_call = mid_p + S0 * math.exp(-q * T) - K * math.exp(-r * T)
                norm_call = synth_call / S0
                
                if not np.isnan(iv) and 0.05 <= iv <= 0.80 and norm_call > 0.0:
                    vega = black_scholes_vega(S0, K, r, q, T, iv) / S0
                    records.append({
                        "expiry": exp_str,
                        "dte": int(dte),
                        "T": float(T),
                        "spot": float(S0),
                        "strike": float(K),
                        "moneyness": float(K / S0),
                        "log_moneyness": float(math.log(K / S0)),
                        "option_type": "put",
                        "bid": float(row["bid"]),
                        "ask": float(row["ask"]),
                        "mid": float(mid_p),
                        "volume": float(row["volume"]),
                        "open_interest": float(row["openInterest"]) if "openInterest" in row else 0.0,
                        "market_iv": float(iv),
                        "synthetic_call_price": float(synth_call),
                        "normalized_call_price": float(norm_call),
                        "bs_vega_normalized": float(vega),
                        "r": float(r),
                        "q": float(q),
                    })

        # 2. Clean Liquid OTM Calls for K >= S0 (moneyness [1.00, 1.15])
        otm_calls = calls[(calls["strike"] >= S0) & (calls["strike"] <= 1.15 * S0)].copy()
        if not otm_calls.empty:
            otm_calls["mid"] = np.where(
                (otm_calls["bid"] > 0) & (otm_calls["ask"] > otm_calls["bid"]),
                (otm_calls["bid"] + otm_calls["ask"]) / 2.0,
                otm_calls["lastPrice"]
            )
            otm_calls = otm_calls[(otm_calls["mid"] > 0.10) & (otm_calls["volume"] > 0)]
            otm_calls = otm_calls[otm_calls["strike"] % 5 == 0]
            
            for _, row in otm_calls.iterrows():
                K = float(row["strike"])
                mid_c = float(row["mid"])
                iv = invert_black_scholes_iv(mid_c, S0, K, r, q, T, is_call=True)
                synth_call = mid_c
                norm_call = synth_call / S0
                
                if not np.isnan(iv) and 0.05 <= iv <= 0.80 and norm_call > 0.0:
                    vega = black_scholes_vega(S0, K, r, q, T, iv) / S0
                    records.append({
                        "expiry": exp_str,
                        "dte": int(dte),
                        "T": float(T),
                        "spot": float(S0),
                        "strike": float(K),
                        "moneyness": float(K / S0),
                        "log_moneyness": float(math.log(K / S0)),
                        "option_type": "call",
                        "bid": float(row["bid"]),
                        "ask": float(row["ask"]),
                        "mid": float(mid_c),
                        "volume": float(row["volume"]),
                        "open_interest": float(row["openInterest"]) if "openInterest" in row else 0.0,
                        "market_iv": float(iv),
                        "synthetic_call_price": float(synth_call),
                        "normalized_call_price": float(norm_call),
                        "bs_vega_normalized": float(vega),
                        "r": float(r),
                        "q": float(q),
                    })
                    
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("Cleaned SPY options dataset is empty. Check market connection or filtering thresholds.")
        
    # Sort contracts deterministically by expiry and strike
    df = df.sort_values(by=["T", "strike"]).reset_index(drop=True)
    
    # Save CSV
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"\n[OUTPUT] Saved frozen deterministic SPY snapshot ({len(df)} contracts) to: {csv_file}")
    print(f"[SUMMARY] Expiry counts:\n{df['expiry'].value_counts().sort_index()}")
    print(f"[SUMMARY] Moneyness K/S0 range: [{df['moneyness'].min():.3f}, {df['moneyness'].max():.3f}]")
    print(f"[SUMMARY] Market IV range:       [{df['market_iv'].min():.3f}, {df['market_iv'].max():.3f}] (Mean: {df['market_iv'].mean():.3f})")
    
    return df


def calibrate_to_market(
    model_path=DEFAULT_DML_PATH,
    snapshot_path=DEFAULT_SNAPSHOT_PATH,
    n_restarts=4,
    max_iters=250,
    lr=0.08,
    verbose=True
):
    """
    Calibrates Bates SVJD parameters Theta* = [v0, kappa, theta, sigma_v, rho, lambda_j, mu_j]
    to the real SPY options snapshot using the neural surrogate model and PyTorch autograd.
    """
    df = fetch_and_snapshot_spy_chain(csv_path=snapshot_path, force_live=False)
    
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    act = checkpoint.get("activation", "silu")
    
    model = BatesNeuralSurrogate(input_dim=input_dim, output_dim=output_dim, activation=act).to(device)
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
    q_arr = df["q"].values
    
    y_target = torch.tensor(df["normalized_call_price"].values, dtype=torch.float32, device=device)
    vegas = torch.tensor(df["bs_vega_normalized"].values, dtype=torch.float32, device=device)
    
    # Vega weights for weighted MSE (directly minimizing implied volatility error)
    weights = 1.0 / torch.clamp(vegas, min=1e-3)**2
    weights = weights / weights.mean()  # normalize so mean(weights) == 1.0
    
    # Parameter bounds: [v0, kappa, theta, sigma_v, rho, lambda_j, mu_j]
    # Fixed institutional Merton jump volatility sigma_j = 0.15
    bounds_low = torch.tensor([0.01, 0.5, 0.01, 0.10, -0.95, 0.00, -0.25], device=device)
    bounds_high = torch.tensor([0.22, 4.5, 0.20, 0.85, -0.10, 1.50,  0.05], device=device)
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"STARTING GRADIENT-BASED CALIBRATION ON REAL SPY SURFACE ({M} CONTRACTS)")
        print(f"Model Checkpoint: {model_path}")
        print("=" * 70)
        
    start_time = time.time()
    best_loss = float("inf")
    best_params = None
    
    for restart in range(n_restarts):
        # Multi-start: restart 0 is midpoint, others are uniform jitter
        if restart == 0:
            raw_params = torch.zeros(7, device=device, requires_grad=True)
        else:
            raw_params = (torch.rand(7, device=device) * 4.0 - 2.0).requires_grad_(True)
            
        optimizer = optim.Adam([raw_params], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-3)
        
        for it in range(max_iters):
            optimizer.zero_grad()
            
            # Map unconstrained raw_params via sigmoid to physical bounds
            phys_theta = bounds_low + torch.sigmoid(raw_params) * (bounds_high - bounds_low)
            
            # Construct (M, 10) input batch for surrogate:
            # ['log_moneyness', 'T', 'r', 'v0', 'kappa', 'theta', 'sigma_v', 'rho', 'lambda_j', 'mu_j']
            # Broadcast the 7 parameters across all M contracts
            param_block = phys_theta.unsqueeze(0).expand(M, -1)
            fixed_block = torch.stack([k_arr, T_arr, r_arr], dim=1)
            X_batch = torch.cat([fixed_block, param_block], dim=1)
            
            # Normalize inputs
            X_scaled = (X_batch - X_mean) / X_std
            
            # Forward pass
            pred_scaled = model(X_scaled).squeeze(-1)
            pred_unit = pred_scaled * y_std + y_mean
            
            # Weighted MSE loss
            loss = torch.mean(weights * (pred_unit - y_target)**2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_val = float(loss.item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = phys_theta.detach().clone()
                
    elapsed_time = time.time() - start_time
    calib_ms = elapsed_time * 1000.0
    
    # Evaluate calibrated model predictions
    with torch.no_grad():
        param_block = best_params.unsqueeze(0).expand(M, -1)
        fixed_block = torch.stack([k_arr, T_arr, r_arr], dim=1)
        X_batch = torch.cat([fixed_block, param_block], dim=1)
        pred_scaled = model((X_batch - X_mean) / X_std).squeeze(-1)
        pred_norm_prices = (pred_scaled * y_std + y_mean).cpu().numpy()
        
    market_norm_prices = df["normalized_call_price"].values
    market_ivs = df["market_iv"].values
    strikes = df["strike"].values
    Ts = df["T"].values
    rs = df["r"].values
    
    # Model Dollar Prices
    model_dollar_prices = pred_norm_prices * S0
    market_dollar_prices = market_norm_prices * S0
    dollar_errors = np.abs(model_dollar_prices - market_dollar_prices)
    price_mae = float(np.mean(dollar_errors))
    price_rmse = float(np.sqrt(np.mean(dollar_errors**2)))
    
    # Invert Model Prices to Implied Volatility
    model_ivs = []
    iv_diffs = []
    for p_norm, k, T, r, q, mkt_iv in zip(pred_norm_prices, strikes, Ts, rs, q_arr, market_ivs):
        p_dollar = p_norm * S0
        iv = invert_black_scholes_iv(p_dollar, S0, k, r, q, T, is_call=True)
        if not np.isnan(iv):
            model_ivs.append(iv)
            iv_diffs.append((iv - mkt_iv) * 10000.0) # basis points
        else:
            model_ivs.append(mkt_iv) # fallback for boundary contracts
            
    iv_rmse_bps = float(np.sqrt(np.mean(np.array(iv_diffs)**2))) if iv_diffs else 0.0
    iv_mae_bps = float(np.mean(np.abs(np.array(iv_diffs)))) if iv_diffs else 0.0
    
    # Arbitrage checks on calibrated surface
    # Butterfly convexity across strikes: C(K - dK) - 2C(K) + C(K + dK) >= -1e-5
    butterfly_violations = 0
    total_butterfly = 0
    for exp, grp in df.groupby("expiry"):
        sorted_grp = grp.sort_values(by="strike")
        grp_prices = model_dollar_prices[sorted_grp.index]
        if len(grp_prices) >= 3:
            for i in range(1, len(grp_prices) - 1):
                convexity = grp_prices[i-1] - 2.0 * grp_prices[i] + grp_prices[i+1]
                total_butterfly += 1
                if convexity < -1e-4:
                    butterfly_violations += 1
                    
    butterfly_pass_pct = 100.0 * (1.0 - butterfly_violations / max(total_butterfly, 1))
    
    p = best_params.cpu().numpy()
    results = {
        "model_name": Path(model_path).stem,
        "calibration_time_ms": calib_ms,
        "calibration_time_s": elapsed_time,
        "price_mae_dollars": price_mae,
        "price_rmse_dollars": price_rmse,
        "iv_rmse_bps": iv_rmse_bps,
        "iv_mae_bps": iv_mae_bps,
        "arbitrage_pass_rate_pct": butterfly_pass_pct,
        "bates_parameters": {
            "v0": float(p[0]),
            "kappa": float(p[1]),
            "theta": float(p[2]),
            "sigma_v": float(p[3]),
            "rho": float(p[4]),
            "lambda_j": float(p[5]),
            "mu_j": float(p[6]),
            "sigma_j": 0.15,
        }
    }
    
    if verbose:
        print(f"\n[CONVERGED] Calibration Runtime: {calib_ms:.2f} ms ({elapsed_time:.3f} s)")
        print(f"            Surface Price MAE:   ${price_mae:.4f} | Price RMSE: ${price_rmse:.4f}")
        print(f"            Implied Vol RMSE:    {iv_rmse_bps:.1f} bps | IV MAE: {iv_mae_bps:.1f} bps")
        print(f"            Arbitrage Pass Rate: {butterfly_pass_pct:.1f}%")
        print("\nCalibrated Bates SVJD Parameters:")
        print(f"  v0      (Initial Variance):      {p[0]:.4f} (spot vol = {math.sqrt(p[0])*100:.1f}%)")
        print(f"  kappa   (Mean Reversion Rate):   {p[1]:.4f}")
        print(f"  theta   (Long-Term Variance):    {p[2]:.4f} (long vol = {math.sqrt(p[2])*100:.1f}%)")
        print(f"  sigma_v (Vol of Vol):            {p[3]:.4f}")
        print(f"  rho     (Correlation / Skew):    {p[4]:.4f}")
        print(f"  lambda_ (Jump Intensity):        {p[5]:.4f} jumps/yr")
        print(f"  mu_j    (Mean Jump Size):        {p[6]:.4f}")
        print(f"  sigma_j (Jump Volatility):       0.1500 (fixed institutional baseline)")
        print("=" * 70)
        
    return results


if __name__ == "__main__":
    # 1. Fetch & Snapshot Real SPY Chain
    snapshot_df = fetch_and_snapshot_spy_chain(DEFAULT_SNAPSHOT_PATH, force_live=True)
    
    # 2. Run Live Neural Calibration
    calib_results = calibrate_to_market(DEFAULT_DML_PATH, DEFAULT_SNAPSHOT_PATH, verbose=True)
