import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from model import BatesNeuralSurrogate
import time

def fetch_and_interpolate_market_data(return_spot=False):
    """
    Build a unit-spot implied pricing surface on the surrogate's fixed grid.

    The surrogate expects prices for maturities of [30, 60, 90, 120, 150] days
    at moneyness 0.8..1.2. For each target maturity we use the CLOSEST listed
    expiration (never 2D-extrapolate across a data hull: griddata with
    fill_value=0 silently produces fake zero prices for uncovered cells) and
    linearly interpolate quoted mid prices across strikes. All prices are
    normalized by the spot, because the surrogate was trained on S0 = 1.0.
    """
    ticker = yf.Ticker("SPY")
    S0 = ticker.history(period="1d")['Close'].iloc[-1]

    target_days = np.array([30, 60, 90, 120, 150])
    target_moneyness = np.linspace(0.8, 1.2, 11)

    # Map each target maturity to the closest available expiration
    exp_days = np.array([(pd.to_datetime(e) - pd.Timestamp.today()).days for e in ticker.options])
    chains = {}  # cache fetched chains
    surface = np.zeros((len(target_days), len(target_moneyness)))

    for i, td in enumerate(target_days):
        j = int(np.argmin(np.abs(exp_days - td)))
        exp = ticker.options[j]
        if exp not in chains:
            opt = ticker.option_chain(exp)
            calls = opt.calls
            rows = calls[(calls['strike'] > 0)].copy()
            mid = np.where((rows['bid'] > 0) & (rows['ask'] > rows['bid']),
                           (rows['bid'] + rows['ask']) / 2.0, rows['lastPrice'])
            rows['mid'] = mid
            rows = rows[rows['mid'] > 0]
            chains[exp] = rows
        rows = chains[exp]
        m = (rows['strike'] / S0).values
        p = (rows['mid'] / S0).values  # normalize to unit spot
        order = np.argsort(m)
        m, p = m[order], p[order]
        # keep only the liquid band around the target grid
        band = (m >= 0.75) & (m <= 1.25)
        m, p = m[band], p[band]
        if len(m) < 2:
            print(f"Warning: insufficient strikes for {exp}, row left as zeros")
            continue
        surface[i, :] = np.interp(target_moneyness, m, p)

    # No-arbitrage floor: a call can never trade below its discounted intrinsic
    # value. Illiquid deep-ITM chains often miss quotes, and np.interp then
    # clamps flat at the last quoted price - which can sit BELOW intrinsic.
    # Floor the target surface so the calibration never chases arbitrage.
    intrinsic = np.maximum(0.0, 1.0 - target_moneyness)[None, :] \
        * np.exp(-0.05 * target_days / 365.0)[:, None]
    surface = np.maximum(surface, intrinsic)

    print(f"Fetching options for SPY (Spot = {S0:.2f})... using expirations "
          f"{sorted(set(int(np.abs(exp_days[np.argmin(np.abs(exp_days - td))])) for td in target_days))} days out")

    if return_spot:
        return surface.flatten(), float(S0)
    return surface.flatten()

def calibrate_to_market():
    # 1. Fetch data
    market_prices = fetch_and_interpolate_market_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load the trained surrogate (dimension-agnostic: 8D legacy or 10D with r, q)
    checkpoint = torch.load('bates_surrogate_dml.pth', map_location=device, weights_only=False)
    input_dim = checkpoint['X_mean'].shape[0]
    model = BatesNeuralSurrogate(input_dim=input_dim, output_dim=55).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # freeze network
    
    # 3. Prepare target
    Y_mean = checkpoint['Y_mean'].to(device)
    Y_std = checkpoint['Y_std'].to(device)
    X_mean = checkpoint['X_mean'].to(device)
    X_std = checkpoint['X_std'].to(device)
    
    Y_target = torch.tensor(market_prices, dtype=torch.float32).to(device)
    Y_target_scaled = (Y_target - Y_mean) / Y_std
    
    # 4. Initialize parameters for optimization
    # Bounds are clipped to the TRAINING distribution of the surrogate
    # (see generate_dataset.py). Optimizing outside it would mean trusting
    # network extrapolation, which is exactly where surrogates fail.
    # [v0, kappa, theta, sigma, rho, lambda_, muJ, vJ]
    bounds_low = torch.tensor([0.01, 0.5, 0.01, 0.1, -0.9, 0.0, -0.2, 0.05], device=device)
    bounds_high = torch.tensor([0.2, 5.0, 0.2, 0.8, 0.0, 1.0, 0.0, 0.3], device=device)
    
    # 10D models take r and q as inputs 9 and 10; keep them FIXED at values
    # inside their training ranges instead of calibrating them.
    if input_dim == 10:
        r_fixed, q_fixed = 0.05, 0.0
        fixed_tail = torch.tensor([[r_fixed, q_fixed]], device=device)
    else:
        fixed_tail = None
    
    criterion = torch.nn.MSELoss()
    
    print("Starting neural calibration with mathematical constraints...")
    start_time = time.time()
    
    # Multi-start: Bates calibration is non-convex and several parameters
    # (kappa, sigma, lambda) are weakly identified from short surfaces,
    # so run a few restarts and keep the best fit.
    best_loss = float('inf')
    best_params = None
    n_restarts = 4
    for restart in range(n_restarts):
        if restart == 0:
            raw_params = torch.zeros(1, 8, device=device, requires_grad=True)  # midpoint start
        else:
            raw_params = (torch.rand(1, 8, device=device) * 4.0 - 2.0).requires_grad_(True)
        optimizer = optim.Adam([raw_params], lr=0.1)
        
        for i in range(400):
            optimizer.zero_grad()
            
            # 1. Enforce strict mathematical bounds (8 optimized Bates params)
            physical_params = bounds_low + torch.sigmoid(raw_params) * (bounds_high - bounds_low)
            
            # 2. Append fixed r, q for 10D models, then scale for the network
            full_params = torch.cat([physical_params, fixed_tail], dim=1) if fixed_tail is not None else physical_params
            X_scaled = (full_params - X_mean) / X_std
            
            # 3. Forward pass through surrogate
            pred_scaled = model(X_scaled)
            
            loss = criterion(pred_scaled, Y_target_scaled.unsqueeze(0))
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = physical_params.detach().clone()
                
    end_time = time.time()
    
    # 5. Parameters are already unscaled (physical)
    final_params = best_params[0]
    
    print(f"\nCalibration finished in {(end_time - start_time)*1000:.2f} ms")
    print(f"Final MSE Loss (Scaled): {best_loss:.6f}")
    
    print("\nCalibrated Bates Parameters:")
    print(f"v0      (Initial Vol): {final_params[0].item():.4f}")
    print(f"kappa   (Mean Rev Rate): {final_params[1].item():.4f}")
    print(f"theta   (Long-term Vol): {final_params[2].item():.4f}")
    print(f"sigma   (Vol of Vol): {final_params[3].item():.4f}")
    print(f"rho     (Correlation): {final_params[4].item():.4f}")
    print(f"lambda_ (Jump Intensity): {final_params[5].item():.4f}")
    print(f"muJ     (Mean Jump Size): {final_params[6].item():.4f}")
    print(f"vJ      (Jump Vol): {final_params[7].item():.4f}")
    print("\nNote: v0/theta are usually well identified; kappa/sigma/lambda are weakly")
    print("identified from short-dated surfaces (flat likelihood), so treat them as")
    print("indicative. Judge the fit by the loss above, not by individual parameters.")

if __name__ == '__main__':
    calibrate_to_market()
