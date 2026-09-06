import time
import torch
import numpy as np
from scipy.optimize import minimize
from bates_fft import price_bates_surface
from live_calibration import fetch_and_interpolate_market_data
from model import BatesNeuralSurrogate

def run_benchmark():
    print("Fetching live market data for DML smoke test...")
    market_prices, S0 = fetch_and_interpolate_market_data(return_spot=True)
    
    target_maturities = np.array([30/365, 60/365, 90/365, 120/365, 150/365])
    target_moneyness = np.linspace(0.8, 1.2, 11)

    # Dimension-agnostic: the shipped checkpoint is 8D; the Kaggle 100k run is
    # 10D (r, q as inputs 9-10). Build bounds/initial guess to match.
    checkpoint = torch.load('bates_surrogate_dml.pth', map_location='cpu', weights_only=False)
    D = checkpoint['X_mean'].shape[0]

    # Bounds match the surrogate's TRAINING distribution (generate_dataset.py)
    bounds_10d = [
        (0.01, 0.2),   # v0
        (0.5, 5.0),    # kappa
        (0.01, 0.2),   # theta
        (0.1, 0.8),    # sigma
        (-0.9, 0.0),   # rho
        (0.0, 1.0),    # lambda_
        (-0.2, 0.0),   # muJ
        (0.05, 0.3),   # vJ
        (0.0, 0.1),    # r
        (0.0, 0.05)    # q
    ]
    guess_10d = [0.04, 2.0, 0.04, 0.3, -0.5, 0.1, -0.1, 0.1, 0.05, 0.01]
    bounds = bounds_10d[:D]
    initial_guess = guess_10d[:D]

    def r_of(p):
        return p[8] if D > 8 else 0.05

    def q_of(p):
        return p[9] if D > 9 else 0.0

    print("\n--- Traditional Optimization (SciPy L-BFGS-B, 200 iters) ---")
    print("Running heavy calculus (Carr-Madan FFT)...")
    def objective_function(params):
        try:
            # Targets are unit-spot normalized, and Bates prices scale exactly
            # linearly in S0, so the FFT leg also prices on S0 = 1.0.
            prices = price_bates_surface(1.0, r_of(params), q_of(params), target_maturities, target_moneyness, params[:8]).flatten()
            return np.mean((prices - market_prices)**2)
        except Exception:
            return 1e6
            
    start_scipy = time.time()
    res = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
    time_scipy = time.time() - start_scipy
    scipy_rmse = float(np.sqrt(res.fun))
    print(f"Traditional Calibration Time: {time_scipy:.2f} seconds | Price RMSE (unit spot): {scipy_rmse:.6f}")
    
    print("\n--- Differential Neural Surrogate Calibration (PyTorch) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BatesNeuralSurrogate(input_dim=D, output_dim=55).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    Y_mean = checkpoint['Y_mean'].to(device)
    Y_std = checkpoint['Y_std'].to(device)
    X_mean = checkpoint['X_mean'].to(device)
    X_std = checkpoint['X_std'].to(device)
    
    # market_prices are already unit-spot normalized (see fetch_and_interpolate_market_data)
    Y_target = torch.tensor(market_prices, dtype=torch.float32, device=device)
    Y_target_scaled = (Y_target - Y_mean) / Y_std
    
    bounds_low = torch.tensor([b[0] for b in bounds], device=device)
    bounds_high = torch.tensor([b[1] for b in bounds], device=device)
    
    criterion = torch.nn.MSELoss()
    
    start_torch = time.time()
    best_loss = float('inf')
    best_params = None
    for restart in range(4):
        raw_params = torch.zeros(1, D, device=device, requires_grad=True) if restart == 0 \
            else (torch.rand(1, D, device=device) * 4.0 - 2.0).requires_grad_(True)
        optimizer = torch.optim.Adam([raw_params], lr=0.1)
        for _ in range(400):
            optimizer.zero_grad()
            physical_params = bounds_low + torch.sigmoid(raw_params) * (bounds_high - bounds_low)
            X_scaled = (physical_params - X_mean) / X_std
            pred_scaled = model(X_scaled)
            loss = criterion(pred_scaled, Y_target_scaled.unsqueeze(0))
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = physical_params.detach().clone()
            
    time_torch = time.time() - start_torch
    with torch.no_grad():
        pred_unit = model((best_params - X_mean) / X_std) * Y_std + Y_mean
        neural_rmse = float(torch.sqrt(torch.mean((pred_unit.squeeze() - Y_target)**2)))
    print(f"Neural Calibration Time: {time_torch:.2f} seconds | Price RMSE (unit spot): {neural_rmse:.6f}")

    # Direct micro-benchmark: cost of ONE pricing-surface evaluation, both ways
    t0 = time.time()
    for _ in range(20):
        price_bates_surface(1.0, r_of(initial_guess), q_of(initial_guess), target_maturities, target_moneyness, initial_guess[:8])
    ms_fft = (time.time() - t0) / 20 * 1000
    xb = torch.randn(256, D, device=device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20):
            model((xb - X_mean) / X_std)
    ms_net = (time.time() - t0) / 20 * 1000 / 256  # amortized cost per 55-price surface, batched
    print(f"Per-evaluation cost: FFT surface {ms_fft:.2f} ms vs network surface {ms_net:.3f} ms "
          f"(batched 256) -> ~{ms_fft / max(ms_net, 1e-9):.0f}x cheaper per evaluation")

    print(f"\n>>> TOTAL TIME (CPU): neural {time_torch:.1f}s vs traditional {time_scipy:.1f}s "
          f"(L-BFGS-B can win on small surfaces with a cheap pricer) <<<")
    print(f">>> FIT QUALITY: neural RMSE {neural_rmse:.6f} vs traditional RMSE {scipy_rmse:.6f} (unit-spot prices) <<<")
    print(">>> The surrogate's edge is per-evaluation cost: it pays off on GPU/batched")
    print("    workloads, high-dimensional parameter spaces, or expensive pricers (PDE/MC). <<<")
    print("DML Smoke test completed successfully!")

if __name__ == '__main__':
    run_benchmark()
