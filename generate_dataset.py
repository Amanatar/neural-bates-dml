import numpy as np
import torch
from bates_fft import price_bates_surface
import time


def _price_one_sample(args):
    """Worker for multiprocessing: prices + central-difference Greeks for one 10D sample."""
    params, maturities, moneyness_grid, h = args
    S0 = 1.0
    p8 = params[:8]
    r_val, q_val = params[8], params[9]
    prices = price_bates_surface(S0, r_val, q_val, maturities, moneyness_grid, p8).flatten()
    grads = np.zeros((prices.shape[0], 10))
    for j in range(10):
        pp = list(params); pp[j] += h
        pm = list(params); pm[j] -= h
        pr_plus = price_bates_surface(S0, pp[8], pp[9], maturities, moneyness_grid, pp[:8]).flatten()
        pr_minus = price_bates_surface(S0, pm[8], pm[9], maturities, moneyness_grid, pm[:8]).flatten()
        grads[:, j] = (pr_plus - pr_minus) / (2 * h)
    return prices, grads


def generate_dataset(num_samples=100000, save_path='dataset_dml.pt', seed=42, n_workers=1):
    """
    10D inputs: [v0, kappa, theta, sigma, rho, lambda_, muJ, vJ, r, q].
    Seeded Latin Hypercube sampling; Greeks via central finite differences.
    n_workers > 1 parallelizes FFT generation (requires fork start method;
    keep n_workers=1 on Windows, use 4 on Kaggle/Linux).
    """
    np.random.seed(seed)

    maturities = np.array([30/365, 60/365, 90/365, 120/365, 150/365])
    moneyness_grid = np.linspace(0.8, 1.2, 11)
    num_prices = len(maturities) * len(moneyness_grid)

    print(f"Generating {num_samples} samples with Greeks (DML), n_workers={n_workers}...")
    start_time = time.time()

    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=10, seed=seed)
    l_bounds = [0.01, 0.5, 0.01, 0.1, -0.9, 0.0, -0.2, 0.05, 0.0, 0.0]
    u_bounds = [0.2, 5.0, 0.2, 0.8, 0.0, 1.0, 0.0, 0.3, 0.1, 0.05]
    X_samples = qmc.scale(sampler.random(n=num_samples), l_bounds, u_bounds)

    h = 1e-4
    tasks = [(X_samples[i].tolist(), maturities, moneyness_grid, h) for i in range(num_samples)]

    X = X_samples
    Y = np.zeros((num_samples, num_prices))
    dY_dX = np.zeros((num_samples, num_prices, 10))

    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            results = pool.map(_price_one_sample, tasks, chunksize=32)
    else:
        results = [_price_one_sample(t) for t in tasks]

    for i, (prices, grads) in enumerate(results):
        Y[i, :] = prices
        dY_dX[i, :, :] = grads
        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{num_samples} in {time.time() - start_time:.2f}s")

    torch.save({
        'X': torch.tensor(X, dtype=torch.float32),
        'Y': torch.tensor(Y, dtype=torch.float32),
        'dY_dX': torch.tensor(dY_dX, dtype=torch.float32),
        'maturities': maturities,
        'moneyness': moneyness_grid,
    }, save_path)
    print(f"DML Dataset saved to {save_path}")


if __name__ == '__main__':
    # Local default: 1,000 samples (~2 min). For the Kaggle 100k GPU run,
    # call generate_dataset(num_samples=100000, n_workers=4) as the notebook does.
    generate_dataset(num_samples=1000, save_path='dataset_1000_dml.pt')
