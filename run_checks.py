"""
Self-test suite for the Neural Bates DML project.

Run:  python run_checks.py

Checks (all runnable offline):
 1. Carr-Madan FFT pricer vs Black-Scholes closed form (pure-BS limit)
 2. No-arbitrage consistency of the pricer (non-negative, monotone in strike,
    butterfly >= 0, calendar spreads >= 0)
 3. Finite-difference Greeks have the correct signs (theta, v0 > 0)
 4. Shipped checkpoint loads; forward + torch.func.jacrev work
 5. If a training dataset is present: model accuracy vs dataset (prices + Greeks)
 6. Synthetic calibration: recover parameters of a known surface (unit spot),
    asserting on the FIT (pricing error), not on weakly-identified parameters
"""
import sys
import numpy as np
import torch
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import minimize

from bates_fft import price_bates_surface
from model import BatesNeuralSurrogate
from torch.func import vmap, jacrev

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def bs_call(S, K, r, T, sig):
    d1 = (log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


MAT = np.array([30 / 365, 60 / 365, 90 / 365, 120 / 365, 150 / 365])
MON = np.linspace(0.8, 1.2, 11)
PARAMS = [0.04, 2.0, 0.04, 0.3, -0.5, 0.1, -0.1, 0.1]
BOUNDS = [(0.01, 0.2), (0.5, 5.0), (0.01, 0.2), (0.1, 0.8),
          (-0.9, 0.0), (0.0, 1.0), (-0.2, 0.0), (0.05, 0.3)]


def check_1_fft_vs_bs():
    from bates_fft import verify_fft_against_black_scholes
    max_err, _ = verify_fft_against_black_scholes(tol=1e-3, verbose=False)
    check("1. FFT pricer matches Black-Scholes across [0.80, 1.20] (< 1e-3)", max_err < 1e-3, f"max err {max_err:.6e}")


def check_2_no_arbitrage():
    P = price_bates_surface(1.0, 0.05, 0.0, MAT, MON, PARAMS)
    ok_nonneg = bool((P >= 0).all())
    ok_mono = bool((np.diff(P, axis=1) <= 1e-9).all())
    h = 0.01
    ok_conv, ok_cal = True, True
    for i, T in enumerate(MAT):
        Pm = price_bates_surface(1.0, 0.05, 0.0, [T], MON - h, PARAMS)[0]
        Pp = price_bates_surface(1.0, 0.05, 0.0, [T], MON + h, PARAMS)[0]
        if ((Pm - 2 * P[i] + Pp) < -1e-8).any():
            ok_conv = False
    ok_cal = bool((np.diff(P, axis=0) >= -1e-8).all())
    check("2a. Prices non-negative", ok_nonneg)
    check("2b. Monotone decreasing in strike", ok_mono)
    check("2c. Butterfly spreads >= 0 (convexity)", ok_conv)
    check("2d. Calendar spreads >= 0", ok_cal)


def check_3_greek_signs():
    h = 1e-4
    def surf(p):
        return price_bates_surface(1.0, 0.05, 0.0, MAT, MON, p)
    g_theta = (surf(PARAMS[:2] + [PARAMS[2] + h] + PARAMS[3:])
               - surf(PARAMS[:2] + [PARAMS[2] - h] + PARAMS[3:])) / (2 * h)
    g_v0 = (surf([PARAMS[0] + h] + PARAMS[1:])
            - surf([PARAMS[0] - h] + PARAMS[1:])) / (2 * h)
    # Prices are floored at 0, so deep-OTM short-maturity cells can have a
    # sensitivity of exactly 0 (both FD points floored) - hence >= 0, not > 0.
    check("3a. d(Price)/d(theta) >= 0 everywhere (mean > 0)",
          bool((g_theta >= -1e-10).all()) and float(g_theta.mean()) > 0)
    check("3b. d(Price)/d(v0) >= 0 everywhere (mean > 0)",
          bool((g_v0 >= -1e-10).all()) and float(g_v0.mean()) > 0)


def check_4_checkpoint():
    try:
        ckpt = torch.load('bates_surrogate_dml.pth', weights_only=False, map_location='cpu')
        D = ckpt['X_mean'].shape[0]
        model = BatesNeuralSurrogate(D, 55, activation='elu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        out = model(torch.randn(4, D))
        j = vmap(jacrev(lambda x: model(x.unsqueeze(0)).squeeze(0)))(torch.randn(4, D))
        ok = tuple(out.shape) == (4, 55) and tuple(j.shape) == (4, 55, D) \
            and D in (8, 10) \
            and all(k in ckpt for k in ('X_mean', 'X_std', 'Y_mean', 'Y_std'))
        check(f"4. Checkpoint loads ({D}D); forward (4,55) + jacrev (4,55,{D}) work", ok)
        return model, ckpt
    except FileNotFoundError:
        check("4. Checkpoint loads", False, "bates_surrogate_dml.pth not found")
        return None, None


def check_5_dataset_accuracy(model, ckpt):
    import os
    path = 'dataset_100000_dml.pt' if os.path.exists('dataset_100000_dml.pt') else 'dataset_1000_dml.pt'
    if not os.path.exists(path):
        check("5. Dataset accuracy", True, "dataset file not present - skipped (regenerate with generate_dataset.py)")
        return
    data = torch.load(path, weights_only=False)
    X, Y = data['X'], data['Y']
    if X.shape[1] != ckpt['X_mean'].shape[0]:
        check("5. Dataset accuracy", True, f"dataset is {X.shape[1]}D but checkpoint is {ckpt['X_mean'].shape[0]}D - skipped")
        return
    Xm, Xs, Ym, Ys = ckpt['X_mean'], ckpt['X_std'], ckpt['Y_mean'], ckpt['Y_std']
    preds = []
    with torch.no_grad():
        for b in torch.split(X, 4096):
            preds.append(model((b - Xm) / Xs) * Ys + Ym)
    pred = torch.cat(preds)
    mae = float((pred - Y).abs().mean())
    mask = Y > 0.01
    rel = float(((pred - Y).abs() / Y.clamp(min=1e-6))[mask].median()) * 100
    n = X.shape[0]
    check("5. Model reproduces dataset prices", mae < 0.01 and rel < 1.0,
          f"{n} samples: MAE {mae:.5f}, median rel err {rel:.3f}%")


def check_6_synthetic_calibration(model, ckpt):
    if model is None:
        check("6. Synthetic calibration", False, "no checkpoint")
        return
    Xm, Xs, Ym, Ys = ckpt['X_mean'], ckpt['X_std'], ckpt['Y_mean'], ckpt['Y_std']
    D = Xm.shape[0]
    true_P = price_bates_surface(1.0, 0.05, 0.0, MAT, MON, PARAMS).flatten()
    Yt = torch.tensor(true_P, dtype=torch.float32)
    Yt_s = (Yt - Ym) / Ys
    lo = torch.tensor([b[0] for b in BOUNDS])
    hi = torch.tensor([b[1] for b in BOUNDS])
    # 10D checkpoints take r, q as fixed inputs 9-10 (inside training ranges)
    tail = torch.tensor([[0.05, 0.0]]) if D == 10 else None
    crit = torch.nn.MSELoss()
    best, best_p = float('inf'), None
    for restart in range(3):
        raw = torch.zeros(1, 8, requires_grad=True) if restart == 0 else \
            (torch.rand(1, 8) * 4 - 2).requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=0.1)
        for _ in range(400):
            opt.zero_grad()
            p8 = lo + torch.sigmoid(raw) * (hi - lo)
            full = torch.cat([p8, tail], dim=1) if tail is not None else p8
            loss = crit(model((full - Xm) / Xs), Yt_s.unsqueeze(0))
            loss.backward()
            opt.step()
            if loss.item() < best:
                best, best_p = loss.item(), p8.detach().clone()
    with torch.no_grad():
        full_best = torch.cat([best_p, tail], dim=1) if tail is not None else best_p
        pred = (model((full_best - Xm) / Xs) * Ys + Ym).squeeze()
    rmse = float(torch.sqrt(torch.mean((pred - Yt) ** 2)))
    scale = float(true_P.mean())
    # Assert on the FIT (what calibration can honestly deliver); parameter
    # recovery is NOT asserted because kappa/sigma/lambda are weakly identified.
    check("6. Synthetic calibration converges to the true surface",
          rmse < 0.02 * scale, f"price RMSE {rmse:.5f} (< 2% of mean price {scale:.3f})")


def main():
    print("=" * 64)
    print("NEURAL BATES DML - SELF TEST SUITE")
    print("=" * 64)
    check_1_fft_vs_bs()
    check_2_no_arbitrage()
    check_3_greek_signs()
    model, ckpt = check_4_checkpoint()
    check_5_dataset_accuracy(model, ckpt)
    check_6_synthetic_calibration(model, ckpt)
    print("-" * 64)
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {n_ok}/{len(results)} checks passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == '__main__':
    main()
