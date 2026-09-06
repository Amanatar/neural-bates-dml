import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bates_fft import price_bates_surface
from model import BatesNeuralSurrogate

# Dark theme for beautiful github visuals
plt.style.use('dark_background')

def generate_plots():
    print("Generating stunning DML visualization plots...")
    
    # 1. Setup the grids
    maturities = np.array([30/365, 60/365, 90/365, 120/365, 150/365])
    moneyness = np.linspace(0.8, 1.2, 11)
    
    X_mesh, Y_mesh = np.meshgrid(moneyness, maturities)
    
    # 2. Pick a random, realistic set of Bates parameters to plot
    # [v0, kappa, theta, sigma, rho, lambda_, muJ, vJ]
    params = [0.04, 2.0, 0.04, 0.3, -0.5, 0.1, -0.1, 0.1]
    
    # 3. Calculate True Prices (Carr-Madan FFT)
    S0 = 1.0
    r = 0.05
    q = 0.0
    print("Calculating True FFT Surface...")
    true_prices = price_bates_surface(S0, r, q, maturities, moneyness, params)
    
    # 4. Calculate DML Surrogate Prices
    print("Calculating DML Surrogate Surface...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('bates_surrogate_dml.pth', map_location=device, weights_only=False)
    
    model = BatesNeuralSurrogate(input_dim=8, output_dim=55).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X_mean = checkpoint['X_mean'].to(device)
    X_std = checkpoint['X_std'].to(device)
    Y_mean = checkpoint['Y_mean'].to(device)
    Y_std = checkpoint['Y_std'].to(device)
    
    x_input = torch.tensor(params, dtype=torch.float32).to(device)
    x_scaled = (x_input - X_mean) / X_std
    
    with torch.no_grad():
        y_scaled = model(x_scaled.unsqueeze(0)).squeeze(0)
        surrogate_prices_flat = (y_scaled * Y_std) + Y_mean
        
    surrogate_prices = surrogate_prices_flat.cpu().numpy().reshape(5, 11)
    
    # --- Plot 1: 3D Surface Comparison ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot True FFT as scatter points (cyan)
    ax.scatter(X_mesh, Y_mesh, true_prices, color='cyan', s=50, label='True Prices (FFT)', alpha=0.8)
    
    # Plot DML Surrogate as a wireframe surface (magenta)
    ax.plot_wireframe(X_mesh, Y_mesh, surrogate_prices, color='magenta', label='DML Surrogate', alpha=0.6)
    
    ax.set_title("Bates Model Option Pricing Surface\nDifferential Neural Surrogate vs Carr-Madan FFT", fontsize=16, pad=20, color='white')
    ax.set_xlabel("Moneyness (K/S0)", fontsize=12, labelpad=10)
    ax.set_ylabel("Maturity (Years)", fontsize=12, labelpad=10)
    ax.set_zlabel("Option Price", fontsize=12, labelpad=10)
    
    # Customize grid and panes for dark theme
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.legend(loc='upper left', fontsize=12, facecolor='black', edgecolor='white')
    ax.view_init(elev=25, azim=225)
    
    plt.tight_layout()
    plt.savefig('surface_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved 'surface_comparison.png'")
    
    # --- Plot 2: Absolute Error Heatmap ---
    fig2 = plt.figure(figsize=(10, 6))
    error = np.abs(true_prices - surrogate_prices)
    
    plt.imshow(error, cmap='magma', aspect='auto', origin='lower',
               extent=[moneyness.min(), moneyness.max(), maturities.min(), maturities.max()])
    
    cbar = plt.colorbar()
    cbar.set_label('Absolute Error', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.title("Absolute Pricing Error (DML vs FFT)", fontsize=16, color='white', pad=15)
    plt.xlabel("Moneyness (K/S0)", fontsize=12)
    plt.ylabel("Maturity (Years)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('error_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved 'error_heatmap.png'")

if __name__ == '__main__':
    generate_plots()
