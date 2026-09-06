import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.func import vmap, jacrev
from model import BatesNeuralSurrogate
import time

def train_dml_surrogate(dataset_path='dataset_100000_dml.pt', epochs=100, batch_size=512, alpha=1.0, seed=42):
    # Seed for reproducible weight init / batch shuffling
    torch.manual_seed(seed)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load DML data
    data = torch.load(dataset_path, weights_only=False)
    X = data['X'].to(device)
    Y = data['Y'].to(device)
    dY_dX = data['dY_dX'].to(device) # Shape: (N, num_prices, num_params)
    
    # Standardize inputs
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    X_scaled = (X - X_mean) / X_std
    
    # Standardize outputs (prices)
    Y_mean = Y.mean(dim=0)
    Y_std = Y.std(dim=0)
    Y_scaled = (Y - Y_mean) / Y_std
    
    # Note: By chain rule, if Y_scaled = (Y - Y_mean)/Y_std and X_scaled = (X - X_mean)/X_std
    # d(Y_scaled)/d(X_scaled) = (dY/dX) * (X_std / Y_std)
    # We must scale the target Greeks to match the scaled network output
    # Reshape for broadcasting
    X_std_reshaped = X_std.view(1, 1, -1)
    Y_std_reshaped = Y_std.view(1, -1, 1)
    
    dY_dX_scaled = dY_dX * (X_std_reshaped / Y_std_reshaped)
    
    dataset = TensorDataset(X_scaled, Y_scaled, dY_dX_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=False)
    
    # Initialize model (input dim taken from the dataset: 8D legacy or 10D with r, q)
    model = BatesNeuralSurrogate(input_dim=X.shape[1], output_dim=Y.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Function for jacrev
    def predict(x_single):
        return model(x_single.unsqueeze(0)).squeeze(0)
    
    compute_batch_jacobian = vmap(jacrev(predict))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    
    print("Starting Differential Machine Learning (Sobolev) training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        mse_price_sum = 0.0
        mse_greeks_sum = 0.0
        
        for batch_X, batch_Y, batch_dY_dX in dataloader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)
            batch_dY_dX = batch_dY_dX.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Predict prices
            outputs = model(batch_X)
            
            # Predict Greeks (Jacobian)
            # Jacobian shape: (batch, num_prices, num_params)
            pred_dY_dX = compute_batch_jacobian(batch_X)
            
            # Sobolev Loss with OTM-weighted price term.
            # IMPORTANT: weight by UNSCALED positive prices. batch_Y is
            # z-scored and takes negative values, so 1/(batch_Y + eps) would
            # produce negative or exploding weights and corrupt the loss.
            prices_unscaled = batch_Y * Y_std + Y_mean
            weights = 1.0 / (prices_unscaled + 0.01)
            weights = weights / weights.mean()  # keep scale comparable across batches
            loss_price = torch.mean(weights * (outputs - batch_Y) ** 2)
            # Standard MSE for Jacobian
            loss_greeks = nn.functional.mse_loss(pred_dY_dX, batch_dY_dX)
            
            loss = loss_price + alpha * loss_greeks
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            mse_price_sum += loss_price.item() * batch_X.size(0)
            mse_greeks_sum += loss_greeks.item() * batch_X.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        avg_price_loss = mse_price_sum / len(dataloader.dataset)
        avg_greeks_loss = mse_greeks_sum / len(dataloader.dataset)
        
        scheduler.step(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': X_mean,
                'X_std': X_std,
                'Y_mean': Y_mean,
                'Y_std': Y_std
            }, 'bates_surrogate_dml.pth')
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Total Loss: {epoch_loss:.6f} | Price MSE: {avg_price_loss:.6f} | Greeks MSE: {avg_greeks_loss:.6f} | {time.time()-start_time:.0f}s")
            
    print(f"Training complete in {time.time() - start_time:.2f}s. Best model saved to 'bates_surrogate_dml.pth'.")

if __name__ == '__main__':
    # 100k samples x 100 epochs x batch 512 ~= 19.6k optimizer steps, sized for
    # a Kaggle GPU session. For the legacy 1k local dataset use epochs=500.
    train_dml_surrogate(epochs=100)
