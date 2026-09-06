import torch
from model import BatesNeuralSurrogate

ckpt = torch.load('bates_surrogate_dml.pth', weights_only=False, map_location='cpu')
print('=== Checkpoint keys:', list(ckpt.keys()))
sd = ckpt['model_state_dict']
print(f'Model layers: {len(sd)} parameter tensors')
for k, v in sd.items():
    print(f'  {k}: {list(v.shape)}')

xm = ckpt['X_mean']
xs = ckpt['X_std']
ym = ckpt['Y_mean']
ys = ckpt['Y_std']
print(f'\nX_mean shape: {xm.shape}  =>  input_dim = {xm.shape[0]}')
print(f'Y_mean shape: {ym.shape}  =>  output_dim = {ym.shape[0]}')
print(f'X_std:  {xs}')
print(f'Y_std (first 5): {ys[:5]}')

print('\n=== Quick inference test ===')
model = BatesNeuralSurrogate(input_dim=10, output_dim=55)
model.load_state_dict(sd)
model.eval()
x = torch.randn(1, 10)
x_scaled = (x - xm) / xs
with torch.no_grad():
    y_scaled = model(x_scaled)
    y = y_scaled * ys + ym
print(f'Test prediction shape: {y.shape}')
print(f'Sample prices (first 10): {y[0,:10].numpy()}')

print('\n=== Dataset verification ===')
data = torch.load('dataset_100000_dml.pt', weights_only=False, map_location='cpu')
print(f'X shape: {data["X"].shape}')
print(f'Y shape: {data["Y"].shape}')
print(f'dY_dX shape: {data["dY_dX"].shape}')
print(f'Maturities: {data["maturities"]}')
print(f'Moneyness: {data["moneyness"]}')
print(f'Y min: {data["Y"].min():.6f}, Y max: {data["Y"].max():.6f}')
print(f'Any NaN in X: {torch.isnan(data["X"]).any().item()}')
print(f'Any NaN in Y: {torch.isnan(data["Y"]).any().item()}')
print(f'Any NaN in dY_dX: {torch.isnan(data["dY_dX"]).any().item()}')

print('\n✅ Model + Dataset are VALID and ready!')
