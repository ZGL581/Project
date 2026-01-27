#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# Set font for Western character support
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams['axes.unicode_minus'] = False
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 1. Echo State Network (ESN) Reservoir Layer with Sparse Connections and Node Isolation
class EchoStateRNNCell:
    def __init__(self, num_units, num_inputs, key_size=32, activation=np.tanh, decay=0.1,
                 spectral_radius=0.8, sparsity_res=0.8,
                 input_sparsity=0.8,
                 input_fraction=0.5,
                 input_scale=1.0, key_scale=0.5, rng=None):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.key_size = key_size
        self.activation = activation
        self.decay = decay
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # Core: Partition reservoir nodes into "input-dedicated" and "output-dedicated" (non-overlapping)
        self.n_input_nodes = int(num_units * input_fraction)
        self.input_node_indices = np.arange(self.n_input_nodes)
        self.output_node_indices = np.arange(self.n_input_nodes, num_units)
        print(
            f"Reservoir Node Partitioning: {self.n_input_nodes} input-dedicated nodes, {len(self.output_node_indices)} output-dedicated nodes (no overlap)")

        # 1.1 Sparse input-reservoir connections (only to input-dedicated nodes)
        self.W_in = self.rng.randn(num_inputs, num_units) * input_scale
        input_mask = np.zeros((num_inputs, num_units))
        for i in range(num_inputs):
            input_mask[i, self.input_node_indices] = (self.rng.rand(self.n_input_nodes) > input_sparsity).astype(float)
        self.W_in *= input_mask

        # 1.2 Key-reservoir connections (only to input-dedicated nodes)
        self.W_key = self.rng.randn(key_size, num_units) * key_scale
        self.W_key *= input_mask

        # 1.3 Sparse internal reservoir connections
        W_res = self.rng.randn(num_units, num_units)
        mask_res = (self.rng.rand(num_units, num_units) > sparsity_res).astype(float)
        W_res *= mask_res
        radius = np.max(np.abs(np.linalg.eigvals(W_res))) if num_units > 0 else 1
        self.W_res = (W_res / radius) * spectral_radius

    def __call__(self, inputs, key_input, state):
        new_state = (1 - self.decay) * state + self.decay * self.activation(
            np.dot(inputs, self.W_in) +
            np.dot(state, self.W_res) +
            np.dot(key_input, self.W_key)
        )
        return new_state, new_state


# 2. Generate Mackey-Glass time series
def generate_mackey_glass(n_samples=10000, tau=17, delta_t=1, x0=1.2, a=0.2, b=0.1, n_timesteps=20):
    x = np.zeros(n_samples + tau)
    x[0] = x0
    for t in range(1, tau):
        x[t] = (1 - b * delta_t) * x[t - 1] + (a * delta_t * x[t - 1]) / (1 + x[t - 1] ** 10)
    for t in range(tau, n_samples + tau):
        x[t] = (1 - b * delta_t) * x[t - 1] + (a * delta_t * x[t - tau]) / (1 + x[t - tau] ** 10)

    series = x[tau:]
    X, y = [], []
    for i in range(n_timesteps, len(series)):
        X.append(series[i - n_timesteps:i])
        y.append(series[i])
    return np.array(X), np.array(y)


# 3. Generate static key
def generate_static_key(key_size, seed=789):
    """Generate fixed static key (binaryized)"""
    rng = np.random.RandomState(seed)
    static_key = rng.rand(key_size)
    static_key = (static_key > 0.5).astype(np.float32).reshape(1, key_size)
    return static_key


def generate_static_key_series(n_timesteps, key_size, seed=789):
    """Generate static key series: n_timesteps identical static keys"""
    static_key = generate_static_key(key_size, seed)
    key_series = [static_key for _ in range(n_timesteps)]
    return key_series


# 4. Collect ESN states
def collect_states_with_timesteps(X_data, esn_cell, key_series):
    n_samples = len(X_data)
    n_timesteps = X_data.shape[1]
    n_neurons = esn_cell.num_units
    all_states = []

    for i in range(n_samples):
        state = np.zeros((1, n_neurons))
        timestep_states = []
        for t in range(n_timesteps):
            input_t = X_data[i, t].reshape(1, -1)
            key_t = key_series[t]
            _, state = esn_cell(input_t, key_t, state)
            timestep_states.append(state.flatten())
        all_states.append(timestep_states)
    return np.array(all_states)


# 5. State dimensionality reduction and visualization
def visualize_reservoir_states(states, title, save_path, key_type, n_timesteps, output_dir):
    """Visualize reservoir states (3D PCA) and save trajectory data"""
    n_samples, _, n_neurons = states.shape
    target_step = n_timesteps - 1
    target_states = states[:, target_step, :]  # Select target timestep states for all samples

    # PCA dimensionality reduction to 3D
    pca = PCA(n_components=3, random_state=42)
    states_3d = pca.fit_transform(target_states)
    print(f"{key_type} PCA dimensionality reduction completed, explained variance ratio: {pca.explained_variance_ratio_}")

    # Save trajectory data to CSV
    trajectory_path = os.path.join(output_dir, 'trajectory_data.csv')
    file_exists = os.path.exists(trajectory_path)
    with open(trajectory_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Key_Type', 'Timestep', 'Dimension_1', 'Dimension_2', 'Dimension_3', 'Sample_Index'])
        for sample_idx in range(n_samples):
            writer.writerow([
                key_type, target_step,
                round(states_3d[sample_idx, 0], 6),
                round(states_3d[sample_idx, 1], 6),
                round(states_3d[sample_idx, 2], 6),
                sample_idx
            ])
    print(f"Trajectory data saved to: {trajectory_path}")

    # Plot 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    scatter = ax.scatter(
        states_3d[:, 0], states_3d[:, 1], states_3d[:, 2],
        c=colors, alpha=0.8, s=50, cmap='viridis'
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sample Index', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('PCA Dimension 1', fontsize=12)
    ax.set_ylabel('PCA Dimension 2', fontsize=12)
    ax.set_zlabel('PCA Dimension 3', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D visualization saved to: {save_path}")


# 6. Helper function to save states to CSV
def save_states_to_csv(states, filename, output_dir, n_neurons):
    """Save reservoir states to CSV file"""
    states_path = os.path.join(output_dir, filename)
    n_samples, n_timesteps, _ = states.shape

    with open(states_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Sample_Index', 'Timestep'] + [f'Neuron_{i}' for i in range(n_neurons)]
        writer.writerow(header)

        for sample_idx in range(n_samples):
            for timestep in range(n_timesteps):
                state = states[sample_idx, timestep, :]
                row = [sample_idx, timestep] + state.tolist()
                writer.writerow(row)
    print(f"{filename} saved to {states_path}")


# 7. Data preprocessing
n_timesteps = 20
X, y = generate_mackey_glass(n_samples=10000, tau=17, n_timesteps=n_timesteps)

scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False
)

# 8. Hyperparameter configuration
n_neurons = 512
n_inputs = 1
key_size = 32
n_outputs = 1
spectral_radius = 1.1
sparsity_res = 0.85
input_sparsity = 0.8
output_sparsity = 0.8
input_fraction = 0.5
epochs = 400
lr = 0.001
output_dir = "esn_mackey_glass_evaluation"  # Removed "fair" from directory name
os.makedirs(output_dir, exist_ok=True)

# 9. Initialize ESN
rng = np.random.RandomState(42)
esn_cell = EchoStateRNNCell(
    num_units=n_neurons,
    num_inputs=n_inputs,
    key_size=key_size,
    activation=np.tanh,
    decay=0.3,
    spectral_radius=spectral_radius,
    sparsity_res=sparsity_res,
    input_sparsity=input_sparsity,
    input_fraction=input_fraction,
    input_scale=0.5,
    key_scale=0.5,
    rng=rng
)

# 10. Pre-generate key series
print("=" * 50)
print("Generating keys...")
correct_key_seed = 789
wrong_key_seed = 234

correct_static_keys = generate_static_key_series(n_timesteps, key_size, seed=correct_key_seed)
wrong_static_keys = generate_static_key_series(n_timesteps, key_size, seed=wrong_key_seed)

print(f"Correct key seed: {correct_key_seed}")
print(f"Wrong key seed: {wrong_key_seed}")

# 11. Collect training states
print("=" * 50)
print("Collecting training states (correct key only)...")
train_states = collect_states_with_timesteps(X_train, esn_cell, correct_static_keys)
train_states_features = np.array([np.concatenate(sample[-5:]) for sample in train_states])

# 12. Collect test states
print("=" * 50)
print("Collecting and saving test states...")

# Collect samples for visualization (first 1000 test samples)
viz_samples = min(1000, len(X_test))
test_states_correct_viz = collect_states_with_timesteps(X_test[:viz_samples], esn_cell, correct_static_keys)
test_states_wrong_viz = collect_states_with_timesteps(X_test[:viz_samples], esn_cell, wrong_static_keys)

# Save complete states to CSV
save_states_to_csv(test_states_correct_viz, 'correct_states.csv', output_dir, n_neurons)
save_states_to_csv(test_states_wrong_viz, 'wrong_states.csv', output_dir, n_neurons)

# 13. 3D visualization and trajectory saving
print("=" * 50)
print("Generating 3D PCA visualizations...")

visualize_reservoir_states(
    states=test_states_correct_viz,
    title=f'Mackey-Glass Reservoir State Distribution with Correct Key',
    save_path=os.path.join(output_dir, 'states_correct.png'),  # Removed "last_timestep_" prefix
    key_type="Correct_Key",
    n_timesteps=n_timesteps,
    output_dir=output_dir
)

visualize_reservoir_states(
    states=test_states_wrong_viz,
    title=f'Mackey-Glass Reservoir State Distribution with Wrong Key',
    save_path=os.path.join(output_dir, 'states_wrong.png'),  # Removed "last_timestep_" prefix
    key_type="Wrong_Key",
    n_timesteps=n_timesteps,
    output_dir=output_dir
)

# 14. Prepare training and test features
print("=" * 50)
print("Preparing predictor data...")

# Training data
train_states_features = np.array([np.concatenate(sample[-5:]) for sample in train_states])

# Test data
test_states_correct_full = collect_states_with_timesteps(X_test, esn_cell, correct_static_keys)
test_states_wrong_full = collect_states_with_timesteps(X_test, esn_cell, wrong_static_keys)

test_states_correct_features = np.array([np.concatenate(sample[-5:]) for sample in test_states_correct_full])
test_states_wrong_features = np.array([np.concatenate(sample[-5:]) for sample in test_states_wrong_full])

# 15. Convert to PyTorch tensors
train_states_t = torch.FloatTensor(train_states_features)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

test_states_correct_t = torch.FloatTensor(test_states_correct_features)
test_states_wrong_t = torch.FloatTensor(test_states_wrong_features)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)


# 16. Sparse regression predictor
class SparseTimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, output_size, n_neurons, n_timesteps_used=5,
                 output_sparsity=0.8, input_node_indices=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_timesteps_used = n_timesteps_used
        self.output_sparsity = output_sparsity
        self.input_node_indices = input_node_indices
        self.output_node_indices = np.setdiff1d(np.arange(n_neurons), input_node_indices)

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.criterion = nn.MSELoss()
        self._apply_output_sparsity_mask()

    def _apply_output_sparsity_mask(self):
        mask = np.zeros_like(self.fc1.weight.data.numpy())
        for t in range(self.n_timesteps_used):
            start = t * self.n_neurons
            end = start + n_neurons
            mask[:, start + self.output_node_indices] = (
                    np.random.rand(mask.shape[0], len(self.output_node_indices)) > self.output_sparsity
            ).astype(float)
        self.fc1_mask = torch.FloatTensor(mask)
        self.fc1.weight.data *= self.fc1_mask

    def forward(self, x, gt_label=None):
        self.fc1.weight.data *= self.fc1_mask
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        output = self.fc3(x)

        if gt_label is not None:
            loss = self.criterion(output, gt_label)
            return {'loss': loss}, loss.item()
        return output


# 17. Training
model = SparseTimeSeriesPredictor(
    input_size=n_neurons * 5,
    output_size=n_outputs,
    n_neurons=n_neurons,
    output_sparsity=output_sparsity,
    input_node_indices=esn_cell.input_node_indices
)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

results = []
header = ['Epoch', 'Train MSE (Correct Key)', 'Test MSE (Correct Key)', 'Test MSE (Wrong Key)', 'Performance_Gap']
results.append(header)
best_test_mse = float('inf')

print("=" * 50)
print("Starting training (correct key data only)...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_metrics, train_mse = model(train_states_t, y_train_t)
    train_metrics['loss'].backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        # Evaluate: predict true values y_test with both keys
        _, test_mse_correct = model(test_states_correct_t, y_test_t)
        _, test_mse_wrong = model(test_states_wrong_t, y_test_t)

    scheduler.step(train_metrics['loss'])

    if test_mse_correct < best_test_mse:
        best_test_mse = test_mse_correct
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    gap = test_mse_wrong / test_mse_correct if test_mse_correct > 0 else float('inf')

    if (epoch + 1) % 20 == 0 or epoch == 0:
        log = f"Epoch {epoch + 1}/{epochs} | " \
              f"Train: {train_mse:.6f} | " \
              f"Test Correct: {test_mse_correct:.6f} | " \
              f"Test Wrong: {test_mse_wrong:.6f} | " \
              f"Gap: {gap:.2f}x"
        print(log)
    results.append([epoch + 1, train_mse, test_mse_correct, test_mse_wrong, gap])

# 18. Save results and final evaluation
with open(os.path.join(output_dir, 'evaluation_results.csv'), 'w', newline='') as f:  # Removed "fair_" prefix
    writer = csv.writer(f)
    writer.writerows(results)

# Load best model
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
model.eval()
with torch.no_grad():
    pred_correct = model(test_states_correct_t).numpy()
    pred_wrong = model(test_states_wrong_t).numpy()

# Inverse standardization
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
pred_correct_original = scaler_y.inverse_transform(pred_correct).flatten()
pred_wrong_original = scaler_y.inverse_transform(pred_wrong).flatten()

# Calculate final MSE
from sklearn.metrics import mean_squared_error

mse_correct_final = mean_squared_error(y_test_original, pred_correct_original)
mse_wrong_final = mean_squared_error(y_test_original, pred_wrong_original)

print("=" * 50)
print("Final Evaluation Results (original value scale):")
print(f"Correct key MSE: {mse_correct_final:.6f}")
print(f"Wrong key MSE: {mse_wrong_final:.6f}")
print(f"Performance gap: {mse_wrong_final / mse_correct_final:.2f}x")

# Save detailed prediction comparison
with open(os.path.join(output_dir, 'prediction_comparison.csv'), 'w', newline='') as f:  # Removed "_fair" suffix
    writer = csv.writer(f)
    writer.writerow(['Timestep', 'True_Value', 'Correct_Key_Prediction', 'Wrong_Key_Prediction',
                     'Correct_Key_Absolute_Error', 'Wrong_Key_Absolute_Error'])
    for i in range(len(y_test_original)):
        writer.writerow([
            i,
            round(y_test_original[i], 6),
            round(pred_correct_original[i], 6),
            round(pred_wrong_original[i], 6),
            round(abs(y_test_original[i] - pred_correct_original[i]), 6),
            round(abs(y_test_original[i] - pred_wrong_original[i]), 6)
        ])

# Visualization comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(y_test_original[:200], label='True Value', linewidth=2, color='black')
plt.plot(pred_correct_original[:200], label='Correct Key', linestyle='--', color='blue')
plt.plot(pred_wrong_original[:200], label='Wrong Key', linestyle=':', color='red', alpha=0.7)
plt.title('Sequence Prediction Comparison (First 200 Steps)')
plt.xlabel('Timestep')
plt.ylabel('Sequence Value')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
errors_correct = np.abs(y_test_original - pred_correct_original)
errors_wrong = np.abs(y_test_original - pred_wrong_original)
plt.hist(errors_correct, bins=50, alpha=0.7, label='Correct Key', color='blue', density=True)
plt.hist(errors_wrong, bins=50, alpha=0.7, label='Wrong Key', color='red', density=True)
plt.xlabel('Absolute Error')
plt.ylabel('Density')
plt.title('Error Distribution')
plt.legend()
plt.yscale('log')

plt.subplot(1, 3, 3)
plt.scatter(y_test_original, pred_correct_original, alpha=0.5, s=10, label='Correct Key', color='blue')
plt.scatter(y_test_original, pred_wrong_original, alpha=0.5, s=10, label='Wrong Key', color='red')
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()], 'k--', label='Perfect Prediction')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Prediction vs True Value Scatter Plot')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'evaluation_summary.png'), dpi=300, bbox_inches='tight')  # Removed "fair_" prefix
plt.close()

print(f"\nAll results saved to: {output_dir}")
print("Generated files:")
print("  - correct_states.csv / wrong_states.csv (complete reservoir states)")
print("  - trajectory_data.csv (PCA trajectory data)")
print("  - states_correct.png / states_wrong.png (3D PCA plots)")
print("  - evaluation_results.csv (training logs)")
print("  - prediction_comparison.csv (prediction comparison)")
print("  - evaluation_summary.png (comprehensive visualization)")