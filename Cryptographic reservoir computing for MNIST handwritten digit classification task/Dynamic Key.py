#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import os
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from numpy.random import RandomState
from scipy.stats import entropy as shannon_entropy
import warnings

warnings.filterwarnings('ignore')

# Set plot font to support Western characters (removed SimHei as it's for Chinese)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 1. Improved ESN Cell
class EchoStateRNNCell:
    def __init__(self, num_units, num_inputs, key_size_per_step=32, activation=np.tanh, decay=0.3,
                 spectral_radius=0.7, sparsity=0.7, input_scale=0.8, key_scale=0.8, rng=None):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.key_size_per_step = key_size_per_step  # Key length per time step
        self.activation = activation
        self.decay = decay
        self.rng = rng if rng is not None else RandomState(42)

        # Input weights
        self.W_in = self.rng.randn(num_inputs, num_units) * input_scale * 0.5

        # Enhance key weight impact (amplify effect of single bit changes)
        self.W_key_correct = self.rng.randn(key_size_per_step, num_units) * key_scale * 2.0
        self.W_key_wrong = self.rng.randn(key_size_per_step, num_units) * key_scale * 2.0

        # Reservoir weights
        W_res = self.rng.randn(num_units, num_units)
        mask = (self.rng.rand(num_units, num_units) > sparsity).astype(float)
        W_res *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W_res)))
        self.W_res = (W_res / radius) * spectral_radius

    def __call__(self, inputs, key_input_step, state, is_correct_key):
        """Adapt to dynamic keys: pass corresponding time step key each step"""
        if is_correct_key:
            key_term = np.dot(key_input_step, self.W_key_correct) * 1.0  # Amplify correct key term
            pre_activation = (
                    np.dot(inputs, self.W_in) * 0.6 +
                    np.dot(state, self.W_res) * 1.1 +
                    key_term
            )
        else:
            key_term = np.dot(key_input_step, self.W_key_wrong) * 1.0  # Amplify wrong key term
            pre_activation = (
                    np.dot(inputs, self.W_in) * 0.6 +
                    np.dot(state, self.W_res) * 1.1 +
                    key_term
            )

        new_state = (1 - self.decay) * state + self.decay * self.activation(pre_activation)
        return new_state, new_state


print("start")
# 2. Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.int64)  # Convert labels to integer type
X = X.reshape(-1, 28, 28).astype(np.float32) / 255.0  # (N, 28, 28)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Hyperparameters
n_neurons = 256
n_inputs = 28
key_size_per_step = 32  # Key length per time step
dynamic_steps = 20  # Dynamic key steps (total dynamic key length = 32×20=640 bits)
n_steps = 28  # Total time steps
n_outputs = 10
batch_size = 128
spectral_radius = 0.7
sparsity = 0.7
epochs = 800
lr = 1e-5
output_dir = "esn_mnist_results_dynamic_key"
temperature = 0.5

# Create subdirectory structure
subdirs = [
    os.path.join(output_dir, "confusion_matrices"),
    os.path.join(output_dir, "single_sample_states"),
    os.path.join(output_dir, "single_sample_3d"),
    os.path.join(output_dir, "multi_sample_final_states"),
    os.path.join(output_dir, "multi_sample_3d")
]
crack_dir = os.path.join(output_dir, "crack_results")
os.makedirs(crack_dir, exist_ok=True)
for subdir in subdirs:
    os.makedirs(subdir, exist_ok=True)

# 4. Initialize ESN
rng = RandomState(42)
esn_cell = EchoStateRNNCell(
    num_units=n_neurons,
    num_inputs=n_inputs,
    key_size_per_step=key_size_per_step,
    activation=np.tanh,
    decay=0.2,
    spectral_radius=spectral_radius,
    sparsity=sparsity,
    input_scale=1.2,
    key_scale=1.5,
    rng=rng
)


# 5. Dynamic key generation
def generate_dynamic_key(batch_size, key_size_per_step, dynamic_steps, total_steps=28, correct_key=True,
                         rng=RandomState(42), base_seed=12345):
    """
    Generate dynamic keys (improved version):
    - correct_key=True: Generate different 32-bit random patterns for first 20 steps based on different seeds
    """
    if correct_key:
        # Deterministic generation: use different seed for each step to ensure fixed but varied 32-bit patterns
        key_base = np.zeros((dynamic_steps, key_size_per_step), dtype=float)
        for step in range(dynamic_steps):
            # Use step + base seed as deterministic seed to generate different random patterns per step
            step_rng = RandomState(base_seed + step)
            key_base[step] = (step_rng.rand(key_size_per_step) > 0.5).astype(float)

        # Reuse step 20 (index 19) key for last 8 steps
        key_extend = np.tile(key_base[-1], (total_steps - dynamic_steps, 1))
        key_full = np.vstack([key_base, key_extend])  # (28, 32)
        key = np.tile(key_full, (batch_size, 1, 1))  # (batch_size, 28, 32)
    else:
        # Wrong dynamic key: fully random generation
        key_random = (rng.rand(batch_size, dynamic_steps, key_size_per_step) > 0.5).astype(float)
        key_extend = np.tile(key_random[:, -1:], (1, total_steps - dynamic_steps, 1))  # Reuse last step
        key = np.concatenate([key_random, key_extend], axis=1)  # (batch_size, 28, 32)
    return key


# 6. Enhanced state collection
def collect_states(X_data, esn_cell, use_correct_key=True, return_all_timesteps=False, custom_key=None,
                   rng=RandomState(42), base_seed=12345):
    n_samples = len(X_data)
    states = []

    for i in range(n_samples):
        state = np.zeros((1, n_neurons), dtype=np.float32)
        # Generate dynamic key
        if custom_key is not None:
            # Custom key must be in (28, 32) shape
            key_inputs = custom_key.reshape(n_steps, key_size_per_step).astype(np.float32)
            is_correct = False
        else:
            key_inputs = generate_dynamic_key(1, key_size_per_step, dynamic_steps, n_steps, use_correct_key,
                                              rng, base_seed).squeeze(0)
            is_correct = use_correct_key

        all_timestates = []
        for t in range(n_steps):
            # Use corresponding time step dynamic key for each step
            key_input_step = key_inputs[t].reshape(1, -1)
            _, state = esn_cell(X_data[i, t].reshape(1, -1), key_input_step, state, is_correct)
            all_timestates.append(state.flatten().astype(np.float32))

        states_array = np.array(all_timestates, dtype=np.float32)

        # Jump detection and smoothing
        diffs = np.linalg.norm(states_array[1:] - states_array[:-1], axis=1)
        if np.any(diffs > np.mean(diffs) * 2):
            def smooth_col(col):
                return np.convolve(col, np.ones(3) / 3, mode='same')

            states_array = np.apply_along_axis(smooth_col, axis=0, arr=states_array)

        if return_all_timesteps:
            states.append(states_array)
        else:
            states.append(np.concatenate(all_timestates))

    return np.array(states)


# 7. Collect training and testing states
print("Collecting states with dynamic key...")
base_seed = 12345  # Base seed, can be changed to replace entire key sequence
train_states_correct = collect_states(X_train, esn_cell, use_correct_key=True, rng=rng, base_seed=base_seed)
# Note: Wrong key states are only used in testing phase, not collected in training phase
test_states_correct = collect_states(X_test, esn_cell, use_correct_key=True, rng=rng, base_seed=base_seed)
test_states_wrong = collect_states(X_test, esn_cell, use_correct_key=False, rng=rng, base_seed=base_seed)

# Convert to PyTorch tensors
train_states_correct_t = torch.FloatTensor(train_states_correct)
y_train_t = torch.LongTensor(y_train)
test_states_correct_t = torch.FloatTensor(test_states_correct)
test_states_wrong_t = torch.FloatTensor(test_states_wrong)
y_test_t = torch.LongTensor(y_test)


# 8. Classifier definition
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.fc(x)


# 9. Inference with temperature scaling
def predict_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=1)
    return torch.multinomial(probs, num_samples=1).squeeze()


# 10. Training configuration
model = Classifier(n_neurons * n_steps, n_outputs)  # Input dimension = 256×28
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Only use dataset with correct key for training
dataset_correct = TensorDataset(train_states_correct_t, y_train_t)
loader_correct = DataLoader(dataset_correct, batch_size=batch_size, shuffle=True)

# 11. Training loop
results = []
header = ['Epoch', 'Train Loss', 'Train Acc (Correct)', 'Test Acc (Correct Key)', 'Test Acc (Wrong Key)']
results.append(header)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Only use batches with correct key for training
    for correct_batch, correct_labels in loader_correct:
        optimizer.zero_grad()

        # Forward propagation
        output_correct = model(correct_batch)

        # Calculate loss
        loss = nn.functional.cross_entropy(output_correct, correct_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * correct_batch.size(0)
        pred_correct = output_correct.argmax(dim=1)
        total_correct += (pred_correct == correct_labels).sum().item()
        total_samples += correct_batch.size(0)

    train_loss = total_loss / total_samples if total_samples > 0 else 0
    train_acc = total_correct / total_samples if total_samples > 0 else 0

    # Validation
    model.eval()
    with torch.no_grad():
        output_test_correct = model(test_states_correct_t)
        pred_test_correct = predict_with_temperature(output_test_correct, temperature)
        test_acc_correct = (pred_test_correct == y_test_t).sum().item() / len(y_test_t)

        output_test_wrong = model(test_states_wrong_t)
        pred_test_wrong = predict_with_temperature(output_test_wrong, temperature)
        test_acc_wrong = (pred_test_wrong == y_test_t).sum().item() / len(y_test_t)

    scheduler.step(train_loss)

    log = f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | " \
          f"Train Acc: {train_acc * 100:.2f}% | Test Acc (Correct): {test_acc_correct * 100:.2f}% | " \
          f"Test Acc (Wrong): {test_acc_wrong * 100:.2f}%"
    print(log)

    results.append([
        epoch + 1,
        train_loss,
        train_acc * 100,
        test_acc_correct * 100,
        test_acc_wrong * 100
    ])

# Save training results
with open(os.path.join(output_dir, 'training_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(results)


# 12. Confusion matrix
def plot_and_save_confusion_matrix(y_true, y_pred, title, save_path_data, save_path_img):
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(save_path_data, cm, delimiter=',', fmt='%d')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path_img)
    plt.close()


model.eval()
with torch.no_grad():
    logits_correct = model(test_states_correct_t)
    pred_correct = predict_with_temperature(logits_correct, temperature).numpy()

    logits_wrong = model(test_states_wrong_t)
    pred_wrong = predict_with_temperature(logits_wrong, temperature).numpy()

plot_and_save_confusion_matrix(
    y_test, pred_correct,
    'Confusion Matrix (Correct Dynamic Key)',
    os.path.join(subdirs[0], 'cm_correct.csv'),
    os.path.join(subdirs[0], 'cm_correct.png')
)

plot_and_save_confusion_matrix(
    y_test, pred_wrong,
    'Confusion Matrix (Wrong Dynamic Key)',
    os.path.join(subdirs[0], 'cm_wrong.csv'),
    os.path.join(subdirs[0], 'cm_wrong.png')
)


# 13. Single sample acquisition
def get_samples_per_class(X, y, n_per_class=1, random_state=42):
    samples = []
    labels = []
    rng = RandomState(random_state)
    for cls in range(10):
        cls_indices = np.where(y == cls)[0]
        selected = rng.choice(cls_indices, size=n_per_class, replace=False)
        samples.append(X[selected])
        labels.append(y[selected])
    return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)


# 14. 3D visualization
def reduce_and_visualize_single(states, label, is_correct, idx):
    time_weights = np.linspace(1, 3, n_steps).reshape(-1, 1)
    states_weighted = states * time_weights

    pca = PCA(n_components=3)
    states_3d = pca.fit_transform(states_weighted)

    key_type = 'correct' if is_correct else 'wrong'
    np.savetxt(
        os.path.join(subdirs[2], f'3d_{key_type}_cls{label}_idx{idx}.csv'),
        states_3d, delimiter=',', fmt='%.18e'
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)

    ax.plot(
        states_3d[:, 0], states_3d[:, 1], states_3d[:, 2],
        color='#1f77b4' if is_correct else '#ff7f0e',
        linewidth=5, alpha=0.9, zorder=2
    )

    ax.scatter(
        states_3d[:, 0], states_3d[:, 1], states_3d[:, 2],
        c=np.linspace(0, 1, n_steps),
        cmap='viridis', s=30, edgecolors='none', zorder=1
    )

    for t in [0, 13, 27]:
        ax.text(
            states_3d[t, 0], states_3d[t, 1], states_3d[t, 2],
            f't={t}', fontsize=10, fontweight='bold', color='black', zorder=3
        )

    ax.set_title(f'3D State Trajectory ({"Correct" if is_correct else "Wrong"} Dynamic Key, Class {label})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(subdirs[2], f'3d_{key_type}_cls{label}_idx{idx}.png'),
        dpi=200
    )
    plt.close()


# 15. Generate single sample results
single_samples, single_labels = get_samples_per_class(X_test, y_test, n_per_class=1, random_state=42)
single_states_correct = collect_states(single_samples, esn_cell, use_correct_key=True, return_all_timesteps=True,
                                       rng=rng, base_seed=base_seed)
single_states_wrong = collect_states(single_samples, esn_cell, use_correct_key=False, return_all_timesteps=True,
                                     rng=rng, base_seed=base_seed)

for i in range(10):
    cls = single_labels[i]
    np.savetxt(
        os.path.join(subdirs[1], f'state_correct_cls{cls}_idx{i}.csv'),
        single_states_correct[i], delimiter=','
    )
    np.savetxt(
        os.path.join(subdirs[1], f'state_wrong_cls{cls}_idx{i}.csv'),
        single_states_wrong[i], delimiter=','
    )
    reduce_and_visualize_single(single_states_correct[i], cls, is_correct=True, idx=i)
    reduce_and_visualize_single(single_states_wrong[i], cls, is_correct=False, idx=i)

# 16. Multi-sample final states
multi_samples, multi_labels = get_samples_per_class(X_test, y_test, n_per_class=20, random_state=42)
multi_states_correct = collect_states(multi_samples, esn_cell, use_correct_key=True, return_all_timesteps=False,
                                      rng=rng, base_seed=base_seed)
multi_states_wrong = collect_states(multi_samples, esn_cell, use_correct_key=False, return_all_timesteps=False,
                                    rng=rng, base_seed=base_seed)

np.savetxt(os.path.join(subdirs[3], 'final_states_correct.csv'), multi_states_correct, delimiter=',')
np.savetxt(os.path.join(subdirs[3], 'labels_correct.csv'), multi_labels, delimiter=',', fmt='%d')
np.savetxt(os.path.join(subdirs[3], 'final_states_wrong.csv'), multi_states_wrong, delimiter=',')
np.savetxt(os.path.join(subdirs[3], 'labels_wrong.csv'), multi_labels, delimiter=',', fmt='%d')


# 17. Multi-sample 3D visualization
def reduce_and_visualize_multi(states, labels, is_correct):
    pca = PCA(n_components=3)
    states_3d = pca.fit_transform(states)

    key_type = 'correct' if is_correct else 'wrong'
    np.savetxt(
        os.path.join(subdirs[4], f'3d_{key_type}_all.csv'),
        states_3d, delimiter=','
    )

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for cls in range(10):
        mask = (labels == cls)
        ax.scatter(
            states_3d[mask, 0], states_3d[mask, 1], states_3d[mask, 2],
            label=f'Class {cls}', alpha=0.7, s=50
        )
    ax.set_title(f'3D Final States ({"Correct" if is_correct else "Wrong"} Dynamic Key)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs[4], f'3d_{key_type}_all.png'))
    plt.close()


reduce_and_visualize_multi(multi_states_correct, multi_labels, True)
reduce_and_visualize_multi(multi_states_wrong, multi_labels, False)

# 18. Probability distribution comparison
prob_dist_dir = os.path.join(output_dir, "probability_distributions")
os.makedirs(prob_dist_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    single_states_correct_prob = collect_states(single_samples, esn_cell, use_correct_key=True,
                                                return_all_timesteps=False, rng=rng, base_seed=base_seed)
    single_states_wrong_prob = collect_states(single_samples, esn_cell, use_correct_key=False,
                                              return_all_timesteps=False, rng=rng, base_seed=base_seed)

    single_states_correct_t = torch.FloatTensor(single_states_correct_prob)
    single_states_wrong_t = torch.FloatTensor(single_states_wrong_prob)

    outputs_correct = model(single_states_correct_t)
    probs_correct = nn.functional.softmax(outputs_correct, dim=1).numpy()

    outputs_wrong = model(single_states_wrong_t)
    probs_wrong = nn.functional.softmax(outputs_wrong, dim=1).numpy()

for i in range(10):
    cls = single_labels[i]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Sample {i} (True Class: {cls}) Prediction Probability Distribution (Dynamic Key)', fontsize=14)

    ax1.bar(range(10), probs_correct[i], color='green', alpha=0.7)
    ax1.set_title('Correct Dynamic Key', fontsize=12)
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('Probability')
    ax1.set_xticks(range(10))
    ax1.set_ylim(0, 1.0)
    ax1.axvline(x=cls, color='red', linestyle='--', label=f'True Class: {cls}')
    ax1.legend()

    ax2.bar(range(10), probs_wrong[i], color='red', alpha=0.7)
    ax2.set_title('Wrong Dynamic Key', fontsize=12)
    ax2.set_xlabel('Predicted Class')
    ax2.set_ylabel('Probability')
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 1.0)
    ax2.axvline(x=cls, color='red', linestyle='--', label=f'True Class: {cls}')
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(prob_dist_dir, f'prob_dist_cls{cls}_idx{i}.png'))
    plt.close()

print(f"\nAll results have been saved to the {output_dir} directory")