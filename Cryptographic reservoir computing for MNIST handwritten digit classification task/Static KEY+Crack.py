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

# Set plot font to support Western characters
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 1. Improved ESN Cell
class EchoStateRNNCell:
    def __init__(self, num_units, num_inputs, key_size=10, activation=np.tanh, decay=0.3,
                 spectral_radius=0.7, sparsity=0.7, input_scale=0.8, key_scale=0.8, rng=None):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.key_size = key_size
        self.activation = activation
        self.decay = decay
        self.rng = rng if rng is not None else RandomState(42)

        # Input weights
        self.W_in = self.rng.randn(num_inputs, num_units) * input_scale * 0.5

        # Key weights (amplify single bit impact)
        self.W_key_correct = self.rng.randn(key_size, num_units) * key_scale * 2.0
        self.W_key_wrong = self.rng.randn(key_size, num_units) * key_scale * 2.0

        # Reservoir weights
        W_res = self.rng.randn(num_units, num_units)
        mask = (self.rng.rand(num_units, num_units) > sparsity).astype(float)
        W_res *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W_res)))
        self.W_res = (W_res / radius) * spectral_radius

    def __call__(self, inputs, key_input, state, is_correct_key):
        """Amplify the impact of single key bits on reservoir states"""
        if is_correct_key:
            key_term = np.dot(key_input, self.W_key_correct) * 1.0  # Amplify correct key term
            pre_activation = (
                    np.dot(inputs, self.W_in) * 0.6 +
                    np.dot(state, self.W_res) * 1.1 +
                    key_term
            )
        else:
            key_term = np.dot(key_input, self.W_key_wrong) * 1.0  # Amplify wrong key term
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
key_size = 32
n_steps = 28
n_outputs = 10
batch_size = 128
spectral_radius = 0.7  # Reduce spectral radius to enhance state stability
sparsity = 0.7  # Reduce sparsity to enhance connections
epochs = 800  # Reduce training epochs to avoid overfitting
lr = 1e-5  # Increase learning rate
output_dir = "esn_mnist_results_static"
temperature = 0.5  # Increase temperature to enhance probability discrimination

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
    key_size=key_size,
    activation=np.tanh,
    decay=0.2,  # Increase decay rate to enhance state updates
    spectral_radius=spectral_radius,
    sparsity=sparsity,
    input_scale=1.2,  # Increase input scaling
    key_scale=1.5,  # Increase key scaling
    rng=rng
)


# 5. Key generation
def generate_key(batch_size, key_size, correct_key=True, rng=RandomState(42)):
    """Generate encryption key: generate specific pattern when correct_key=True, random otherwise"""
    if correct_key:
        # Custom 32-bit correct key
        custom_key = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1,
                      0]  # 32-bit example
        key = np.array(custom_key, dtype=int)  # Convert to numpy array
        key = np.tile(key, (batch_size, 1))  # Extend to batch dimension without changing key itself
    else:
        key = rng.rand(batch_size, key_size)
        key = (key > 0.5).astype(float)
    return key


# 6. Enhanced state collection (use float32 to save memory)
def collect_states(X_data, esn_cell, use_correct_key=True, return_all_timesteps=False, custom_key=None,
                   rng=RandomState(42)):
    n_samples = len(X_data)
    states = []

    for i in range(n_samples):
        state = np.zeros((1, n_neurons), dtype=np.float32)
        if custom_key is not None:
            key_input = custom_key.reshape(1, -1).astype(np.float32)
            is_correct = False
        else:
            key_input = generate_key(1, key_size, use_correct_key, rng).astype(np.float32)
            is_correct = use_correct_key

        all_timestates = []  # Defined variable name is all_timesteps
        for t in range(n_steps):
            _, state = esn_cell(X_data[i, t].reshape(1, -1), key_input, state, is_correct)
            all_timestates.append(state.flatten().astype(np.float32))

        states_array = np.array(all_timestates, dtype=np.float32)

        # Optional: Keep jump detection (smooth abnormal states)
        diffs = np.linalg.norm(states_array[1:] - states_array[:-1], axis=1)
        if np.any(diffs > np.mean(diffs) * 2):
            def smooth_col(col):
                return np.convolve(col, np.ones(3) / 3, mode='same')

            states_array = np.apply_along_axis(smooth_col, axis=0, arr=states_array)

        if return_all_timesteps:
            states.append(states_array)
        else:
            last_states = all_timestates[-28:]
            states.append(np.concatenate(last_states))

    return np.array(states)


# 7. Collect training and testing states
print("Collecting states...")
# Training set: only collect states with correct key
train_states_correct = collect_states(X_train, esn_cell, use_correct_key=True, rng=rng)
# Test set: collect states with both correct and wrong keys (for evaluation comparison)
test_states_correct = collect_states(X_test, esn_cell, use_correct_key=True, rng=rng)
test_states_wrong = collect_states(X_test, esn_cell, use_correct_key=False, rng=rng)

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
model = Classifier(n_neurons * 28, n_outputs)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Only use dataset with correct key for training
dataset_correct = TensorDataset(train_states_correct_t, y_train_t)
loader_correct = DataLoader(dataset_correct, batch_size=batch_size, shuffle=True)

# 11. Training loop (modified: only train with correct key, remove wrong key training logic)
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
    'Confusion Matrix (Correct Key)',
    os.path.join(subdirs[0], 'cm_correct.csv'),
    os.path.join(subdirs[0], 'cm_correct.png')
)

plot_and_save_confusion_matrix(
    y_test, pred_wrong,
    'Confusion Matrix (Wrong Key)',
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


# 14. Enhanced 3D visualization
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

    ax.set_title(f'3D State Trajectory ({"Correct" if is_correct else "Wrong"} Key, Class {label})')
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
                                       rng=rng)
single_states_wrong = collect_states(single_samples, esn_cell, use_correct_key=False, return_all_timesteps=True,
                                     rng=rng)

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
                                      rng=rng)
multi_states_wrong = collect_states(multi_samples, esn_cell, use_correct_key=False, return_all_timesteps=False, rng=rng)

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
    ax.set_title(f'3D Final States ({"Correct" if is_correct else "Wrong"} Key)')
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
                                                return_all_timesteps=False, rng=rng)
    single_states_wrong_prob = collect_states(single_samples, esn_cell, use_correct_key=False,
                                              return_all_timesteps=False, rng=rng)

    single_states_correct_t = torch.FloatTensor(single_states_correct_prob)
    single_states_wrong_t = torch.FloatTensor(single_states_wrong_prob)

    outputs_correct = model(single_states_correct_t)
    probs_correct = nn.functional.softmax(outputs_correct, dim=1).numpy()

    outputs_wrong = model(single_states_wrong_t)
    probs_wrong = nn.functional.softmax(outputs_wrong, dim=1).numpy()

for i in range(10):
    cls = single_labels[i]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Sample {i} (True Class: {cls}) Prediction Probability Distribution', fontsize=14)

    ax1.bar(range(10), probs_correct[i], color='green', alpha=0.7)
    ax1.set_title('Correct Key', fontsize=12)
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('Probability')
    ax1.set_xticks(range(10))
    ax1.set_ylim(0, 1.0)
    ax1.axvline(x=cls, color='red', linestyle='--', label=f'True Class: {cls}')
    ax1.legend()

    ax2.bar(range(10), probs_wrong[i], color='red', alpha=0.7)
    ax2.set_title('Wrong Key', fontsize=12)
    ax2.set_xlabel('Predicted Class')
    ax2.set_ylabel('Probability')
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 1.0)
    ax2.axvline(x=cls, color='red', linestyle='--', label=f'True Class: {cls}')
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(prob_dist_dir, f'prob_dist_cls{cls}_idx{i}.png'))
    plt.close()

# ===================== Improved Multi-round Key Cracking Module =====================
print("\n===================== Starting Multi-round Key Cracking =====================")


# 19. Entropy calculation, avoid scipy axis pitfalls
def calculate_prediction_entropy(logits_or_probs):
    """
    Calculate Shannon entropy of classifier prediction probabilities (correct implementation)
    Returns scalar entropy value, range [0, log2(n_classes)]
    """
    if isinstance(logits_or_probs, torch.Tensor):
        probs = torch.softmax(logits_or_probs, dim=-1).numpy()
    else:
        probs = np.array(logits_or_probs)

    # Safety handling
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    # Manually calculate entropy per sample, then average
    entropy_per_sample = -np.sum(probs * np.log2(probs), axis=-1)
    return float(np.mean(entropy_per_sample))


# 20. Corrected difference calculation function - optimized memory usage
def calculate_state_diff(custom_states, correct_states):
    """
    Calculate normalized state difference to enhance discrimination (memory optimized version)
    """
    # Calculate norm along feature axis (norm per sample), keep dimension for broadcasting
    custom_norms = np.linalg.norm(custom_states, axis=1, keepdims=True) + 1e-8
    correct_norms = np.linalg.norm(correct_states, axis=1, keepdims=True) + 1e-8

    # Normalize states (broadcasting handles automatically, avoid large memory allocation)
    custom_norm = custom_states / custom_norms
    correct_norm = correct_states / correct_norms

    # Calculate difference
    diff = np.linalg.norm(custom_norm - correct_norm, axis=1)
    return np.mean(diff)


# 21. Evaluate model accuracy on complete test set
def evaluate_model_accuracy(model, esn_cell, X_test, y_test, candidate_key, rng=RandomState(42)):
    """
    Evaluate model accuracy on complete test set using given candidate key
    """
    model.eval()
    with torch.no_grad():
        # Collect test set states using candidate key
        test_states = collect_states(
            X_test, esn_cell, use_correct_key=False,
            custom_key=candidate_key, return_all_timesteps=False, rng=rng
        )
        test_states_t = torch.FloatTensor(test_states)

        # Prediction
        logits = model(test_states_t)
        preds = torch.argmax(logits, dim=1).numpy()

        # Calculate accuracy
        accuracy = np.mean(preds == y_test) * 100.0

    return accuracy


# 22. Bit-by-bit key cracking function
def crack_key_bit_by_bit(model, esn_cell, X_test, y_test, key_size=32,
                         initial_key=None, rng=RandomState(42)):
    """
    Improved version: crack key only based on prediction entropy, no difference information
    Returns: (cracked_key, true_key, crack_log, final_accuracy, final_entropy)
    """
    # Step 1: Get correct key and states
    true_key = generate_key(1, key_size, correct_key=True, rng=rng).reshape(-1)
    true_states = collect_states(
        X_test, esn_cell, use_correct_key=True,
        return_all_timesteps=False, rng=rng
    )

    model.eval()
    with torch.no_grad():
        true_states_tensor = torch.FloatTensor(true_states)
        true_logits = model(true_states_tensor)
        true_entropy = calculate_prediction_entropy(true_logits)
        print(f"Baseline entropy of correct key: {true_entropy:.4f}")

    # Step 2: Initialize candidate key
    if initial_key is None:
        candidate_key = rng.randint(0, 2, size=key_size, dtype=int)
        print(f"Initial random key for first round: {candidate_key}")
    else:
        candidate_key = initial_key.copy()
        print(f"Initial key for current round (inherited from previous round): {candidate_key}")

    crack_log = []
    bit_indices = rng.permutation(key_size)

    # Step 3: Test each bit
    for idx, bit_idx in enumerate(bit_indices):
        print(f"\n--- Cracking bit {bit_idx} ---")

        # Save original value of current bit
        original_bit = candidate_key[bit_idx]

        # Test bit as 0
        candidate_key[bit_idx] = 0
        states_0 = collect_states(
            X_test, esn_cell, custom_key=candidate_key,
            return_all_timesteps=False, rng=rng
        )

        with torch.no_grad():
            states_0_tensor = torch.FloatTensor(states_0)
            logits_0 = model(states_0_tensor)
            entropy_0 = calculate_prediction_entropy(logits_0)
            accuracy_0 = evaluate_model_accuracy(model, esn_cell, X_test, y_test, candidate_key, rng)

        # Test bit as 1
        candidate_key[bit_idx] = 1
        states_1 = collect_states(
            X_test, esn_cell, custom_key=candidate_key,
            return_all_timesteps=False, rng=rng
        )

        with torch.no_grad():
            states_1_tensor = torch.FloatTensor(states_1)
            logits_1 = model(states_1_tensor)
            entropy_1 = calculate_prediction_entropy(logits_1)
            accuracy_1 = evaluate_model_accuracy(model, esn_cell, X_test, y_test, candidate_key, rng)

        # Step 4: Select optimal bit
        best_bit = 0 if entropy_0 < entropy_1 else 1

        # Update candidate key
        candidate_key[bit_idx] = best_bit

        # Step 5: Record log
        crack_log.append({
            'bit_idx': bit_idx,
            'bit_0_entropy': entropy_0,
            'bit_0_accuracy': accuracy_0,
            'bit_1_entropy': entropy_1,
            'bit_1_accuracy': accuracy_1,
            'best_bit': best_bit,
            'best_entropy': entropy_0 if best_bit == 0 else entropy_1,
            'best_accuracy': accuracy_0 if best_bit == 0 else accuracy_1,
            'current_accuracy': accuracy_0 if best_bit == 0 else accuracy_1,
            'true_bit': true_key[bit_idx]
        })

        print(f"Bit {bit_idx} | Entropy for 0: {entropy_0:.4f} Accuracy: {accuracy_0:.2f}% | "
              f"Entropy for 1: {entropy_1:.4f} Accuracy: {accuracy_1:.2f}% | "
              f"Selected: {best_bit} True value: {true_key[bit_idx]}")

    # Calculate final accuracy and final entropy
    correct_bits = np.sum(candidate_key == true_key)
    final_accuracy = correct_bits / key_size * 100
    final_entropy = crack_log[-1]['best_entropy'] if crack_log else float('inf')

    crack_log = sorted(crack_log, key=lambda x: x['bit_idx'])
    return candidate_key, true_key, crack_log, final_accuracy, final_entropy


# 23. Visualize prediction entropy changes during cracking
def plot_crack_entropy(crack_log, save_path, round_num):
    bit_indices = [log['bit_idx'] for log in crack_log]
    best_entropies = [log['best_entropy'] for log in crack_log]
    true_bits = [log['true_bit'] for log in crack_log]
    best_bits = [log['best_bit'] for log in crack_log]

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot entropy change curve
    ax.plot(bit_indices, best_entropies, color='#1f77b4', linewidth=2, zorder=2)
    # Mark correct bits (green) and wrong bits (red)
    for i, (bit_idx, entropy, tb, bb) in enumerate(zip(bit_indices, best_entropies, true_bits, best_bits)):
        if bb == tb:
            ax.scatter(bit_idx, entropy, color='green', s=50, zorder=3, label='Correct Bit' if i == 0 else "")
        else:
            ax.scatter(bit_idx, entropy, color='red', s=50, zorder=3, label='Wrong Bit' if i == 0 else "")

    ax.set_xlabel('Key Bit Index')
    ax.set_ylabel('Prediction Probability Entropy (bits)')
    ax.set_title(f'Round {round_num} Cracking - Bit-by-bit Prediction Entropy Change (Green=Correct, Red=Wrong)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# 24. Visualize accuracy progress
def plot_accuracy_progress(crack_log, save_path, round_num):
    bit_indices = [log['bit_idx'] for log in crack_log]
    accuracies = [log['current_accuracy'] for log in crack_log]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bit_indices, accuracies, color='#2ca02c', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Cracked Bit Index')
    ax.set_ylabel('Complete Test Set Accuracy (%)')
    ax.set_title(f'Round {round_num} Cracking - Complete Test Set Accuracy Change')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# 25. Execute adaptive multi-round cracking
print(f"\nExecuting adaptive key cracking on complete test set (entropy decrease stopping criterion)...")
print(f"Stopping condition: Stop when current round entropy >= previous round entropy (first round runs compulsorily)")

round_results = []
current_key = None
prev_entropy = float('inf')  # Initially set to infinity to ensure first round runs
round_idx = 1

while True:
    print(f"\n{'=' * 60}")
    print(f"Starting Round {round_idx} cracking...")
    print(f"{'=' * 60}")

    if round_idx > 1:
        print(f"Previous round entropy: {prev_entropy:.4f}")

    # Execute current round cracking
    cracked_key, true_key, crack_log, final_accuracy, final_entropy = crack_key_bit_by_bit(
        model, esn_cell, X_test, y_test,
        key_size=key_size, initial_key=current_key, rng=rng
    )

    # Store current round results
    round_info = {
        'round': round_idx,
        'initial_key': current_key.copy() if current_key is not None else 'Randomly Generated',
        'cracked_key': cracked_key,
        'true_key': true_key,
        'final_accuracy': final_accuracy,
        'final_entropy': final_entropy,  # Record final entropy of this round
        'crack_log': crack_log,
        'entropy_decreased': final_entropy < prev_entropy if round_idx > 1 else True
    }
    round_results.append(round_info)

    # Generate visualization charts for current round
    plot_crack_entropy(
        crack_log,
        os.path.join(crack_dir, f'crack_entropy_round{round_idx}.png'),
        round_idx
    )
    plot_accuracy_progress(
        crack_log,
        os.path.join(crack_dir, f'crack_accuracy_round{round_idx}.png'),
        round_idx
    )

    # Print current round results
    print(f"\nRound {round_idx} completed!")
    print(f"Final entropy of current round: {final_entropy:.4f}")
    print(f"Final cracking accuracy: {final_accuracy:.2f}%")
    print(f"Cracked key: {cracked_key}")
    print(f"True key: {true_key}")

    # Determine stopping condition: first round runs compulsorily, subsequent rounds continue only if entropy decreases
    if round_idx == 1:
        # First round: continue compulsorily, update baseline entropy
        prev_entropy = final_entropy
        current_key = cracked_key.copy()
        round_idx += 1
    else:
        # Subsequent rounds: check if entropy decreased
        if final_entropy < prev_entropy:
            print(f"Entropy decreased ({prev_entropy:.4f} -> {final_entropy:.4f}), continuing next round...")
            prev_entropy = final_entropy
            current_key = cracked_key.copy()
            round_idx += 1
        else:
            print(f"Entropy did not decrease ({prev_entropy:.4f} -> {final_entropy:.4f}), stopping cracking.")
            print(f"Best result in Round {round_idx - 1} with entropy {prev_entropy:.4f}")
            break

# 26. Save multi-round cracking results to Excel file
print("\n===================== Saving Multi-round Cracking Results to Excel =====================")

excel_path = os.path.join(crack_dir, 'multi_round_crack_results.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Create separate sheet for each round
    for round_info in round_results:
        round_num = round_info['round']
        crack_log = round_info['crack_log']

        # Prepare data
        data = []
        for log_entry in crack_log:
            data.append({
                'Bit Index': log_entry['bit_idx'],
                'Accuracy Before Cracking (%)': log_entry['current_accuracy'],
                'Prediction Entropy for 0': log_entry['bit_0_entropy'],
                'Test Set Accuracy for 0': f"{log_entry['bit_0_accuracy']:.2f}%",
                'Prediction Entropy for 1': log_entry['bit_1_entropy'],
                'Test Set Accuracy for 1': f"{log_entry['bit_1_accuracy']:.2f}%",
                'Optimal Value': log_entry['best_bit'],
                'Optimal Prediction Entropy': log_entry['best_entropy'],
                'Optimal Test Set Accuracy': f"{log_entry['best_accuracy']:.2f}%",
                'True Value': log_entry['true_bit'],
                'Is Correct': 'Yes' if log_entry['best_bit'] == log_entry['true_bit'] else 'No'
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Bit Index').reset_index(drop=True)

        # Write to Excel
        sheet_name = f'Round {round_num} Cracking'
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Set column width
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            worksheet.column_dimensions[chr(65 + idx)].width = 12

    # Create summary sheet
    summary_data = []
    for round_info in round_results:
        initial_key_str = ''.join(map(str, round_info['initial_key'])) if isinstance(round_info['initial_key'],
                                                                                     np.ndarray) else str(
            round_info['initial_key'])
        summary_data.append({
            'Round': round_info['round'],
            'Initial Key': initial_key_str,
            'Cracked Key': ''.join(map(str, round_info['cracked_key'])),
            'True Key': ''.join(map(str, round_info['true_key'])),
            'Final Accuracy (%)': round_info['final_accuracy'],
            'Final Entropy': round_info['final_entropy'],
            'Entropy Decreased': 'Yes' if round_info['entropy_decreased'] else 'No',
            'Correct Bits': np.sum(round_info['cracked_key'] == round_info['true_key']),
            'Total Bits': len(round_info['true_key'])
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

    # Set summary table column width
    worksheet = writer.sheets['Summary Statistics']
    col_widths = [10, 35, 35, 35, 15, 12, 10, 10, 10]
    for idx, width in enumerate(col_widths):
        worksheet.column_dimensions[chr(65 + idx)].width = width

print(f"\nMulti-round cracking results saved to: {excel_path}")

# 27. Output final cracking results summary
print("\n===================== Multi-round Cracking Final Summary =====================")
print(f"True key: {true_key}")
print(f"Actual rounds executed: {len(round_results)}")
print(f"Stopping reason: {'Entropy did not decrease' if len(round_results) > 1 and not round_results[-1]['entropy_decreased'] else 'Completed'}")

for round_info in round_results:
    status = "✓" if round_info['entropy_decreased'] else "✗"
    print(f"Round {round_info['round']} [{status}]: Entropy={round_info['final_entropy']:.4f}, "
          f"Accuracy={round_info['final_accuracy']:.2f}%, "
          f"Cracked Key={round_info['cracked_key']}")

# Find best round (round with minimum entropy)
best_round = min(round_results, key=lambda x: x['final_entropy'])
print(f"\nBest Result: Round {best_round['round']}")
print(f"Best Entropy: {best_round['final_entropy']:.4f}")
print(f"Best Accuracy: {best_round['final_accuracy']:.2f}%")
print(f"Best Cracked Key: {best_round['cracked_key']}")

print(f"\nAll results saved to {output_dir} directory")