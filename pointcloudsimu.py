import numpy as np
import matplotlib.pyplot as plt


def vector_field(xi, X, k_a, k_r):
    attractive = -k_a * xi
    repulsive = np.zeros_like(xi)
    for xj in X:
        if np.any(xi != xj):
            diff = xi - xj
            repulsive += k_r * diff / np.linalg.norm(diff) ** 3
    return attractive + repulsive


def simulate(X, k_a, k_r, lr=0.01, steps=1000, checkpoints=[0.2, 0.4, 0.6, 0.8]):
    history = [X.copy()]
    checkpoint_steps = [int(steps * cp) for cp in checkpoints]
    step_count = 0

    for _ in range(steps):
        for i in range(len(X)):
            X[i] += lr * vector_field(X[i], X, k_a, k_r)
        step_count += 1
        if step_count in checkpoint_steps:
            history.append(X.copy())

    history.append(X.copy())  # Final state
    return history


def initialize_gaussian(mean, cov, num_samples):
    return np.random.multivariate_normal(mean, cov, num_samples)

# Parameters for Gaussian distribution
mean = [-10, 0]
cov = [[1, 0], [0, 12]]
num_samples = 50

# Initialize points from Gaussian distribution
np.random.seed(0)
X = initialize_gaussian(mean, cov, num_samples)

# Parameters
k_a = 1.0
k_r = 3.0
steps = 200
checkpoints = [0.2, 0.4, 0.6, 0.8]

# Simulate
history = simulate(X, k_a, k_r, steps=steps, checkpoints=checkpoints)

x_min, y_min = -12, -12
x_max, y_max = 12, 12

# Ensure a square aspect ratio
padding = 0.1 * max(x_max - x_min, y_max - y_min)
x_min -= padding
x_max += padding
y_min -= padding
y_max += padding

# Plotting
checkpoint_labels = ["Initial"] + [str(int(100*ch)) + "% Steps" for ch in checkpoints] + ["Final"]
num_plots = len(history)
fig, axes = plt.subplots(1, num_plots, figsize=(20, 5))

for i, (ax, X_snap) in enumerate(zip(axes, history)):
    ax.scatter(X_snap[:, 0], X_snap[:, 1], c='blue')
    ax.set_title(checkpoint_labels[i])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(False)

plt.tight_layout()
plt.show()
