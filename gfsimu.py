import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import broyden1 as broyden

def circular_vf(x, y):
    x_1 = x# - 2
    x_2 = y# - 2
    u = -x_2 * (x_1 ** 2 + x_2 ** 2)**.5
    v = x_1 * (x_1 ** 2 + x_2 ** 2)**.5
    return u, v

def circular_vf_tensor(x):
    if torch.all(x == 0):
        return torch.tensor([0., 0.])
    else:
        x = x# - 2
        u = -x[1] * (x[0] ** 2 + x[1] ** 2)**.5
        v = x[0] * (x[0] ** 2 + x[1] ** 2)**.5
        return torch.tensor([u, v])

def make_f_trajectory(start, n_iter=100):
    vector = start
    trajectory = [vector]
    for i in range(n_iter):
        x_1 = vector[0]
        x_2 = vector[1]
        next_x_1 = -x_2 * (x_1 ** 2 + x_2 ** 2)**.5
        next_x_2 = x_1 * (x_1 ** 2 + x_2 ** 2)**.5
        vector = np.array([next_x_1, next_x_2])
        trajectory.append(vector)
    trajectory = np.vstack(trajectory)
    return vector, trajectory

def make_broyden_trajectory(start):
    trajectory = [start]
    def broyden_callback(x, f):
        trajectory.append(x)
    def broyden_functional(x):
        norm_x = np.linalg.norm(x)
        obj = x - norm_x * np.array([-x[1], x[0]])
        return obj
    vector = broyden(broyden_functional, start, f_tol=1e-6, callback=broyden_callback)
    trajectory = np.vstack(trajectory)
    return vector, trajectory

def make_newton_trajectory(start, tol=1e-6, max_iter=1000):
    trajectory = [start]
    vector = start
    err = np.linalg.norm(vector)
    def inverse_jac(x): # compute the inverse of the Jacobian of the objective function
        norm_x = np.linalg.norm(x)
        x1 = x[0]
        x2 = x[1]
        jac = np.eye(2) - norm_x*np.array([[-x1*x2/norm_x**2, -1-x2**2/norm_x**2], [1+x1**2/norm_x**2, x1*x2/norm_x**2]])
        inv_jac = np.linalg.inv(jac)
        return inv_jac
    for i in range(max_iter):
        if err <= tol:
            print(f'Termination after {i+1} iterations; accuracy below tol')
            break
        vector = vector - np.matmul(inverse_jac(vector), vector - np.array(circular_vf(*vector)))
        trajectory.append(vector)
        err = np.linalg.norm(vector)
    trajectory = np.vstack(trajectory)
    return vector, trajectory



def circular_objective(x):
    target = circular_vf_tensor(x)
    obj = torch.norm(x - target)**2 / 2
    return obj

def gradient(x):
    #x = x - 2
    norm_x_squared = x[0]**2 + x[1]**2
    grad = np.array([x[0] + 2*x[0]*norm_x_squared, x[1]+2*x[1]*norm_x_squared])
    return grad

def gradient_descent(start, gradient=gradient, learn_rate=0.001, n_iter=5000, tolerance=1e-6):
    vector = start
    trajectory = [vector]
    for i in range(n_iter):
        grad = gradient(vector)
        next_vector = vector - learn_rate * grad
        trajectory.append(next_vector)
        if np.linalg.norm(next_vector - vector) < tolerance:
            print(f'Termination after {i+1} iterations; accuracy below tol')
            break
        vector = next_vector
    trajectory = np.vstack(trajectory)
    return vector, trajectory



def gradient_descent_torch(objective_function, start, learn_rate=0.001, n_iter=5000, tolerance=1e-6):
    vector = start.clone().detach()
    vector.requires_grad = True
    trajectory = [vector]
    for i in range(n_iter):
        # Zero the gradients from the previous step
        if vector.grad is not None:
            vector.grad.zero_()

        # Compute the objective function value
        objective_value = objective_function(vector)

        # Compute the gradient
        objective_value.backward()

        # Update the vector
        with torch.no_grad():
            next_vector = vector - learn_rate * vector.grad
            trajectory.append(next_vector)

        # Check convergence
        if torch.norm(next_vector - vector) < tolerance:
            print(f'Termination after {i+1} iterations; accuracy below tol')
            break

        # Update the vector
        vector = next_vector.clone().detach().requires_grad_(True)

    return vector, torch.stack(trajectory).detach()

# Meshgrid
X, Y = np.meshgrid(np.linspace(-7, 7, 15),
                   np.linspace(-7, 7, 15))

# Directional vectors
U, V = circular_vf(X, Y)

# Plotting Vector Field with QUIVER
plt.quiver(X, Y, U, V, color='g')
plt.title('Methods with #iterations')

start = np.array([-4., -1.])

_, gd_trajectory = gradient_descent(start=start, n_iter=5000, learn_rate=0.01)
n_iter = len(gd_trajectory)
traj_x = gd_trajectory[:, 0]
traj_y = gd_trajectory[:, 1]
plt.plot(traj_x, traj_y, marker='o', linestyle='-', color='b', label=f'direct GD ({n_iter-1})', markersize=1, linewidth=1)

_, gd_trajectory_torch = gradient_descent_torch(circular_objective, torch.tensor(start), n_iter=5000, learn_rate=0.01)
n_iter = len(gd_trajectory_torch)
traj_x = gd_trajectory_torch[:, 0]
traj_y = gd_trajectory_torch[:, 1]
plt.plot(traj_x.numpy(), traj_y.numpy(), marker='o', linestyle='-', color='r', label=f'torch GD ({n_iter-1})', markersize=1, linewidth=1)

_, vf_trajectory = make_f_trajectory(start, 100)
traj_x = vf_trajectory[:, 0]
traj_y = vf_trajectory[:, 1]
plt.plot(traj_x, traj_y, marker='o', linestyle='-', color='gray', label='apply vf', markersize=1, linewidth=1)

_, broyden_trajectory = make_broyden_trajectory(start)
n_iter = len(broyden_trajectory)
traj_x = broyden_trajectory[:, 0]
traj_y = broyden_trajectory[:, 1]
plt.plot(traj_x, traj_y, marker='o', linestyle='-', color='orange', label=f'broyden ({n_iter-1})', markersize=1, linewidth=1)

_, newton_trajectory = make_newton_trajectory(start)
n_iter = len(newton_trajectory)
traj_x = newton_trajectory[:, 0]
traj_y = newton_trajectory[:, 1]
plt.plot(traj_x, traj_y, marker='o', linestyle='-', color='black', label=f'newton ({n_iter-1})', markersize=1, linewidth=1)


# Setting x_1, x_2 boundary limits
plt.xlim(-7, 7)
plt.ylim(-7, 7)

# Show plot with grid
plt.grid()
plt.legend()
plt.show()