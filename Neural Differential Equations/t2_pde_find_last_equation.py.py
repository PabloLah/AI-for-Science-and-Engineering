import numpy as np
from sklearn.feature_selection import f_regression
import jax.numpy as jnp
from jax import grad, jit
import optax

# Load data from file
data3 = np.load('3.npz')
u3 = data3['u']
v3 = data3['v']
x3 = data3['x']
y3 = data3['y']
t3 = data3['t']

# Parameters
dt3 = 0.05
dx3 = 0.078125
dy3 = 0.078125

# Define feature extraction functions
def compute_derivatives(u3, v3, dt, dx, dy):
    # Compute derivatives for u
    u3_t = (u3[..., 1:-1] - u3[..., :-2]) / dt
    u3_x = (u3[:, :-2, 1:-1] - u3[:, 2:, 1:-1]) / (2 * dx)
    u3_xx = (u3[:, :-2, 1:-1] - 2 * u3[:, 1:-1, 1:-1] + u3[:, 2:, 1:-1]) / dx**2
    u3_y = (u3[:, 1:-1, :-2] - u3[:, 1:-1, 2:]) / (2 * dy)
    u3_yy = (u3[:, 1:-1, :-2] - 2 * u3[:, 1:-1, 1:-1] + u3[:, 1:-1, 2:]) / dy**2

    # Compute derivatives for v
    v3_t = (v3[..., 1:-1] - v3[..., :-2]) / dt
    v3_x = (v3[:, :-2, 1:-1] - v3[:, 2:, 1:-1]) / (2 * dx)
    v3_xx = (v3[:, :-2, 1:-1] - 2 * v3[:, 1:-1, 1:-1] + v3[:, 2:, 1:-1]) / dx**2
    v3_y = (v3[:, 1:-1, :-2] - v3[:, 1:-1, 2:]) / (2 * dy)
    v3_yy = (v3[:, 1:-1, :-2] - 2 * v3[:, 1:-1, 1:-1] + v3[:, 1:-1, 2:]) / dy**2
    
    return u3_t, u3_x, u3_xx, u3_y, u3_yy, v3_t, v3_x, v3_xx, v3_y, v3_yy

# Extract derivatives
u3_t, u3_x, u3_xx, u3_y, u3_yy, v3_t, v3_x, v3_xx, v3_y, v3_yy = compute_derivatives(u3, v3, dt3, dx3, dy3)

# Define feature combinations with their names

features = {
    "u3_1": u3[:, 1:-1, 1:-1],
    "u3_2": u3[:, 1:-1, 1:-1]**2,
    "v3_1": v3[:, 1:-1, 1:-1],
    "v3_2": v3[:, 1:-1, 1:-1]**2,  
    "u3_1*v3_1": u3[:, 1:-1, 1:-1] * v3[:, 1:-1, 1:-1],
    "u3_x": u3_x,
    "u3_xx": u3_xx,
    "v3_x": v3_x,
    "v3_xx": v3_xx,
    "u3_y": u3_y,
    "u3_yy": u3_yy,
    "v3_y": v3_y,
    "v3_yy": v3_yy,
    "u3_1*u3_x": u3[:, 1:-1, 1:-1] * u3_x,
    "u3_1*u3_y": u3[:, 1:-1, 1:-1] * u3_y,
    "u3_1*v3_x": u3[:, 1:-1, 1:-1] * v3_x,
    "u3_1*v3_y": u3[:, 1:-1, 1:-1] * v3_y,
    "v3_1*u3_x": v3[:, 1:-1, 1:-1] * u3_x,
    "v3_1*v3_x": v3[:, 1:-1, 1:-1] * v3_x,
    "v3_1*u3_y": v3[:, 1:-1, 1:-1] * u3_y,
    "v3_1*v3_y": v3[:, 1:-1, 1:-1] * v3_y,
    "v3_x*u3_y": v3_x * u3_y,
}

# Stack the features into a single matrix
X = np.stack(list(features.values()), axis=-1)  # Shape: (256, 254, 199, number of features)
print("shape of X: ", X.shape)
# Reshape X for feature selection (flattening spatial dimensions)
X_reshaped = X.reshape(-1, X.shape[-1])  # Shape: (256*254*199, number of features)
print("shape of X_reshaped: ", X_reshaped.shape)

# Calculate correlation coefficients (or other suitable metric)
# Here, we use f_regression to compute scores (correlation coefficients in this case)
scores, _ = f_regression(X_reshaped, np.zeros(X_reshaped.shape[0]))  # Using zeros as target for feature importance

# Sort indices of features based on their scores
sorted_indices = np.argsort(scores)[::-1]  # Descending order

# Select top k features
k = 10  # Example: Select top 10 features
selected_indices = sorted_indices[:k]

#UPDATE: decided not to use the best feature selection.
# Retrieve the selected features and their names
#selected_features = [list(features.values())[i] for i in selected_indices]
#selected_feature_names = [list(features.keys())[i] for i in selected_indices]
#print(selected_feature_names)

#alternatively: prepare all features
all_features = [list(features.values())[i] for i in sorted_indices]
all_feature_names = [list(features.keys())[i] for i in selected_indices]

# Construct matrix Theta
Theta = jnp.stack(all_features, axis=-1)  # Shape: (256, 254, 199, k)
print(f"Shape of Theta: {Theta.shape}")

# Reshape Theta for the solver (flattening spatial dimensions)
Theta_flattened = Theta.reshape(-1, Theta.shape[-1])  # Shape: (256*254*199, k)
print(f"Shape of Theta_flattened: {Theta_flattened.shape}")

# Define equations to solve
Q3 = jnp.transpose(Theta_flattened)
print(f"Shape of Q3: {Q3.shape}")

# Flatten the spatial dimensions of derivatives for use in the solver
u3_t_flat = u3_t[..., 1:-1, :].reshape(-1)  # Shape: (256*254*199,)
v3_t_flat = v3_t[..., 1:-1, :].reshape(-1)  # Shape: (256*254*199,)
U3 = jnp.stack([u3_t_flat, v3_t_flat], axis=-1)  # Shape: (256*254*199, 2)
print(f"Shape of u3_t: {u3_t.shape}")
print(f"Shape of v3_t: {v3_t.shape}")

# Print shapes for verification
print(f"Shape of u3_t_flat: {u3_t_flat.shape}")
print(f"Shape of v3_t_flat: {v3_t_flat.shape}")
print(f"Shape of U3: {U3.shape}")


# Initialize parameters
xi3 = jnp.ones((Q3.shape[0], 2))  # Shape: (k, 2)

optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(xi3)

lambda1 = 1.0
lambda2 = 1.0
lambda3 = 1.0
# JIT-compiled loss function and gradient function
@jit
def loss3(xi3):
    "Computes mean squared error between PDE predictions and observed data"
    predicted_ut = jnp.dot(Q3.T, xi3)
    loss = lambda1 * jnp.mean((predicted_ut - U3) ** 2)
    loss += lambda2 * jnp.linalg.norm(xi3[:, 0], ord=0)
    loss += lambda3 * jnp.linalg.norm(xi3[:, 1], ord=0)
    return loss

@jit
def grad_loss3(xi3):
    "Computes gradient of the loss function"
    return grad(loss3)(xi3)

# Optimization loop
epochs = 1000
threshold = 1e-2

for i in range(epochs):
    # Compute gradient
    gradient = grad_loss3(xi3)
    
    # Update parameters
    updates, opt_state = optimizer.update(gradient, opt_state)
    xi3 = optax.apply_updates(xi3, updates)

    #truncate xi3 values
    xi3 = jnp.where(jnp.abs(xi3) < threshold, 0.0, xi3)

    
    # Print loss periodically
    if (i + 1) % 100 == 0 or i == 0:
        loss_value = loss3(xi3)
        print(f"[{i + 1}/{epochs}] loss: {loss_value}")

print("Estimated coefficients (xi3):")
print(xi3)