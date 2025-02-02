import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

M = 1.0
m = 0.1
l = 1.0
g = 9.81
x_t0 = 0.0
x_d_t0 = 0.0
theta_t0 = np.pi / 4.0
theta_d_t0 = 0.0
T_N = 5.0
T_0 = 0.0
N = 100

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        # Normalize input to range [-1, 1]
        t_normalized = (t - T_0) / (T_N - T_0) * 2 - 1
        return self.layers(t_normalized)

# Define the system dynamics with the forcing function F_model
def pendulum_dynamics(t, state, F_model):
    x, x_d, theta, theta_d = state
    F_t = F_model(t.unsqueeze(0)) 
    x_dd = (F_t - m * g * torch.cos(theta) * torch.sin(theta) + m * l * theta_d**2 * torch.sin(theta)) / (M + m - m * torch.cos(theta)**2)
    theta_dd = (g * torch.sin(theta) - x_dd * torch.cos(theta)) / l
    return torch.stack([x_d, x_dd.squeeze(), theta_d, theta_dd.squeeze()])

# Define the loss function with regularization terms
def loss(F_model, initial_state, T_0, T_N, N):
    time_points = torch.linspace(T_0, T_N, N, requires_grad=True)
    solution = odeint(lambda t, y: pendulum_dynamics(t, y, F_model), initial_state, time_points, method='rk4')
    
    theta = solution[:, 2]
    theta_d = solution[:, 3]
    
    lambda1 = 0.2 
    lambda2 = 1.0
    lambda3 = 0.1
    lambda4 = 0.2
    

    #penalty 1
    mse_loss_x = lambda1 * torch.mean(solution[int(0.9 * N):, 0]**2)
    #penalty 2
    reg_loss_theta = lambda2 * torch.mean(theta[int(0.5 * N):]**2)  
    #penalty 3
    reg_loss_theta_d = lambda3 * torch.mean(theta_d[int(0.5 * N):]**2) 
    #penalty 4
    reg_loss_theta_end = lambda4 * torch.mean(theta[int(0.9 * N):]**2)
    
    loss_value = mse_loss_x + reg_loss_theta + reg_loss_theta_d + reg_loss_theta_end
    return loss_value

F_model = MLP()
optimizer = optim.Adam(F_model.parameters(), lr=1e-2)
initial_state = torch.tensor([x_t0, x_d_t0, theta_t0, theta_d_t0], dtype=torch.float32, requires_grad=True)

num_epochs = 1000
loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss_value = loss(F_model, initial_state, T_0, T_N, N)
    loss_value.backward()
    optimizer.step()
    loss_history.append(loss_value.item())
    print(f"Epoch {epoch + 1}, Loss: {loss_value.item()}")

#final solution
time_points = torch.linspace(T_0, T_N, N)
solution = odeint(lambda t, y: pendulum_dynamics(t, y, F_model), initial_state, time_points, method='rk4')

t = time_points.detach().numpy()
x = solution[:, 0].detach().numpy()
theta = solution[:, 2].detach().numpy()
theta_d = solution[:, 3].detach().numpy()
F_values = np.array([F_model(torch.tensor([[ti]], dtype=torch.float32)).item() for ti in t])

# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, x, label='x(t)')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(t, theta, label='theta(t)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(t, F_values, label='F(t)')
plt.legend()
plt.xlabel('Time')
plt.show()
