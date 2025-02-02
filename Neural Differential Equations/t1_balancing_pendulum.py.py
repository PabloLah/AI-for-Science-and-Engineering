import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import matplotlib.pyplot as plt

mass_cart = 1.0
mass_pendulum = 0.1
length = 1.0
gravity = 9.81

def force_function(t):
    return 10 * jnp.sin(t)

def pendulum_dynamics(t, state, args):
    pos_cart, vel_cart, angle, angular_vel = state
    force = force_function(t)
    acc_cart = (force - mass_pendulum * gravity * jnp.cos(angle) * jnp.sin(angle) + mass_pendulum * length * angular_vel**2 * jnp.sin(angle)) / (mass_cart + mass_pendulum - mass_pendulum * jnp.cos(angle)**2)
    angular_acc = (gravity * jnp.sin(angle) - acc_cart * jnp.cos(angle)) / length
    return jnp.array([vel_cart, acc_cart, angular_vel, angular_acc])

initial_angle = jnp.pi / 4
initial_pos_cart = 0.0
initial_vel_cart = 0.0
initial_angular_vel = 0.0
initial_state = jnp.array([initial_pos_cart, initial_vel_cart, initial_angle, initial_angular_vel])

# Time interval for simulation
start_time = 0.0
end_time = 5.0
time_points = jnp.linspace(start_time, end_time, 100)

ode_term = ODETerm(pendulum_dynamics)
solver = Tsit5()
save_at = SaveAt(ts=time_points)

solution = diffeqsolve(ode_term, solver, t0=start_time, t1=end_time, dt0=0.1, y0=initial_state, saveat=save_at)

cart_positions = solution.ys[:, 0]
pendulum_angles = solution.ys[:, 2]

fig, ax1 = plt.subplots(figsize=(8, 5))  

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)', color='tab:blue')
ax1.plot(time_points, cart_positions, label='Cart Position (x(t))', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Angle (rad)', color='tab:red')
ax2.plot(time_points, pendulum_angles, label='Pendulum Angle (θ(t))', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
plt.title('Cart Position and Pendulum Angle Over Time', loc='center')  # Center-align the title
fig.tight_layout()
fig.legend(loc='upper right')
plt.show()

