import jax
import jax.numpy as jnp
from jax import jit, vmap
import equinox as eqx
import optax


data1 = jnp.load('1.npz')
u1 = data1['u']
x1 = data1['x']
t1 = data1['t']

#----------Equation 1---------------
#notice the values in the data are given by a grid with time steps 0.1 and x-steps 0.0625
#compute finite differences
dt1 = 0.1
dx1 = 0.0625 #grid spacing in data
u1_t = (u1[1:-1,0:-2]-u1[1:-1,2:])/(2*dt1)
u1_x = (u1[0:-2,1:-1]-u1[2:,1:-1])/(2*dx1)
u1_xx = (u1[0:-2,1:-1]-2*u1[1:-1,1:-1]+u1[2:,1:-1])/dx1**2
u1_xxx = 3*(u1[0:-2,1:-1]-2*u1_x*dx1+u1[2:,1:-1])/dx1**3
u1_1 = u1[1:-1,1:-1]
u1_1x = u1_1*u1_x
u1_1xx = u1_1*u1_xx
u1_1xxx = u1_1*u1_xxx
u1_2 = u1_1*u1_1
u1_2x = u1_1x*u1_2
u1_2xx = u1_1xx*u1_2
u1_2xxx = u1_1xxx*u1_2

#define relevant matrices
Q1 = jnp.array([u1_1, u1_x, u1_xx, u1_xxx, u1_1x, u1_1xx, u1_1xxx, u1_2, u1_2x, u1_2xx, u1_2xxx])
Q1 = jnp.reshape(Q1,(11,254*99))
U1 = jnp.reshape(u1_t,(254*99))


#do ridge regression training
lambda1 = 1
lambda2 = .1
eps = 1e-3
cond = vmap(jax.lax.cond, (0,None,None,0),0)

@jit
def f(x0, xi):
    return xi@x0

def loss1(x0, f, fargs, X_obs):
    "Computes mean squared error between NDE solution and observed data"
    sol = f(x0, *fargs)
    loss = lambda1*jnp.mean((sol-X_obs)**2)+lambda2*jnp.linalg.norm(*fargs, ord=0)
    return loss

def grad1(x0, f, fargs, X_obs):
    "Computes gradient of loss function wrt fargs"
    partial = lambda fargs: loss1(x0, f, fargs, X_obs)
    return eqx.filter_value_and_grad(partial)(fargs)

@eqx.filter_jit
def step(x0, f, fargs, opt_state, opt_update, X_obs):
    "Performs one gradient descent step on fargs"
    lossval, grads = grad1(x0, f, fargs, X_obs)
    updates, opt_state = opt_update(grads, opt_state)
    fargs = eqx.apply_updates(fargs, updates)
    fargs = (cond(jnp.abs(fargs[0])<eps, lambda x: 0.0, lambda x: x, fargs[0]),)
    return lossval, fargs, opt_state

# initialise learnable parts
xi1 = jnp.ones(11)
fargs = (xi1,)

# define optimizer
optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(eqx.filter(fargs, eqx.is_array))
opt_update = optimizer.update
epochs = 40000

# train NDE
lossvals1 = []
for i in jnp.arange(epochs):
    lossval, fargs, opt_state = step(Q1, f, fargs, opt_state, opt_update, U1)
    lossvals1.append(lossval)
    if (i+1)%1000==0 or i==0:
        print(f"[{i+1}/{epochs}] loss: {lossval}")

xi1 = fargs[0]
print(xi1,'\n')


#-------Equation2------------
data2 = jnp.load('2.npz')
u2 = data2['u']
x2 = data2['x']
t2 = data2['t']

dt2 = 0.1
dx2 = 0.1171875
u2_t = (u2[1:-1,0:-2]-u2[1:-1,2:])/(2*dt2)
u2_x = (u2[0:-2,1:-1]-u2[2:,1:-1])/(2*dx2)
u2_xx = (u2[0:-2,1:-1]-2*u2[1:-1,1:-1]+u2[2:,1:-1])/dx2**2
u2_xxx = 3*(u2[0:-2,1:-1]-2*u2_x*dx2+u2[2:,1:-1])/dx2**3
u2_1 = u2[1:-1,1:-1]
u2_1x = u2_1*u2_x
u2_1xx = u2_1*u2_xx
u2_1xxx = u2_1*u2_xxx
u2_2 = u2_1*u2_1
u2_2x = u2_1x*u2_2
u2_2xx = u2_1xx*u2_2
u2_2xxx = u2_1xxx*u2_2

Q2 = jnp.array([u2_1, u2_x, u2_xx, u2_xxx, u2_1x, u2_1xx, u2_1xxx, u2_2, u2_2x, u2_2xx, u2_2xxx])
Q2 = jnp.reshape(Q2,(11,510*199))
U2 = jnp.reshape(u2_t,(510*199))

# initialise learnable parts
xi2 = jnp.ones(11)
fargs = (xi2,)

# define optimizer
optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(eqx.filter(fargs, eqx.is_array))
opt_update = optimizer.update
epochs = 30000

# train NDE
lossvals2 = []
for i in jnp.arange(epochs):
    lossval, fargs, opt_state = step(Q2, f, fargs, opt_state, opt_update, U2)
    lossvals2.append(lossval)
    if (i+1)%1000==0 or i==0:
        print(f"[{i+1}/{epochs}] loss: {lossval}")

xi2 = fargs[0]
print(xi2,'\n')







