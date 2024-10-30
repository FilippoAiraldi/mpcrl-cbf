import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(this_dir))
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from env import ConstrainedLtiEnv as Env
from scipy.io import savemat
from tqdm import tqdm
from lti_example.controllers.mpc import create_mpc

# get the MPC controller
mpc = create_mpc(
    horizon=20,  # approximately 10 should be long enough, but increase it a bit
    soft=False,
    bound_initial_state=False,
    dlqr_terminal_cost=False,
)

# compute the value function
n_elm = 100
grid = np.linspace(-Env.x_soft_bound, Env.x_soft_bound, n_elm)
X = np.asarray(np.meshgrid(grid, grid, indexing="ij"))
V = np.zeros((n_elm, n_elm))
for i, j in tqdm(product(range(n_elm), range(n_elm)), total=n_elm * n_elm):
    sol = mpc.solve(pars={"x_0": X[:, i, j]})
    V[i, j] = np.inf if sol.infeasible else sol.f

# save and plot
savemat(os.path.join(this_dir, "value_func.mat"), {"grid": grid, "V": V})
V_ = V[np.isfinite(V)]
Vmin, Vmax = V_.min(), V_.max()
fig = plt.figure(figsize=(10, 5), constrained_layout=True)
ax1 = fig.add_subplot(1, 2, 1)
CS = ax1.contour(X[0], X[1], V, vmin=Vmin, vmax=Vmax, cmap="RdBu_r")
ax1.clabel(CS, inline=True, fontsize=10)
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X[0], X[1], V, vmin=Vmin, vmax=Vmax, cmap="RdBu_r")
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_zlabel("$V(x)$")
plt.show()
