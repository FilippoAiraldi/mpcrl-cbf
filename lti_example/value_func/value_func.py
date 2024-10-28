import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(this_dir))
from itertools import product

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from env import ConstrainedLtiEnv as Env
from scipy.io import savemat
from tqdm import tqdm

# define the parameters
N = 20  # approximately an horizon of 10 should be enough, but increase it a bit
A = Env.A
B = Env.B
Q = Env.Q
R = Env.R
ns, na = B.shape
a_bnd = Env.a_bound
x_bnd = Env.x_soft_bound

# create the MPC problem
mpc = Mpc(Nlp("MX"), prediction_horizon=N, shooting="single")
mpc.state("x", ns)
u, _ = mpc.action("u", na, lb=-a_bnd, ub=a_bnd)
mpc.set_linear_dynamics(A, B)
x = mpc.states["x"]
mpc.constraint("lbx", x, ">=", -x_bnd)
mpc.constraint("ubx", x, "<=", x_bnd)
J = sum(cs.bilin(Q, x[:, i]) + cs.bilin(R, u[:, i]) for i in range(N))
mpc.minimize(J)
opts = {
    "error_on_fail": False,
    "expand": True,
    "print_time": False,
    "printLevel": "none",
}
mpc.init_solver(opts, "qpoases")

# compute the value function
n_elm = 100
grid = np.linspace(-x_bnd, x_bnd, n_elm)
X = np.asarray(np.meshgrid(grid, grid, indexing="ij"))
V = np.zeros((n_elm, n_elm))
for i, j in tqdm(product(range(n_elm), range(n_elm)), total=n_elm * n_elm):
    sol = mpc.solve(pars={"x_0": X[:, i, j]})
    V[i, j] = np.inf if sol.infeasible else sol.f

# save and plot
savemat(os.path.join(this_dir, "value_func.mat"), {"grid": grid, "V": V})
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
V_ = V[np.isfinite(V)]
V[V == np.inf] = V_.max()
ax.plot_surface(X[0], X[1], V, vmin=V_.min(), vmax=V_.max(), cmap="RdBu_r")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$V(x)$")
plt.show()
