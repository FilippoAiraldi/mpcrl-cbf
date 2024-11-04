import sys
from pathlib import Path

dir = Path(__file__).parent
lti_example_dir = dir.parent
sys.path.extend((str(lti_example_dir.parent), str(lti_example_dir)))

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from env import ConstrainedLtiEnv as Env
from scipy.io import savemat
from tqdm import tqdm
from controllers.mpc import create_mpc
from matplotlib.ticker import LogLocator, FuncFormatter, MaxNLocator

# get the MPC controller
mpc = create_mpc(
    horizon=10,  # 10 should be long enough
    dcbf=True,
    soft=True,
    bound_initial_state=False,  # doesn't matter when `dcbf=True`
    dlqr_terminal_cost=False,
)

# compute the value function
n_elm = 200
grid = np.linspace(-Env.x_soft_bound, Env.x_soft_bound, n_elm)
X = np.asarray(np.meshgrid(grid, grid, indexing="ij"))
V = np.zeros((n_elm, n_elm))
for i, j in tqdm(product(range(n_elm), range(n_elm)), total=n_elm * n_elm):
    sol = mpc.solve(pars={"x_0": X[:, i, j]})
    V[i, j] = np.inf if sol.infeasible else sol.f

# save
savemat(str(dir / "value_func.mat"), {"grid": grid, "V": V})

# plot
V_ = V[np.isfinite(V)]
vmin = V_.min()
vmax = V_.max()
cmap = "RdBu_r"

fig = plt.figure(figsize=(10, 5), constrained_layout=True)
ax1 = fig.add_subplot(1, 2, 1)
locator = LogLocator(subs="auto")
CS = ax1.contourf(X[0], X[1], V, locator=locator, vmin=vmin, vmax=vmax, cmap=cmap)
# ax1.clabel(CS, CS.levels[::10], inline=True, fontsize=10, colors='k')
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
SF = ax2.plot_surface(
    X[0], X[1], np.log10(V), vmin=np.log10(vmin), vmax=np.log10(vmax), cmap=cmap
)
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_zlabel("$V(x)$")
# thanks to https://stackoverflow.com/a/67774238/19648688
ax2.zaxis.set_major_formatter(FuncFormatter(lambda val, _: f"$10^{{{int(val)}}}$"))
ax2.zaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
