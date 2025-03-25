# MPC-RL with Probabilistic CBF

This repository contains the source code used to produce the results in TODO.

In this work, we propose to tackle safety-critical stochastic Reinforcement Learning
(RL) tasks via a model-based sample-based approach. The proposed method integrates into
a Model Predictive Control (MPC) framework a probabilistic Control Barrier Function
(CBF) condition that guarantees safety at the trajectory-level with high probability. To
render the problem tractable, a sample-based approximation is introduced alongside a
learnable terminal cost-to-go function to reduce computational complexity.

If you find the paper or this repository useful, please consider citing:

```bibtex
TODO
```

---

## Installation

The code was created with `Python 3.12.6`. To access it, clone the repository

```bash
git clone https://github.com/FilippoAiraldi/mpcrl-cbf.git
cd mpcrl-cbf
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way:

- **`lti`** contains all the code for the first numerical experiment on a stochastic
constrained LTI system. The API of the environment (**`env.py`**) follows the standard
OpenAI's `gym` style. Code for training, evaluation and plotting can be found in
**`train.py`**, **`eval.py`**, and **`plot.py`**, respectively. The subfolders are
  - **`controllers`** contains the implementation of various controllers, including
  optimal control ones (which are based on
  [csnlp](https://github.com/FilippoAiraldi/casadi-nlp))
  - **`agents`** contains the MPC-RL agents (which are implemented via
  [mpcrl](https://github.com/FilippoAiraldi/mpc-reinforcement-learning))
  - **`data`** contains the simulation data saved to disk.
  - **`explicit_sol`** contains scripts to compute the explicit solution to the
  constrained LTI problem and compare it to the learned one.
- **`quadrotor`** contains the source code for the second numerical experiment of a
nonlinear stochastic quadrotor platform. The structure and contents of the folder are
similar to those of the **`lti`** folder (however, there is not explicit solution for
this case).
- **`util`** contains utility classes and functions for, e.g., constants, plotting, etc.
- **`simple_examples`** contains implementations and experimentations with other (often
simpler) CBF-based controllers.

---

## Experiments

Training and evaluation simulations can easily be launched via the command below. The
provided arguments are set to reproduce the same main results found in the paper,
assuming there are no major discrepancies due to OS, CPU, etc.. For help about the
effect of each different argument, run, e.g.,

```bash
python lti/eval.py --help
```

Note that in what follows we will use multiple variables to, e.g., denote the number of
parallel jobs to run, the filename to save the results, etc. These variables should be
set according to the user's needs.

### 1. Constrained Stochastic LTI System

#### 1.1 Simulation

For the constrained stochastic LTI system, we can simulate two different kind of
controllers: a non-learning MPC one (i.e., a baseline controller with sufficiently long
horizon) and our proposed MPC-based RL one. We show how to do so below. Note that you
can also find the results of our simulations in the `lti/data` folder, if you do not
intend to run the simulations yourself.

##### Non-Learning MPC

As this controller does not require training, we only need to evaluate it with

```bash
python lti/eval.py scmpc --horizon=12 --dcbf --soft --n-eval=100 --save=${eval_scmpc_fn} --n-jobs=${number_of_parallel_jobs}
```

##### MPC-based RL

To train the proposed Q-learning MPC-RL agent, run

```bash
python lti/train.py lstd-ql --horizon=1 --dcbf --soft --terminal-cost=pwqnn --save=${train_scmpcrl_fn} --n-jobs=${number_of_parallel_jobs}
```

Once the training is completed, we can evalue the outcomes with

```bash
python lti/eval.py scmpc --from-train=${train_scmpcrl_fn} --n-eval=100 --save=${eval_scmpcrl_fn} --n-jobs=${number_of_parallel_jobs}
```

#### 1.2 Visualization

Here we show how to plot the results of the simulations. Note that you can also find our
pre-computed simulation results in the `lti/data` folder.

To plot the results of a training simulation, run

```bash
python lti/plot.py ${train_scmpcrl_fn} --training --terminal-cost
```

The `training` flag will show the evolution of the temporal difference error during
learning as well as the evolution of each parameter of the MPC parametrisation. The
`terminal-cost` flag will show the evolution of the terminal cost-to-go function as the
learning progresses (takes a bit longer). More interesting is the plotting of the
evaluation results after training. To compare it with the non-learning MPC controller,
run

```bash
python lti/plot.py ${eval_scmpc_fn} ${eval_scmpcrl_fn} --returns --solver-time --state-action
```

The `returns` flag will show the returns obtained by the two controllers, side by side.
The same for the `solver-time` flag, which will show the computation times of the two
controllers. The `state-action` flag will show the state-action trajectories (can be
messy if too many episodes are plotted).

The figures in the paper are generated by saving to disk the data in .dat format (see
`pgfplotstables` flag), loading them in the LaTeX document via PGFplotsTable, and then
plotting them with PGFplots.

### 2. Quadrotor Platform

#### 2.1 Simulation

#### 2.2 Visualization

TODO
