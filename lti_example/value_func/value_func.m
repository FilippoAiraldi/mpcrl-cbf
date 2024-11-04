%% Define the control problem
clearvars, close all

% define LTI system
A = [1.0, 0.4; -0.1, 1.0];
B = [1.0, 0.05; 0.5, 1.0];
model = LTISystem('A', A, 'B', B);

% add penalties
Q = eye(model.nx);
R = 0.1 * eye(model.nu);
model.x.penalty = QuadFunction(Q);
model.u.penalty = QuadFunction(R);

% add constraints
u_bnd = 0.5;
model.u.min = [-u_bnd; -u_bnd];
model.u.max = [u_bnd; u_bnd];
x_bnd = 3;
model.x.min = [-x_bnd; -x_bnd];
model.x.max = [x_bnd; x_bnd];


%% Get the vertices of the (bounded) feasible set
% vertices = explicit_mpc.partition.convexHull.V;
% disp(vertices)
% or better, compute the maximal invariant set
inv_set = model.invariantSet();
inv_set.plot();
disp(inv_set.V)


%% Solve the explicit MPC problem
% create MPC problem
N = 4;
explicit_mpc = MPCController(model, N).toExplicit();
% explicit_mpc.partition.plot()
% explicit_mpc.cost.fplot()
% xlim([-x_bnd, x_bnd])
% ylim([-x_bnd, x_bnd])
% zlim([-5, 80])


%% Extract the optimal value function and policy over the state space
n_elements = 5;
ls = linspace(-x_bnd, x_bnd, n_elements);
[X1, X2] = meshgrid(ls, ls);
V = zeros(n_elements, n_elements);
U = zeros(size(B, 2), n_elements, n_elements);
fprintf("Progress: %3d%%\n", 0);
for i = 1:n_elements
    fprintf("\b\b\b\b%3.0f%%", 100 * i / n_elements);
    for j = 1:n_elements
        x = [X1(i, j); X2(i, j)];
        [u,~,openloop] = explicit_mpc.evaluate(x);
        V(i, j) = openloop.cost;
        U(:, i, j) = u;
    end
end


%% Plot and save to disk
% surf(X1, X2, V, 'EdgeColor','none');
% title('Value Function')
% xlabel("x_1")
% ylabel("x_2")
% zlabel("V(x)")
% save(sprintf("value_func_data_%i.mat", N), "X1", "X2", "V")
