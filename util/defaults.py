SOLVER_OPTS = {
    "fatrop": {
        "error_on_fail": False,
        "expand": True,
        "print_time": False,
        "record_time": True,
        "bound_consistency": True,
        "calc_lam_p": False,
        "calc_lam_x": True,
        "fatrop": {"max_iter": 500, "print_level": 0},
    },
    "ipopt": {
        "error_on_fail": False,
        "expand": True,
        "print_time": False,
        "record_time": True,
        "bound_consistency": True,
        "calc_lam_p": False,
        "calc_lam_x": True,
        "ipopt": {
            "max_iter": 500,
            "print_level": 0,
            "sb": "yes",
            "linear_solver": "ma27",
        },
    },
    "qpoases": {
        "error_on_fail": False,
        "expand": True,
        "record_time": True,
        "print_time": False,
        "printLevel": "none",
    },
    "osqp": {
        "error_on_fail": False,
        "expand": True,
        "record_time": True,
        "print_time": False,
        "warm_start_primal": True,
        "osqp": {
            "verbose": False,
            "scaling": 30,
            "max_iter": 10000,
            "polish": True,
            "polish_refine_iter": 30,
            "scaled_termination": True,
            "eps_abs": 1e-9,
            "eps_rel": 1e-9,
            "eps_prim_inf": 1e-10,
            "eps_dual_inf": 1e-10,
        },
    },
    "gurobi": {
        "error_on_fail": False,
        "expand": True,
        "record_time": True,
        "gurobi": {"OutputFlag": 0},
    },
}
PWQNN_HIDDEN = 16
DCBF_GAMMA = 0.95
TIME_MEAS = "t_proc_total"  # or t_wall_total
PSDNN_HIDDEN = KAPPANN_HIDDEN = (16, 16)
