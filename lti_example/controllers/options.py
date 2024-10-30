"""File containing options for various optimization solvers."""

OPTS = {
    "ipopt": {
        "error_on_fail": True,
        "expand": True,
        "print_time": False,
        "bound_consistency": True,
        "calc_lam_p": False,
        "calc_lam_x": True,
        "ipopt": {"max_iter": 500, "print_level": 0, "sb": "yes"},
    },
    "fatrop": {
        "error_on_fail": True,
        "expand": True,
        "print_time": False,
        "bound_consistency": True,
        "calc_lam_p": False,
        "calc_lam_x": True,
        "fatrop": {"max_iter": 500, "print_level": 0},
    },
    "qpoases": {
        "error_on_fail": False,
        "expand": True,
        "print_time": False,
        "printLevel": "none",
    },
}
