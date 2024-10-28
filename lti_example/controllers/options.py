"""File containing options for various optimization solvers."""

OPTS = {
    "fatrop": {
        "error_on_fail": True,
        "expand": True,
        "print_time": False,
        "bound_consistency": True,
        "calc_lam_p": False,
        "calc_lam_x": False,
        "fatrop": {"max_iter": 500, "print_level": 0},
    },
    "qpoases": {
        "error_on_fail": False,
        "expand": True,
        "print_time": False,
        "printLevel": "none",
    },
}
