from numpy import asarray, concatenate

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
TIME_MEAS = "t_proc_total"  # or t_wall_total

DCBF_GAMMA = 0.95

PWQNN_HIDDEN = 16
PSDNN_HIDDEN = KAPPANN_HIDDEN = (16, 16)

NORMALIZATION = {
    "state_mean": asarray(
        [
            6.0236548373104934,
            6.151740949800862,
            8.93540450257289,
            1.7912016729274385,
            1.7951859511426536,
            1.6033835275569344,
        ]
    ),
    "state_std": asarray(
        [
            3.8305345178751216,
            3.867992622023662,
            3.444267861725363,
            1.0424283860886545,
            1.0149382660368655,
            1.5038041427261817,
        ]
    ),
    "action_mean": asarray(
        [
            10.895486281952921,
            -0.003770499934022998,
            0.004818233523290866,
            -0.000921798165769234,
        ]
    ),
    "action_std": asarray(
        [
            9.255642102619968,
            0.21482505173097652,
            0.23155736550237246,
            0.05589278934746118,
        ]
    ),
    "pos_obs_mean": asarray(
        [
            2.1501183931938646,
            3.149195729198508,
            0.01915161847424313,
            8.070382473003617,
            0.07263722684384825,
            7.910910323305432,
            11.86892471734808,
            11.975422194254207,
            0.06951498743674385,
        ]
    ),
    "pos_obs_std": asarray(
        [
            0.9882564008971457,
            1.0240661424691657,
            1.0029827469791857,
            1.0409915674956833,
            1.0364195836037582,
            0.9836045719212586,
            0.9964865665158052,
            0.977362603387804,
            0.9814155602746699,
        ]
    ),
    "dir_obs_mean": asarray(
        [
            5.830065365259106e-5,
            -0.006040709069421122,
            0.9895049792346912,
            -0.0014712701304950841,
            0.9896505187192365,
            -0.0057590560373283915,
            -0.010106393569256784,
            -0.0034225069279508052,
            0.989990141410835,
        ]
    ),
    "dir_obs_std": asarray(
        [
            0.10226346768120201,
            0.10180337023361205,
            0.009558758700248624,
            0.10257391678649148,
            0.011094264318978846,
            0.09990384993548733,
            0.09555828673217691,
            0.10316701077724287,
            0.009853353658495698,
        ]
    ),
}
CONTEXT_NORMALIZATION = tuple(
    concatenate(
        [
            NORMALIZATION[name + suffix]
            for name in ("state", "action", "pos_obs", "dir_obs")
        ]
    )
    for suffix in ("_mean", "_std")
)
