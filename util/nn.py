import casadi as cs
from csnn.convex import PwqNN


def nn2function(net: PwqNN, name: str = "nn") -> cs.Function:
    """Converts the neural network model to a CasADi function.

    Parameters
    ----------
    net : PwqNN
        The neural network that must be converted to a function.
    name : str
        Name of the function.

    Returns
    -------
    cs.Function
        A CasADi function with signature `"input" x "parameters" -> "output"`.
    """
    in_features = net.input_layer.in_features
    x = net.sym_type.sym("x", in_features, 1)
    y = cs.simplify(cs.cse(net.forward(x.T).T))
    p = dict(net.parameters(skip_none=True))
    return cs.Function(
        name, [x] + list(p.values()), [y], ["x"] + list(p.keys()), ["y"], {"cse": True}
    )
