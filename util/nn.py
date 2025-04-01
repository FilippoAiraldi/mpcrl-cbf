import casadi as cs
from csnn.convex import PwqNN


def nn2function(net: PwqNN, prefix: str) -> cs.Function:
    """Converts a neural network model into a CasADi function.

    Parameters
    ----------
    net : PwqNN
        The neural network that must be converted to a function.
    prefix : str
        Prefix to add in front of the net's parameters. Used also to name the CasADi
        function.

    Returns
    -------
    cs.Function
        A CasADi function with signature `"input" x "parameters" -> "output"`.
    """
    if isinstance(net, PwqNN):
        in_features = (
            net.input_layer.in_features
            if isinstance(net, PwqNN)
            else net.layers[0].in_features
        )
        x = net.sym_type.sym("x", in_features, 1)

        inputs = [x]
        input_names = ["x"]
        outputs = [net.forward(x.T)]
        output_names = ["V"]

    else:
        raise ValueError("The neural network model must be an instance of PwqNN.")

    parameters = dict(net.parameters(prefix=prefix, skip_none=True))
    return cs.Function(
        prefix,
        inputs + list(parameters.values()),
        [cs.simplify(o) for o in outputs],
        input_names + list(parameters.keys()),
        output_names,
        {"cse": True},
    )
