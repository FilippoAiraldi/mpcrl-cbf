import casadi as cs
from csnn import Module


def nn2function(net: Module, prefix: str) -> cs.Function:
    """Converts a neural network model into a CasADi function.

    Parameters
    ----------
    net : Module
        The neural network that must be converted to a function. Must either possess
        an `input_layer` attribute or a `layers` attribute.
    prefix : str
        Prefix to add in front of the net's parameters. Used also to name the CasADi
        function.

    Returns
    -------
    cs.Function
        A CasADi function with signature `"input" x "parameters" -> "output"`.

    Raises
    ------
    ValueError
        If the neural network model input layer cannot be identified.
    """
    if hasattr(net, "input_layer"):
        in_features = net.input_layer.in_features
    elif hasattr(net, "layers"):
        in_features = net.layers[0].in_features
    else:
        raise ValueError(
            "The neural network model must be an instance of Mlp, PwqNN or PsdNN."
        )
    x = net.sym_type.sym("x", in_features, 1)

    output = net.forward(x.T)
    if output.is_row():
        output = output.T

    parameters = dict(net.parameters(prefix=prefix, skip_none=True))
    return cs.Function(
        prefix,
        [x] + list(parameters.values()),
        [cs.simplify(output)],
        ["x"] + list(parameters.keys()),
        ["y"],
        {"cse": True},
    )
