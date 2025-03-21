import casadi as cs
from csnn.convex import PsdNN, PwqNN
from csnn.feedforward import Mlp


def nn2function(net: Mlp | PsdNN | PwqNN, prefix: str) -> cs.Function:
    """Converts a neural network model into a CasADi function.

    Parameters
    ----------
    net : Mlp, PsdNN, or PwqNN
        The neural network that must be converted to a function.
    prefix : str
        Prefix to add in front of the net's parameters. Used also to name the CasADi
        function.

    Returns
    -------
    cs.Function
        A CasADi function with signature `"input" x "parameters" -> "output"`.
    """
    if isinstance(net, (Mlp, PwqNN)):
        in_features = (
            net.input_layer.in_features
            if isinstance(net, PwqNN)
            else net.layers[0].in_features
        )
        x = net.sym_type.sym("x", in_features, 1)

        inputs = [x]
        input_names = ["x"]
        raw_output = net.forward(x.T)

    elif isinstance(net, PsdNN):
        in_features = net.hidden_layers[0].in_features
        out_features = net.ref_head.weight.size1()
        context = net.sym_type.sym("x", in_features, 1)
        x = net.sym_type.sym("x", out_features, 1)

        inputs = [x, context]
        input_names = ["x", "context"]
        raw_output = net.quadform(x.T, context.T)

    else:
        raise ValueError(
            "The neural network model must be an instance of Mlp, PwqNN or PsdNN."
        )

    outputs = [cs.simplify(raw_output.T if raw_output.is_row() else raw_output)]
    parameters = dict(net.parameters(prefix=prefix, skip_none=True))
    return cs.Function(
        prefix,
        inputs + list(parameters.values()),
        outputs,
        input_names + list(parameters.keys()),
        ["V"],
        {"cse": True},
    )
