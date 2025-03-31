from collections.abc import Generator, Sequence
from typing import Literal, TypeVar

import casadi as cs
import numpy as np
from csnn import Linear, Module, ReLU
from csnn.convex import PsdNN, PwqNN
from csnn.convex.psd import _reshape_mat
from csnn.feedforward import Mlp
from numpy.typing import ArrayLike, NDArray

SymType = TypeVar("SymType", cs.SX, cs.MX)


class DynTanh(Module[SymType]):
    """Dynamic tanh normalization layer from https://arxiv.org/abs/2503.10622.

    Parameters
    ----------
    num_features : int
        Number of input features.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.factor = self.sym_type.sym("factor", 1, num_features)
        self.scale = self.sym_type.sym("scale", 1, num_features)
        self.offset = self.sym_type.sym("offset", 1, num_features)

    def _custom_init(
        self, *_, **__
    ) -> Generator[tuple[str, NDArray[np.floating]], None, None]:
        """Custom initialization of the parameters."""
        yield "factor", np.full(self.factor.shape, 0.5)
        yield "scale", np.ones(self.scale.shape)
        yield "offset", np.zeros(self.offset.shape)

    def forward(self, input: SymType) -> SymType:
        """Computes the normalized output.

        Parameters
        ----------
        input : SymType
            The input tensor of shape `(batch_size, num_features)`.

        Returns
        -------
        SymType
            The output tensor of shape `(batch_size, num_features)`.
        """
        y = cs.tanh(self.factor * input)
        return self.offset + self.scale * y

    def extra_repr(self) -> str:
        return f"{self.num_features}"


class ConLTIKappaNN(Module[SymType]):
    """Class Kappa function neural network for the constrained LTI environment.

    Parameters
    ----------
    in_features : int
        Number of input features (i.e., size of the context).
    hidden_features : sequence of int
        Number of features in each hidden linear layer.
    output_features : int
        Number of output features (i.e., size of the constraints).
    """

    def __init__(
        self, in_features: int, hidden_features: Sequence[int], output_features: int
    ) -> None:
        super().__init__()
        self.normalization = DynTanh(in_features)
        self.mlp = Mlp(
            (in_features, *hidden_features, output_features),
            [ReLU] * len(hidden_features) + [None],
        )

    def forward(self, input: SymType) -> SymType:
        gamma_unscaled = self.mlp(self.normalization(input))
        return (cs.tanh(gamma_unscaled) + 1) / 2


class QuadrotorNN(PsdNN):
    """Network for the `QuadrotorEnv` that combines positive semidefinite terminal
    cost approximation and a class Kappa function approximation (with shared
    parameters).

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : sequence of ints
        Number of features in each hidden linear layer.
    out_size : int
        Size of the quadratic form elements (i.e., the side length of the PSD matrix).
    out_shape : {"flat", "triu", "tril"}
        Shape of the output PSD matrix. If "flat", the output is not reshaped in any
        matrix. If "triu" or "tril", the output is reshaped as an upper or lower
        triangular, but does not support batched inputs.
    act : type of activation function, optional
        Class of the activation function. By default, `ReLU` is used.
    eps : array-like, optional
        Value to add to the PSD matrix, e.g., to ensure it is positive definite. Should
        be broadcastable to the shape `(out_size, out_size)`. By default, an identity
        matrix with `1e-4` is used. Only used in the `quadform` method.

    Raises
    ------
    ValueError
        Raises if the number of hidden layers is less than 1.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_size: int,
        out_shape: Literal["flat", "triu", "tril"],
        act: type[Module] = ReLU,
        eps: ArrayLike | None = None,
    ) -> None:
        super().__init__(in_features, hidden_features, out_size, out_shape, act, eps)
        self.normalization = DynTanh(in_features)
        self.cbf_head = Linear(hidden_features[-1], 1)

    def _forward(self, input: SymType) -> tuple[SymType, SymType, SymType]:
        h = self.hidden_layers(self.normalization(input))
        ref = self.ref_head(h)
        mat_flat = self.mat_head(h)
        gamma = (cs.tanh(self.cbf_head(h)) + 1) / 2
        return mat_flat, ref, gamma

    def forward(self, input: SymType) -> tuple[SymType, SymType, SymType]:
        mat, ref, gamma = self._forward(input)
        if self._out_shape != "flat":
            mat = _reshape_mat(mat, self._eps.size1(), self._out_shape)
        return mat, ref, gamma

    def terminal_cost_and_kappa(
        self, x: SymType, context: SymType
    ) -> tuple[SymType, SymType]:
        """Computes the terminal cost quadratic form `(x - x_ref)' Q (x - x_ref)`, where
        `Q` is the predicted the PSD matrix and `x_ref` the reference point, as well as
        the class Kappa function's linear factor.

        Parameters
        ----------
        x : SymType
            The value at which the quadratic form is evaluated.
        context : SymType
            The context passed as input to the neural network for the prediction of the
            PSD matrix and the reference point.

        Returns
        -------
        SymType
            The value of the quadratic form.
        """
        L_flat, ref, gamma = self._forward(context)
        L = _reshape_mat(L_flat, self._eps.size1(), "tril")
        return cs.bilin(L @ L.T + self._eps, x - ref), gamma


def nn2function(net: ConLTIKappaNN | QuadrotorNN | PwqNN, prefix: str) -> cs.Function:
    """Converts a neural network model into a CasADi function.

    Parameters
    ----------
    net : ConLTIKappaNN, QuadrotorNN, or PwqNN
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

    elif isinstance(net, ConLTIKappaNN):
        in_features = net.mlp.layers[0].in_features
        x = net.sym_type.sym("x", in_features, 1)

        inputs = [x]
        input_names = ["context"]
        outputs = [net.forward(x.T)]
        output_names = ["gamma"]

    elif isinstance(net, QuadrotorNN):
        in_features = net.hidden_layers[0].in_features
        out_features = net.ref_head.weight.size1()
        context = net.sym_type.sym("x", in_features, 1)
        x = net.sym_type.sym("x", out_features, 1)

        inputs = [x, context]
        input_names = ["x", "context"]
        outputs = net.terminal_cost_and_kappa(x.T, context.T)
        output_names = ["V", "gamma"]

    else:
        raise ValueError(
            "The neural network model must be an instance of ConLTIKappaNN, "
            "QuadrotorNN, or PwqNN."
        )

    parameters = dict(net.parameters(prefix=prefix, skip_none=True))
    return cs.Function(
        prefix,
        inputs + list(parameters.values()),
        [cs.simplify(o) for o in outputs],
        input_names + list(parameters.keys()),
        output_names,
        {"cse": True},
    )
