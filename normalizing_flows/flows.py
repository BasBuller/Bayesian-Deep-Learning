import torch
import torch.nn as nn


# Flow defintions
class PlanarFlow(nn.Module):
    """Simple planar flow, according to f(z) = z + uh(w^{T}z + b)"""

    def __init__(self, n_points, act_func="relu"):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.Tensor(n_points, 1))
        nn.init.kaiming_normal_(self.u)

        self.w = nn.Parameter(torch.Tensor(n_points, 1))
        nn.init.kaiming_normal_(self.w)

        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, 0)

        # Set activation function
        if act_func == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_func == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.w @ x + self.b
        y = self.activation(y)
        y = x + self.u * y
        return y

    def backward(self, x):
        return x


class NormalizingFlowModel(nn.Module):
    """Combines a set of flows into a single model."""

    def __init__(self, flows):
        self.flows = nn.ModuleList(flows)
