import torch
import torch.nn as nn


###############################
# Residual defintions
###############################
class ResidualFlow(nn.Module):
    r"""Base class for other normalizing flows."""

    def __init__(self, act_func):
        super(Flow, self).__init__()

        # Set activation function
        if act_func == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_func == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act_func == "sigmoid":
            self.activation = torch.sigmoid


class PlanarFlow(ResidualFlow):
    r"""Planar flow, according to f(z) = z + uh(w^{T}z + b)"""

    def __init__(self, n_points, act_func="relu"):
        super(PlanarFlow, self).__init__(act_func)
        self.u = nn.Parameter(torch.Tensor(n_points, 1))
        nn.init.kaiming_normal_(self.u)

        self.w = nn.Parameter(torch.Tensor(1, n_points))
        nn.init.kaiming_normal_(self.w)

        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, 0)

    def forward(self, x):
        # Run input through activation
        y = self.w @ x + self.b
        y = self.activation(y)

        # Get Jacobian of activation function
        jacobian = y.grad_fn(torch.ones_like(y))
        det = 1 + jacobian @ self.u.t() @ self.w.t()

        # Complete flow calculation
        y = x + self.u * y

        return y, det


class SylvesterFlow(ResidualFlow):
    r"""Planar flow extended to M hidden units, according to f(z) = z + Vh(W^{T}z + b)"""

    def __init__(self, n_points, act_func="relu"):
        super(SylvesterFlow, self).__init__(act_func)

    def forward(self, x):
        return


class RadialFlow(ResidualFlow):
    r"""Radial flow, according to f(z) = z + beta / (alpha + r(z)) * (z - z_0) where r(z) = ||z - z_0||"""

    def __init__(self, n_points, act_func="relu"):
        super(RadialFlow, self).__init__(act_func)

    def forward(self, x):
        return


###############################
# Utilities
###############################
class NormalizingFlowStack(nn.Module):
    r"""Combines a set of flows into a single model."""

    def __init__(self, prior, flows):
        super(NormalizingFlowStack, self).__init__()
        self.flows = nn.ModuleList(flows)
        self.priod = prior

    def forward(self, x):
        complete_det = 1
        for flow in self.flows:
            x, det = flow(x)
            complete_det *= det
        return x, complete_det


###############################
# Loss defintions
###############################
def forward_kld(log_p, log_det):
    return -(log_p + log_det).mean()


if __name__ == "__main__":
    a = torch.rand(5, requires_grad=True)
    plan_flow = PlanarFlow(5, act_func="sigmoid")

    b, det = plan_flow(a)

    print("Flow output value: ", b)
    print("Determinant value: ", det)
