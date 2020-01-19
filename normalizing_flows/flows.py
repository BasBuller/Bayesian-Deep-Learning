import torch
import torch.nn as nn


###############################
# Flow defintions
###############################
class PlanarFlow(nn.Module):
    r"""Simple planar flow, according to f(z) = z + uh(w^{T}z + b)"""

    def __init__(self, n_points, act_func="relu"):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.Tensor(n_points, 1))
        nn.init.kaiming_normal_(self.u)

        self.w = nn.Parameter(torch.Tensor(1, n_points))
        nn.init.kaiming_normal_(self.w)

        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, 0)

        # Set activation function
        if act_func == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_func == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act_func == "sigmoid":
            self.activation = torch.sigmoid

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


class NormalizingFlowStack(nn.Module):
    r"""Combines a set of flows into a single model."""

    def __init__(self, flows):
        self.flows = nn.ModuleList(flows)


if __name__ == "__main__":
    a = torch.rand(5, requires_grad=True)
    plan_flow = PlanarFlow(5, act_func="sigmoid")

    b, det = plan_flow(a)

    print("Flow output value: ", b)
    print("Determinant value: ", det)
