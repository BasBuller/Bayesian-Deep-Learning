""" 
Normalizing flows applied to the moon dataset from sklearn.

Citation:
    Variational inference with Normalizing Flows
    https://arxiv.org/abs/1505.05770
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from normalizing_flows.flows import PlanarFlow, NormalizingFlowStack, forward_kld


# Argument parsing
parser = ArgumentParser(description="Training normalizing flows")

parser.add_argument("--bsize", type=int, default=128)
parser.add_argument("--nsamples", type=int, default=250)
parser.add_argument("--noise", type=float, default=0.05)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=9)
parser.add_argument("--data", type=str, action="store", default="moons")
parser.add_argument("--flow", type=str, action="store", default="planar")
parser.add_argument("--nlayers", type=int, default=4)
parser.add_argument("--activation", type=str, action="store", default="relu")


class MoonDataset:
    r"""Simple wrapper around sklean moons dataset."""
    def __init__(self, noise=0.05):
        self.noise = noise

    def __call__(self, n_samples):
        data, _ = make_moons(n_samples=n_samples, noise=self.noise)
        return torch.from_numpy(data).float()


class MoonNet(nn.Module):
    def __init__(self, n_layers, act_func="relu"):
        flows = [PlanarFlow(2, act_func=act_func) for _ in n_layers]
        self.flows = NormalizingFlowStack(flows)

    def forward(self, x):
        for flow in self.flows:
            x = flow(x)
        return x


def main():
    """Execution of training and visualization of results"""

    args = parser.parse_args()

    # Load dataset
    if args.data == "moons":
        dataset = MoonDataset(args.noise)

    model = MoonNet(args.nlayers, act_func=args.activation)
    criterion = 5


if __name__ == "__main__":
    main()
