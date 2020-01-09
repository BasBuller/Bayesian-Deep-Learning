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


# Argument parsing
parser = ArgumentParser(description="Training normalizing flows")

parser.add_argument("--bsize", type=int, default=128)
parser.add_argument("--nsamples", type=int, default=250)
parser.add_argument("--noise", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=9)
parser.add_argument("--data", type=str, action="store", default="moons")
parser.add_argument("--flow", type=str, action="store", default="planar")


class MoonDataset:
    """Simple wrapper around sklean moons dataset."""
    def __init__(self, noise=0.05):
        self.noise = noise

    def __call__(self, n_samples):
        data, _ = make_moons(n_samples=n_samples, noise=self.noise)
        return torch.from_numpy(data).float()


def main():
    """Execution of training and visualization of results"""

    args = parser.parse_args()

    # Load dataset
    if args.data == "moons":
        dataset = MoonDataset(args.noise)


if __name__ == "__main__":
    main()
