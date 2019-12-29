""" Normalizing flows applied to the moon dataset from sklearn, and to the MNIST dataset.

Citation:
    Variational inference with Normalizing Flows
    https://arxiv.org/abs/1505.05770
"""

from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_train_test_funcs import mnist_dataloaders, moon_dataset


# Argument parsing
parser = ArgumentParser(description="Training normalizing flows")

parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--mnist", action="store_true", default=False)
parser.add_argument("--nworkers", type=int, default=0)
parser.add_argument("--bsize", type=int, default=64)
parser.add_argument("--nsamples", type=int, default=250)
parser.add_argument("--noise", type=float, default=0.05)


# Flow defintions
class PlanarFlow(nn.Module):
    def __init__(self):
        super(PlanarFlow, self).__init__()

    def forward(self, x):
        return x

    def backward(self, x):
        return x


# Main execution loop
def main(args):

    # Load dataset
    if args.minst:
        train_loader, _ = mnist_dataloaders(batch_size=args.bsize, num_workers=args.nworkers)
    else:
        train_data = moon_dataset(args.nsamples, args.noise)
        train_loader = DataLoader(train_data, batch_size=args.bsize, num_workers=args.nworkers)

    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
