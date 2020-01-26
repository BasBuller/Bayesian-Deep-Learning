""" 
Normalizing flows applied to the moon dataset from sklearn.

Citation:
    Variational inference with Normalizing Flows
    https://arxiv.org/abs/1505.05770
"""

from argparse import ArgumentParser

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flows import PlanarFlow, SylvesterFlow, RadialFlow, NormalizingFlowStack, forward_kld
plt.style.use("ggplot")


# Argument parsing
parser = ArgumentParser(description="Training normalizing flows")

parser.add_argument("--bsize", type=int, default=128)
parser.add_argument("--nsamples", type=int, default=1024)
parser.add_argument("--noise", type=float, default=0.05)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=9)
parser.add_argument("--data", type=str, action="store", default="moons")
parser.add_argument("--flow", type=str, action="store", default="planar")
parser.add_argument("--nlayers", type=int, default=4)
parser.add_argument("--activation", type=str, action="store", default="relu")


class MoonDataset:
    r"""Simple wrapper around sklean moons dataset."""
    def __init__(self, n_samples, noise=0.05):
        data, _ = make_moons(n_samples=n_samples, noise=noise)
        self.samples = [torch.from_numpy(d).float().unsqueeze(1) for d in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MoonNet(NormalizingFlowStack):
    def __init__(self, n_layers, flow_type="planar", act_func="relu"):

        # Define base PDF
        self.pdf = torch.distributions.Normal(0, 1)

        # Select flow constructor
        if flow_type == "planar":
            flow = PlanarFlow
        elif flow_type == "radial":
            flow =  RadialFlow
        elif flow_type == "sylvester":
            flow == SylvesterFlow

        # Generate flow stack
        flows = [flow(2, act_func=act_func) for _ in range(n_layers)]
        super(MoonNet, self).__init__(flows)

    def forward(self, x):
        x, det = super(MoonNet, self).forward(x)
        log_p = self.pdf.log_prob(x)
        log_det = torch.log(det)
        return log_p, log_det


def main():
    r"""Execution of training and visualization of results"""

    args = parser.parse_args()

    # Load dataset
    if args.data == "moons":
        dataset = MoonDataset(args.nsamples, args.noise)
    dataloader = DataLoader(dataset, batch_size=args.bsize)

    # Set model, criterion, and optimizer
    model = MoonNet(args.nlayers, flow_type=args.flow, act_func=args.activation)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = forward_kld

    # Training
    losses = []
    progbar = tqdm(desc="Training progress", total=args.epochs)
    for _ in range(args.epochs):
        epoch_loss = 0
        for batch in dataloader:
            optim.zero_grad()

            # Forward pass and loss
            log_p, log_det = model(batch) 
            loss = criterion(log_p, log_det)

            loss.backward()
            optim.step()

            epoch_loss += loss.item()

        # Update progressbar
        progbar.set_postfix({"Epoch loss": epoch_loss}, refresh=False)
        progbar.update()
        losses.append(epoch_loss)

    # Plot results
    plt.figure(figsize=(15, 8))
    plt.plot(losses)
    plt.title("Loss progression during training")
    plt.show()


if __name__ == "__main__":
    main()
