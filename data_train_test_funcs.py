import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.datasets import make_moons

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid


# Vectorized dataset container class
class VectorizedDataset(Dataset):
    """Class to contain matrices of vectorized dataset into a PyTorch compatible dataset.
    
    Make sure the first dimension of the Tensors is the sample dimension.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem(self, idx):
        return self.x[idx], self.y[idx]


# sklearn moon dataset wrapped in PyTorch dataset
def moon_dataset(n_samples, noise, seed=9):
    """Returns a PyTorch dataset containing sklearn's moon data."""
    x, y = make_moons(n_samples=n_samples, shuffle=True, noise=noise, random_state=seed)
    return VectorizedDataset(x, y)


# Quick loading of datasets
def mnist_dataloaders(batch_size=128, num_workers=4, pin_memory=False):
    r"""Prepare mnist dataloaders"""

    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    train_mnist = datasets.MNIST("./data", train=True, transform=transf)
    train_loader = DataLoader(train_mnist, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    test_mnist = datasets.MNIST("./data", train=False, transform=transf)
    test_loader = DataLoader(test_mnist, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader


# Simple training loop
def train(train_step, models, dataloader, epochs=20, gpu=True):

    # Initialize device
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # Push model(s) to device and set to training mode
    for model in models:
        model.train(mode=True)
        model.to(device)

    # training
    bar = tqdm.tqdm(total=epochs)
    for _ in range(epochs):
        for b_idx, batch in enumerate(dataloader):
            batch = [i.to(device) for i in batch]

            # Run forward and backward pass
            loss_dict = train_step(batch, b_idx)

        # Update progress bar
        bar.set_postfix(ordered_dict=loss_dict, refresh=False)
        bar.update()


# Simple testing loop
def test(model, dataloader, gpu=True):
    pass


# Image visualizations
def sample_images(generator, latent_samples, out_shape=(1, 28, 28)):
    samples = latent_samples.cpu()
    generator = generator.cpu()
    generator.train(mode=False)
    with torch.no_grad():

        # Draw samples from generative model 
        samples = generator(latent_samples).detach()
        ims = make_grid(samples.view(-1, *out_shape), nrow=8, padding=10)
        ims = ims.numpy()

        # Plot figures 
        plt.figure(figsize=((latent_samples.shape[0]// 8)*5, 20))
        plt.imshow(np.transpose(ims, (1, 2, 0)), interpolation="nearest")
        plt.show()

