# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from array import array
from typing import Set, List
import numpy as np

import csv
import numpy as np
from typing import Set,Tuple, List
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision
NoneType = type(None)
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import vgg11
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
import time

# <editor-fold desc="Excercise 4">

class Generator(nn.Module):
    """
    Generator class for the GAN
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output
class Discriminator(nn.Module):
    """
    Discriminator class for the GAN
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output
def train_gan(batch_size: int = 64, num_epochs: int = 100, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    import torchvision
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
    except:
        torchvision.datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
             'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
             'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
             '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
             'ec29112dd5afa0611ce80d1b7f02629c')
        ]
        train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    loss_function = nn.BCELoss()
    lr = 0.0001
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            current_batch_size = real_samples.size(0)

            real_samples = real_samples.to(device)
            real_labels = torch.ones((current_batch_size, 1)).to(device)

            noise = torch.randn((current_batch_size, 100)).to(device)
            fake_samples = generator(noise)
            fake_labels = torch.zeros((current_batch_size, 1)).to(device)

            all_samples = torch.cat((real_samples, fake_samples))
            all_labels = torch.cat((real_labels, fake_labels))

            # Train discriminator
            discriminator.zero_grad()
            output = discriminator(all_samples)
            loss_discriminator = loss_function(output, all_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Train generator
            noise = torch.randn((current_batch_size, 100)).to(device)
            generator.zero_grad()
            generated = generator(noise)
            output = discriminator(generated)
            loss_generator = loss_function(output, real_labels)
            loss_generator.backward()
            optimizer_generator.step()

            if n == len(train_loader) - 1:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss D: {loss_discriminator.item():.4f}, Loss G: {loss_generator.item():.4f}")
                samples = generated.detach().cpu().numpy()
                fig = plt.figure(figsize=(6, 6))
                for i in range(16):
                    sub = fig.add_subplot(4, 4, i + 1)
                    sub.imshow(samples[i].reshape(28, 28), cmap="gray_r")
                    sub.axis('off')
                fig.suptitle(f"Generated Images at Epoch {epoch + 1}")
                fig.tight_layout()
                clear_output(wait=True)
                display(fig)
if __name__ == "__main__":
    train_gan(batch_size=64, num_epochs=100)

# </editor-fold>

