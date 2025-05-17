
"""
ANSWER:

Structural Bug:
The code uses a fixed `batch_size` value to create label tensors, but in the last batch of the DataLoader, the actual number of samples might be smaller than the defined batch size.
This mismatch between the size of the generated data and labels leads to a runtime error such as:
    ValueError: Using a target size (torch.Size([128, 1])) that is different to the input size (torch.Size([96, 1]))

Fix:
Instead of using the fixed `batch_size`, use the actual size of the current batch:
    `current_batch_size = real_samples.size(0)`

Cosmetic Bug:
The condition `if n == batch_size - 1:` is logically incorrect because `n` is the batch index, and `batch_size` is the number of samples per batch.
This condition may never trigger. To ensure visualization happens at the last batch of each epoch, use:
    `if n == len(train_loader) - 1:`
"""

import matplotlib

matplotlib.use('TkAgg')

import csv
import numpy as np
NoneType = type(None)
import matplotlib.pyplot as plt

# <editor-fold desc="Exercise 3">
# You can copy this code to your personal pipeline project or execute it here.
def plot_data(csv_file_path: str):
    """
    This code plots the precision-recall curve based on data from a .csv file,
    where precision is on the x-axis and recall is on the y-axis.
    It it not so important right now what precision and recall means.

    :param csv_file_path: The CSV file containing the data to plot.


    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    | ``1   For some reason the plot is not showing correctly, can you find out what is going wrong?``
    | ``2   How could this be fixed?``

    This example demonstrates the issue.
    It first generates some data in a csv file format and the plots it using the ``plot_data`` method.
    If you manually check the coordinates and then check the plot, they do not correspond.

    >>> f = open("data_file.csv", "w")
    >>> w = csv.writer(f)
    >>> _ = w.writerow(["precision", "recall"])
    >>> w.writerows([[0.013,0.951],
    ...              [0.376,0.851],
    ...              [0.441,0.839],
    ...              [0.570,0.758],
    ...              [0.635,0.674],
    ...              [0.721,0.604],
    ...              [0.837,0.531],
    ...              [0.860,0.453],
    ...              [0.962,0.348],
    ...              [0.982,0.273],
    ...              [1.0,0.0]])
    >>> f.close()
    >>> plot_data('data_file.csv')
    """
    # load data
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            if len(row) != 0:
                results.append(row)
        results = np.stack(results)

    # plot precision-recall curve
    # x , y
    plt.plot(results[:, 0], results[:, 1])
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()
if __name__ == "__main__":

    f = open("data_file.csv", "w")
    w = csv.writer(f)
    _ = w.writerow(["precision", "recall"])
    w.writerows([[0.013,0.951], #[x, y]
                 [0.376,0.851],
                 [0.441,0.839],
                 [0.570,0.758],
                 [0.635,0.674],
                 [0.721,0.604],
                 [0.837,0.531],
                 [0.860,0.453],
                 [0.962,0.348],
                 [0.982,0.273],
                 [1.0,0.0]])
    f.close()
    plot_data('data_file.csv')
# </editor-fold>


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

