"""
Domain adversarial training.
"""

from torchvision.datasets import MNIST
from mnist_m import MNISTM
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Source
    source = MNIST('../data/mnist/train', download=True)

    # Target
    target = MNISTM('../data/mnistm/', download=True)

    # Plot a sample
    sample, _ = target[0]
    print(sample)
    plt.imshow(np.asarray(sample))
    plt.show()

    # Plot a sample
    sample, _ = source[0]
    print(sample)
    plt.imshow(np.asarray(sample))
    plt.show()


