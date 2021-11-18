import torch
from mnist_m import MNISTM

if __name__ == '__main__':
    ds = MNISTM('../data/mnistm/', train=True, download=True, transform=None)
    print(ds[0])


