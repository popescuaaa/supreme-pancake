"""
    Training and evaluation
"""
# Models
from models import Baseline
import torch.optim as optim
import torch.nn as nn

# Data
from torchvision.datasets import MNIST
from mnist_m import MNISTM
from torch.utils.data import DataLoader
from torchvision import transforms

# Utils
import matplotlib.pyplot as plt
import numpy as np
import torch

transform_source = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

transform_target = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    lr = 1e-5
    baseline = Baseline().to('cuda')
    optimizer = optim.Adam(baseline.parameters(), lr=lr)
    loss_class = torch.nn.NLLLoss().to('cuda')
    loss_domain = torch.nn.NLLLoss().to('cuda')

    # Source
    source = MNIST('../data/mnist/train', download=True, transform=transform_source)
    source_dl = torch.utils.data.DataLoader(
                        dataset=source,
                        batch_size=128,
                        shuffle=True,
                        num_workers=8
                    )

    # Target
    target = MNISTM('../data/mnistm/', download=True, transform=transform_target)
    target_dl = torch.utils.data.DataLoader(
                        dataset=target,
                        batch_size=128,
                        shuffle=True,
                        num_workers=8
                    )
    
    # Training

    for epoch in range(100):

        min_iter_limit = min(len(source_dl), len(target_dl))
        
        data_source_iterator = iter(dataloader_source)
        data_target_iterator = iter(dataloader_target)

        counter = 0
        
        while counter < min_iter_limit:
            
            p = float(i + epoch * min_iter_limit) / 100 / min_iter_limit
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Train on source data
            data = data_source_iterator.next()
            data, label = data

            baseline.zero_grad()
            batch_size = len(label)

            input_img = torch.FloatTensor(batch_size, 3, 28, 28)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()