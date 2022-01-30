from models.baseline import DANN as Baseline
from models.image_utils import transform_source, transform_target

# Data
from torchvision.datasets import MNIST
from models.mnist_m import MNISTM
from torch.utils.data import DataLoader

import torch.optim as optim
from typing import Dict

import torch
import numpy as np
import os


# Fixed random seed
torch.random.manual_seed(42)

class BaselineTrainer:
    def __init__(self, cfg: Dict or None):
        self.cfg = cfg

        # Setup environment for training

        self.lr = float(cfg['lr'])
        self.num_epochs = int(cfg['num_epochs'])
        self.baseline = Baseline().to('cuda')
        self.optimizer = optim.Adam(self.baseline.parameters(), lr=self.lr)
        self.loss_class = torch.nn.NLLLoss().to('cuda')
        self.loss_domain = torch.nn.NLLLoss().to('cuda')

        # Source
        source = MNIST('../../data/mnist/train', download=True, transform=transform_source)
        self.source_dl = torch.utils.data.DataLoader(
                            dataset=source,
                            batch_size=128,
                            shuffle=True,
                            num_workers=8
                        )
        # Source test
        source = MNIST('../../data/mnist/test', download=True, transform=transform_source)
        self.source_test_dl = torch.utils.data.DataLoader(
                            dataset=source,
                            batch_size=128,
                            shuffle=True,
                            num_workers=8
                        )

        # Target
        target = MNISTM('../../data/mnistm/mnist_m_train.pt', download=True, transform=transform_target)
        self.target_dl = torch.utils.data.DataLoader(
                            dataset=target,
                            batch_size=128,
                            shuffle=True,
                            num_workers=8
                        )
        # Target test
        target = MNISTM('../../data/mnistm/mnist_m_test.pt', download=True, transform=transform_target)
        self.target_test_dl = torch.utils.data.DataLoader(
                            dataset=target,
                            batch_size=128,
                            shuffle=True,
                            num_workers=8
                        )
                
    def train_baseline(self):

        # Training
        for epoch in range(self.num_epochs):

            min_iter_limit = min(len(self.source_dl), len(self.target_dl))
            
            data_source_iterator = iter(self.source_dl)
            data_target_iterator = iter(self.target_dl)

            counter = 0
            
            while counter < min_iter_limit:
                
                p = float(counter + epoch * min_iter_limit) / 100 / min_iter_limit
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # Train on source data
                data = data_source_iterator.next()
                data, label = data

                self.baseline.zero_grad()
                batch_size = len(label)

                domain_label = torch.zeros(batch_size)
                domain_label = domain_label.long()

                data = data.to('cuda')
                label = label.to('cuda')
                domain_label = domain_label.to('cuda')
                
                class_output, domain_output = self.baseline(x=data, alpha=alpha)
                err_source_label = self.loss_class(class_output, label)
                err_source_domain = self.loss_domain(domain_output, domain_label)

                # Training on target data
                target_data = data_target_iterator.next()
                target_data, _ = target_data

                batch_size = len(target_data)

                domain_label = torch.ones(batch_size)
                domain_label = domain_label.long()

                target_data = target_data.to('cuda')
                domain_label = domain_label.to('cuda')

                _, domain_output = self.baseline(x=target_data, alpha=alpha)
                err_target_domain = self.loss_domain(domain_output, domain_label)

                # Computing full error
                err = err_source_domain + err_target_domain + err_source_label
                err.backward()
                self.optimizer.step()

                counter += 1

            print('epoch: %d, [iter: %d / all %d], err_source_label: %f, err_source_domain: %f, err_target_domain: %f' \
                % (epoch, counter, min_iter_limit, err_source_label.cpu().data.numpy(),
                    err_source_domain.cpu().data.numpy(), err_target_domain.cpu().data.numpy()))

        # Save baseline model
        self.baseline.to('cpu')
        torch.save(self.baseline.state_dict(), 'models/saved_models/baseline.pth')

    def test_baseline(self):
        # Test model accuracy (classification task)
        
        # Load model
        loaded_baseline = Baseline()
        loaded_baseline.load_state_dict(torch.load('models/saved_models/baseline.pth'))
        loaded_baseline.to('cuda')
        loaded_baseline.eval()

        min_iter_limit = min(len(self.source_test_dl), len(self.source_test_dl))
            
        data_source_iterator = iter(self.source_test_dl)
        data_target_iterator = iter(self.source_test_dl)

        i = 0
        n_total = 0
        n_correct = 0

        data_target_iterator = iter(self.source_test_dl)
        data_target_iterator = iter(self.target_test_dl)

        while i < min_iter_limit:

            # test model using target data
            data = data_target_iterator.next()
            target_data, target_label = data

            batch_size = len(target_label)

            target_data = target_data.to('cuda')
            target_label = target_label.to('cuda')
        
            class_output, _ = loaded_baseline(x=target_data, alpha=0)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(target_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size

            i += 1

        accu = n_correct.data.numpy() * 1.0 / n_total

        print ('accuracy of the %s dataset: %f' % ("mnist_m", accu))
