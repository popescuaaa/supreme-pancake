"""
 DANN: Unsupervised Domain Adaptation by Backpropagation
"""

import torch.nn as nn
from torch import Tensor
from models.functions import ReverseLayerF
import torch

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax()
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: Tensor, alpha: float):
        x = x.expand(x.data.shape[0], 3, 28, 28)
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    @property
    def device(self):
        return next(self.parameters()).device

if __name__ == "__main__":
    dann = DANN()
    x = torch.rand((28, 28))
    c, d = dann(x=x, alpha=0.12)
    assert c.shape == torch.Size([28, 10]), "The model failed to produce correct shape for output"
    assert d.shape == torch.Size([28, 2]), "The model failed to produce correct shape for output"