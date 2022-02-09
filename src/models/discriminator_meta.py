"""
 
 Baseline implementation (DANN), adapted for meta alignment: https://arxiv.org/pdf/2103.13575.pdf

"""


import torch.nn as nn
from utils.torch_funcs import grl_hook
from torch import Tensor

class Discriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int, out_feature: int = 1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_size, out_feature)
        )

        self._init_params()

    def forward(self, x: Tensor, coeff: float):
        x = x * 1.  # to avoid affect the grad from another pipeline to x_0
        x.register_hook(grl_hook(coeff))
        y = self.net(x)
        return y

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1., 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult': 10}]