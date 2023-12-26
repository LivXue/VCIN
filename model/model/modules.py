import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GaussianEncoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc21 = nn.Linear(output_dim, output_dim)  # mu values
        self.fc22 = nn.Linear(output_dim, output_dim)  # sigma values
        # setup the non-linearities
        self.act = nn.GELU()

    def forward(self, x, sampling: int = 0):
        if x is None:
            if sampling:
                return None, None, None
            return None, None
        # then compute the hidden units
        hidden = self.act(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mean = self.fc21(hidden)
        z_var = torch.exp(self.fc22(hidden))

        if sampling > 0:
            if self.training:
                noise = torch.randn([z_var.shape[0], sampling, z_var.shape[1]]).to(z_var.device)
                return z_mean, z_var, z_mean.unsqueeze(1) + torch.exp(self.fc22(hidden) / 2).unsqueeze(1) * noise
            else:
                return z_mean, z_var, z_mean.unsqueeze(1)

        return z_mean, z_var
