import torch.nn as nn
import numpy as np
import torch
from modules.sti_net import StackedSTINet

class Full_Model(nn.Module):
    def __init__(self, net, w_size, n_dims, mode, bn_mode =True):
        super(Full_Model, self).__init__()

        # define STI-net encoder with decoder
        self.bn= bn_mode
        self.mode = mode
        self.relu = nn.ReLU()
        self.R = net[0]
        self.T=net[1]
        self.E = net[2]
        self.linear = nn.Linear(n_dims*w_size, 256)
        self.linear1 = nn.Linear(256, 257, bias=False)    #include 1 uncetainty feature
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(256)

    def forward(self, masked_batch):
        if isinstance(self.R, StackedSTINet) or isinstance(self.E, StackedSTINet):
            # Check input length should evenly divided for 2^level
            assert self.input_len % (np.power(2,
                                              self.num_levels)) == 0  # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)


        r_output = self.R(masked_batch)

        if isinstance(self.R, StackedSTINet):
            t_output = self.T(r_output.permute(0, 2, 1))
        else:
            t_output = self.T(r_output)

        if isinstance(self.E, StackedSTINet):
            x = self.E(t_output.permute(0, 2, 1))
        else:
            x = self.E(t_output)

        x = self.flat(x)
        x = self.linear(x)
        if self.bn:
            x = self.batchnorm(x)
        x = self.relu(x)
        # x = self.drop(x)
        x= self.linear1(x)

        features = x[:, :-1]
        u = x[:, -1]
        u = torch.unsqueeze(u, 1)

        if self.mode ==1:
            return r_output, features, u
        elif self.mode ==2:
            return features, u