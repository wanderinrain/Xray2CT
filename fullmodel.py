"""
/*
 * Created on Tue Nov 19 2024
 *
 * Copyright (c) 2024 - Yiran Sun (ys92@rice.edu)
 */
"""

"""
This file is about to fuse features from each view of X-ray image and get the final prediction
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def creategrid(z, y, x):

    tensor = tuple([torch.linspace(start = -1, end = 1, steps = z), torch.linspace(start = -1, end = 1, steps = y), torch.linspace(start = -1, end = 1, steps = x)])
    grid = torch.stack(torch.meshgrid(*tensor), dim=-1)
    grid = grid.reshape(-1, 3)

    return grid


class Linear(nn.Module):

    def __init__(self, in_feat, out_feat, activate=True):
        super(Linear, self).__init__()

        self.main = [nn.Linear(in_feat, out_feat)]

        if activate:
            self.main.append(nn.ReLU(inplace = True))

        self.main = nn.Sequential(*self.main)

    def forward(self, x):

        output = self.main(x)

        return output


"""
Siren Layer
"""

class SirenLinear(nn.Module):

    def __init__(self, in_feat, out_feat, bias=True, is_first=False, omega_0=50):

        """
        in_feat: input features number
        out_feat: output features number
        is_first: if it's the first layer
        omega_0: frequency
        """

        super(SirenLinear, self).__init__()

        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.is_first = is_first
        self.omega_0 = omega_0
        self.init_weights()

    def init_weights(self):

        with torch.no_grad():

            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_feat, 1 / self.in_feat)

            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_feat) / self.omega_0, np.sqrt(6 / self.in_feat) / self.omega_0)

    def forward(self, input):

        return torch.sin(self.omega_0 * self.linear(input))


class FullModel(nn.Module):

    def __init__(self, ae_model, mlp_model1, mlp_model2, z, y, x, hidden_dim, out_dim, num_layers):

        super(FullModel, self).__init__()

        self.z = z
        self.y = y
        self.x = x
        self.num_layers = num_layers
        self.ae_model = ae_model
        self.mlp_model1 = mlp_model1
        self.mlp_model2 = mlp_model2
        self.ops = [SirenLinear(hidden_dim, hidden_dim, bias=True, is_first=False, omega_0=50) for i in range(num_layers)]
        self.ops = nn.Sequential(*self.ops)
        self.out = Linear(hidden_dim, out_dim, activate=False)

    def forward(self, img1, img2):

        w1 = self.ae_model(img1)
        w2 = self.ae_model(img2)
        pred1 = self.mlp_model1(w1, creategrid(128, 128, 128).to(device))
        pred2 = self.mlp_model2(w2, creategrid(128, 128, 128).to(device))

        pred = (pred1 + pred2) / 2.0
        pred = self.out(self.ops(pred))
        pred = pred.reshape(1, 128 * 128 * 128, -1).permute(0, 2, 1)

        return w1, w2, pred.reshape(pred.shape[0], self.z, self.y, self.x)
