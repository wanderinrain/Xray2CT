"""
/*
 * Created on Tue Nov 19 2024
 *
 * Copyright (c) 2024 - Yiran Sun (ys92@rice.edu)
 */
"""

"""
This file is about for each X-ray image, how to extend the feature image of X-ray image to 3D dimension and incorporate with 3D space coordinates
"""



import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""
Create coordinates grid
"""
def creategrid(z, y, x):

    tensor = tuple([torch.linspace(start = -1, end = 1, steps = z), torch.linspace(start = -1, end = 1, steps = y), torch.linspace(start = -1, end = 1, steps = x)])
    grid = torch.stack(torch.meshgrid(*tensor), dim=-1)
    grid = grid.reshape(-1, 3)
    return grid


"""
Positional Encoding: Gaussian Fourier feature method.

Please refer https://arxiv.org/pdf/2006.10739 for more encoding options.
"""

def encode_position(C):

    """Encodes the position into its corresponding Fourier feature.
    Args:
        C: The input coordinate (coords_num, 3)
    Returns:
        Fourier features tensors of the position.
    """
    C = C.to(device)
    positions = [C]
    B = torch.tensor([[-8.9113, 19.1638, -8.8575],
        [ 3.9577, 10.2506,  3.8891],
        [12.1395,  5.0181, 12.4339]]).to(device)

    for fn in [torch.cos, torch.sin]:
        positions.append(fn((2.0 * np.pi * C) @ B.T))

    return torch.cat(positions, axis=-1)



class Linear(nn.Module):

    def __init__(self, in_feat, out_feat, activate=True):
        super(Linear, self).__init__()

        self.main = [nn.Linear(in_feat, out_feat)]

        if activate:
            self.main.append(nn.ReLU(inplace = True))

        self.main = nn.Sequential(*self.main)

    def forward(self, x):

        return self.main(x)

"""
Siren Layer, please refer code and paper at https://arxiv.org/pdf/2006.09661
"""

class SirenLinear1(nn.Module):

    def __init__(self, in_feat, out_feat, bias=True, is_first=False, omega_0=50):

        """
        in_feat: input features number
        out_feat: output features number
        is_first: if it's the first layer
        omega_0: frequency
        """

        super(SirenLinear1, self).__init__()

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

        return self.omega_0 * self.linear(input) # remove the activation operation here for next residual block



"""
Residual Block
"""
class BasicBlock(nn.Module): # except last layer, each block

    def __init__(self, dim):

      """
      Each block contains two linear layer, every linear layer follows by a relu nonlinearity layer, for hidden layers

      Block:

      x --> Linear layer1 --> ACTIVATE --> Linear layer2 --> dx --> ACTIVATE -- > out
         |                                                   |
         |                                                   |
         |                                                   |
         |___________________________________________________|
                                   plus
      """

      super(BasicBlock, self).__init__()

      self.dim = dim

      self.fc = SirenLinear1(dim, dim, bias=True, is_first=False, omega_0=50)


    def forward(self, x):

        identity = x
        dx = self.fc(x)
        dx = torch.sin(dx)
        dx = self.fc(dx)
        out = dx + identity
        out = torch.sin(out)

        return out


"""
MLP Network
"""
class Resnet(nn.Module):

    def __init__(self, hidden_dim, num_blocks, view_direction):

        """
        hidden_dim: hidden layer dimension
        num_blocks: the number we want residual block number repeat
        view_direction: the angle of input xray
        """

        super(Resnet, self).__init__()

        self.num_blocks = num_blocks
        self.view_direction = view_direction
        self.fc = SirenLinear1(137, hidden_dim, bias=True, is_first=True, omega_0=50)
        self.blocks = nn.ModuleList([BasicBlock(hidden_dim) for i in range(num_blocks)])

    def forward(self, w, coords):
        # coords: (n_coord, 3). Assuming z-coordinate first, y-coordinate second, x-coordinate third
        # w: (batch, n_w_feat, H, W)

        batch, n_w_feat, H, W = w.shape  # batch size, feature number, image downsize area, such as 1, 128, 128, 128

        """
        repeat feature images into a feature volume along the view dimension
        
        assume your CT volume is in order (z, y, x)
        
        then along y should be frontal view, along x should be lateral view
        """
        if self.view_direction == 'frontal':

            w = w.unsqueeze(3).repeat(1, 1, 1, 128, 1)

        else:

            w = w.unsqueeze(4).repeat(1, 1, 1, 1, 128)

        # w has shape: (batch, n_w_feat, n_coord). But this isn't a good shape to do the
        # linear operation, since it expects shape of (*, n_w_feat). Let's permute and
        # reshape to get a shape of (batch * n_coord, n_w_feat).
        w = w.permute(0, 2, 3, 4, 1).reshape(-1, n_w_feat) # (128*128*128, 128)

        coords = encode_position(coords)  # (128*128*128, 9)

        x = torch.cat((w, coords), dim=1)

        x = self.fc(x)

        x = torch.sin(x)

        for i in range(self.num_blocks):

            x = self.blocks[i](x)

        return x.unsqueeze(0)




