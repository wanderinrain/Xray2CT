"""
/*
 * Created on Tue Nov 19 2024
 *
 * Copyright (c) 2024 - Yiran Sun (ys92@rice.edu)
 */
"""


"""
This file is to extract the feature Image from X-ray Image, this is adapted from a U-Net
"""

import torch
import torch.nn as nn


def double_conv(in_chan, out_chan):
  # (Width/height - Kernel_size + 2 * padding) / stride + 1

  conv = nn.Sequential(
      nn.Conv2d(in_chan, out_chan, kernel_size = 5, stride = 1, padding = 2),
      nn.LeakyReLU(0.2),
      nn.Conv2d(out_chan, out_chan, kernel_size = 5, stride = 1, padding = 2),
      nn.LeakyReLU(0.2)
  )

  return conv


def crop_img(tensor, target_tensor):

  target_size = target_tensor.size()[2]
  tensor_size = tensor.size()[2]
  delta = tensor_size - target_size
  delta = delta // 2

  return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class unet(nn.Module):

    def __init__(self):

        super(unet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/ 2 input size

        # down sample

        self.down_conv1 = double_conv(1, 64)  # Medical images only have one channel, so our input channel is 1
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)

        # up sample

        self.up_trans1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2) # == upsample(2)
        self.up_conv1 = double_conv(512, 256)
        self.up_trans2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2) # == upsample(2)
        self.up_conv2 = double_conv(256, 128)
        self.up_trans3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2) # == upsample(2)

        self.out = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)

    def forward(self, image):

        # encoder
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)


        # decoder
        x = self.up_trans1(x7)
        y = crop_img(x5, x)
        x = self.up_conv1(torch.cat([x, y], 1))
        x = self.up_trans2(x)
        y = crop_img(x3, x)
        x = self.up_conv2(torch.cat([x, y], 1))
        x = self.up_trans3(x)

        # output
        x = self.out(x) # (batch, feature, H, W)

        return x


