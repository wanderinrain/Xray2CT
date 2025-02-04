"""
/*
 * Created on Tue Nov 19 2024
 *
 * Copyright (c) 2024 - Yiran Sun (ys92@rice.edu)
 */
"""

"""
This file is about to train the full model
"""


from fullmodel import FullModel
from feature_extract import unet
from feature_fuse import Resnet
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

"""
Check if the system has GPU, else use CPU
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


"""
Create Dataloaders

You need load your (xray1, xray2, CT) pair dataset here!!!
"""
# PUT your data here:

trainloader = torch.utils.data.DataLoader(xray_ct_traindata, shuffle=True, batch_size=1, num_workers=1)

"""
Train the Model
"""

ae_model = unet(input_channel=1)
ae_model.to(device)

mlp1 = Resnet(hidden_dim=128, num_blocks=3, view_direction='frontal')
mlp2 = Resnet(hidden_dim=128, num_blocks=3, view_direction='lateral')
mlp1.to(device)
mlp2.to(device)

full_model = FullModel(ae_model=ae_model, mlp_model1=mlp1, mlp_model2=mlp2, z=128, y=128, x=128, hidden_dim=128, out_dim=1, num_layers=4)
full_model.to(device)

"""
Optimizer and Critirion
"""
lr = 3e-5
optim_G = torch.optim.Adam(full_model.parameters(), lr=lr)
crit = nn.MSELoss()
num_epochs = 100
print(num_epochs)

print("Start training model!")

"""
Training Process
"""
for epoch in range(num_epochs):

    print('epoch', epoch)

    if (epoch+1) % 50 == 0:
        lr = 0.1*lr
        for group in optim_G.param_groups:
            group['lr'] = lr

    for i, data in enumerate(trainloader, 0):

        x_ray_train1, x_ray_train2, ct_train = data[2].to(device), data[0].to(device), data[4].to(device) # please make sure x_ray_train1 is fronatal, x_ray_train2 is lateral
        _, _, fake = full_model(img1=x_ray_train1, img2=x_ray_train2)
        loss_recon = crit(fake, ct_train)
        full_model.zero_grad()
        loss_recon.backward()
        optim_G.step()
