import numpy as np 
import random
import torch
import os, sys
import abc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    def __init__(self, hidden_dim=4, macro_dim=6, L=8, box_L=8):
        super().__init__()

        assert L % box_L == 0, "L must be divisible by box_L"

        self.hidden_dim = hidden_dim
        self.macro_dim = macro_dim
        self.L = L
        self.box_L = box_L
        self.d = int(L / box_L)
       
        num_downsamples = int(np.log2(box_L // 2)) # 2
        self.num_downsamples = num_downsamples
        layers = []
        in_channels = 2
        out_channels = 32 
        output_size = box_L 
        
        for _ in range(num_downsamples):
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.MaxPool3d(2, stride=2))

            in_channels = out_channels
            out_channels *= 2
            output_size //= 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels * output_size ** 3, hidden_dim))
        self.model = nn.Sequential(*layers)    

        # self.register_buffer('min_val', torch.zeros(macro_dim + hidden_dim), persistent=False)
        # self.register_buffer('max_val', torch.ones(macro_dim + hidden_dim), persistent=False)

        self.register_buffer('mean', torch.zeros(macro_dim + hidden_dim), persistent=False)
        self.register_buffer('std', torch.ones(macro_dim + hidden_dim), persistent=False)
        
        self.apply(weights_init)
            
    def forward(self, x, z_macro):   
        # x: [B, 2, 8, 8, 8] 

        z_hat = self.model(x)  # [B, hidden_dim]
        z = torch.cat([z_macro, z_hat], -1)  # [B, macro_dim + hidden_dim]
        return z


class Decoder(nn.Module):
    def __init__(self, hidden_dim=3, macro_dim=3, L=8, box_L=8, num_classes=4): # Added L
        super().__init__()
        # assert box_L in [4, 8, 16, 32], "box_L must be in [4, 8, 16, 32]"
        assert L % box_L == 0, "L must be divisible by box_L"
        self.num_classes = num_classes
        self.macro_dim = macro_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.box_L = box_L
        self.d = int(L / box_L)

        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim + self.macro_dim, 128 * 2 * 2 * 2), 
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 2, 2, 2)), # [B, 128, 2, 2, 2]

            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 4, 4, 4]
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(64, 2 * num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 2, 8, 8, 8]
            # nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, z):
        # z: [B, macro_dim + hidden_dim]
        # return shape: [B, INPUT_DIM]

        x = self.model(z) # [B, 2 * 4, 8, 8, 8]
        x = x.reshape(-1, 2, self.num_classes, self.L, self.L, self.L)  # [B, 2, num_classes, L, L, L]
        return x

    
class Conv3DAutoencoder(nn.Module):
    def __init__(self, hidden_dim=3, macro_dim=3, L=32, box_L=16, num_classes=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.macro_dim = macro_dim
        
        self.encoder = Encoder(hidden_dim, macro_dim, L, box_L)
        self.decoder = Decoder(hidden_dim, macro_dim, L, box_L, num_classes)
    
    def forward(self, x, z_macro=None):
        # x: [B, 2, 8, 8, 8]
        z = self.encoder(x, z_macro)
        x = self.decoder(z) 
        return x
    
    def encode(self, x, z_macro=None):
        # x: [B, 2, 8, 8, 8]
        z = self.encoder(x, z_macro)
        return z
