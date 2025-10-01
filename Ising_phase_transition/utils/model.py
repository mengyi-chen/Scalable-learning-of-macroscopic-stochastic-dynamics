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
    def __init__(self, closure_dim=2, macro_dim=6, L=32, patch_L=16, h=0.0):
        super().__init__()
        assert L % patch_L == 0, "L must be divisible by patch_L"

        self.closure_dim = closure_dim
        self.macro_dim = macro_dim
        self.L = L
        self.patch_L = patch_L
        self.d = int(L / patch_L)
        self.h = h
        
        if patch_L == 32:
            num_downsamples = 3
        else:
            num_downsamples = int(np.log2(patch_L // 2))
        self.num_downsamples = num_downsamples
        layers = []
        in_channels = 1
        out_channels = 32 
        output_size = patch_L 
        
        for _ in range(num_downsamples):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.MaxPool2d(2, stride=2))

            in_channels = out_channels
            out_channels *= 2
            output_size //= 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels * output_size ** 2, closure_dim))
        self.model = nn.Sequential(*layers)

        self.register_buffer('min_val', torch.zeros(macro_dim + closure_dim), persistent=False)
        self.register_buffer('max_val', torch.ones(macro_dim + closure_dim), persistent=False)
        
        self.apply(weights_init)
    
    @abc.abstractmethod
    def cal_macro(self, x):
        pass 
            
    def forward(self, x):  
        # x: [B, 1, 16, 16]
        z_macro = self.cal_macro(x) # [B, 1, 16, 16] -> [B, macro_dim]

        x = x.reshape(-1, self.d, self.patch_L, self.d, self.patch_L)
        x = x.permute(0, 1, 3, 2, 4) # [B, 2, 2, 8, 8]
        x = x.flatten(0, 2).unsqueeze(1) # [B * 4, 1, 8, 8]
        z_hat = self.model(x).reshape(-1, self.d ** 2, self.closure_dim) # [B, 4, closure_dim]
        z_hat = torch.mean(z_hat, dim=1) # [B, closure_dim]
        z = torch.cat([z_macro, z_hat], -1) # [B, macro_dim + closure_dim]
        return z
    
    def encode(self, x0, x1, partial=False, index=None):
    
        # x: [B, 1, 32, 32, 32]

        if partial == False: 
            z0 = self.forward(x0) # [B, macro_dim + closure_dim]
            z1 = self.forward(x1) # [B, macro_dim + closure_dim]

            z0 = (z0 - self.min_val) / (self.max_val - self.min_val) # [B, macro_dim + closure_dim]
            z1 = (z1 - self.min_val) / (self.max_val - self.min_val) # [B, macro_dim + closure_dim]
            return z0, z1

        if partial == True:
            assert index is not None, "index must be provided when partial is True"
            assert index.shape[0] == x0.shape[0], "index must have the same batch size as x"

            z0 = self.forward(x0)

            x0 = x0.reshape(-1, self.d, self.patch_L, self.d, self.patch_L)
            x0 = x0.permute(0, 1, 3, 2, 4) # [B, 2, 2, 32, 32]
            x0 = x0.flatten(1, 2) # [B, 4, 32, 32]
            batch_indices = torch.arange(x0.shape[0], device=x0.device)

            x0 = x0[batch_indices, index].unsqueeze(1) # [B, 1, 32, 32] 

            z0_macro = self.cal_macro(x0) # [B-1, 1, 16, 16, 16] -> [B-1, macro_dim]
            z1_macro = self.cal_macro(x1) # [B-1, 1, 16, 16, 16] -> [B-1, macro_dim]
            z0_hat = self.model(x0) # [B-1, 1, 16, 16, 16] -> [B-1, closure_dim]

            z1_hat = self.model(x1) # [B-1, 1, 16, 16, 16] -> [B-1, closure_dim]
            z0_partial = torch.cat([z0_macro, z0_hat], -1) # [B-1, macro_dim + closure_dim]
            z1_partial = torch.cat([z1_macro, z1_hat], -1) # [B-1, macro_dim + closure_dim]

            z1_hat = z0 + (z1_partial - z0_partial) # [B-1, macro_dim + closure_dim]

            z0 = (z0 - self.min_val) / (self.max_val - self.min_val)
            z1_hat = (z1_hat - self.min_val) / (self.max_val - self.min_val) # [B-1, macro_dim + closure_dim]

            return z0, z1_hat

class EncoderIsing(Encoder):
    def __init__(self, closure_dim=2, macro_dim=6, L=16, patch_L=8, h=0.0):
        super().__init__(closure_dim, macro_dim, L, patch_L, h)

    def cal_macro(self, x):
        # magnetization 
        mag = torch.mean(x, dim=(1, 2, 3)) # [B]

        # # domain wall density
        shift_x = torch.roll(x, shifts=1, dims=2)
        shift_y = torch.roll(x, shifts=1, dims=3)

        dw_x = 0.5 * (1 - x * shift_x)
        dw_y = 0.5 * (1 - x * shift_y)

        total_dw = dw_x + dw_y
        rho_dw = total_dw.mean(dim=(1, 2, 3)) / 2
        z_macro = torch.stack([mag, rho_dw], dim=-1) # [B, 2]
        return z_macro
        # return mag.unsqueeze(-1)  # [B, 1] for magnetization only
    

class Decoder(nn.Module):
    def __init__(self, closure_dim=2, macro_dim=6, L=32, patch_L=16): # Added L
        super().__init__()
        assert L % patch_L == 0, "L must be divisible by patch_L"
        assert L >= 4, "L must be at least 4"
        assert patch_L >= 4, "patch_L must be at least 4"
        
        self.macro_dim = macro_dim
        self.closure_dim = closure_dim
        self.L = L
        self.patch_L = patch_L
        self.d = int(L / patch_L)

        # Calculate number of upsampling layers needed to reach L from initial 4x4
        if L == 48:
            num_upsamples = int(np.log2(L // 3))
            initial_size = (128, 3, 3)
        else:
            num_upsamples = int(np.log2(L // 4))
            initial_size = (128, 4, 4)

        layers = []
        
        # Initial linear layer and unflatten
        layers.append(nn.Linear(self.closure_dim + self.macro_dim, np.prod(initial_size)))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Unflatten(1, initial_size))  # [B, 128, 4, 4]
        
        # Build upsampling layers dynamically
        in_channels = 128
        current_size = 4
        
        for i in range(num_upsamples):
            out_channels = max(16, in_channels // 2) if i < num_upsamples - 1 else 1
            
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            current_size *= 2
            
            # Add batch norm and activation for all layers except the last one
            if i < num_upsamples - 1:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(0.2))
            else:
                # Final layer uses sigmoid activation
                layers.append(nn.Sigmoid())
            
            in_channels = out_channels
        
        self.model = nn.Sequential(*layers)

        # Ensure min/max are registered correctly
        self.apply(weights_init)

    def forward(self, z):
        # z: [B, macro_dim + closure_dim]
        # return shape: [B, 1, L, L]

        x = self.model(z) 
        x = x * 2 - 1
        return x

    
class Conv2DAutoencoder(nn.Module):
    def __init__(self, closure_dim=2, macro_dim=6, L=16, patch_L=8, h=0.0):
        super().__init__()
        self.closure_dim = closure_dim
        self.macro_dim = macro_dim
        
        self.encoder = EncoderIsing(closure_dim, macro_dim, L, patch_L, h)
        self.decoder = Decoder(closure_dim, macro_dim, L, patch_L)
    
    def forward(self, x):
        # x: [B, 1, 16, 16]
        z = self.encoder(x)
        x = self.decoder(z) 
        return x
    
    def encode(self, x0, x1, partial=False, index=None):
        z = self.encoder.encode(x0, x1, partial=partial, index=index)
        return z
        
    def cal_macro(self, x):
        z_macro = self.encoder.cal_macro(x)
        return z_macro




