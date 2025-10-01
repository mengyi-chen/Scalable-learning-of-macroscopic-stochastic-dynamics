import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, sys
sys.path.append('..')
import argparse
import matplotlib.pyplot as plt 
from datetime import datetime
import torch.utils.data as data
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from utils.utils import set_seed
from utils.models import Autoencoder
import yaml
torch.set_default_dtype(torch.float32)
set_seed(42)

with open('../config/config.yaml', 'r') as file:
    params = yaml.safe_load(file)

# General arguments
parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--macro_dim', default=params['macro_dim'], type=int, help='Dimension of macroscopic variables')
parser.add_argument('--closure_dim', default=params['closure_dim'], type=int, help='Dimension of closure variables')
parser.add_argument('--gpu_idx', default=params['gpu_idx'], type=int, help='GPU index')
parser.add_argument('--input_dim', default=200, type=int, help='Input dimension')
parser.add_argument('--train_bs', default=256, type=int, help='Batch size for training')
parser.add_argument('--num_epoch', default=10, type=int, help='Number of training epochs')
parser.add_argument('--n_patch', default=5, type=int, help='Number of parts to divide the grid into')
parser.add_argument('--ckpt_path', default='../checkpoints', type=str, help='Path to save checkpoints')
args = parser.parse_args()


if __name__ == "__main__":
    
    device = torch.device(f'cuda:{args.gpu_idx}')
    os.makedirs(args.ckpt_path, exist_ok=True)

    # ============ Load data ==============
    x0_train = torch.load('../raw_data_upsample/X0_train_grid_200_upscaled.pt', weights_only=True, map_location=device) # x_t: 
    idx_train_partial = torch.load('../raw_data_upsample/idx_train_partial.pt', map_location=device) # index of partial labels $\mathcal{I}$
    x1_train_partial = torch.load('../raw_data_upsample/X1_train_partial.pt', weights_only=True, map_location=device) # x_{t+dt, I}

    print('x0 shape:', x0_train.shape)
    print('x1_partial shape:', x1_train_partial.shape)
    print('idx_train_partial shape:', idx_train_partial.shape)

    x0_val = torch.load('../raw_data/X0_val_grid_200.pt', weights_only=True, map_location=device)
    x1_val = torch.load('../raw_data/X1_val_grid_200.pt', weights_only=True, map_location=device)
    print('x0_val shape:', x0_val.shape)
    print('x1_val shape:', x1_val.shape) 

    date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    
    # Macroscopic dimension = the dimension of macroscopic variables + the dimension of closuere variables
    folder = os.path.join(args.ckpt_path,f'AE_dim_{args.closure_dim+args.macro_dim}_{date}')
    if not os.path.exists(folder):
        os.mkdir(folder) 

    # ============ Begin training the autoencoder ==============
    AE = Autoencoder(args.input_dim, args.closure_dim, args.n_patch).to(device)
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.95))
    metric = nn.MSELoss()

    dataset = data.TensorDataset(x0_train.flatten(0, 1))
    dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True)

    dataset_val = data.TensorDataset(x0_val.flatten(0, 1))
    dataloader_val = DataLoader(dataset_val, batch_size=args.train_bs, shuffle=False)

    for epoch in range(args.num_epoch):
        AE.eval()
        val_mse = []
        with torch.no_grad():
            for _, (batch_X, ) in enumerate(dataloader_val):
                pred = AE(batch_X)
                loss = metric(batch_X, pred)
                val_mse.append(loss.item())
        loss_mean_val = sum(val_mse) / len(val_mse)

        # training 
        AE.train()
        train_mse = []
        for step, (batch_X, ) in enumerate(dataloader):
            # train
            optimizer.zero_grad()

            # reconstruction loss
            pred = AE(batch_X)
            loss = metric(batch_X, pred)

            train_mse.append(loss.item())
          
            loss.backward()
            optimizer.step()

        loss_mean = sum(train_mse) / len(train_mse)
        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training epoch {epoch+1}, training mse:{loss_mean}, val mse:{loss_mean_val}" 
        if epoch % 1 == 0:
            print(message)

        if epoch % 10 ==0:
            torch.save(AE,os.path.join(folder,f'{epoch}.pt'))

    # ============ save partial data ==============
    z0_train = []
    z1_train_partial = []

    z0_train_naive = []
    z1_train_naive = []
    
    for i in tqdm(range(x0_train.shape[0])):
        x0 = x0_train[i] # [B, 2, 100]
        x1_partial = x1_train_partial[i] # [B, 2, 10]
        idx_partial = idx_train_partial[i] # [B]

        # data for our method 
        z0, z1_hat, z0_naive, z1_naive = AE.encode_pairs(x0, x1_partial, partial=True, index=idx_partial)
        z0_train.append(z0)
        z1_train_partial.append(z1_hat)

        # data for the baseline method (naive)
        z0_train_naive.append(z0_naive)
        z1_train_naive.append(z1_naive)
    
    z0_train = torch.stack(z0_train, 0)
    z1_train_partial = torch.stack(z1_train_partial, 0)
    z0_train_naive = torch.stack(z0_train_naive, 0)
    z1_train_naive = torch.stack(z1_train_naive, 0)

    mean, std = torch.mean(z0_train, dim=(0, 1)), torch.std(z0_train, dim=(0, 1))
    AE.mean.copy_(mean)
    AE.std.copy_(std)
    torch.save(AE,os.path.join(folder,'model.pt'))
    print('mean:', mean)
    print('std:', std)

    # Save the model with updated normalization parameters
    torch.save(AE,os.path.join(folder,'model.pt'))

    # Apply normalization to the data
    z0_train = (z0_train - mean) / std
    z1_train_partial = (z1_train_partial - mean) / std
    z0_train_naive = (z0_train_naive - mean) / std
    z1_train_naive = (z1_train_naive - mean) / std

    print('z0_train shape:', z0_train.shape)
    print('z1_train_partial shape:', z1_train_partial.shape)
    print('z0_train_naive shape:', z0_train_naive.shape)
    print('z1_train_naive shape:', z1_train_naive.shape)

    if not os.path.exists('../data'):
        os.mkdir('../data')
    torch.save(z0_train, '../data/z0_train.pt')
    torch.save(z1_train_partial, '../data/z1_train_partial.pt')
    torch.save(z0_train_naive, '../data/z0_train_naive.pt')
    torch.save(z1_train_naive, '../data/z1_train_naive.pt')

    # ========== val latent ==============
    z0_val = []
    z1_val = []
    for i in range(x0_val.shape[0]):
        x0 = x0_val[i]
        x1 = x1_val[i]

        z0, z1 = AE.encode_pairs(x0, x1, partial=False)
        z0_val.append(z0)
        z1_val.append(z1)
    z0_val = torch.stack(z0_val, 0)  
    z1_val = torch.stack(z1_val, 0)

    print('z0_val shape:', z0_val.shape)
    print('z1_val shape:', z1_val.shape)

    torch.save(z0_val, '../data/z0_val.pt')
    torch.save(z1_val, '../data/z1_val.pt')

    # ========== visualize ==============
    z0_val = z0_val.detach().cpu().numpy()
    fig = plt.figure(figsize=(32, 6))
    for i in range(4):
        axes = fig.add_subplot(1, 4, i+1)
        for j in range(10):
            axes.plot(z0_val[j, :, i])
    
    plt.savefig(os.path.join(folder,'z0_val.png'))

