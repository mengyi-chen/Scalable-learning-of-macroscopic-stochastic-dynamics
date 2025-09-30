import torch
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import logging
import torch.nn as nn
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.models import Conv2DAutoencoder       
from utils.utils import set_seed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import yaml
from tqdm import tqdm
torch.set_default_dtype(torch.float32)
set_seed(0)

  
# Load parameters from YAML configuration file
with open('../config/config.yaml', 'r') as file:
    params = yaml.safe_load(file)

# General arguments
parser = argparse.ArgumentParser(description='Autoencoder Pretraining')
parser.add_argument('--gpu_idx', default=params['gpu_idx'], type=int, help='GPU index to use')
parser.add_argument('--patch_L', default=params['patch_L'], type=int, help='Patch size')
parser.add_argument('--L', default=64, type=int, help='Lattice size')
parser.add_argument('--h', default=params['h'], type=float, help='External field strength')
parser.add_argument('--T', default=params['T'], type=float, help='Temperature')
parser.add_argument('--steps', default=params['steps'], type=int, help='Number of steps')

parser.add_argument('--macro_dim', default=params['macro_dim'], type=int, help='Dimension of macro variable')
parser.add_argument('--closure_dim', default=params['closure_dim'], type=int, help='Dimension of closure variable')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
parser.add_argument('--num_epoch', default=10, type=int, help='Number of training epochs')
parser.add_argument('--train_bs', default=512, type=int, help='Training batch size')
parser.add_argument('--lr', default=2e-4, type=float, help='Learning rate')
args = parser.parse_args()

class Trainer():
    def __init__(self, args):

        set_seed(args.seed)
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_idx}") if torch.cuda.is_available() else torch.device('cpu')
        self.d = int(args.L / args.patch_L)

        self.data_path_val = f'../raw_data/L{args.L}_MC{args.steps}_h{args.h}_T{args.T:.2f}'
        self.save_path = f'../raw_data_upsample/scaleup_patch_L_{args.patch_L}_L{args.L}_h{args.h}_T{args.T:.2f}'

        start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Initializing."
        print(start_message)

        date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.folder = os.path.join(f'../checkpoints',f'AE_num_epoch_{args.num_epoch}_patch_L_{args.patch_L}_L_{args.L}_{date}')
        if not os.path.exists(self.folder):
           os.makedirs(self.folder, exist_ok=True) 

        # ========== log ==============
        self.log_path = os.path.join(self.folder,'train.log')
        logging.basicConfig(filename=self.log_path,
                    filemode='a',
                    format='%(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
        logging.info("Training log")
        self.logger = logging.getLogger('')
        self.logger.info(start_message)
        for arg in vars(args):
            self.logger.info(f'{arg}:{getattr(args,arg)}')
            print(f'{arg}:{getattr(args,arg)}')
                    
        # ========== initialize ==============
        self.load_data()
        self.model = Conv2DAutoencoder(args.closure_dim, args.macro_dim, args.L, args.patch_L).to(self.device)
            
        print('*********model structure**********')
        print(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0)
        self.metric = nn.BCELoss()
        
    def load_data(self):
        
        # load data 

        self.X0_train = torch.load(os.path.join(self.save_path, 'X0_train.pt'), map_location=self.device).unsqueeze(2) # [n_tra, length_per_tra, 1, L, L]
        self.X1_train_partial = torch.load(os.path.join(self.save_path, f'X1_train_partial.pt'), map_location=self.device).unsqueeze(2) # [n_tra, length_per_tra, 1, L, L]
        self.idx_train_partial = torch.load(os.path.join(self.save_path, f'idx_train_partial.pt'), map_location=self.device).to(torch.int64) # [n_tra, length_per_tra]

        self.X0_val = torch.load(os.path.join(self.data_path_val, 'X0_val.pt'), map_location=self.device).unsqueeze(2) # [n_tra, length_per_tra, 1, L, L]
        self.X1_val = torch.load(os.path.join(self.data_path_val, 'X1_val.pt'), map_location=self.device).unsqueeze(2) # [n_tra, length_per_tra, 1, L, L]


        print('X0_train shape:', self.X0_train.shape)
        print('X1_train_partial shape:', self.X1_train_partial.shape)
        print('idx_train_partial shape:', self.idx_train_partial.shape)
        print('X0_val shape:', self.X0_val.shape)
        print('X1_val shape:', self.X1_val.shape)

        dataset = TensorDataset(self.X0_train.flatten(0, 1))
        self.dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True, drop_last=True)

        dataset_val = TensorDataset(self.X0_val.flatten(0, 1))
        self.dataloader_val = DataLoader(dataset_val, batch_size=args.train_bs, shuffle=False, drop_last=True)

        # save one of the true spin configuration for visualization
        self.plot = self.X0_train[0, 0].unsqueeze(0).to(self.device).to(torch.float32) # [1, 1, 16, 16]
        fig = plt.figure(figsize=(8, 6))
        axes = fig.add_subplot(1, 1, 1)
        plot = self.plot.detach().cpu().numpy()
        plt.imshow(plot[0, 0])
        plt.colorbar()
        axes.axis('off')
        plt.axis('tight')
        plot_name = os.path.join(self.folder,f'true.png') 
        plt.savefig(plot_name)
        plt.close()


    def train(self):
        # ========== training ==============
        start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training started."
        print(start_message)
        self.logger.info(start_message)

        for num_epoch in range(args.num_epoch):

            # validation
            self.model.eval()
            val_mse = []
            with torch.no_grad():
                for _, (X, ) in enumerate(self.dataloader_val):
                    X = X.to(self.device).to(torch.float32)
                    X_pred = self.model(X)
                    loss = self.metric((X_pred + 1) / 2, (X + 1) / 2)
                    val_mse.append(loss.item())
            loss_mean_val = sum(val_mse) / len(val_mse)

            # training 
            self.model.train()
            train_mse = []
            for _, (X, ) in enumerate(self.dataloader):
                X = X.to(self.device).to(torch.float32)
                self.optimizer.zero_grad()
                X_pred = self.model(X)
                loss = self.metric((X_pred + 1) / 2, (X + 1) / 2)
                train_mse.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            loss_mean = sum(train_mse)/len(train_mse)

            message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training num_epoch {num_epoch+1}, training mse:{loss_mean}, val mse:{loss_mean_val}" 
            if num_epoch % 1 == 0:
                print(message)
                self.logger.info(message)
                
                # save one of the recovered configuration for comparison 
                fig = plt.figure(figsize=(8, 6))
                axes = fig.add_subplot(1, 1, 1)
                X_pred = self.model(self.plot) 
                X_pred = X_pred.detach().cpu().numpy()
                plt.imshow(X_pred[0, 0])
                axes.axis('off')
                plt.axis('tight')
                plt.colorbar()
                plot_name = os.path.join(self.folder,f'predict_{num_epoch}.png') 
                plt.savefig(plot_name)
                plt.close()


            if num_epoch % 1 == 0:
                self.model.eval()
                model_path = os.path.join(self.folder, f'num_epoch_{num_epoch}.pt')
                torch.save(self.model, model_path)

        self.model.eval()
        model_path = os.path.join(self.folder, f'num_epoch_{args.num_epoch}.pt')
        torch.save(self.model, model_path)
        logging.shutdown()
    
    def process(self):
        # process after training 
        self.model.eval()
        with torch.no_grad():

            # ========== save latent ==============
            z0_train = []
            z1_train_partial = []

            z0_train_naive = []
            z1_train_naive = []
            for i in tqdm(range(self.X0_train.shape[0])):
                X0 = self.X0_train[i].to(self.device).to(torch.float32)
                X1_partial = self.X1_train_partial[i].to(self.device).to(torch.float32)
                idx_partial = self.idx_train_partial[i].to(self.device)

                z0, z1_partial, z0_naive, z1_naive = self.model.encode_pairs(X0, X1_partial, partial=True, index=idx_partial)

                z0_train.append(z0)
                z1_train_partial.append(z1_partial)

                z0_train_naive.append(z0_naive)
                z1_train_naive.append(z1_naive)

            z0_train = torch.stack(z0_train)
            z1_train_partial = torch.stack(z1_train_partial)

            z0_train_naive = torch.stack(z0_train_naive)
            z1_train_naive = torch.stack(z1_train_naive)

            print('z0_train shape:', z0_train.shape) # [n_tra, length_per_tra, macro_dim + closure_dim]
            print('z1_train_partial shape:', z1_train_partial.shape) # [n_tra, length_per_tra, macro_dim + closure_dim]

            print('z0_train_naive shape:', z0_train_naive.shape) # [n_tra, length_per_tra, macro_dim + closure_dim]
            print('z1_train_naive shape:', z1_train_naive.shape) # [n_tra, length_per_tra, macro_dim + closure_dim]

            # ========== val latent ==============
    
            z0_val = []
            z1_val = []
            for i in range(self.X0_val.shape[0]):
                X0 = self.X0_val[i].to(self.device).to(torch.float32)
                X1 = self.X1_val[i].to(self.device).to(torch.float32)

                z0, z1 = self.model.encode_pairs(X0, X1, partial=False)

                z0_val.append(z0)
                z1_val.append(z1)
            z0_val = torch.stack(z0_val)
            z1_val = torch.stack(z1_val)

            print('z0_val shape:', z0_val.shape) # [n_tra, length_per_tra, macro_dim + closure_dim]
            print('z1_val shape:', z1_val.shape) # [n_tra, length_per_tra, macro_dim + closure_dim]

            # ========== normalization ==============
            min_val, max_val = torch.amin(z0_val[..., 1:], dim=(0, 1)), torch.amax(z0_val[..., 1:], dim=(0, 1))
            min_val = torch.cat([torch.tensor([0.], device=min_val.device), min_val])
            max_val = torch.cat([torch.tensor([1.], device=max_val.device), max_val])

            print('min_val shape:', min_val.shape)
            print('max_val shape:', max_val.shape)
            print('min_val:', min_val)
            print('max_val:', max_val)

            self.model.encoder.min_val.copy_(min_val)
            self.model.encoder.max_val.copy_(max_val)
            torch.save(self.model, os.path.join(self.folder, 'model.pt'))

            z0_train = (z0_train - min_val) / (max_val - min_val)
            z1_train_partial = (z1_train_partial - min_val) / (max_val - min_val)

            z0_train_naive = (z0_train_naive - min_val) / (max_val - min_val)
            z1_train_naive = (z1_train_naive - min_val) / (max_val - min_val)

            z0_val = (z0_val - min_val) / (max_val - min_val)
            z1_val = (z1_val - min_val) / (max_val - min_val)

            torch.save(z0_val, os.path.join(self.save_path, 'z0_val.pt'))
            torch.save(z1_val, os.path.join(self.save_path, f'z1_val.pt'))
            self.plot_tra(z0_val, os.path.join(self.save_path, 'z0_val.png'))

            torch.save(z0_train_naive, os.path.join(self.save_path, 'z0_train_naive.pt'))
            torch.save(z1_train_naive, os.path.join(self.save_path, 'z1_train_naive.pt'))
            self.plot_tra(z0_train_naive, os.path.join(self.save_path, 'z0_train_naive.png'))

            torch.save(z0_train, os.path.join(self.save_path, 'z0_train.pt'))
            torch.save(z1_train_partial, os.path.join(self.save_path, f'z1_train_partial.pt'))
            self.plot_tra(z0_train, os.path.join(self.save_path, 'z0_train.png'))



    def plot_tra(self, latent, save_path):
        
        titles  = ["Magnetization", "Domain Wall Density", "closure variable 1", "closure variable 2"]

        fig = plt.figure(figsize=(8 * 4, 6))
        plot = latent.detach().cpu().numpy() # [n_tra, length_per_tra, closure_dim]
        t = np.arange(plot.shape[1])
        for idx in range(4):
            axes = fig.add_subplot(1, 4, idx + 1)
            for i in range(plot.shape[0]):
                axes.plot(t, plot[i, :, idx])
                # break
            axes.set_title(titles[idx], fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(save_path)
        plt.close()



if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()
    trainer.process()
    print('*********Data generation done*********')