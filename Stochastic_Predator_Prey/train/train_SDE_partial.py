import torch 
import torch.nn
import numpy as np
import sys,os
import matplotlib.pyplot as plt 
import argparse
from datetime import datetime
import logging
import sys
import torch.nn as nn
sys.path.append('..')
from utils.models import SDE_Net
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from utils.utils import set_seed
import warnings
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)


# General arguments
parser = argparse.ArgumentParser(description='Identify macroscopic dynamics')
parser.add_argument('--gpu_idx', default=5, type=int, help='GPU index')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
parser.add_argument('--hidden_dim', default=4, type=int, help='latent dimension')
parser.add_argument('--L', default=200, type=int, help='number of grid points for the large system')
parser.add_argument('--box_L', default=40, type=int, help='box size for local averaging')
parser.add_argument('--dt', default=0.1, type=float, help='time step')
parser.add_argument('--n_patch', default=5, type=int, help='Number of parts to divide the grid into')
# training parameters
parser.add_argument('--train_bs', default=1024, type=int, help='batch size for training')
parser.add_argument('--val_bs', default=1024, type=int, help='batch size for validation')
parser.add_argument('--num_epoch', default=100, type=int, help='number of epochs')
parser.add_argument('--patience', default=5, type=int, help='patience for lr scheduler')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--mode', default='arbitrary', type=str, choices=['arbitrary', 'diagonal', 'constant_diagonal', 'constant'], help='structure of diffusion term')
parser.add_argument('--epsilon', default=1e-5, type=float, help='small constant for numerical stability')
parser.add_argument('--coeff', default=None, type=float, help='coefficient for the diffusion term')
parser.add_argument('--method', default='ours', type=str, choices=['ours', 'naive'], help='method for training')
args = parser.parse_args()
    
class Trainer():
    def __init__(self, args):
        set_seed(args.seed)

        start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Initializing."
        print(start_message)
        date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.d = args.n_patch
        if args.coeff is not None:
            self.coeff = args.coeff
        else:
            if args.method == 'ours':
                self.coeff = self.d
            elif args.method == 'naive':
                self.coeff = 1
            else:
                raise ValueError(f"Method {args.method} is not supported.")
        
        self.folder = os.path.join('../checkpoints',f'SDE_method_{args.method}_coeff_{self.coeff}_seed_{args.seed}')
        if not os.path.exists(self.folder):
            os.mkdir(self.folder) 
                        
        self.log_path = os.path.join(self.folder,'train.log')
        logging.basicConfig(filename=self.log_path,
                    filemode='a',
                    format='%(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
        logging.info("Training log for DNA")
        self.logger = logging.getLogger('')
        self.logger.info(start_message)
        for arg in vars(args):
            self.logger.info(f'{arg}:{getattr(args,arg)}')
            print(f'{arg}:{getattr(args,arg)}')
            
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_idx}") if torch.cuda.is_available() else torch.device('cpu')
        
        self.dt = torch.tensor(args.dt, device=self.device)
        self.load_data()

        self.model = SDE_Net(self.dt, mode=args.mode, n_dim=args.hidden_dim, epsilon=args.epsilon).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=0.5,threshold_mode='rel',patience=args.patience,cooldown=0,min_lr=5e-6)
        
        print('coeff:', self.coeff)

 
    def load_data(self):
        

        # ========= load train data =========
        if args.method == 'ours':
            
            self.z0_train = torch.load('../data/z0_train.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]
            self.z1_train = torch.load('../data/z1_train_partial.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]

        elif args.method == 'naive':

            self.z0_train = torch.load('../data/z0_train_naive.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]
            self.z1_train = torch.load('../data/z1_train_naive.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]

     
        z0_train = self.z0_train.flatten(0, 1)
        z1_train = self.z1_train.flatten(0, 1)

        print('z0_train shape:', z0_train.shape)
        print('z1_train shape:', z1_train.shape)

        self.z0_val = torch.load(f'../data/z0_val.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]
        self.z1_val = torch.load(f'../data/z1_val.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]  

        z0_val = self.z0_val.flatten(0, 1)
        z1_val = self.z1_val.flatten(0, 1)

        print('z0_val shape:', z0_val.shape)
        print('z1_val shape:', z1_val.shape)

        # ========= dataloader =========
    
        dataset = TensorDataset(z0_train, z1_train) 
        self.dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True)

        dataset_val = TensorDataset(z0_val, z1_val)
        self.dataloader_val = DataLoader(dataset_val, batch_size=args.val_bs, shuffle=True)

        # ========= visualize =========
        fig = plt.figure(figsize=(8*3, 6))
        plot = self.z0_val.detach().cpu().numpy() # [n_tra, length_per_tra, hidden_dim]
        t = np.arange(plot.shape[1]) * self.dt.item()

        axes = fig.add_subplot(1, 3, 1)
        for i in range(10):
            axes.plot(t, plot[i, :, 0])
            
        axes = fig.add_subplot(1, 3, 2)
        for i in range(10):
            axes.plot(t, plot[i, :, 1])

        axes = fig.add_subplot(1, 3, 3)
        for i in range(10):
            axes.plot(plot[i, :, 0], plot[i, :, 1])
        plot_name = os.path.join(self.folder,f'true.png') 
        plt.savefig(plot_name)
        plt.close()
        
    def train(self):
        start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training started."
        print(start_message)
        self.logger.info(start_message)
        
        for epoch in range(args.num_epoch):
            val_mse = []
            self.model.eval()
            for _, (X, Y) in enumerate(self.dataloader_val):
                with torch.no_grad():
                    loss =  self.model.custom_loss(X,Y)
                    val_mse.append(loss.item())
            loss_mean_val = sum(val_mse) / len(val_mse)

            self.model.train()
            train_mse = []
            for _, (X, Y) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                loss =  self.model.custom_loss(X, Y, coeff=self.coeff)

                train_mse.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            loss_mean = sum(train_mse)/len(train_mse)
            last_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(loss_mean)

            message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training epoch {epoch+1}, training mse:{loss_mean}, val mse:{loss_mean_val}, lr:{last_lr}" 
            if epoch % 1 == 0:
                print(message)
                self.logger.info(message)
            
            if epoch > 0 and epoch % 10 == 0:
                self.plot_tra(epoch)

                model_path = os.path.join(self.folder, f'epoch_{epoch}.pt')
                torch.save(self.model, model_path) 
                
       
        model_path = os.path.join(self.folder,f'model.pt')
        torch.save(self.model,model_path)        
        self.plot_tra(args.num_epoch)  
        logging.shutdown()
    
    def plot_tra(self, epoch):
        # plot the predicted trajectories
        with torch.no_grad():
  
            initial = self.z0_val[:, 0]
            predict_tra = self.model.predict(initial, self.z0_val.shape[1])
            plot = predict_tra.detach().cpu().numpy()

            t = np.arange(predict_tra.shape[1]) * self.dt.item()
            fig = plt.figure(figsize=(8*3, 6))
            axes = fig.add_subplot(1, 3, 1)
            for i in range(10):
                axes.plot(t, plot[i, :, 0])
                
            axes = fig.add_subplot(1, 3, 2)
            for i in range(10):
                axes.plot(t, plot[i, :, 1])

            axes = fig.add_subplot(1, 3, 3)
            for i in range(10):
                axes.plot(plot[i, :, 0], plot[i, :, 1])

            plot_name = os.path.join(self.folder,f'predict_{epoch}.png') 
            plt.savefig(plot_name)
            plt.close()

if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()


    