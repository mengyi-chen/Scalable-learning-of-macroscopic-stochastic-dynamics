import torch 
import torch.nn
import numpy as np
import sys,os
import matplotlib.pyplot as plt 
import argparse
from datetime import datetime
import logging
import sys
from functools import partial
import torch.nn as nn
sys.path.append('..')
from utils.models import S_OnsagerNet, SDE_Net
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
parser = argparse.ArgumentParser(description='Stochastic OnsagerNet')
parser.add_argument('--gpu_idx', default=5, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--Task_NAME', default='NbMoTa_alloy', type=str)
parser.add_argument('--n_dim', default=6, type=int)
parser.add_argument('--embedding_dim', default=6, type=int)
parser.add_argument('--L', default=16, type=int)
parser.add_argument('--patch_L', default=8, type=int)
parser.add_argument('--N_atoms', default=8192, type=int)
parser.add_argument('--model_name', default='S_OnsagerNet', type=str)

# training parameters
parser.add_argument('--train_bs', default=1024, type=int)
parser.add_argument('--val_bs', default=1024, type=int)
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--patience', default=20, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--mode', default='arbitrary', type=str)
parser.add_argument('--epsilon', default=1e-4, type=float)
parser.add_argument('--dt', default=0.001, type=float)
parser.add_argument('--coeff', default=8, type=float)
args = parser.parse_args()

class Trainer():
    def __init__(self, args):
        set_seed(args.seed)
        
        self.plot_T_list = [400, 600, 700, 800, 1000, 1200, 1400, 1800, 2400, 3000]
        
        start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Initializing."
        print(start_message)
        date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        os.makedirs('../checkpoints', exist_ok=True)
        self.folder = os.path.join('../checkpoints',f'model_atoms_{args.N_atoms}_{args.model_name}_{date}')
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
        print('Using device:', self.device)
        self.d = int(args.L / args.patch_L)
        
        self.dt = torch.tensor(args.dt, device=self.device)
        self.load_data()

        if args.model_name == 'S_OnsagerNet':
            self.model = S_OnsagerNet(self.dt, mode=args.mode, n_dim=args.hidden_dim, epsilon=args.epsilon, embedding_dim=args.embedding_dim).to(self.device)
            # self.model = S_OnsagerNet(self.dt, mode=args.mode, n_dim=args.n_dim, epsilon=args.epsilon, embedding_dim=args.embedding_dim, forcing=False).to(self.device)
        elif args.model_name == 'SDE_Net':
            self.model = SDE_Net(self.dt, mode=args.mode, n_dim=args.n_dim, epsilon=args.epsilon, embedding_dim=args.embedding_dim).to(self.device)
        else:
            raise ValueError(f"Model {args.model_name} is not supported.")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=0.5,threshold_mode='rel',patience=args.patience,cooldown=0,min_lr=5e-6)

        self.d = int(args.L / args.patch_L)
        if args.coeff is not None:
            self.coeff = args.coeff
        else:
            self.coeff = self.d**3
        print('self.coeff:', self.coeff)

    
 
    def load_data(self):
      
        # ========= load train data =========
        self.z0_train = torch.load(f'../data/partial_sampling_atoms_{args.N_atoms}/z0_train.pt', map_location=self.device).to(torch.float32)
        self.z1_train_partial = torch.load(f'../data/partial_sampling_atoms_{args.N_atoms}/z1_train_partial.pt', map_location=self.device).to(torch.float32)
        self.time_step = torch.load(f'../data/partial_sampling_atoms_{args.N_atoms}/time_step.pt', map_location=self.device).unsqueeze(-1).to(torch.float32)
        self.train_T = torch.load(f'../data/partial_sampling_atoms_{args.N_atoms}/T.pt', map_location=self.device).to(torch.float32)

        Z0_train = self.z0_train.flatten(0, 1)
        T0_train = self.train_T.flatten(0, 1)
        Z1_train = self.z1_train_partial.flatten(0, 1)
        delta_t = self.time_step.flatten(0, 1)


        print('Z0_train shape:', Z0_train.shape)
        print('T0_train shape:', T0_train.shape)
        print('Z1_train shape:', Z1_train.shape)
        print('delta_t shape:', delta_t.shape)

        # ========= load val data =========
        self.plot_data = torch.load(f'../data/atoms_1024/macro_state.pt', map_location=self.device).to(torch.float32)
        self.plot_T = torch.load(f'../data/atoms_1024/T_state.pt', map_location=self.device).to(torch.float32)
        self.plot_time = torch.load(f'../data/atoms_1024/time_state_scaled.pt', map_location=self.device).unsqueeze(-1).to(torch.float32)

        # ========= dataloader =========
        dataset = TensorDataset(Z0_train, T0_train, Z1_train, delta_t)
        self.dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True)

        for T in self.plot_T_list:
            index = torch.unique(torch.where(self.plot_T == T)[0])
            fig = plt.figure(figsize=(48, 6))
            plot = self.plot_data[index].detach().cpu().numpy() # [n_tra, length_per_tra, n_dim]
            true_t = self.plot_time[index].detach().cpu().numpy() # [n_tra, length_per_tra]
            for idx in range(6):
                axes = fig.add_subplot(1, 8, idx+1)
                for i in range(true_t.shape[0]):
                    axes.plot(true_t[i], plot[i, :, idx])
                axes.set_xlabel("time (t)", fontsize=20)
                axes.set_ylim(-3.5, 3.5)
            plot_name = os.path.join(self.folder,f'true_{T}.png') 
            plt.savefig(plot_name)
            plt.clf()
            plt.close()
        
    def train(self):
        start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training started."
        print(start_message)
        self.logger.info(start_message)
        
        for epoch in range(args.num_epoch):

            self.model.train()
            train_mse = []
            for _, (X, T_X, Y, dt) in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                loss =  self.model.custom_loss(X, T_X, Y, dt, coeff=self.coeff)
                train_mse.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            loss_mean = sum(train_mse)/len(train_mse)
            last_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(loss_mean)

            message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training epoch {epoch+1}, training mse:{loss_mean}, lr:{last_lr}" 
            if epoch % 1 == 0:
                print(message)
                self.logger.info(message)
            
            if epoch % 10 == 0:
                self.model.eval()
                self.plot_tra(epoch)

            if epoch % 10 == 0:
                self.model.eval()

                model_path = os.path.join(self.folder, f'epoch_{epoch}.pt')
                torch.save(self.model, model_path) 
                
        model_path = os.path.join(self.folder,f'model.pt')
        torch.save(self.model,model_path)        
        self.plot_tra(args.num_epoch)  
        logging.shutdown()
    
    def plot_tra(self, epoch):
        # plot the predicted trajectories
        with torch.no_grad():
            for T in self.plot_T_list:
                index = torch.unique(torch.where(self.plot_T == T)[0])

                fig = plt.figure(figsize=(48, 6))
                if index is not None:
                    true_tra = self.plot_data[index].detach().cpu().numpy()
                    true_t = self.plot_time[index].detach().cpu().numpy()
                dt = 1 / 2000

                initial = torch.zeros((10, self.plot_data.shape[2]), device=self.device)
                T_initial = T * torch.ones((10, ), device=self.device)  

                predict_tra = self.model.predict(initial, T_initial, self.plot_data.shape[1], dt=torch.tensor(dt, device=self.device))
                predict_tra = predict_tra.detach().cpu().numpy()
                predict_tra_mean = np.mean(predict_tra, axis=0)
                
                t = np.arange(true_tra.shape[1]) * dt
                for idx in range(6):
                    axes = fig.add_subplot(1, 8, idx+1)
                    if index is not None: 
                        for i in range(true_t.shape[0]):
                            axes.plot(true_t[i], true_tra[i, :, idx], 'tab:blue', alpha=0.5)
                    for i in range(predict_tra.shape[0]):
                        axes.plot(t, predict_tra[i, :, idx])
                    axes.set_xlabel("time (t)", fontsize=20)
                    axes.plot(t, predict_tra_mean[:, idx], 'k--', linewidth=2)
                    axes.set_ylim(-3.5, 3.5)
                plot_name = os.path.join(self.folder, f'epoch_{epoch}_predict_{T}.png')
                plt.savefig(plot_name)
                plt.clf()
                plt.close()

if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()


    