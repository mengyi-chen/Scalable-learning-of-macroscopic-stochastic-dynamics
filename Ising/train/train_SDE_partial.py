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
from utils.onsagernet_pytorch import S_OnsagerNet, SDE_Net
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
parser.add_argument('--Task_NAME', default='ising', type=str)
parser.add_argument('--hidden_dim', default=4, type=int)
parser.add_argument('--L', default=64, type=int)
parser.add_argument('--box_L', default=64, type=int)
parser.add_argument('--h', default=0.1, type=float)
parser.add_argument('--T', default=2.5, type=float)
# training parameters
parser.add_argument('--train_bs', default=1024, type=int)
parser.add_argument('--val_bs', default=1024, type=int)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--patience', default=5, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--mode', default='arbitrary', type=str)
parser.add_argument('--epsilon', default=1e-4, type=float)
# parser.add_argument('--coeff', default=64, type=float)
parser.add_argument('--method', default='ours', type=str, choices=['ours', 'naive'])
parser.add_argument('--model_name', default='SDE_Net', type=str, choices=['SDE_Net', 'S_OnsagerNet'])
args = parser.parse_args()
    
class Trainer():
    def __init__(self, args):
        set_seed(args.seed)

        start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Initializing."
        print(start_message)
        date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.folder = os.path.join('../checkpoints',f'SDE_method_{args.method}_box_L_{args.box_L}_L_{args.L}_seed_{args.seed}_{date}')
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
        self.d = int(args.L / args.box_L)
        
        self.load_data()
        self.dt = torch.tensor(1, device=self.device)
        
        if args.model_name == 'S_OnsagerNet':
            self.model = S_OnsagerNet(self.dt, mode=args.mode, n_dim=args.hidden_dim, epsilon=args.epsilon).to(self.device)
        elif args.model_name == 'SDE_Net':
            self.model = SDE_Net(self.dt, mode=args.mode, n_dim=args.hidden_dim, epsilon=args.epsilon).to(self.device)
        else:
            raise ValueError(f"Model {args.model_name} is not supported.")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=0.5,threshold_mode='rel',patience=args.patience,cooldown=0,min_lr=5e-6)
        if args.method == 'ours':
            self.coeff = self.d**2
        elif args.method == 'naive':
            self.coeff = 1
        else:
            raise ValueError(f"Method {args.method} is not supported.")
        
        # self.coeff = args.coeff
        print('coeff:', self.coeff)

 
    def load_data(self):
        
        data_path = f'../raw_data_upscale/scaleup_box_L_{args.box_L}_L{args.L}_h{args.h}_T{args.T:.2f}'

        # ========= load train data =========
        if args.method == 'ours':
            
            self.z0_train = torch.load(f'{data_path}/z0_train.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]
            self.z1_train = torch.load(f'{data_path}/z1_train_partial.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]

        elif args.method == 'naive':
            
            self.z0_train = torch.load(f'{data_path}/z0_train_naive.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]
            self.z1_train = torch.load(f'{data_path}/z1_train_naive.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]

        self.train_dt = torch.load(os.path.join(data_path, 'time_step_train.pt'), map_location=self.device) # [n_tra, length_per_tra]

        z0_train = self.z0_train.flatten(0, 1)
        z1_train = self.z1_train.flatten(0, 1)
        train_dt = self.train_dt.flatten(0, 1).unsqueeze(-1)

        print('z0_train shape:', z0_train.shape)
        print('z1_train shape:', z1_train.shape)
        print('train_dt shape:', train_dt.shape)

        val_data_path = f'../raw_data/L{args.L}_MC500_h{args.h}_T{args.T:.2f}'
        self.z0_val = torch.load(f'{data_path}/z0_val.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]
        self.z1_val = torch.load(f'{data_path}/z1_val.pt', map_location=self.device) # [n_tra, length_per_tra, hidden_dim]  
        self.val_dt = torch.load(f'{val_data_path}/time_step_val.pt', map_location=self.device)

        z0_val = self.z0_val.flatten(0, 1)
        z1_val = self.z1_val.flatten(0, 1)
        val_dt = self.val_dt.flatten(0, 1).unsqueeze(-1)

        print('z0_val shape:', z0_val.shape)
        print('z1_val shape:', z1_val.shape)
        print('val_dt shape:', val_dt.shape)

        mean_val_dt = torch.mean(val_dt)
        print('mean of train_dt:', torch.mean(train_dt).item())
        print('mean of val_dt:', torch.mean(val_dt).item())

        # self.train_dt = train_dt / mean_train_dt
        # self.val_dt = val_dt / mean_train_dt
        train_dt = train_dt / mean_val_dt
        val_dt = val_dt / mean_val_dt
        self.mean_dt = torch.mean(val_dt)
        print('mean_dt:', self.mean_dt)

        # ========= dataloader =========
    
        dataset = TensorDataset(z0_train, z1_train, train_dt)
        self.dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True)

        dataset_val = TensorDataset(z0_val, z1_val, val_dt)
        self.dataloader_val = DataLoader(dataset_val, batch_size=args.val_bs, shuffle=True)

        # Plot the true trajectories
        fig = plt.figure(figsize=(8*4, 6))
        titles  = ["Magnetization", "Domain Wall Density", "closure variable 1", "closure variable 2"]
        plot = self.z0_val.detach().cpu().numpy() # [n_tra, length_per_tra, hidden_dim]
        t = np.arange(plot.shape[1])

        for idx in range(4):
            axes = fig.add_subplot(1, 4, idx + 1)
            for i in range(plot.shape[0]):
                axes.plot(t, plot[i, :, idx])

            axes.set_xlabel("time (t)", fontsize=20)
            axes.set_title(titles[idx], fontsize=20)
            if idx == 0:
                axes.set_ylim(-1, 1)
            else:
                axes.set_ylim(0, 1)
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
            for _, (X, Y, dt) in enumerate(self.dataloader_val):
                with torch.no_grad():
                    loss =  self.model.custom_loss(X,Y,dt=dt)
                    val_mse.append(loss.item())
            loss_mean_val = sum(val_mse) / len(val_mse)

            self.model.train()
            train_mse = []
            for _, (X, Y, dt) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                loss =  self.model.custom_loss(X, Y, dt=dt,coeff=self.coeff)

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
            
            if epoch > 0 and epoch % 25 == 0:
                self.plot_tra(epoch)

            # if epoch % 50 == 0:

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
            predict_tra = self.model.predict(initial, 500, dt=self.mean_dt)
            plot = predict_tra.detach().cpu().numpy()

            t = np.arange(predict_tra.shape[1]) * self.mean_dt.item()
            fig = plt.figure(figsize=(8*4, 6))
            titles  = ["Magnetization", "Domain Wall Density", "closure variable 1", "closure variable 2"]

            for idx in range(4):
                axes = fig.add_subplot(1, 4, idx + 1)
                for i in range(plot.shape[0]):
                    axes.plot(t, plot[i, :, idx])

                axes.set_xlabel("time (t)", fontsize=20)
                axes.set_title(titles[idx], fontsize=20)
                if idx == 0:
                    axes.set_ylim(-1, 1)
                else:
                    axes.set_ylim(0, 1)
            plot_name = os.path.join(self.folder,f'predict_{epoch}.png') 
            plt.savefig(plot_name)
            plt.close()

if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()


    