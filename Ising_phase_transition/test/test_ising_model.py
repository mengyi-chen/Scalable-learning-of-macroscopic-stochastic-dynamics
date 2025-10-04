import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os, sys
sys.path.append('../')
import seaborn as sns
import warnings
from utils.utils import set_seed, cal_binder_cumulant, cal_mag_susceptibility
import pickle
import argparse

warnings.filterwarnings("ignore")
set_seed(42)


parser = argparse.ArgumentParser(description='Process Ising model predictions for different temperatures')
parser.add_argument('--L', type=int, default=64, help='Lattice size L for the Ising model (default: 64)')
parser.add_argument('--patch_L', type=int, default=16, help='Patch size for the model (default: 16)')
parser.add_argument('--length', type=int, default=32000, help='Length of the simulation (default: 32000)')
parser.add_argument('--gpu_idx', type=int, default=6, help='GPU index to use (default: 6)')
args = parser.parse_args()

def main():
    
    L = args.L
    length = args.length
    print(f"Processing with lattice size L = {L}")

    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    M_predict = {}
    T_list = [2.25, 2.26, 2.27, 2.28, 2.29]
    # T_list = [2.26]
    for T in T_list:
        M_predict[f'T{T}'] = []
    
    T_unique = []
    M_mean = []
    X_mean = []
    U_mean = []
    
    with torch.no_grad():
        for T in T_list:
            print(f'Processing T={T}...')
            folder = f'../checkpoints_T_{T}'
            
            if not os.path.exists(folder):
                print(f"Warning: Checkpoint folder {folder} does not exist! Skipping...")
                continue
            
            paths = os.listdir(folder)
            ckpt_paths = [os.path.join(folder, path) for path in paths if f'L_{L}' in path and 'SDE' in path]

            ckpt_paths = [os.path.join(path, 'model.pt') for path in ckpt_paths]
            
            M_list = []
            X_list = []
            U_list = []
            for path in ckpt_paths: 
                model = torch.load(path, map_location=device)
                z0_val = torch.load(f'../data/patch_L_16_L_{L}_T_{T}/z0_val.pt', map_location=device)
                z0_train = torch.load(f'../data/patch_L_16_L_{L}_T_{T}/z0_train.pt', map_location=device)
                train_dt = torch.load(f'../raw_data_upsample/scaleup_patch_L_{args.patch_L}_L{L}_h0.0_T{T}/time_step_train.pt', map_location=device)

                val_dt = torch.load(f'../raw_data/L{L}_MC32000_h0.0_T{T}/time_step_val.pt', map_location=device)
                mean_train_dt = torch.mean(train_dt)
                val_dt = val_dt / mean_train_dt
                mean_dt = torch.mean(val_dt)

                initial = z0_train[:, 0].repeat(2, 1)
                predict = model.predict(initial, length, dt=mean_dt)

                nan_mask = torch.isnan(predict).any(dim=(1, 2))
                if nan_mask.any():
                    print(f'Found {nan_mask.sum().item()} trajectories with NaN values, removing them...')
                    predict = predict[~nan_mask]
                    print(f'Remaining trajectories: {predict.shape[0]}')
                
                magnetization = predict[:, (length // 2):, 0].detach().cpu().numpy()
                magnetization = np.abs(magnetization)
                magnetization = np.clip(magnetization, 0, 1)
                M_predict[f'T{T}'].append(magnetization)

                M = np.mean(magnetization)
                X = cal_mag_susceptibility(magnetization, L, T)
                binder_cumulant = cal_binder_cumulant(magnetization)

                M_list.append(M)
                X_list.append(X)
                U_list.append(binder_cumulant)

            T_unique.append(T)
            M_mean.append(np.mean(M_list))
            X_mean.append(np.mean(X_list))
            U_mean.append(np.mean(U_list))

            print(f'Temperature: {T}, M: {np.mean(M_list)}, X: {np.mean(X_list)}, Binder Cumulant: {np.mean(U_list)}')


    macro_dict = {
        'T': np.array(T_unique),
        'M': np.array(M_mean),
        'X': np.array(X_mean),
        'U': np.array(U_mean)
    }
    # Save results
    output_prefix = f'L_{L}'
    os.makedirs('./plot_data', exist_ok=True)
    with open(f'./plot_data/M_predict_{output_prefix}.pkl', 'wb') as f:
        pickle.dump(M_predict, f)
    with open(f'./plot_data/macro_{output_prefix}.pkl', 'wb') as f:
        pickle.dump(macro_dict, f)
    
    print(f"Results saved to M_predict_{output_prefix}.pkl")

if __name__ == "__main__":
    main()