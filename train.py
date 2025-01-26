import torch
import torch.nn as nn
from model import Conditional3DVAE
from dataset import BrainDataset
import pdb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
import glob
from sklearn.decomposition import PCA
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import chain
from scipy import stats
import argparse

class AgeLoss(nn.Module):
    def __init__(self):
        super(AgeLoss, self).__init__()
        
    def forward(self, pred, uncertainty, target):
        """
        Custom loss function that combines MSE with uncertainty prediction
        Args:
            pred: predicted age
            uncertainty: predicted uncertainty
            target: true age
        """
        loss = 0.5 * torch.exp(-uncertainty) * (pred - target)**2 + 0.5 * uncertainty
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, kl_weight=0.1, age_weight=1.0, recon_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.kl_weight = kl_weight
        self.age_weight = age_weight
        self.recon_weight = recon_weight
        self.age_criterion = AgeLoss()
        
    def forward(self, recon_x, x, z_mean, z_log_var, pred_age, uncertainty, true_age):
        # Reconstruction loss (L1 loss)
        recon_loss = F.l1_loss(recon_x, x, reduction='mean') * self.recon_weight
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) * self.kl_weight
        
        # Age prediction loss with uncertainty
        age_loss = self.age_criterion(pred_age, uncertainty, true_age.float()) * self.age_weight
        
        total_loss = recon_loss + kl_loss + age_loss
        return total_loss, recon_loss, kl_loss, age_loss
    
def plotlatent(latent_vectors, ages, save_path=None):
    pca = PCA(n_components = 2)
    latent_2d = pca.fit_transform(latent_vectors)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=ages, cmap='jet',
                         alpha=0.6, s=15)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Age', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_predictions(type, true_ages, predicted_ages, uncertainties, epoch, save_dir='plots'):
    """Plot and save age predictions with uncertainty"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    
    # Scatter plot with error bars
    plt.errorbar(true_ages, predicted_ages, 
                yerr=2*np.sqrt(uncertainties),  # 2 standard deviations
                fmt='o', alpha=0.3, elinewidth=0.5)
    
    # Plot perfect prediction line
    min_age = min(min(true_ages), min(predicted_ages))
    max_age = max(max(true_ages), max(predicted_ages))
    plt.plot([min_age, max_age], [min_age, max_age], 'r--', label='Perfect prediction')
    
    # Calculate metrics
    mae = np.mean(np.abs(true_ages - predicted_ages))
    correlation = np.corrcoef(true_ages, predicted_ages)[0, 1]
    
    plt.title(f'Age Prediction (Epoch {epoch})\nMAE: {mae:.2f}, Correlation: {correlation:.2f}')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.grid(True)
    
    plt.savefig(f'{save_dir}/age_prediction_epoch_{epoch}_{type}.png')
    plt.close()
    
    return mae, correlation

def train_model(config, model, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
    criterion = CombinedLoss(kl_weight=0.1, age_weight=1.0, recon_weight=1.0)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    r_squared_best = 0
    
    for epoch in range(config["num_epochs"]):
        # Training Phase
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0, 'age': 0}
        train_true_ages = []
        train_pred_ages = []
        train_uncertainties = []
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{config["num_epochs"]}')
        for batch in pbar:
            x = batch['image'].to(device)
            age = batch['age'].to(device)
            
            # Forward pass
            recon_x, z_mean, z_log_var, pred_age, uncertainty, latent = model(x)
            
            # Calculate losses
            loss, recon_loss, kl_loss, age_loss = criterion(
                recon_x, x, z_mean, z_log_var, pred_age, uncertainty, age
            )
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Store losses and predictions
            epoch_losses['total'] += loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['age'] += age_loss.item()
            
            train_true_ages.extend(age.cpu().numpy())
            train_pred_ages.extend(pred_age.detach().cpu().numpy())
            train_uncertainties.extend(uncertainty.detach().cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'age': f'{age_loss.item():.4f}'
            })
        
        # Plot training predictions
        plot_predictions(config['type'],
            np.array(train_true_ages).flatten(),
            np.array(train_pred_ages).flatten(),
            np.array(train_uncertainties).flatten(),
            epoch,
            save_dir='plots/train'
        )
        
        # Validation Phase
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0, 'age': 0}
        val_true_ages = []
        val_pred_ages = []
        val_uncertainties = []
        latents = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Validation Epoch {epoch+1}/{config["num_epochs"]}')
            for batch in pbar:
                x = batch['image'].to(device)
                age = batch['age'].to(device)
                
                # Forward pass
                recon_x, z_mean, z_log_var, pred_age, uncertainty, latent = model(x)
                
                # Calculate losses
                loss, recon_loss, kl_loss, age_loss = criterion(
                    recon_x, x, z_mean, z_log_var, pred_age, uncertainty, age
                )
                
                # Store results
                val_losses['total'] += loss.item()
                val_true_ages.extend(age.cpu().numpy())
                val_pred_ages.extend(pred_age.cpu().numpy())
                val_uncertainties.extend(uncertainty.cpu().numpy())
                latents.extend(latent.cpu().numpy())
                
                pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_recon': f'{recon_loss.item():.4f}',
                    'val_kl': f'{kl_loss.item():.4f}',
                    'val_age': f'{age_loss.item():.4f}'
                })
        
        # Calculate validation metrics
        val_true_ages = np.array(val_true_ages).flatten()
        val_pred_ages = np.array(val_pred_ages).flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(val_true_ages, val_pred_ages)
        r_squared = r_value ** 2
        
        # Plot validation results
        plot_predictions(config['type'],
            val_true_ages,
            val_pred_ages,
            np.array(val_uncertainties).flatten(),
            epoch,
            save_dir='plots/val'
        )
        
        # Plot latent space
        plotlatent(latents, val_true_ages, save_path=f'plots/latent/epoch_{epoch}.png')
        data_type = config['type']
        # Generate example brains
        if epoch % 10 == 0:
            recon = model.genBrain(torch.tensor([25, 60]).to(device))
            nib.save(nib.Nifti1Image(recon[0], np.eye(4)), f'samples/young_{data_type}_epoch_{epoch}.nii.gz')
            nib.save(nib.Nifti1Image(recon[1], np.eye(4)), f'samples/old_{data_type}_epoch_{epoch}.nii.gz')
        
        # Save best model
        avg_val_loss = val_losses['total'] / len(test_loader)
        torch.save(model.state_dict(), f'checkpoints/{data_type}_model.pt')
        if r_squared > r_squared_best:
            torch.save(model.state_dict(), f'checkpoints/{data_type}_best_model.pt')
            r_squared_best = r_squared
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {epoch_losses["total"]/len(train_loader):.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'R-squared: {r_squared:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

# Create necessary directories
Path('plots/train').mkdir(parents=True, exist_ok=True)
Path('plots/val').mkdir(parents=True, exist_ok=True)
Path('plots/latent').mkdir(parents=True, exist_ok=True)
Path('samples').mkdir(parents=True, exist_ok=True)
Path('checkpoints').mkdir(parents=True, exist_ok=True)

# Main training loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brain Age Training')
    parser.add_argument('--type', type=str, default='r_thickmap', help='Training Data Type (r_thickmap, r_T1w_norm_noskull)')
    args = parser.parse_args()

    config = {
        'num_epochs':100,
        'learning_rate':1e-4,
        'input_size': [144,176,128], #9,11,8
        'num_workers': 8,
        'train_ratio': 0.8,
        'batch_size': 8,
        'num_young': 4,
        'num_elderly': 4, 
        'dataset': ['camcan', 'HCP_aging', 'NIMH-IRP'],
        'type': args.type, # , r_T1w_norm_noskull
    }
    # 'dataset': ['camcan', 'HCP_aging', 'NIMH-IRP',  'ds003097-download', 'ds002168-download',  'ds003592-download',
    #                     'ds002382-download', 'ds002872-download', 'ds003639-download', 'ds003745-download', 'ds004173-download',
    #                     'ds004215-download', 'ds004466-download', 'ds004604-download', 'ds004636-download', 'ds004725-download',
    #                     'ds004856-download', 'ds005026-download', 'ds005123-download', 'ds005237-download', 'ds005270-download',
    #                     'ds005364-download', 'ds005374-download', 'ds005418-download'],
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Setting up dataloader ########################################################
    # df_camcan = pd.read_csv('/ix1/haizenstein/jil202/studies/camcan/derivatives/report/study_report.csv')
    # df_hcp = pd.read_csv('/ix1/haizenstein/jil202/studies/HCP_aging/derivatives/report/study_report.csv')
    # df_nimh = pd.read_csv('/ix1/haizenstein/jil202/studies/NIMH-IRP/derivatives/report/study_report.csv')
    # df_ds002168 = pd.read_csv('/ix1/haizenstein/jil202/studies/ds002168-download/derivatives/report/study_report.csv')
    # df_ds003097 = pd.read_csv('/ix1/haizenstein/jil202/studies/ds003097-download/derivatives/report/study_report.csv')
    # df_ds003592 = pd.read_csv('/ix1/haizenstein/jil202/studies/ds003592-download/derivatives/report/study_report.csv')

    # df = pd.concat([df_camcan, df_hcp, df_nimh, df_ds002168, df_ds003097, df_ds003592], ignore_index=True)
    dfs = []
    niipaths = []
    data_type = config['type']
    for study in config['dataset']:
        dfs.append(f'/ix1/haizenstein/jil202/studies/{study}/derivatives/report/study_report.csv')
        niipaths.extend(glob.glob(f'/ix1/haizenstein/jil202/studies/{study}/derivatives/thickness/*/{data_type}.nii.gz'))

    dataset = BrainDataset(report_paths=dfs, nii_paths=natsorted(niipaths), type=data_type, image_size=config['input_size'],)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    ##################################################################################
    
    model = Conditional3DVAE(config).to(device)
    model_path = f'/ix1/haizenstein/jil202/cortical_VAE_2025_01_07/gmba/checkpoints/{data_type}_best_model.pt'
    model.load_state_dict(torch.load(model_path), strict=False)
    train_model(config, model, train_loader, test_loader, device)