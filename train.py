import torch
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
    

def plot_predictions(true_ages, predicted_ages, epoch, save_dir='plots'):
    """Plot and save age predictions"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(true_ages, predicted_ages, alpha=0.5)
    
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
    
    # Save plot
    plt.savefig(f'{save_dir}/age_prediction_epoch_{epoch}.png')
    plt.close()
    
    return mae, correlation

config = {
        'num_epochs':100,
        'learning_rate':1e-4,
        'input_size': [144,176,128], #9,11,8
        'num_workers': 0,
        'train_ratio': 0.8,
        'batch_size': 16,
        'num_young': 4,
        'num_elderly': 4, 
        'dataset': ['camcan', 'HCP_aging', 'NIMH-IRP'],
        'type': 'r_thickmap' #r_thickmap, r_T1w_norm_noskull
    }

## Setting up dataloader ########################################################
df_camcan = pd.read_csv('/ix1/haizenstein/jil202/studies/camcan/derivatives/report/study_report.csv')
df_hcp = pd.read_csv('/ix1/haizenstein/jil202/studies/HCP_aging/derivatives/report/study_report.csv')
df_nimh = pd.read_csv('/ix1/haizenstein/jil202/studies/NIMH-IRP/derivatives/report/study_report.csv')
df = pd.concat([df_camcan, df_hcp, df_nimh], ignore_index=True)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conditional3DVAE(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
best_val_loss = float('inf') 

train_losses = []
val_losses = []
maes = []
correlations = []

for epoch in range(config["num_epochs"]):
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_label_loss = 0
    true_age = []
    predicted_ages = []
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{config["num_epochs"]}')
    for batch in pbar:
        x = batch['image'].to(device)
        age = batch['age'].to(device).long()
        age_onehot = F.one_hot(age.squeeze(), num_classes=100).float().to(device)
        recon_x, z_mean, z_log_var, predicted_age, latent = model(x)
        true_age.extend(age.detach().cpu().numpy())
        recon_loss = torch.nn.functional.l1_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        age_loss = torch.nn.L1Loss()(age, predicted_age)

        predicted_ages.extend(predicted_age.detach().cpu().numpy())
        loss = (recon_loss + 0.1 * kl_loss + age_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()

        pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
            })
    
    plt.scatter(true_age, predicted_ages)
    plt.savefig('train_regression_brain.png')
    plt.close()
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    model.eval()
    val_epoch_loss = 0
    val_epoch_recon_loss = 0
    val_epoch_kl_loss = 0
    val_epoch_label_loss = 0
    
    true_ages = []
    predicted_ages = []
    latents = []
    true_age = []
    predicted_ages = []
    r_squared_best = 0
    pbar = tqdm(test_loader, desc=f'Testing Epoch {epoch+1}/{config["num_epochs"]}')
    for batch in pbar:
        x = batch['image'].to(device)
        age = batch['age'].to(device).long()
        age_onehot = F.one_hot(age.squeeze(), num_classes=100).float().to(device)
        
        with torch.no_grad():
            recon_x, z_mean, z_log_var, predicted_age, latent = model(x)
            true_age.extend(age.detach().cpu().numpy())
            predicted_ages.extend(predicted_age.detach().cpu().numpy())
            latents.extend(latent.detach().cpu().numpy()) 
            true_ages.extend(age.detach().cpu().numpy())
            recon_loss = torch.nn.functional.l1_loss(recon_x, x, reduction='mean')
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
            loss = (recon_loss +  kl_loss ).mean()
            
        val_loss = (recon_loss + kl_loss).mean()

        pbar.set_postfix({
                'loss': f'{val_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',})
        
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(true_age).flatten(), np.array(predicted_ages).flatten())
    r_squared = r_value ** 2

    plotlatent(latents, true_ages, save_path='latent.png')
    
    recon = model.genBrain(torch.tensor([25, 60]))
    nib.save(nib.Nifti1Image(recon[0], np.eye(4)), f'25_{data_type}.nii.gz')
    nib.save(nib.Nifti1Image(recon[1], np.eye(4)), f'60_{data_type}.nii.gz')
    
    avg_val_loss = val_epoch_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    print(f'\nEpoch {epoch+1} Summary:')
    print(f'Train Loss: {avg_train_loss:.4f}')
    print(f'Val Loss: {avg_val_loss:.4f}')

    if r_squared > r_squared_best:
        torch.save(model.state_dict(), f'{data_type}_best.pt')
        r_squared_best = r_squared

        plt.scatter(true_age, predicted_ages)
        plt.savefig(f'test_regression_{data_type}_best.png')
        plt.close()