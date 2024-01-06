import numpy as np
import torch
import nibabel as nib
import os
import glob
from natsort import natsorted
import torch.nn as nn
from torchsummary import summary 
import pdb
from math import prod

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(ch_in, ch_out, kernel_size = k_size, stride=stride, padding=padding),
                                  nn.BatchNorm3d(ch_out),
                                  nn.ReLU(inplace=True))
        
    def forward(self, x):
        out = self.conv(x)
        return out

class conv_up(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, padding=1):
        super(conv_up, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=padding),
                                  nn.BatchNorm3d(ch_out),
                                  nn.ReLU(inplace=True))
    
    def forward(self, x):
        out = self.conv(x)
        return out

class ResConvBlock(nn.Module):
    def __init__(self, ch, k_size, stride=1, padding=1):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(5, stride=2),
            nn.BatchNorm3d(ch),
            
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(5, stride=2),
            nn.BatchNorm3d(ch),
        )
    
    def forward(self, x):
        out = self.conv(x) + x
        return out
    
class Encoder(nn.Module):
    def __init__(self, chs=[16, 32, 64, 128], k_sizes=[3,3,3,3,3]):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(conv_block(1, chs[0], k_sizes[0]))
        self.layers.append(ResConvBlock(ch=chs[0], k_size=k_sizes[0]))
        self.layers.append(nn.MaxPool3d(3, stride=2, padding=1))
        for idx in range(len(chs)-1):
            self.layers.append(conv_block(chs[idx], chs[idx+1], k_sizes[idx+1]))
            self.layers.append(ResConvBlock(ch=chs[idx+1], k_size=k_sizes[idx+1]))
            self.layers.append(nn.MaxPool3d(3, stride=2, padding=1))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self,latent_dim, latent_dim_arr, chs=[128, 64, 32, 16], k_sizes=[4,4,4,4], stride=[2,2,2,2]):
        super(Decoder, self).__init__()
        self.chs = chs
        self.latent_dim = latent_dim   
        self.latent_x, self.latent_y, self.latent_z = latent_dim_arr
        self.latent_linear = nn.Linear(latent_dim, chs[0]*prod(latent_dim_arr)) # change here if you want to change dimension
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        
        for idx in range(len(chs)-1):
            self.layers.append(conv_up(chs[idx], chs[idx+1], k_size=k_sizes[idx], stride=stride[idx]))
            self.layers.append(ResConvBlock(ch=chs[idx+1], k_size=k_sizes[idx]-1))
        
        self.layers.append(conv_up(chs[-1], 1, k_size=k_sizes[-1], stride=stride[-1])) 
        self.layers.append(ResConvBlock(ch=1, k_size=k_sizes[-1]-1)) 
            

    def forward(self, x):
        x = self.latent_linear(x)
        x = self.relu(x)
        x = x.view(-1, self.chs[0], self.latent_x, self.latent_y, self.latent_z) # change here if you want to change dimension
        for layer in self.layers:
            x = layer(x)
        x = self.relu(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, out_features):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features//2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features//2, in_features//4)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features//4, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, latent_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(latent_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()

        x = x.view(-1, 128)
        x = self.fc1(x).relu()
        x = self.fc2(x)        
        return x
    
class VAE_regressor(nn.Module):
    def __init__(self, latent_dim=np.prod([6,6,6]), encoder_ch = 128, encoder_latent_array=[6,6,6], conditional=False):
        super(VAE_regressor, self).__init__()
        self.latent_dim = latent_dim
        self.z_mean = nn.Linear(encoder_ch*latent_dim, latent_dim) # change here if you want to change dimension
        self.z_log_sigma = nn.Linear(encoder_ch*latent_dim, latent_dim) # change here if you want to change dimension

        self.r_mean = nn.Linear(encoder_ch*latent_dim, latent_dim)
        self.r_log_sigma = nn.Linear(encoder_ch*latent_dim, latent_dim)

        self.epsilon = torch.normal(size=(1,latent_dim), mean=0, std=1)
        self.encoder = Encoder(chs=[16, 32, 64, 128])
        self.regressor = CNN(2*np.prod(encoder_latent_array))


        if conditional: 
            self.decoder = Decoder(chs=[128, 64, 32, 16], latent_dim=latent_dim+100, latent_dim_arr=encoder_latent_array) #age up to 100 y.o. + 2 for sex
        else:
            self.decoder = Decoder(latent_dim, encoder_latent_array)

    def reparameterize(self, z_mean, z_log_sigma):
        std = torch.exp(0.5*z_log_sigma)
        eps = torch.randn_like(std)
        return z_mean + eps*std

    def forward(self, x):
        img = x[0]
        age = x[1]
        
        x = self.encoder(img)        
        x = torch.flatten(x, start_dim=1)
        z_mean = self.z_mean(x)
        z_log_sigma = self.z_log_sigma(x)
        r_predict = self.regressor(torch.unsqueeze(torch.cat((z_mean, z_log_sigma),1),2))
        z = self.reparameterize(z_mean, z_log_sigma)
        z_cond = torch.cat((z, age), 1)
        y = self.decoder(z_cond)
        return y, z_mean, z_log_sigma, r_predict

# model = Encoder().to('cuda')
# summary(model, (1,160,192,192))
# model = Decoder(latent_dim=125, latent_dim_arr=[5,5,5], chs=[128,64,32,16,8]).to('cuda')
# summary(model, (1,125))  

# model = VAE(conditional=True).to('cuda')
# model.load_state_dict(torch.load('/bgfs/tibrahim/jil202/07-Myelin_mapping/VAE/trained_model/pytorch/weight/full3d_sex+age.pth'))

# z = torch.randn((1, 1000))
# age = 30
# sex = 1 # male for 0 female for 1
# Age = torch.tensor(age).unsqueeze(0)
# Sex = torch.tensor(sex).unsqueeze(0)
# Age = idx2onehot(Age, 100)
# Sex = idx2onehot(Sex, 2)
# z_cond = torch.cat((z, Age, Sex), 1)
# pdb.set_trace()
# out = model.decoder(z_cond)
# out = out.squeeze().detach().cpu().numpy()
# out = nib.Nifti1Image(out, affine)
# nib.save(out, f'./cvae_full3D_{age}_FEMALE.nii.gz')
