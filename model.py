import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(channels)
        self.in2 = nn.InstanceNorm3d(channels)
        
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.in1(self.conv1(x)), 0.2)
        x = self.in2(self.conv2(x))
        x += residual
        return F.leaky_relu(x, 0.2)
    
class AgePredictor(nn.Module):
    def __init__(self, config):
        super(AgePredictor, self).__init__()
        
        self.config = config
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        
        self.flatten_size = self._get_flatten_size()
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def _get_flatten_size(self):
        input_x, input_y, input_z = self.config['input_size']
        x = torch.randn(1, 1, input_x, input_y, input_z)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x.flatten(1).shape[1]
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.flatten(1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, config, in_channels=1, latent_dim=128):
        super(Encoder, self).__init__()
        
        self.config = config
        
        # Increase initial channels and use smaller kernel for better detail preservation
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Instance Normalization instead of Batch Normalization for better style preservation
        self.in1 = nn.InstanceNorm3d(64)
        self.in2 = nn.InstanceNorm3d(128)
        self.in3 = nn.InstanceNorm3d(256)
        self.in4 = nn.InstanceNorm3d(512)
        
        # Add residual blocks for better feature extraction
        self.res1 = ResBlock3D(64)
        self.res2 = ResBlock3D(128)
        self.res3 = ResBlock3D(256)
        self.res4 = ResBlock3D(512)
        
        self.flatten_size = self._get_flatten_size()
        
        # Add more layers in the bottleneck
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def _get_flatten_size(self):
        input_x, input_y, input_z = self.config['input_size']
        x = torch.randn(1, 1, input_x, input_y, input_z)
        x = self.encode_conv(x)
        return x.flatten(1).shape[1]
    
    def encode_conv(self, x):
        # First block
        x1 = F.leaky_relu(self.in1(self.conv1(x)), 0.2)
        x1 = self.res1(x1)
        
        # Second block
        x2 = F.leaky_relu(self.in2(self.conv2(x1)), 0.2)
        x2 = self.res2(x2)
        
        # Third block
        x3 = F.leaky_relu(self.in3(self.conv3(x2)), 0.2)
        x3 = self.res3(x3)
        
        # Fourth block
        x4 = F.leaky_relu(self.in4(self.conv4(x3)), 0.2)
        x4 = self.res4(x4)
        
        return x4
    
    def forward(self, x):
        x = self.encode_conv(x)
        x = x.flatten(1)
        
        # Enhanced bottleneck
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, config, latent_dim=128):
        super(Decoder, self).__init__()
        
        input_x, input_y, input_z = config['input_size']
        self.init_size = (512, input_x//2**4, input_y//2**4, input_z//2**4)
        
        # Enhanced bottleneck
        self.fc1 = nn.Linear(latent_dim + 64, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, np.prod(self.init_size))
        
        # Increase channels in deconvolution layers
        self.deconv1 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Instance Normalization
        self.in1 = nn.InstanceNorm3d(256)
        self.in2 = nn.InstanceNorm3d(128)
        self.in3 = nn.InstanceNorm3d(64)
        
        # Add residual blocks
        self.res1 = ResBlock3D(256)
        self.res2 = ResBlock3D(128)
        self.res3 = ResBlock3D(64)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, z):
        # Enhanced bottleneck
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        
        x = x.view(-1, *self.init_size)
        
        # First block
        x = F.leaky_relu(self.in1(self.deconv1(x)), 0.2)
        x = self.res1(x)
        
        # Second block
        x = F.leaky_relu(self.in2(self.deconv2(x)), 0.2)
        x = self.res2(x)
        
        # Third block
        x = F.leaky_relu(self.in3(self.deconv3(x)), 0.2)
        x = self.res3(x)
        
        # Final convolution with Tanh activation for better stability
        x = self.deconv4(x)
        
        return x

class Conditional3DVAE(nn.Module):
    def __init__(self, config, in_channels=1, latent_dim=128, condition_dim=100):
        super(Conditional3DVAE, self).__init__()
        
        self.encoder = Encoder(config, in_channels, latent_dim)
        self.decoder = Decoder(config, latent_dim)
        self.age_predictor = AgePredictor(config)
        self.condition_embedding = nn.Linear(condition_dim, 64)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_latent(self, x):
        predicted_age = self.age_predictor(x)
        age = torch.clamp(predicted_age, 0, 99).long()
        age_onehot = F.one_hot(age.squeeze(), num_classes=100).float()
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        condition_encoded = F.relu(self.condition_embedding(age_onehot))
        z = torch.cat([z, condition_encoded], dim=1)

        return z

    def genBrain(self, age):
        age = torch.clamp(age, 0, 99).long()
        age_onehot = F.one_hot(age.squeeze(), num_classes=100).float().to('cuda')
        condition_encoded = F.relu(self.condition_embedding(age_onehot))
        z = torch.rand(len(age), 128).to('cuda')
        z = torch.cat([z, condition_encoded], dim=1)

        return self.decoder(z).detach().cpu().numpy().squeeze()

    def forward(self, x):
        predicted_age = self.age_predictor(x)
        age = torch.clamp(predicted_age, 0, 99).long()
        age_onehot = F.one_hot(age.squeeze(), num_classes=100).float()
        condition_encoded = F.relu(self.condition_embedding(age_onehot))
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, condition_encoded], dim=1)
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar, predicted_age, z