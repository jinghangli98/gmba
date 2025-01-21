import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D inputs"""
    def __init__(self, channels, reduction_ratio=8):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y

class ResBlock3D(nn.Module):
    def __init__(self, channels, use_se=True):
        super(ResBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(channels)
        self.in2 = nn.InstanceNorm3d(channels)
        self.se = SEBlock3D(channels) if use_se else None
        
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.in1(self.conv1(x)), 0.2)
        x = self.in2(self.conv2(x))
        if self.se:
            x = self.se(x)
        x += residual
        return F.leaky_relu(x, 0.2)

class AgePredictor(nn.Module):
    def __init__(self, config):
        super(AgePredictor, self).__init__()
        
        self.config = config
        # Reduced initial channels
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Group Normalization
        self.gn1 = nn.GroupNorm(4, 16)
        self.gn2 = nn.GroupNorm(8, 32)
        self.gn3 = nn.GroupNorm(8, 64)
        self.gn4 = nn.GroupNorm(16, 128)
        
        # SE blocks
        self.se1 = SEBlock3D(16)
        self.se2 = SEBlock3D(32)
        self.se3 = SEBlock3D(64)
        self.se4 = SEBlock3D(128)
        
        self.flatten_size = self._get_flatten_size()
        
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_age = nn.Linear(128, 1)
        self.fc_uncertainty = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def _get_flatten_size(self):
        input_x, input_y, input_z = self.config['input_size']
        x = torch.randn(1, 1, input_x, input_y, input_z)
        x = self._forward_features(x)
        return x.flatten(1).shape[1]
    
    def _forward_features(self, x):
        x = F.leaky_relu(self.gn1(self.conv1(x)), 0.2)
        x = self.se1(x)
        x = F.leaky_relu(self.gn2(self.conv2(x)), 0.2)
        x = self.se2(x)
        x = F.leaky_relu(self.gn3(self.conv3(x)), 0.2)
        x = self.se3(x)
        x = F.leaky_relu(self.gn4(self.conv4(x)), 0.2)
        x = self.se4(x)
        return x
    
    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(1)
        
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.2))
        
        age_pred = self.fc_age(x)
        uncertainty = torch.exp(self.fc_uncertainty(x))
        
        return age_pred, uncertainty

class Encoder(nn.Module):
    def __init__(self, config, in_channels=1, latent_dim=128):
        super(Encoder, self).__init__()
        
        self.config = config
        
        # Reduced channel counts
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.in1 = nn.InstanceNorm3d(32)
        self.in2 = nn.InstanceNorm3d(64)
        self.in3 = nn.InstanceNorm3d(128)
        self.in4 = nn.InstanceNorm3d(256)
        
        self.res1 = ResBlock3D(32, use_se=True)
        self.res2 = ResBlock3D(64, use_se=True)
        
        self.flatten_size = self._get_flatten_size()
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def _get_flatten_size(self):
        input_x, input_y, input_z = self.config['input_size']
        x = torch.randn(1, 1, input_x, input_y, input_z)
        x = self.encode_conv(x)
        return x.flatten(1).shape[1]
    
    def encode_conv(self, x):
        x = F.leaky_relu(self.in1(self.conv1(x)), 0.2)
        x = self.res1(x)
        x = F.leaky_relu(self.in2(self.conv2(x)), 0.2)
        x = self.res2(x)
        x = F.leaky_relu(self.in3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.in4(self.conv4(x)), 0.2)
        return x
    
    def forward(self, x):
        x = self.encode_conv(x)
        x = x.flatten(1)
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, config, latent_dim=128):
        super(Decoder, self).__init__()
        
        input_x, input_y, input_z = config['input_size']
        self.init_size = (256, input_x//2**4, input_y//2**4, input_z//2**4)
        
        self.fc1 = nn.Linear(latent_dim + 64, 512)
        self.fc2 = nn.Linear(512, np.prod(self.init_size))
        
        # Reduced channel counts in deconv layers
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.in1 = nn.InstanceNorm3d(128)
        self.in2 = nn.InstanceNorm3d(64)
        self.in3 = nn.InstanceNorm3d(32)
        
        self.res1 = ResBlock3D(128, use_se=True)
        self.res2 = ResBlock3D(64, use_se=True)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        
        x = x.view(-1, *self.init_size)
        
        x = F.leaky_relu(self.in1(self.deconv1(x)), 0.2)
        x = self.res1(x)
        x = F.leaky_relu(self.in2(self.deconv2(x)), 0.2)
        x = self.res2(x)
        x = F.leaky_relu(self.in3(self.deconv3(x)), 0.2)
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
    
    def genBrain(self, age):
        age = torch.clamp(age, 0, 99).long()
        age_onehot = F.one_hot(age.squeeze(), num_classes=100).float().to('cuda')
        condition_encoded = F.relu(self.condition_embedding(age_onehot))
        z = torch.rand(len(age), 128).to('cuda')
        z = torch.cat([z, condition_encoded], dim=1)

        return self.decoder(z).detach().cpu().numpy().squeeze()
    
    def forward(self, x):
        predicted_age, uncertainty = self.age_predictor(x)
        age = torch.clamp(predicted_age, 0, 99).long()
        age_onehot = F.one_hot(age.squeeze(), num_classes=100).float()
        condition_encoded = F.relu(self.condition_embedding(age_onehot))
        
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        if len(condition_encoded.shape) == 1:
            condition_encoded = condition_encoded.unsqueeze(0)
            
        z = torch.cat([z, condition_encoded], dim=1)
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar, predicted_age, uncertainty, z