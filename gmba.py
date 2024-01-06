#!/usr/bin/env python3

import torch
import pandas as pd
from dataset import *
import pdb
from medcam import medcam
import matplotlib.pyplot as plt
from itertools import chain
from tqdm import tqdm
from vae_3d import VAE_regressor
from bids.layout import BIDSLayout
from sklearn.linear_model import LinearRegression
import seaborn as sns
import sys

img_folder = sys.argv[1]
output = sys.argv[2]
print(f'Reading images from folder: {img_folder}')
print(f'Output will be saved to: {output}')

'''

'''
gm_path = natsorted(glob.glob(f'{img_folder}/*.nii.gz'))
gm_dataset = ratio_dataset(gm_path)
gm_loader = DataLoader(gm_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class agePrediction(torch.nn.Module):
    def __init__(self, pretrainedModel):
        super(agePrediction, self).__init__()
        pretrainedModel.eval()
        self.encoder = pretrainedModel.encoder
        self.z_mean = pretrainedModel.z_mean
        self.z_log_sigma = pretrainedModel.z_log_sigma
        self.regressor = pretrainedModel.regressor
        
    def forward(self, x):
        img = x
        x = self.encoder(img)        
        x = torch.flatten(x, start_dim=1)
        z_mean = self.z_mean(x)
        z_log_sigma = self.z_log_sigma(x)
        r_predict = self.regressor(torch.unsqueeze(torch.cat((z_mean, z_log_sigma),1),2))
        
        return r_predict

gm_model = torch.load('/ix1/haizenstein/jil202/paper/gm_brainAge/model/camcan_ctx_96x96x96.pth', map_location=device)    
gm_model = gm_model.to(device)
gm_model.eval()

age_prediction = []
IDs = []
gm_model = agePrediction(gm_model)
for idx, [img, ID] in enumerate(tqdm(gm_loader)):
    ID = [id.split('_')[0].split('-')[0] for id in ID]
    IDs.append(list(ID))
    img = torch.unsqueeze(img, 1).float().to(device)
    
    brainAge = [age.item() for age in gm_model(img).detach().cpu()]
    age_prediction.append(brainAge)
    
age_prediction = list(chain.from_iterable(age_prediction))
IDs = list(chain.from_iterable(IDs))
predictedModel = {"SubjectID": IDs, "gm_brainAge":age_prediction}
predictedModel = pd.DataFrame(predictedModel)
predictedModel.to_csv(f'{output}')