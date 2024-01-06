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

study = 'GIA_IBR'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
df = pd.read_csv(f'/ix1/tibrahim/jil202/studies/{study}/derivatives/report/study_report.csv')
df['SubjectID'] = [ID.split('_')[0].split('-')[0] for ID in df['ID']]
try:
    df = df.drop('Unnamed: 0', axis=1)
except:
    pass

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
gm_path = natsorted(glob.glob(f'/ix1/tibrahim/jil202/studies/{study}/derivatives/thickness/*/r_*.nii.gz'))

gm_dataset = ratio_dataset(gm_path)
gm_loader = DataLoader(gm_dataset, batch_size=2, shuffle=False)

age_prediction = []
IDs = []
gm_model = agePrediction(gm_model)

for idx, [img, ID] in enumerate(tqdm(gm_loader)):
    ID = [id.split('_')[0].split('-')[0] for id in ID]
    IDs.append(list(ID))
    img = torch.unsqueeze(img, 1).float().to(device)
    
    brainAge = [age.item() for age in gm_model(img).detach().cpu()]
    age_prediction.append(brainAge)
pdb.set_trace()
age_prediction = list(chain.from_iterable(age_prediction))
IDs = list(chain.from_iterable(IDs))
predictedModel = {"SubjectID": IDs, "gm_brainAge":age_prediction}
predictedModel = pd.DataFrame(predictedModel)

merged_df = pd.merge(df, predictedModel, on='SubjectID', how='outer')
gm_df = pd.merge(df, predictedModel, on='SubjectID', how='inner')
gm_df = gm_df.dropna(subset=['age'])
model = LinearRegression()
model.fit(np.expand_dims(gm_df['age'].values, 1), np.expand_dims(gm_df['gm_brainAge'].values, 1))
slope = model.coef_[0]
intercept = model.intercept_
brain_age_corrected = [((age - intercept)/slope)[0] for age in gm_df['gm_brainAge'].values]
brain_age_gap = brain_age_corrected - gm_df['age'].values
gm_df['gm_age(corrected)'] = brain_age_corrected 
gm_df['gm_gap(corrected)'] = brain_age_gap
gm_df = gm_df[['gm_age(corrected)','SubjectID', 'gm_gap(corrected)']]
merged_df = pd.merge(merged_df, gm_df, on='SubjectID', how='outer')
merged_df.to_csv(f'/ix1/tibrahim/jil202/studies/{study}/derivatives/report/study_report_110423.csv')

#####Plotting######
plt.figure(figsize=(8, 6))
sns.boxplot(x='sex', y='gm_gap(corrected)', data=merged_df)
plt.title(f'{study} Boxplot of Brain Age Gap by Sex')
plt.xlabel('Sex')
plt.ylabel('Brain Age Gap')
plt.savefig(f'{study}_sex_BA_gap_boxplot.png')

# Create a scatter plot for Chronological Age vs. Brain Age Corrected
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='gm_age(corrected)', data=merged_df, hue='sex')
plt.title(f'{study} Scatter Plot of Chronological Age vs. Brain Age Corrected')
plt.xlabel('Chronological Age')
plt.ylabel('Brain Age Corrected')
plt.savefig(f'{study}_sex_BA_scatterplot_corrected.png')

plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='gm_brainAge', data=merged_df, hue='sex')
plt.title(f'{study} Scatter Plot of Chronological Age vs. Brain Age Corrected')
plt.xlabel('Chronological Age')
plt.ylabel('Brain Age Corrected')
plt.savefig(f'{study}_sex_BA_scatterplot.png')
