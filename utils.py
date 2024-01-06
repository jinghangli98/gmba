from tqdm import tqdm
import numpy as np
import torch
import pdb
from itertools import chain
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
f = open('./model_specs.json')
specs = json.load(f)

def train(model, regressor, dataloader, optimizer, criterion_rec, criterion_dis, criterion_label, epoch, device):
    model.train()
    
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
    total_loss = 0
    recon_loss = 0
    label_loss = 0
    KL_div = 0

    age_prediction = []
    chrono_age = []
    bps = []
    IDs = []
    for idx, [img, age, sex, bp, ID] in enumerate(dataloader):
        
        
        chrono_age.append(np.int16(age))
        IDs.append(list(ID))
        bps.append(list(bp))
        Age = torch.from_numpy(np.int16(age))
        img = torch.unsqueeze(img, 1).float().to(device)
        Age = idx2onehot(Age, 100).to(device)
        sex = idx2onehot(sex, 2).to(device)
        age = torch.from_numpy(np.int16(age)).to(device)
        
        optimizer.zero_grad()
        input = [img, Age]
        if regressor == "False":
            y, z_mean, z_log_sigma = model(input)
            loss_rec_batch = criterion_rec(img, y)
            loss_KL_batch = criterion_dis(z_mean, z_log_sigma)
            total_loss_batch = loss_rec_batch + loss_KL_batch

            recon_loss += loss_rec_batch
            KL_div += loss_KL_batch
            total_loss += total_loss_batch

            batch_size = specs['batch_size']
            lr = optimizer.param_groups[0]['lr']
            batch_bar.set_postfix(
                epoch=f'Epoch: {epoch}',
                reconloss='{:.04f}'.format(recon_loss/(idx + 1)),
                KL_div='{:.04f}'.format(KL_div/(idx + 1)),
                lr='{:.04f}'.format(float(lr))
            )
        elif regressor == "True":
            y, z_mean, z_log_sigma, r_predict = model(input)
            age_prediction.append(r_predict.detach().cpu().numpy())
            loss_rec_batch = criterion_rec(img, y)
            loss_KL_batch = criterion_dis(z_mean, z_log_sigma)
            label_loss_batch = criterion_label(age.float(), r_predict.flatten())
            total_loss_batch =  loss_rec_batch +  loss_KL_batch +  label_loss_batch
            # print(f'Predicted Age: {r_predict.flatten()[0]} | Real Age: {torch.argmax(age,1).float()[0]}')

            recon_loss += loss_rec_batch
            KL_div += loss_KL_batch
            label_loss += label_loss_batch
            total_loss += total_loss_batch

            batch_size = specs['batch_size']
            lr = optimizer.param_groups[0]['lr']
            batch_bar.set_postfix(
                epoch=f'Epoch: {epoch}',
                reconloss='{:.04f}'.format(recon_loss/(idx + 1)),
                KL_div='{:.04f}'.format(KL_div/(idx + 1)),
                label_loss='{:.04f}'.format(label_loss/(idx + 1)),
                lr='{:.04f}'.format(float(lr))
            )
            

        total_loss_batch.backward()
        optimizer.step()
        batch_bar.update()

    IDs = list(chain.from_iterable(IDs))
    flatten_age_prediction = [list(item.ravel()) for item in age_prediction]
    flatten_chrono_age = [list(item.ravel()) for item in chrono_age]
    flatten_bps = list(chain.from_iterable(bps))
    flatten_age_prediction = list(chain.from_iterable(flatten_age_prediction))
    flatten_chrono_age = list(chain.from_iterable(flatten_chrono_age))
    data = pd.read_excel('./camcan-brainAge-plot-crc.xlsx')
    
    newdata = {'SubjectID': IDs, 'age': flatten_chrono_age, 'predicted_age': flatten_age_prediction, 'blood_pressure': flatten_bps}
    newdata = pd.DataFrame(newdata)
    merged_df = pd.merge(data, newdata, on='SubjectID', how='left')
    normal_bp = merged_df[merged_df['BP'] == 1]
    elevated_bp = merged_df[merged_df['BP'] == 2]
    high_bp = merged_df[merged_df['BP'] == 3] 
    none_bp = merged_df[merged_df['BP'] == 0]     
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(0,100), range(0,100))
    sns.scatterplot(x='age', y='predicted_age', data=normal_bp, color='blue', label='Normal Blood Pressure')
    sns.scatterplot(x='age', y='predicted_age', data=elevated_bp, color='green', label='Elevated Blood Pressure')
    sns.scatterplot(x='age', y='predicted_age', data=high_bp, color='red', label='High Blood Pressure')
    sns.scatterplot(x='age', y='predicted_age', data=none_bp, color='black', label='N/A')
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.title('Age vs. Predicted Age by Blood Pressure Type')
    plt.legend()
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.grid(True)
    plt.show()
    plt.savefig(f'./regressionPlot/{epoch}_train.png')
    plt.close()
    
    oldWMH=merged_df#[merged_df['Age'] > 70]
    plt.figure(figsize=(10, 6))
    plt.plot(range(0,100), range(0,100))
    sns.scatterplot(x=oldWMH['WMH(L)'], y=oldWMH['predicted_age']-oldWMH['Age'], hue=oldWMH['Sex'])
    plt.xlabel('WMH(L)')
    plt.ylabel('Age GAP')
    plt.title('Brain Age differences with respect to WMH')
    plt.legend()
    # plt.ylim(-20, 20)
    plt.xlim(0, np.max(oldWMH['WMH(L)'])+0.01)
    plt.grid(True)
    plt.show()
    plt.savefig(f'./regressionPlot/{epoch}_wmh_train.png')
    plt.close()
    
    oldWMH=merged_df[merged_df['Age'] > 70]
    plt.figure(figsize=(10, 6))
    plt.plot(range(0,100), range(0,100))
    sns.scatterplot(x=oldWMH['WMH(L)'], y=oldWMH['predicted_age']-oldWMH['Age'], hue=oldWMH['Sex'])
    plt.xlabel('WMH(L)')
    plt.ylabel('Age GAP')
    plt.title('Brain Age differences with respect to WMH')
    plt.legend()
    # plt.ylim(-20, 20)
    plt.xlim(0, np.max(oldWMH['WMH(L)'])+0.01)
    plt.grid(True)
    plt.show()
    plt.savefig(f'./regressionPlot/{epoch}_wmh_train>70.png')
    plt.close()
    
    batch_bar.close()

    return recon_loss/(idx + 1), KL_div/(idx + 1), label_loss/(idx+1), total_loss/(idx + 1)

def validate(model, regressor, dataloader, optimizer, criterion_rec, criterion_dis, criterion_label, epoch, device):
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    total_loss = 0
    recon_loss = 0
    label_loss = 0
    KL_div = 0
    
    age_prediction = []
    chrono_age = []
    bps = []
    IDs = []
    for idx, [img, age, sex, bp, ID] in enumerate(dataloader):
        IDs.append(list(ID))
        chrono_age.append(np.int16(age))
        bps.append(list(bp))
        Age = torch.from_numpy(np.int16(age))
        img = torch.unsqueeze(img, 1).float().to(device)
        Age = idx2onehot(Age, 100).to(device)
        sex = idx2onehot(sex, 2).to(device)
        age = torch.from_numpy(np.int16(age)).to(device)

        with torch.inference_mode():
            input = [img, Age]
            if regressor == "False":
                y, z_mean, z_log_sigma = model(input)
                loss_rec_batch = criterion_rec(img, y)
                loss_KL_batch = criterion_dis(z_mean, z_log_sigma)
                total_loss_batch = loss_rec_batch + loss_KL_batch

                recon_loss += loss_rec_batch
                KL_div += loss_KL_batch
                total_loss += total_loss_batch

                lr = optimizer.param_groups[0]['lr']
                batch_bar.set_postfix(
                    epoch=f'Epoch: {epoch}',
                    reconloss='{:.04f}'.format(recon_loss/(idx + 1)),
                    KL_div='{:.04f}'.format(KL_div/(idx + 1)),
                    lr='{:.04f}'.format(float(lr))
                )
            elif regressor == "True":
                y, z_mean, z_log_sigma, r_predict = model(input)
                
                age_prediction.append(r_predict.detach().cpu().numpy())
                loss_rec_batch = criterion_rec(img, y)
                loss_KL_batch = criterion_dis(z_mean, z_log_sigma)
                label_loss_batch = criterion_label(age.float(), r_predict.flatten())
                total_loss_batch = loss_rec_batch + loss_KL_batch + label_loss_batch

                recon_loss += loss_rec_batch
                label_loss += label_loss_batch
                KL_div += loss_KL_batch
                total_loss += total_loss_batch

                lr = optimizer.param_groups[0]['lr']
                batch_bar.set_postfix(
                    epoch=f'Epoch: {epoch}',
                    reconloss='{:.04f}'.format(recon_loss/(idx + 1)),
                    KL_div='{:.04f}'.format(KL_div/(idx + 1)),
                    label_loss='{:.04f}'.format(label_loss/(idx + 1)),
                    lr='{:.04f}'.format(float(lr))
                )

        batch_bar.update()
    
    IDs = list(chain.from_iterable(IDs))
    flatten_age_prediction = [list(item.ravel()) for item in age_prediction]
    flatten_chrono_age = [list(item.ravel()) for item in chrono_age]
    flatten_bps = list(chain.from_iterable(bps))
    flatten_age_prediction = list(chain.from_iterable(flatten_age_prediction))
    flatten_chrono_age = list(chain.from_iterable(flatten_chrono_age))
    
    data = pd.read_excel('./camcan-brainAge-plot-crc.xlsx')
    newdata = {'SubjectID': IDs, 'age': flatten_chrono_age, 'predicted_age': flatten_age_prediction, 'blood_pressure': flatten_bps}
    newdata = pd.DataFrame(newdata)
    merged_df = pd.merge(newdata, data, on='SubjectID', how='left')
    merged_df.to_csv(f'./csv/{epoch}_agePrediction.csv')
    merged_df = merged_df #[merged_df['age'] > 45]
     
    normal_bp = merged_df[merged_df['BP'] == 1]
    elevated_bp = merged_df[merged_df['BP'] == 2]
    high_bp = merged_df[merged_df['BP'] == 3] 
    none_bp = merged_df[merged_df['BP'] == 0]   
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(0,100), range(0,100))
    sns.scatterplot(x='age', y='predicted_age', data=normal_bp, color='blue', label='Normal Blood Pressure')
    sns.scatterplot(x='age', y='predicted_age', data=elevated_bp, color='green', label='Elevated Blood Pressure')
    sns.scatterplot(x='age', y='predicted_age', data=high_bp, color='red', label='High Blood Pressure')
    sns.scatterplot(x='age', y='predicted_age', data=none_bp, color='black', label='N/A')
    
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.title('Age vs. Predicted Age by Blood Pressure Type')
    plt.legend()
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.grid(True)
    plt.show()
    plt.savefig(f'./regressionPlot/{epoch}_test.png')
    plt.close()
    
    oldWMH=merged_df#[merged_df['Age'] > 70]
    plt.figure(figsize=(10, 6))
    plt.plot(range(0,100), range(0,100))
    sns.scatterplot(x=oldWMH['WMH(L)'], y=oldWMH['predicted_age']-oldWMH['Age'], hue=oldWMH['Sex'])
    plt.xlabel('WMH(L)')
    plt.ylabel('Age GAP')
    plt.title('Brain Age differences with respect to WMH')
    plt.legend()
    # plt.ylim(-20, 20)
    plt.xlim(0, np.max(oldWMH['WMH(L)'])+0.01)
    plt.grid(True)
    plt.show()
    plt.savefig(f'./regressionPlot/{epoch}_wmh_test.png')
    plt.close()
    
    oldWMH=merged_df[merged_df['Age'] > 70]
    plt.figure(figsize=(10, 6))
    plt.plot(range(0,100), range(0,100))
    sns.scatterplot(x=oldWMH['WMH(L)'], y=oldWMH['predicted_age']-oldWMH['Age'], hue=oldWMH['Sex'])
    plt.xlabel('WMH(L)')
    plt.ylabel('Age GAP')
    plt.title('Brain Age differences with respect to WMH')
    plt.legend()
    # plt.ylim(-20, 20)
    plt.xlim(0, np.max(oldWMH['WMH(L)'])+0.01)
    plt.grid(True)
    plt.show()
    plt.savefig(f'./regressionPlot/{epoch}_wmh_test>70.png')
    plt.close()
    
    batch_bar.close()
    return recon_loss/(idx + 1), KL_div/(idx + 1), label_loss/(idx+1), total_loss/(idx + 1)
            

def idx2onehot(idx, n):

    idx = idx.type(torch.int64)  
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

def zeropad_z(x, target):
    assert(len(x.shape) == 3)
    x = torch.tensor(x)
    pad_front = (target - x.shape[2]) // 2
    pad_back = target - x.shape[2] - pad_front
    x_padded = torch.nn.functional.pad(x, (pad_front, pad_back, 0, 0, 0, 0))
    return x_padded.detach().cpu().numpy()
