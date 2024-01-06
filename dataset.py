import torch
import numpy as np
import glob
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, random_split
import pdb
import random
import torchio as tio
from natsort import natsorted
import pandas as pd
import ants

template = '/ix1/tibrahim/jil202/07-Myelin_mapping/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a.nii'
template = ants.image_read(template)

class ratio_dataset(Dataset):
    def __init__(self, nii_dir, transform=None):
        self.nii_dir = nii_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.nii_dir)
    
    def __getitem__(self, idx):
        
        mi = ants.image_read(self.nii_dir[idx])
        mytx = ants.registration(fixed=template, moving=mi, type_of_transform = 'Rigid' )
        nii_img = mytx['warpedmovout'].numpy()
        
        nii_img[nii_img<0] = 0
        nii_img[nii_img>10] = 0
        nii_img = np.float32(nii_img)
   
        transform = tio.Resize((96,96,96))        
        image = transform(np.expand_dims(nii_img, axis=0))
        return np.squeeze(image), self.nii_dir[idx].split('/')[-1]