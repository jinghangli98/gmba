import torch
import numpy as np
import glob
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from natsort import natsorted
import pandas as pd
import random
from pathlib import Path
import pdb

class BrainDataset(Dataset):
    def __init__(self, report_paths, nii_paths, type, transform=None, image_size=[96,96,96]):
        """
        Args:
            report_paths (list): List of paths to report CSV files
            nii_paths (list): List of paths to NIfTI files
            transform (callable, optional): Optional transform to be applied
        """
        self.transform = transform
        self.report_paths = report_paths 
        self.type = type
        try:
            self.df = pd.concat([pd.read_csv(file) for file in report_paths])
        except:
            self.df = pd.read_csv(report_paths)
        self.df.reset_index(drop=True, inplace=True)
        self.nii_paths = nii_paths

        input_x, input_y, input_z = image_size
        self.preprocessing = tio.Compose([tio.transforms.CropOrPad((input_x, input_y, input_z)),])

    def __len__(self):
        return len(self.nii_paths)

    def __getitem__(self, idx):
        nii_path = self.nii_paths[idx]
        nii_img = nib.load(nii_path)
        image = nii_img.get_fdata()
        
        subject_id = Path(nii_path).parent.name
        subject_data = self.df[self.df['SubjectID'] == subject_id].iloc[0]
        
        age = subject_data['age']
        sex = self._convert_sex(subject_data['sex'])
        
        image = self._preprocess_image(image)
        
        image = torch.FloatTensor(image)
        image = image.unsqueeze(0)  
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'age': torch.FloatTensor([age]), 'sex': torch.FloatTensor([sex]), 'ID': subject_id} 
    
    def _convert_sex(self, sex):
        """Convert sex to numerical value"""
        if sex in ['F', 'Female', 'FEMALE']:
            return 0
        elif sex in ['M', 'Male', 'MALE']:
            return 1
        return -1
    
    def _preprocess_image(self, image):
        """Preprocess the image data"""
        processed = self.preprocessing(np.expand_dims(image, 0))
        if self.type == 'r_T1w_norm_noskull':
        
            return np.squeeze(processed)/processed.max()
        elif self.type == 'r_thickmap':
        
            return np.squeeze(processed)
