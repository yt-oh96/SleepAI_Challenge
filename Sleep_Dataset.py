import os
import torch 
import pandas as pd
import random
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils import crop_image, shift, crop_image2


class Sleep_Dataset(Dataset):
    def __init__(self, csv_file, transform, size=None, mode='origin', drop_idx=None, data_root_dir='/DATA/'):
        
        self.size = size
        self.mode = mode
        self.drop_idx = drop_idx
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file, header=None)
        # self.data_len = len(self.data) // 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # offset = np.random.choice(4)
        # idx = idx + offset*self.data_len
        file_path = self.data_root_dir + self.data[0][idx] +'/'+ self.data[1][idx]
        target = self.data[2][idx]
        #print(file_path, target)
        target = self._target_label(target)
        
        if not os.path.exists(file_path):
            print('dose not exist '+file_path)
        
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        image = crop_image(image, size=self.size, mode = self.mode, k=2, drop_idx = self.drop_idx)
        # image = crop_image2(image, mode=self.mode, k=2, drop_idx=self.drop_idx)
        if self.mode != 'origin':
            # image = shift(image)
            # image = np.flip(image, 1).copy()
            pass

        sample = {'image':image, 'label':target}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def _target_label(self,target):
        if target == 'Wake' : return 0
        if target == 'N1' : return 1
        if target == 'N2' : return 2
        if target == 'N3' : return 3
        if target == 'REM' : return 4


class Sleep_Test_Dataset(Dataset):
    def __init__(self, csv_file, transform, size=None, mode='origin', drop_idx=None, flip=False, data_root_dir='/DATA/'):
        self.size = size
        self.mode= mode
        self.drop_idx=drop_idx
        self.flip = flip
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data_root_dir + self.data[0][idx] +'/'+ self.data[1][idx]
        #target = self.data[2][idx]
        #print(file_path, target)
        #target = self._target_label(target)

        if not os.path.exists(file_path):
            print('dose not exist '+file_path)

        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        image = crop_image(image, size=self.size, mode = self.mode, drop_idx=self.drop_idx)
        # image = crop_image2(image, mode=self.mode, k=2, drop_idx=self.drop_idx)
        if self.flip: # flip_lr
            image = np.flip(image, 1).copy()
            # print(image.shape)

        sample = {'image':image, 'code':self.data[0][idx], 'num':self.data[1][idx]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
