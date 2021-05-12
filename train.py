import torch
import os
import random
import numpy as np
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import createFolder, seed_everything, crop_image
from Sleep_Dataset import Sleep_Dataset
from torchvision import models
import torch.nn as nn

import torchvision
import time

from classifier_utils import train,test

seed=20
seed_everything(seed)


batch_size = 96
n_epochs = 3
lr = 3e-4
# lr = 2.5e-5
 
csv_file_train = '/DATA/trainset-for_user.csv'
csv_file_test = '/DATA/testset-for_user.csv'


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [transforms.ToTensor(),
        normalize])

mode = 'drop'

train_set = Sleep_Dataset(csv_file_train, transform, size=387, mode=mode)
valid_set = Sleep_Dataset(csv_file_train, size=387, transform)
test_set = Sleep_Dataset(csv_file_test, transform)


num_train = len(train_set)

train_indices = np.random.choice(num_train, int(0.8*num_train), replace=False)
val_indices = np.setdiff1d(range(num_train), train_indices)

#train_indices = np.random.choice(num_train, int(500), replace=False)
#val_indices = np.setdiff1d(range(int(100)), train_indices)


trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=4)
valloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices),num_workers=4)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)


classes = ('Wake', 'N1', 'N2', 'N3', 'REM')

print(f'num_train : {len(trainloader)}, num_val : {len(valloader)}, num_test : {len(testloader)}')

# model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features,5) 
# model.load_state_dict(torch.load('./model/trial16_epoch3.pth'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

weight_decay = 1e-5
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
print(f'n_epoch:{n_epochs}, lr:{lr}, batch_size:{batch_size}')

save_dir = '/model/test/'
createFolder(save_dir)

trial = 'trial17'

best_model = train(model, n_epochs, trainloader, valloader, criterion, optimizer, scheduler, trial, device)


print("done")





        
    






