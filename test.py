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
from utils import createFolder, seed_everything, crop_image, pred_to_label
from Sleep_Dataset import Sleep_Test_Dataset
from torchvision import models
import torch.nn as nn

import torchvision
import time

from classifier_utils import submit, submit_probs

seed=20
seed_everything(seed)


batch_size = 128
 
csv_file_test = '/DATA/testset-for_user.csv'


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [transforms.ToTensor(),
        normalize])


test_set = Sleep_Test_Dataset(csv_file_test, transform, size=387)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


classes = ('Wake', 'N1', 'N2', 'N3', 'REM')


file_name = 'infer'

# model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features,5)
model.load_state_dict(torch.load( file_name + '.pth'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

drop = True
flip_lr = False
make_csv = True
n_groups = 14
save_dir = './probs_result/' + file_name + '/'
createFolder(save_dir)
print('batch_size', batch_size)

def make_csv(probs, file_name, prefix='' ):
    # o - origin, f - flip, d - drop
    # df - drop flip, d36 - sum drop 3-6 
    prefix = prefix + '_'
    pred = np.argmax(probs, 1)
    label = pred_to_label(pred)
    results_df = pd.DataFrame(label)
    results_df.to_csv('inference_result1.csv', header=False, index=False)
    print('make!!')


probs = 0
if not drop:
    if False:
        print('Full submit')
        test_set = Sleep_Test_Dataset(csv_file_test, transform, size=387, mode='origin')
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        print(f'num_test : {len(testloader)}')

        probs = submit_probs(model, testloader, device)
        # np.save(save_dir + 'probs_full', probs)
        # make_csv(probs, file_name, 'orig')
    probs = np.load(save_dir + 'probs_full.npy')
    if flip_lr:
        print('Full_flip submit')
        test_set = Sleep_Test_Dataset(csv_file_test, transform,size=387, mode='origin', flip=True)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

        flip_probs = submit_probs(model, testloader, device)
        # np.save(save_dir + 'flip_probs_full', flip_probs)
        # make_csv(flip_probs, file_name, 'f')
        probs += flip_probs
        # make_csv(probs, file_name, 'ensem_of')


else:
    print('Drop submit')
    drop_list = [3,4,5,6]
    for drop_idx in range(n_groups):
        if drop_idx not in drop_list:
            continue
        print('drop_idx', drop_idx)
        test_set = Sleep_Test_Dataset(csv_file_test, transform, size=387,mode='drop', drop_idx=drop_idx)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        print(f'num_test : {len(testloader)}')

        drop_probs = submit_probs(model, testloader, device)
        # np.save(save_dir + 'probs_drop' + str(drop_idx), drop_probs)
        # make_csv(drop_probs, file_name, 'd' + str(drop_idx))
        probs += drop_probs
        if drop_idx != drop_list[0]:
            # make_csv(probs, file_name, 'ensem_d3' + str(drop_idx))
            pass

        if flip_lr:
            print('Drop_flip submit')
            test_set = Sleep_Test_Dataset(csv_file_test, transform, size=387,mode='drop', drop_idx=drop_idx, flip=True)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

            flip_probs = submit_probs(model, testloader, device)
            # np.save(save_dir + 'flip_probs_drop' + str(drop_idx), flip_probs)
            # make_csv(flip_probs, file_name, 'df' + str(drop_idx))
            probs += flip_probs
            # make_csv(probs, file_name, 'ensem_df3' + str(drop_idx))

make_csv(probs, file_name, '')

print("done")





        
    






