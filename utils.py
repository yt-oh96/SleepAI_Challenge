import os
import random
import numpy as np
import torch
import cv2
import pandas as pd

def make_wake_n1_csv(wake_n1_idx, csv_path='/DATA/trainset-for_user.csv',save_dir='./'):
    csv = np.array(pd.read_csv(csv_path, header=None))

    results_df = pd.DataFrame()
    for idx, wn_idx in enumerate(wake_n1_idx):
        row = [csv[wn_idx][0],csv[wn_idx][1],csv[wn_idx][2]]
        row_df = pd.DataFrame([row])
        results_df = pd.concat([results_df, row_df])
        
        if idx%100 == 99: print(idx)
    results_df.to_csv(save_dir+'wake_n1.csv',header=False, index=False)


def wake_n1_idx(csv_path='/DATA/trainset-for_user.csv'):
    
    csv = pd.read_csv(csv_path, header=None) 
    csv_file = np.array(csv[2])
    
    idx = []
    for i in range(len(csv_file)):
        if csv_file[i] !=  'Wake' and csv_file[i] != 'N1':
            idx.append(i)

    wake_n1_idx = np.setdiff1d(range(len(csv_file)), idx)

    return wake_n1_idx



def pred_to_label(pred_label):

    label = []
    for i in range(len(pred_label)):
        pred = pred_label[i]

        if pred == 0 : label.append('Wake')
        if pred == 1 : label.append('N1')
        if pred == 2 : label.append('N2')
        if pred == 3 : label.append('N3')
        if pred == 4 : label.append('REM')
    return np.array(label)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    #print(random.random())
    if torch.cuda.is_available():
        print(f'seed : {seed_value}')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def shift(image, shfit_scale=48): # width 480
    H, W, _ = image.shape
    scale = np.random.randint(-shfit_scale, shfit_scale)
    if scale >= 0: # shfit right
        image = image[:, :480-scale]
    else:
        image = image[:, abs(scale):]
    image = cv2.resize(image, dsize=(W,H))
    return image

def crop_image(image, size=None, mode='origin', k=2, drop_idx=None):
    
    if size is None:
        h = 10.75
        origin_image = image
    else:
        h = 43
        origin_image = cv2.resize(image, dsize=(480,1080))
    if mode == 'drop':
        origin_image = drop_features(origin_image, k=k, heights=h, drop_idx=drop_idx)
    elif mode == 'dropG':
        origin_image = drop_groups(origin_image, k=k, heights=h, drop_idx=drop_idx)
    
    if size is None:
        group1 = origin_image[0:97, :] # 9
        group2 = origin_image[97:183, :] # 17
        group3 = origin_image[183:269, :] # 25
    else:
        group1 = origin_image[0:387,:]
        group2 = origin_image[387:731,:]
        group3 = origin_image[731:1075,:]
    
    
    #print(group1.shape, group2.shape, group3.shape)
    if size is None:
        group1 = cv2.resize(group1, dsize=(480,100))
        group2 = cv2.resize(group2, dsize=(480,100))
        group3 = cv2.resize(group3, dsize=(480,100))
    else:
        group1 = cv2.resize(group1, dsize=(480,387))
        group2 = cv2.resize(group2, dsize=(480,387))
        group3 = cv2.resize(group3, dsize=(480,387))

    img = np.stack([group1, group2, group3], axis=-1)

    return img

def crop_image2(image, mode='origin', k=2, drop_idx=None):
    
    image = cv2.resize(image, dsize=(480,1080))
    if mode == 'drop':
        image = drop_features(image, k=k, heights=43, drop_idx=drop_idx, fs=True)
    elif mode == 'dropG':
        image = drop_groups(image, k=k, heights=43, drop_idx=drop_idx, fs=True)
    
    x = np.zeros((3,43*4,480), np.float32)
    x[0,:,:] = image[43*3:43*7,:]           # EEG
    x[1,0:43*2,:] = image[43*7:43*9,:]      # EOG
    x[1,43*2:43*3,:] = image[43*9:43*10,:]  # Chin EMG
    x[2,0:43*3,:] = image[43*11:43*14,:]    # Flow
    x[2,43*3:,:] = image[43*15:43*16,:]     # Thorax
    x[1,43*3:,:] = image[43*16:43*17,:]     # Abdomen
    
    x = x.transpose(1,2,0)
    x = cv2.resize(x, dsize=(480,120))

    return x


def drop_features(img, k = 1, heights = 43, drop_idx=None, fs=False):
    n_features = 21
    start_idx = [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,23] # 21
    end_idx = [1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,23,25]
    if fs: # feature selection
        n_features = 10
        start_idx = [3,4,5,6,7,8,9,11,15,16]
        end_idx = [4,5,6,7,8,9,10,14,16,17]
    if drop_idx is None: # random - train
        # c = np.random.choice(k+1)
        c = k
        drop_idx_list = np.random.choice(range(n_features), c, replace=False)
    else: # fix - test
        drop_idx_list = [drop_idx]
#     print(drop_idx_list)
    for drop_idx in drop_idx_list:
        start = int(np.round(start_idx[drop_idx])*heights)
        end = int(np.round(end_idx[drop_idx])*heights)
        img[start:end, :] = 0
        
    return img

def drop_groups(img, k = 1, heights = 43, drop_idx=None):
    n_groups = 14
    start_idx = [0,3,7,9,10,11,14,15,16,17,19,20,21,23,25] # n(feature group + end) = 15
    if drop_idx is None: # random - train
        c = np.random.choice(k+1)
        drop_idx_list = np.random.choice(range(n_groups), c, replace=False)
    else: # fix - test
        drop_idx_list = [drop_idx]
#     print(drop_idx_list)
    for drop_idx in drop_idx_list:
        start = int(np.round(start_idx[drop_idx])*heights)
        end = int(np.round(start_idx[drop_idx+1])*heights)
        img[start:end, :] = 0

    return img

def print_metrics(pred, label):
    metrics = get_metrics(pred, label)
    for i in range(5):
        print(metrics[i])

def eval_metrics (pred, label):
    metrics = get_metrics (pred, label)
    avg_f1 = 0
    for m in metrics:
        avg_f1 += m['f1']
        
    return np.round(avg_f1 / len(metrics), 5)

def _confusion_matrix(pred, label, positive_class=1):  
    '''
    (pred, label)
    TN (not p_class, not p_class) / FN (not p_class, p_class) / FP (p_class, not p_class) / TP (p_class, p_class)
    ex)
    TN (0,0) / FN (0,1)/ FP (1,0) / TP (1,1)
    '''
    TN, FN, FP, TP = 0, 0, 0, 0

    for y_hat, y in zip(pred, label):
        if y_hat != positive_class:
            if y != positive_class:
                    TN = TN + 1
            else:
                    FN = FN + 1
        elif y_hat == positive_class:
            if y != positive_class:
                FP = FP + 1
            else:
                TP = TP + 1
    return TN, FN, FP, TP

def confusion_5(pred, label):
    n_classes = 5
    pred = np.argmax(pred, 1)
    confusion = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            confusion[i,j] = np.sum(pred[label==i]==j)
    return confusion.astype('int')

def get_metrics (pred, label, num_class=5, eps=1e-5):
    '''
    label : 0,1,2,3,4
    pred : softmax
    '''
    all_metrics = []
    
    pred = np.argmax(pred, 1)
    for p_class in range(num_class):
        metrics = dict()
        num_P, num_N = np.sum(label == p_class), np.sum(label != p_class)
        
        TN, FN, FP, TP = _confusion_matrix(pred, label, p_class)
        metrics['prec'] = TP / (TP + FP + eps) ## ppv
        metrics['recall'] = TP / (TP + FN + eps) ## sensitivive
        metrics['f1'] = 2*(metrics['prec'] * metrics['recall'])/(metrics['prec'] + metrics['recall'] + eps)
        
        all_metrics.append(metrics)

    return all_metrics
