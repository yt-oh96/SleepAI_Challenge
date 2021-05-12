import numpy as np
import torch
import pandas as pd
from copy import deepcopy
import time
import csv
from utils import eval_metrics, pred_to_label


def train(model, n_epochs, trainloader,valloader, criterion, optimizer, scheduler,tri ,device):
    
    best_model = deepcopy(model)
    best_f1 = -np.inf
 
    for epoch in range(n_epochs):
        model.train()
        start_perf_counter = time.perf_counter()
        start_process_time = time.process_time()
        print(tri)
        print(f'n_epoch:{epoch}, lr:{scheduler.get_last_lr()}')

        running_loss = 0.0
        running_f1 = 0.0

        train_loss = 0.0
        train_f1 = 0.0
        for i,data in enumerate(trainloader, 0):

            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            
            proba = torch.nn.Softmax(dim=1)(outputs)

            running_loss += loss.item()
            running_f1 += eval_metrics(np.array(proba.detach().cpu()),np.array(labels.detach().cpu()))
            
            train_loss += loss.item()
            train_f1 += eval_metrics(np.array(proba.detach().cpu()), np.array(labels.detach().cpu()))

            if i%100 == 99:
                print(f'[epoch_{epoch+1}, batch_{i+1}] loss: {running_loss/100}, f1; {running_f1/100}')
                running_loss = 0.0
                running_f1 = 0.0

        end_perf_counter = time.perf_counter()-start_perf_counter
        end_process_time = time.process_time()-start_process_time
        
        print('train_loss:',train_loss/len(trainloader), 'train_f1:',train_f1/len(trainloader))
    
        print(f'perf_counter : {end_perf_counter}')
        print(f'process_time : {end_process_time}')
    
        test_loss, test_f1, probas, labels = test(model, valloader, criterion, device)
        total_test_f1 = eval_metrics(probas, labels)
        print('test_loss:',test_loss, 'test_f1:',test_f1, 'total_test_f1', total_test_f1)
        
        # torch.save(model.state_dict(), './model/'+tri+'_epoch'+str(epoch)+'.pth')
        scheduler.step()
        
        if test_f1 > best_f1:
            best_model = deepcopy(model)
            # torch.save(model.state_dict(), './model/'+tri+'_best_epoch'+str(epoch)+'_'+time.strftime('%H:%M:%S')+'.pth')
            pass
    
    torch.save(model.state_dict(), 'infer.pth')    

    return best_model


def test(model, data_loader, criterion, device):
    model.eval()
    total_loss=0
    total_f1=0
    total_proba = []
    total_label = []
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            proba = torch.nn.Softmax(dim=1)(outputs)
            total_proba.append(proba.cpu().numpy())
            total_label.append(labels.cpu().numpy())
            total_f1 += eval_metrics(np.array(proba.cpu()),np.array( labels.cpu()))
            '''for idx,_ in enumerate(labels):
                print('predicted_labels:', torch.max(outputs.data, dim=1).indices[idx], 'label:', labels[idx].cpu())'''

        total_proba = np.concatenate(total_proba, 0)
        total_label = np.concatenate(total_label, 0)
    return total_loss/len(data_loader), total_f1/len(data_loader), total_proba, total_label
    
def submit(model, file_name, data_loader, device):
    model.eval()
    
    results_df = pd.DataFrame()
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs = data['image']
            code,num = data['code'], data['num']

            inputs = inputs.to(device)

            outputs = model.forward(inputs)

            proba = torch.nn.Softmax(dim=1)(outputs)
            
            pred = np.argmax(np.array(proba.cpu()),1)
            label = pred_to_label(pred)
            
            
            for idx,_ in enumerate(inputs):
                row = [label[idx]]
                row_df = pd.DataFrame([row])
                results_df = pd.concat([results_df, row_df])
            
            if i%100 == 99:
                print(i)
                

    results_df.to_csv('./result/'+file_name+'.csv', header=False, index=False)
    
        
def submit_probs(model, data_loader, device):
    model.eval()

    probs = []
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs = data['image']
            code,num = data['code'], data['num']

            inputs = inputs.to(device)

            outputs = model.forward(inputs)

            prob = torch.nn.Softmax(dim=1)(outputs)
            probs.append(prob.cpu().numpy())
            
            if i%100 == 99:
                print(i)
        
        probs = np.concatenate(probs, 0)
    
    return probs
