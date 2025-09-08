import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
#from tqdm import tqdm
from termcolor import colored
from utils import func as sq
import torch.nn.functional as F
import os
import random
from torch.optim import AdamW



seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



def train_model(net, train_loader, test_loader, params, outdir, device):
    os.makedirs(outdir, exist_ok=True)
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    opt = optim.Adam(net.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    criterion_CE = nn.CrossEntropyLoss()
    best_acc = 0
    
    for epoch in range(params['epoch']):        
        net.train()
        loss_ls = []
        t0 = time.time()

        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            outputs = net(seq)
            loss = criterion_CE(outputs, label)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_ls.append(loss.item())

            
        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data = sq.evalue_model(net, train_loader, device)
            test_performance, test_roc_data, test_prc_data = sq.evalue_model(net, test_loader, device)
            
        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Train Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
        + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC\tAP]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            train_performance[0], train_performance[1], train_performance[2], train_performance[3],
            train_performance[4], train_performance[5],train_performance[6]) + '\n' + '=' * 60
        print(results)
        test_acc = test_performance[0]
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            
            test_results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
            test_results += f'Train: {test_performance[0]:.4f}, time: {time.time() - t0:.2f}'
            test_results += '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(epoch + 1) + '=' * 16 \
            + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC\tAP]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[1], test_performance[2], test_performance[3],
            test_performance[4], test_performance[5],test_performance[6]) + '\n' + '=' * 60

            model_name = net.__class__.__name__
            torch.save(net.state_dict(), outdir+'/'+ model_name+'_best_model.pth')
            print(f'Saved new best model at epoch {best_epoch}')
            print(test_results)

