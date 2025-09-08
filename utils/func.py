import sys
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')  
import os 
from sklearn.metrics import (accuracy_score, roc_curve, auc, 
                            precision_recall_curve, average_precision_score,
                            recall_score, matthews_corrcoef, confusion_matrix)
matplotlib.use('Agg')  
import random
seed = 42  
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


sq = sys.modules[__name__]

def read_fasta(fasta_file):
    """Read sequences and labels from FASTA file
    Args:
        fasta_file: Path to FASTA file
    Returns:
        tuple: (sequences, labels) where sequences is a list of DNA sequences
              and labels is a list of 0/1 labels (0 for negative, 1 for positive)
    """
    seq = []
    label = []
    with open(fasta_file) as fasta:
        for line in fasta:
            line = line.replace('\n', '')
            if line.startswith('>'):
                if 'neg' in line:
                    label.append(0)
                else:
                    label.append(1)
            else:
                seq.append(line.replace('U', 'T'))
    return seq, label

def encode_sequence_1mer(sequences, max_seq):
    k = 1
    overlap = False

    all_kmer = [''.join(p) for p in itertools.product(['A', 'C', 'G', 'T', 'N'], repeat=k)]
    kmer_dict = {all_kmer[i]: i for i in range(len(all_kmer))}

    encoded_sequences = []
    if overlap:
        max_length = max_seq - k + 1

    else:
        max_length = max_seq // k

    for seq in sequences:
        encoded_seq = []
        start_site = len(seq) // 2 - max_length // 2
        for i in range(start_site, start_site + max_length, k):
            encoded_seq.append(kmer_dict[seq[i:i+k]])

        encoded_sequences.append(encoded_seq+[0]*(max_length-len(encoded_seq)))
    return np.array(encoded_sequences)

def load_data_fasta(fasta_file, batch_size, max_seq, shuffle=True):
    max_seq = max_seq
    seq, label = read_fasta(fasta_file)
    seq_1mer = encode_sequence_1mer(seq, max_seq)
    seq_1mer_dataset = TensorDataset(torch.tensor(seq_1mer), torch.tensor(label))
    seq_1mer_dataset_loader = DataLoader(seq_1mer_dataset, batch_size=batch_size, shuffle=shuffle)
    return seq_1mer_dataset_loader

def evalue_model(model, test_iter, device):

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_preds_probs = []

    with torch.no_grad():
        for x, labels in test_iter:
            inputs = x.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            probs = outputs[:, 1]
    
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds_probs.extend(probs.cpu().numpy())
            

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_preds_probs)        
    
    

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)  
    
    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs, pos_label=1)
    
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds)  # Sensitivity = Recall
    specificity = tn / (tn + fp) 
    mcc = matthews_corrcoef(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_probs, average='macro', pos_label=1, sample_weight=None)
    
    ACC = float(tp + tn) / len(all_preds)
    Sensitivity_Recall = Sensitivity = float(tp) / (tp + fn)
    Specificity = float(tn) / (tn + fp)
    BACC = 0.5 * Sensitivity + 0.5 * Specificity
    #MCC = float(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    AP = average_precision_score(all_labels, all_probs, average='macro', pos_label=1, sample_weight=None)
    
    performance = [ACC, BACC, Sensitivity_Recall, Specificity, mcc, roc_auc, AP]
    roc_data = [fpr, tpr]
    prc_data = [recall, precision]
    

    return performance, roc_data, prc_data




########plot ROC$###########
def plot_roc(fpr, tpr, roc_auc, save_path):
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

#precision, recall

def plot_prc(recall, precision, AP, save_path):
    fig = plt.figure()
    plt.plot(recall,precision, color='darkblue', lw=2, label=f'PR curve (AP = {AP:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


#valid_performance

def plot_performance(valid_performance, save_path):

    metrics = ['ACC', 'SE', 'SP', 'MCC', 'AUC']
    values = [valid_performance[0], valid_performance[2], valid_performance[3], 
          valid_performance[4], valid_performance[5]]

    fig = plt.figure()

    bars = plt.bar(metrics, values, width=0.4, color='skyblue', edgecolor='black', linewidth=1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')

    plt.title('Model Performance Metrics', fontsize=14)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)

    plt.ylim(0, 1.1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    plt.tick_params(axis='both', which='both', right=False, top=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


