#!/usr/bin/env python3
import argparse
from utils import func as sq
import torch
from models import step_model3
from train import train_model
import random
import numpy as np
import torch
from torchsummary import summary
import os 
import pandas as pd
seed = 42  
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


MODELS = {
    "s1_CNN": step_model3.s1_CNN,
    "s2_CNN_res_blocks": step_model3.s2_CNN_res_blocks,
    "s3_CNN_res_blocks_LSTM": step_model3.s3_CNN_res_blocks_LSTM,
    "s3_CNN_res_blocks_LSTM_SelfAttention": step_model3.s3_CNN_res_blocks_LSTM_SelfAttention,
    "RNAModificationModel": step_model3.RNAModificationModel,   
}
#model_key = 'CNN_res_blocks_GRU'

#model_key = args.model_name

#model_name = MODELS[model_key].__name__


def load_model(model_name: str, **kwargs):

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name](**kwargs) 



def main():
    parser = argparse.ArgumentParser(description='DNA sequence classification and motif analysis tool', usage='dna-cnn <command> [options]')
    
    subparsers = parser.add_subparsers(dest='command', required=True, metavar='<command>')


    train_parser = subparsers.add_parser('train', help='训练模型', usage='\n%run ./cli.py train -t data/train.fasta -v data/valid.fasta -e 10')

    train_parser.add_argument('-p', '--print-model', action='store_true', help='仅打印模型结构，跳过训练')
    train_parser.add_argument('-a', '--model-name',  default='SequenceCNN', help='test models')
    train_parser.add_argument('-t', '--train',  metavar='FILE', help='训练数据路径(FASTA格式)')
    train_parser.add_argument('-v', '--valid',  metavar='FILE', help='验证数据路径(FASTA格式)')
    train_parser.add_argument('-e', '--epochs', type=int, default=20, metavar='INT', help='训练轮数')
    train_parser.add_argument('-l', '--lr', type=float, default=0.0001, metavar='FLOAT', help='学习率')
    train_parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='INT', help='批量大小')
    train_parser.add_argument('-s', '--max-seq', type=int, default=101, metavar='INT', help='序列最大长度')
    train_parser.add_argument('-m', '--model-arch', default='SequenceCNN', help='模型架构')
    train_parser.add_argument('-o', '--output', default='save_model', metavar='DIR', help='输出目录')

    eval_parser = subparsers.add_parser('eval', help='评估模型', usage = '\n %run ./cli.py eval -m save_model/best_model.pth -d data/test.fasta -p')
    eval_parser.add_argument('-a', '--model-name',  default='SequenceCNN', help='test models')
    eval_parser.add_argument('-m', '--model', required=True, metavar='FILE', help='模型权重路径')
    eval_parser.add_argument('-d', '--data', required=True, metavar='FILE', help='测试数据路径(FASTA格式)')
    eval_parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='INT', help='批量大小(128)')
    eval_parser.add_argument('-s', '--max-seq', type=int, default=101, metavar='INT', help='序列最大长度(201)')
    eval_parser.add_argument('-o', '--output', default='save_model', metavar='DIR', help='输出目录')
    eval_parser.add_argument('-p', '--plt', action='store_true', help='是否绘图')


    args = parser.parse_args()

    if args.command == 'train':
        model_key = args.model_name
        
        model_name = MODELS[model_key].__name__

        model = load_model(model_key, seq_length=args.max_seq)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device) 

        if args.print_model:
            summary(model, input_size=(args.max_seq,))

        else:
            if not all([args.train, args.valid]):
                train_parser.error("The training and validation data paths must be provided (unless using -p to only view the model).")
            
            train_loader = sq.load_data_fasta(args.train, args.batch_size, args.max_seq)
            valid_loader = sq.load_data_fasta(args.valid, args.batch_size, args.max_seq, shuffle=False)
            params = {'epoch': args.epochs, 'lr': args.lr}
            train_model(model, train_loader, valid_loader, params, args.output, device)  


    elif args.command == 'eval':

        model_key = args.model_name
        model_name = MODELS[model_key].__name__

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        device = 'cpu'
        model = load_model(model_key, seq_length=args.max_seq).to(device)
        model.load_state_dict(torch.load(args.model,weights_only=True, map_location=device))
        test_loader = sq.load_data_fasta(args.data, args.batch_size, args.max_seq, shuffle=False)
        performance, roc_data, prc_data = sq.evalue_model(model, test_loader,device)

        print(performance)
        columns = ["ACC", "BACC","Sensitivity_Recall", "Specificity", "mcc", "roc_auc", "AP"]
        df = pd.DataFrame([performance], columns=columns)
        

        # 1. save performance
        csv_path_p = os.path.join(args.output, f"{model_name}_performance_metrics.csv")
        plot_path_p = os.path.join(args.output, f"{model_name}_performance_metrics.pdf")
        df.to_csv(csv_path_p, index=False)
        print(f"performance save: {csv_path_p}")
        if args.plt:
            sq.plot_performance(performance, plot_path_p)
            print(f"performance save: {plot_path_p}")

        # 2. save roc_data
        csv_path_roc = os.path.join(args.output,f"{model_name}_roc_curve.csv")
        plot_path_roc = os.path.join(args.output,f"{model_name}_roc_curve.pdf")
        pd.DataFrame({"FPR": roc_data[0], "TPR": roc_data[1]}).to_csv(csv_path_roc, index=False)
        print(f"roc_data save: {csv_path_roc}")
        if args.plt:
            sq.plot_roc(roc_data[0], roc_data[1], performance[5], plot_path_roc)
            print(f"roc_plot save: {plot_path_roc}")   

        # 3. save prc_data
        csv_path_prc = os.path.join(args.output,f"{model_name}_pr_curve.csv")
        plot_path_prc = os.path.join(args.output,f"{model_name}_pr_curve.pdf")
        pd.DataFrame({"Recall": prc_data[0], "Precision": prc_data[1]}).to_csv(csv_path_prc, index=False)
        print(f"prc_data save: {csv_path_prc}")
        if args.plt:
            sq.plot_prc(prc_data[0], prc_data[1], performance[6], plot_path_prc)
            print(f"prc_plot save: {plot_path_prc}")



if __name__ == '__main__':
    main()
