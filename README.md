# InoDREAM: A Deep Learning Model for RNA Inosine Site Prediction

## Project Overview
InoDREAM is a deep learning-based tool for predicting RNA inosine modification sites. This project utilizes convolutional neural networks (CNN) and recurrent neural networks (RNN) to accurately predict inosine modification sites by analyzing RNA sequence features. This tool is suitable for RNA epigenetics research and can help researchers quickly identify potential inosine modification sites.

## Core Features
- Deep learning-based RNA inosine site prediction
- Multiple model architecture options (CNN, Residual Network, LSTM, etc.)
- Model training, evaluation, and visualization
- Sequence motif analysis

## Installation Guide

### Environment Requirements
 - matplotlib==3.7.1
 - numpy==1.23.5
 - optuna==4.4.0
 - pandas==2.0.0
 - scikit_learn==1.3.0
 - scipy==1.10.1
 - seaborn==0.13.2
 - termcolor==3.1.0
 - torch==2.8.0
 - torchsummary==1.5.1
 - tqdm==4.67.1


### Installation Steps
1. Clone the project repository
```bash
git clone https://github.com/sqhan-whu/InoDREAM.git
cd InoDREAM
```

2. Create and activate a virtual environment (optional but recommended)
```bash
conda create -n inodream python=3.8
conda activate inodream
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
Prepare RNA sequence data in FASTA format, divided into training, validation, and test sets. Ensure that sequence labels are identified in the FASTA headers (containing "neg" indicates negative samples, others indicate positive samples).

### Model Training
```bash
python main.py train -t data/train.fa -v data/valid.fa -e 20 -a RNAModificationModel -l 0.0001 -b 128 -s 101 -o save_model
```
Parameter explanations:
- `-t`, `--train`: Training data path
- `-v`, `--valid`: Validation data path
- `-e`, `--epochs`: Number of training epochs
- `-a`, `--model-name`: Model name
- `-l`, `--lr`: Learning rate
- `-b`, `--batch-size`: Batch size
- `-s`, `--max-seq`: Maximum sequence length
- `-o`, `--output`: Model saving directory

### Model Evaluation
```bash
python main.py eval -m save_model/RNAModificationModel_best_model.pth -d data/test.fa -p
```
Parameter explanations:
- `-m`, `--model`: Model weight path
- `-d`, `--data`: Test data path
- `-p`, `--plt`: Whether to plot ROC and PR curves

Parameter explanations:
- `-m`, `--model`: Model weight path
- `-d`, `--data`: Input data path
- `-o`, `--output`: Output directory

## Model Architectures
This project provides multiple model architecture options, ranging from simple to complex:

1. `s1_CNN`: Basic CNN model with one convolutional layer and fully connected layer
2. `s2_CNN_res_blocks`: CNN model with multi-scale residual blocks
3. `s3_CNN_res_blocks_LSTM`: Hybrid model combining CNN residual blocks and bidirectional LSTM
4. `RNAModificationModel`: Advanced model fusing CNN, residual network, and self-attention mechanism

All model definitions are located in the `models/` directory.

## Evaluation Metrics
Model evaluation metrics include:
- Accuracy (ACC)
- Balanced Accuracy (BACC)
- Sensitivity (Recall)
- Specificity
- Matthews Correlation Coefficient (MCC)
- ROC curve and AUC
- PR curve and AP

Evaluation results are saved in the specified output directory, including CSV files and visualization charts.

## Directory Structure
```
InoDREAM/
├── data/                 # Data directory
├── models/               # Model definitions
├── save_model/           # Model saving path
├── utils/                # Utility functions
├── main.py               # Main program entry
├── train.py              # Training script
└── README.md             # English README
```

## Citation
If you use this tool, please cite our paper:
> [Deciphering the regulatory code of RNA inosine through enzymatic precision mapping and explainable deep learning model] [2025]
