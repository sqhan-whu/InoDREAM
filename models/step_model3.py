import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
from torchsummary import summary
import os 
import pandas as pd

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

set_seed(42)  # Called at the beginning of the code



def init_weights(model):
    """Reuse initialization logic from s3"""
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            if isinstance(m, nn.ReLU):  # Assuming activation function is bound
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GELU, nn.SiLU)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            # Special initialization for BiLSTM
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)  # Orthogonal initialization for recurrent weights
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Initialize forget gate bias to 1 (classic LSTM technique)
                    l = getattr(m, 'hidden_size', 128)
                    param.data[l:2*l].fill_(1.0)



    print("Model parameters initialized successfully.")


class s1_CNN(nn.Module):
    def __init__(self, input_channels=4, num_classes=2, seq_length=101):
        super().__init__()
        self.seq_length = seq_length
        
        # 1. Maintain original CNN structure (strictly named conv1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding='same'),
            nn.GELU()
        )

        # 5. Classification head (maintain high ACC design)
        self.classifier = nn.Sequential(
            nn.Linear(64 * seq_length, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

        init_weights(self)

    def forward(self, x):
        # Input processing (compatible with one-hot and integer inputs)
        x = F.one_hot(x.to(torch.int64), num_classes=4).transpose(1, 2).float()
        
        # First CNN layer (maintain structure)
        x = self.conv1(x)  # [B,64,101]


        x = x.reshape(x.size(0), -1)  # [B, 256*101]
        return self.classifier(x)



class s2_CNN_res_blocks(nn.Module):
    def __init__(self, input_channels=4, num_classes=2, seq_length=101):
        super().__init__()
        self.seq_length = seq_length
        
        # 1. Maintain original CNN structure (strictly named conv1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding='same'),
            nn.GELU()
        )
        # 2. Multi-scale residual blocks (fix dimension issues)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, dilation=d, padding='same'),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding='same'),
                nn.BatchNorm1d(64)
            ) for d in [1, 2, 4]  # Different dilation rates
        ])
        self.res_activation = nn.GELU()

        # 5. Classification head (maintain high ACC design)
        self.classifier = nn.Sequential(
            nn.Linear(64 * seq_length, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        init_weights(self)

    def forward(self, x):
        # Input processing (compatible with one-hot and integer inputs)
        x = F.one_hot(x.to(torch.int64), num_classes=4).transpose(1, 2).float()
        
        # First CNN layer (maintain structure)
        x = self.conv1(x)  # [B,64,101]

        residual = x
        for block in self.res_blocks:
            x = x + block(x)
        x = self.res_activation(x + residual)  # [B,64,101]

        x = x.reshape(x.size(0), -1)  # [B, 256*101]
        return self.classifier(x)


class s3_CNN_res_blocks_LSTM(nn.Module):
    def __init__(self, input_channels=4, num_classes=2, seq_length=101):
        super().__init__()

        self.seq_length = seq_length
        
        # 1. Maintain original CNN structure (strictly named conv1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding='same'),
            nn.GELU()
        )

        # 2. Multi-scale residual blocks (fix dimension issues)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, dilation=d, padding='same'),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding='same'),
                nn.BatchNorm1d(64)
            ) for d in [1, 2, 4]  # Different dilation rates
        ])
        self.res_activation = nn.GELU()

        # 3. Bidirectional GRU (corrected dimension handling)
        self.bilstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)

        # 5. Classification head (maintain high ACC design)
        self.classifier = nn.Sequential(
            nn.Linear(256 * seq_length, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),

        )
        init_weights(self)

    def forward(self, x):
        # Input processing (compatible with one-hot and integer inputs)
        x = F.one_hot(x.to(torch.int64), num_classes=4).transpose(1, 2).float()
        
        # First CNN layer (maintain structure)
        x = self.conv1(x)  # [B,64,101]

        residual = x
        for block in self.res_blocks:
            x = x + block(x)
        x = self.res_activation(x + residual)  # [B,64,101]

        x = x.permute(0, 2, 1)  # [B,101,64]
        x, _ = self.bilstm(x)     # [B,101,256]

        x = x.reshape(x.size(0), -1)  # [B, 256*101]
        return self.classifier(x)



class TransformerAttention(nn.Module):
    """
    Multi-head attention implementation from the paper (8 heads)
    Key modifications: Removed residual connections and LayerNorm
    Input: [batch_size, seq_len, in_features]
    Output: [batch_size, seq_len, out_features]
    """
    def __init__(self, in_features=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert in_features % num_heads == 0, "in_features必须能被num_heads整除"
        
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        
        # Q/K/V projection matrices (paper equation 8)
        self.Wq = nn.Linear(in_features, in_features)
        self.Wk = nn.Linear(in_features, in_features)
        self.Wv = nn.Linear(in_features, in_features)
        
        # Output projection matrix (paper equation 9)
        self.Wo = nn.Linear(in_features, in_features)
        
        # Only keep dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 1. Linear projection to get Q,K,V
        Q = self.Wq(x)  # [B, L, D]
        K = self.Wk(x)  # [B, L, D]
        V = self.Wv(x)  # [B, L, D]
        
        # 2. Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        
        # 3. Calculate attention scores (scaled dot product)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 4. Apply attention weights
        context = torch.matmul(attn_weights, V)  # [B, H, L, D/H]
        
        # 5. Combine multiple heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.Wo(context)  # [B, L, D]
        
        # Save attention weights for visualization
        self.last_attn_weights = attn_weights.detach()
        
        return output  # 直接返回，无残差连接


class s3_CNN_res_blocks_LSTM_SelfAttention(nn.Module):
    def __init__(self, input_channels=4, num_classes=2, seq_length=101):
        super().__init__()
        
        # Initial convolution layer (keep as original)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding='same'),
            nn.GELU(),
        )
        
        # Multi-scale residual blocks (keep as original)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, dilation=d, padding='same'),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding='same'),
                nn.BatchNorm1d(64)
            ) for d in [1, 2, 4]
        ])
        

        # 2. BiLSTM layer 
        self.bilstm = nn.LSTM(
            input_size=64, 
            hidden_size=128, 
            bidirectional=True,
            batch_first=True
        )

        # 4. Pure Transformer multi-head attention (no residual)
        self.transformer_attn = TransformerAttention(
            in_features=256,  # BiLSTM输出是2*128
            num_heads=2,
            dropout=0.1
        )

        # Classification head (adjust input dimensions)
        self.classifier = nn.Sequential(
            nn.Linear(256 * seq_length, 1024),  # 注意保持维度一致
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        # Initialize weights
        init_weights(self)

    def forward(self, x):
        # 输入处理
        x = F.one_hot(x.to(torch.int64), num_classes=4).transpose(1, 2).float()
        
        # 1. Initial convolution
        x = self.conv1(x)
        residual = x
        
        # 2. Residual blocks
        for block in self.res_blocks:
            x = x + block(x)
        x = F.gelu(x + residual)
        
        
        # Adjust dimensions [B, C, L] -> [B, L, C]
        x = x.permute(0, 2, 1)


        # 4. BiLSTM
        x, _ = self.bilstm(x)  # [B, L, 256]
        
        x = self.transformer_attn(x) 

        # Flatten for classification
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)




import torch
import torch.nn as nn
import torch.nn.functional as F

class DNACNN(nn.Module):
    def __init__(self, seq_length=201, num_classes=2):
        super(DNACNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First convolution block
            nn.Conv2d(1, 128, kernel_size=(8,4), padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Second convolution block
            nn.Conv2d(128, 128, kernel_size=(8,1), padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2,1)),
            
            # Third convolution block
            nn.Conv2d(128, 64, kernel_size=(3,1), padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Fourth convolution block
            nn.Conv2d(64, 64, kernel_size=(3,1), padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Dropout(0.5)
        )
        
        # Calculate feature map size after convolution
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_length, 4)
            dummy_output = self.conv_layers(dummy_input)
            self.flattened_size = dummy_output.numel() // dummy_output.shape[0]
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Input x shape: [batch, seq_len]
        
        # 1. one-hot encoding [batch, seq_len] -> [batch, seq_len, 4]
        x = F.one_hot(x.to(torch.int64), num_classes=4).float()
        
        # 2. Adjust dimension order [batch, seq_len, 4] -> [batch, 201, 4] (assuming seq_len=201)
        # 3. Add channel dimension [batch, 201, 4] -> [batch, 1, 201, 4]
        x = x.unsqueeze(1)  # 直接在seq_len维度前添加通道维度
        
        # 4. Convolution layer processing
        x = self.conv_layers(x)
        
        # 5. Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for RNA sequences"""
    def __init__(self, d_model, max_len=501):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AttentionPooling(nn.Module):
    """Interpretable attention pooling layer"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, L, _ = x.shape
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, -1)
        return out.mean(dim=1), attn_weights

class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3 parallel convolutions: short, medium, long scales
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(out_channels//2),
            nn.GELU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//2, kernel_size=11, padding='same'),
            nn.BatchNorm1d(out_channels//2),
            nn.GELU()
        )
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        x5 = self.conv5(x)
        x9 = self.conv9(x)
        x = torch.cat([x5, x9], dim=1)  # [B,128,L]
        return x + residual  # 残差连接

class RNAModificationModel(nn.Module):
    def __init__(self, input_channels=4, num_classes=2, seq_length=101):
        super().__init__()
        self.seq_length = seq_length
        
        # 1. Initial single-layer convolution (keep original for motif analysis)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=7, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. Added multi-scale residual block (3 parallel convolutions + residual connection)
        self.multiscale_res = nn.Sequential(
            MultiScaleResidualBlock(in_channels=256, out_channels=256),
            nn.Dropout(0.2)
        )
        
        # The rest remains the same
        self.pos_encoder = PositionalEncoding(256, max_len=seq_length)
        self.bigru = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, 
                           bidirectional=True, batch_first=True, dropout=0.2)
        self.attention = AttentionPooling(dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def forward(self, x, return_attn=False):
        # Input processing [B,L]->[B,4,L]
        x = F.one_hot(x.long(), num_classes=4).permute(0, 2, 1).float()
        
        # 1. Initial convolution (keep output for motif analysis)
        conv1_out = self.conv1(x)  # [B,128,L]
        
        # 2. Multi-scale residual block
        features = self.multiscale_res(conv1_out)  # [B,128,L]
        
        # Subsequent processing
        features = features.permute(0, 2, 1)  # [B,L,128]
        features = self.pos_encoder(features)
        features, _ = self.bigru(features)
        pooled, attn_weights = self.attention(features)
        
        logits = self.classifier(pooled)
        return (logits, attn_weights, conv1_out) if return_attn else logits
    
    def _init_weights(self):
        """Initialization scheme"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        l = param.size(0)//4
                        param.data[l:2*l].fill_(1.0)

