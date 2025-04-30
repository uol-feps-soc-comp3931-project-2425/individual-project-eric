import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

class TimeAwarePositionalEncoding(nn.Module):
    """
    Custom positional encoding that incorporates temporal information such as
    hour of day and day of week to capture temporal patterns in cryptocurrency markets.
    """
    def __init__(self, d_model, max_seq_len=120, dropout=0.1):
        super(TimeAwarePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        self.hour_embedding = nn.Embedding(24, d_model // 4)
        self.day_embedding = nn.Embedding(7, d_model // 4)
        self.time_projection = nn.Linear(d_model // 2, d_model)
        
    def forward(self, x, time_features):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            time_features: Tensor containing hour and day information [batch_size, seq_len, 2]
                          time_features[:,:,0] = hour of day (0-23)
                          time_features[:,:,1] = day of week (0-6)
        """
        batch_size, seq_len, _ = x.shape
        
        pos_encoding = self.pe[:, :seq_len, :]
        
        hour_embed = self.hour_embedding(time_features[:,:,0].long())
        day_embed = self.day_embedding(time_features[:,:,1].long())
        
        time_embed = torch.cat([hour_embed, day_embed], dim=-1)
        time_embed = self.time_projection(time_embed)
        
        time_aware_pe = pos_encoding + time_embed.unsqueeze(1)
        
        x = x + time_aware_pe
        
        return self.dropout(x)

class MultiScaleFeatureExtraction(nn.Module):
    """
    Processes the input sequence at multiple resolutions to capture patterns
    at different time scales.
    """
    def __init__(self, input_dim, d_model):
        super(MultiScaleFeatureExtraction, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, d_model // 4, kernel_size=1)
        self.conv3 = nn.Conv1d(input_dim, d_model // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, d_model // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, d_model // 4, kernel_size=7, padding=3)
        
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        """
        x_conv = x.transpose(1, 2)
        
        x1 = F.gelu(self.conv1(x_conv))
        x3 = F.gelu(self.conv3(x_conv))
        x5 = F.gelu(self.conv5(x_conv))
        x7 = F.gelu(self.conv7(x_conv))
        
        x_multi = torch.cat([x1, x3, x5, x7], dim=1)
        
        x_multi = x_multi.transpose(1, 2)
        
        return self.projection(x_multi)

class CrossCryptocurrencyAttention(nn.Module):
    """
    Modified attention mechanism that computes attention scores between
    features of different cryptocurrencies.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossCryptocurrencyAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_btc = nn.Linear(d_model, d_model)
        self.k_btc = nn.Linear(d_model, d_model)
        self.v_btc = nn.Linear(d_model, d_model)
        
        self.q_eth = nn.Linear(d_model, d_model)
        self.k_eth = nn.Linear(d_model, d_model)
        self.v_eth = nn.Linear(d_model, d_model)
        
        self.q_sol = nn.Linear(d_model, d_model)
        self.k_sol = nn.Linear(d_model, d_model)
        self.v_sol = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        """Split the last dimension into (n_heads, head_dim)"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch_size, n_heads, seq_len, head_dim]
    
    def combine_heads(self, x):
        """Combine the heads back into original shape"""
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3)  # [batch_size, seq_len, n_heads, head_dim]
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, x_btc, x_eth, x_sol):
        """
        Compute cross-cryptocurrency attention
        
        Args:
            x_btc: BTC features [batch_size, seq_len, d_model]
            x_eth: ETH features [batch_size, seq_len, d_model]
            x_sol: SOL features [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x_btc.shape
        
        q_btc = self.split_heads(self.q_btc(x_btc))
        k_btc = self.split_heads(self.k_btc(x_btc))
        v_btc = self.split_heads(self.v_btc(x_btc))
        
        q_eth = self.split_heads(self.q_eth(x_eth))
        k_eth = self.split_heads(self.k_eth(x_eth))
        v_eth = self.split_heads(self.v_eth(x_eth))
        
        q_sol = self.split_heads(self.q_sol(x_sol))
        k_sol = self.split_heads(self.k_sol(x_sol))
        v_sol = self.split_heads(self.v_sol(x_sol))
        
        attn_btc_eth = torch.matmul(q_btc, k_eth.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_btc_sol = torch.matmul(q_btc, k_sol.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        attn_eth_btc = torch.matmul(q_eth, k_btc.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_eth_sol = torch.matmul(q_eth, k_sol.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        attn_sol_btc = torch.matmul(q_sol, k_btc.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_sol_eth = torch.matmul(q_sol, k_eth.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        attn_btc_btc = torch.matmul(q_btc, k_btc.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_eth_eth = torch.matmul(q_eth, k_eth.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_sol_sol = torch.matmul(q_sol, k_sol.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        attn_btc_btc = F.softmax(attn_btc_btc, dim=-1)
        attn_btc_eth = F.softmax(attn_btc_eth, dim=-1)
        attn_btc_sol = F.softmax(attn_btc_sol, dim=-1)
        
        attn_eth_btc = F.softmax(attn_eth_btc, dim=-1)
        attn_eth_eth = F.softmax(attn_eth_eth, dim=-1)
        attn_eth_sol = F.softmax(attn_eth_sol, dim=-1)
        
        attn_sol_btc = F.softmax(attn_sol_btc, dim=-1)
        attn_sol_eth = F.softmax(attn_sol_eth, dim=-1)
        attn_sol_sol = F.softmax(attn_sol_sol, dim=-1)
        
        attn_btc_btc = self.dropout(attn_btc_btc)
        attn_btc_eth = self.dropout(attn_btc_eth)
        attn_btc_sol = self.dropout(attn_btc_sol)
        
        attn_eth_btc = self.dropout(attn_eth_btc)
        attn_eth_eth = self.dropout(attn_eth_eth)
        attn_eth_sol = self.dropout(attn_eth_sol)
        
        attn_sol_btc = self.dropout(attn_sol_btc)
        attn_sol_eth = self.dropout(attn_sol_eth)
        attn_sol_sol = self.dropout(attn_sol_sol)
        
        out_btc_btc = torch.matmul(attn_btc_btc, v_btc)
        out_btc_eth = torch.matmul(attn_btc_eth, v_eth)
        out_btc_sol = torch.matmul(attn_btc_sol, v_sol)
        
        out_eth_btc = torch.matmul(attn_eth_btc, v_btc)
        out_eth_eth = torch.matmul(attn_eth_eth, v_eth)
        out_eth_sol = torch.matmul(attn_eth_sol, v_sol)
        
        out_sol_btc = torch.matmul(attn_sol_btc, v_btc)
        out_sol_eth = torch.matmul(attn_sol_eth, v_eth)
        out_sol_sol = torch.matmul(attn_sol_sol, v_sol)
        
        out_btc = out_btc_btc + out_btc_eth + out_btc_sol
        out_eth = out_eth_btc + out_eth_eth + out_eth_sol
        out_sol = out_sol_btc + out_sol_eth + out_sol_sol
        
        out_btc = self.fc_out(self.combine_heads(out_btc))
        out_eth = self.fc_out(self.combine_heads(out_eth))
        out_sol = self.fc_out(self.combine_heads(out_sol))
        
        return out_btc, out_eth, out_sol

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network as described in the Transformer paper.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with multi-head self-attention, layer normalization,
    position-wise feed-forward network, and residual connections.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.cross_attn = CrossCryptocurrencyAttention(d_model, n_heads, dropout)
        self.norm1_btc = nn.LayerNorm(d_model)
        self.norm1_eth = nn.LayerNorm(d_model)
        self.norm1_sol = nn.LayerNorm(d_model)
        
        self.ff_btc = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_eth = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_sol = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm2_btc = nn.LayerNorm(d_model)
        self.norm2_eth = nn.LayerNorm(d_model)
        self.norm2_sol = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_btc, x_eth, x_sol):
        attn_btc, attn_eth, attn_sol = self.cross_attn(x_btc, x_eth, x_sol)
        
        x_btc = self.norm1_btc(x_btc + self.dropout(attn_btc))
        x_eth = self.norm1_eth(x_eth + self.dropout(attn_eth))
        x_sol = self.norm1_sol(x_sol + self.dropout(attn_sol))
        
        ff_btc = self.ff_btc(x_btc)
        ff_eth = self.ff_eth(x_eth)
        ff_sol = self.ff_sol(x_sol)
        
        x_btc = self.norm2_btc(x_btc + self.dropout(ff_btc))
        x_eth = self.norm2_eth(x_eth + self.dropout(ff_eth))
        x_sol = self.norm2_sol(x_sol + self.dropout(ff_sol))
        
        return x_btc, x_eth, x_sol

class CryptoTransformer(nn.Module):
    """
    Complete Transformer model for cryptocurrency price prediction.
    """
    def __init__(self, 
                 input_dim, 
                 d_model=128, 
                 n_heads=8, 
                 n_layers=3, 
                 d_ff=512, 
                 max_seq_len=120, 
                 dropout=0.1):
        super(CryptoTransformer, self).__init__()
        
        self.feature_extraction = MultiScaleFeatureExtraction(input_dim, d_model)
        
        self.pos_encoding = TimeAwarePositionalEncoding(d_model, max_seq_len, dropout)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        
        self.output_btc = nn.Linear(64, 1)
        self.output_eth = nn.Linear(64, 1)
        self.output_sol = nn.Linear(64, 1)
        
    def forward(self, x_btc, x_eth, x_sol, time_features):
        """
        Forward pass through the Transformer model.
        
        Args:
            x_btc: BTC features [batch_size, seq_len, input_dim]
            x_eth: ETH features [batch_size, seq_len, input_dim]
            x_sol: SOL features [batch_size, seq_len, input_dim]
            time_features: Temporal features [batch_size, seq_len, 2]
        """
        x_btc = self.feature_extraction(x_btc)
        x_eth = self.feature_extraction(x_eth)
        x_sol = self.feature_extraction(x_sol)
        
        x_btc = self.pos_encoding(x_btc, time_features)
        x_eth = self.pos_encoding(x_eth, time_features)
        x_sol = self.pos_encoding(x_sol, time_features)
        
        for encoder_block in self.encoder_blocks:
            x_btc, x_eth, x_sol = encoder_block(x_btc, x_eth, x_sol)
        
        x_btc = x_btc.transpose(1, 2)
        x_eth = x_eth.transpose(1, 2)
        x_sol = x_sol.transpose(1, 2)
        
        x_btc = self.global_avg_pool(x_btc).squeeze(-1)
        x_eth = self.global_avg_pool(x_eth).squeeze(-1)
        x_sol = self.global_avg_pool(x_sol).squeeze(-1)
        
        x_btc = F.gelu(self.fc1(x_btc))
        x_btc = self.dropout(x_btc)
        x_btc = F.gelu(self.fc2(x_btc))
        
        x_eth = F.gelu(self.fc1(x_eth))
        x_eth = self.dropout(x_eth)
        x_eth = F.gelu(self.fc2(x_eth))
        
        x_sol = F.gelu(self.fc1(x_sol))
        x_sol = self.dropout(x_sol)
        x_sol = F.gelu(self.fc2(x_sol))
        
        out_btc = self.output_btc(x_btc)
        out_eth = self.output_eth(x_eth)
        out_sol = self.output_sol(x_sol)
        
        return out_btc, out_eth, out_sol

class DirectionalAccuracyLoss(nn.Module):
    """
    Custom loss function that combines MSE and directional accuracy.
    """
    def __init__(self, alpha=0.7):
        super(DirectionalAccuracyLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, prev):
        """
        Args:
            pred: Predicted values
            target: Target values
            prev: Previous values used to calculate direction
        """
        mse_loss = self.mse(pred, target)
        
        pred_direction = torch.sign(pred - prev)
        target_direction = torch.sign(target - prev)
        
        dir_accuracy = (pred_direction == target_direction).float().mean()
        
        loss = self.alpha * mse_loss + (1 - self.alpha) * (1 - dir_accuracy)
        
        return loss

def train_crypto_transformer(model, train_loader, val_loader, epochs=100, lr=1e-4, weight_decay=0.01):
    """
    Training function for the CryptoTransformer model.
    
    Args:
        model: CryptoTransformer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW optimizer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = DirectionalAccuracyLoss(alpha=0.7)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data_btc, data_eth, data_sol, time_features, prev_btc, prev_eth, prev_sol, target_btc, target_eth, target_sol) in enumerate(train_loader):
            data_btc, data_eth, data_sol = data_btc.to(device), data_eth.to(device), data_sol.to(device)
            time_features = time_features.to(device)
            prev_btc, prev_eth, prev_sol = prev_btc.to(device), prev_eth.to(device), prev_sol.to(device)
            target_btc, target_eth, target_sol = target_btc.to(device), target_eth.to(device), target_sol.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                pred_btc, pred_eth, pred_sol = model(data_btc, data_eth, data_sol, time_features)
                
                loss_btc = criterion(pred_btc, target_btc, prev_btc)
                loss_eth = criterion(pred_eth, target_eth, prev_eth)
                loss_sol = criterion(pred_sol, target_sol, prev_sol)
                
                loss = (loss_btc + loss_eth + loss_sol) / 3
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data_btc, data_eth, data_sol, time_features, prev_btc, prev_eth, prev_sol, target_btc, target_eth, target_sol in val_loader:
                data_btc, data_eth, data_sol = data_btc.to(device), data_eth.to(device), data_sol.to(device)
                time_features = time_features.to(device)
                prev_btc, prev_eth, prev_sol = prev_btc.to(device), prev_eth.to(device), prev_sol.to(device)
                target_btc, target_eth, target_sol = target_btc.to(device), target_eth.to(device), target_sol.to(device)
                
                pred_btc, pred_eth, pred_sol = model(data_btc, data_eth, data_sol, time_features)
                
                loss_btc = criterion(pred_btc, target_btc, prev_btc)
                loss_eth = criterion(pred_eth, target_eth, prev_eth)
                loss_sol = criterion(pred_sol, target_sol, prev_sol)
                
                loss = (loss_btc + loss_eth + loss_sol) / 3
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_crypto_transformer.pt')
            print(f'Model saved with validation loss: {val_loss:.6f}')
    
    return model

if __name__ == "__main__":
    input_dim = 32  # Number of features per cryptocurrency
    d_model = 128
    n_heads = 8
    n_layers = 3
    d_ff = 512
    max_seq_len = 120
    dropout = 0.2
    
    model = CryptoTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
