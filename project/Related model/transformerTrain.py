import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np 

print("PyTorch imports complete.")

n_features = 10 
n_timesteps = 60 
n_outputs = 3    

d_model = 64          
nhead = 4             
num_encoder_layers = 2 
dim_feedforward = 128 
dropout = 0.2         

learning_rate = 1e-4
weight_decay = 1e-2



print("\n--- Starting Defining Transformer Model ---")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
             pe[:, 1::2] = torch.cos(position * div_term)
        else:
             pe[:, 1::2] = torch.cos(position * div_term[:-1]) 

        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_dim, sequence_length):
        super(TransformerPredictor, self).__init__()
        if d_model % nhead != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def forward(self, src):
        # src shape: [batch_size, sequence_length, input_dim]
        src = self.input_projection(src) * math.sqrt(self.d_model)
        # Permute for Positional Encoding: [sequence_length, batch_size, d_model]
        src_permuted = src.permute(1, 0, 2)
        src_permuted = self.pos_encoder(src_permuted)
        src = src_permuted.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        # output shape: [batch_size, sequence_length, d_model]
        output = output[:, -1, :] 
        output = self.output_layer(output) 
        return output

transformer_model = TransformerPredictor(input_dim=n_features,
                                       d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       output_dim=n_outputs,
                                       sequence_length=n_timesteps)

print("\nTransformer Model Structure:")
print(transformer_model)
num_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
print(f"\nApproximate number of trainable parameters: {num_params}")


print("\n--- Starting etting up Loss and Optimizer for Transformer ---")


criterion = nn.MSELoss()
print(f"Loss function set to: {criterion}")

optimizer = optim.AdamW(transformer_model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay)
print(f"Optimizer set to: AdamW with lr={learning_rate}, weight_decay={weight_decay}")



print("\n====== TRANSFORMER MODEL DEFINITION & SETUP FINISHED ======")
print("Transformer model ('transformer_model'), loss function ('criterion'),")
print("and optimizer ('optimizer') are now defined.")
print("Next steps: Prepare PyTorch DataLoaders, write the training loop (including loss calculation, backpropagation, optimizer steps), and evaluation loop.")