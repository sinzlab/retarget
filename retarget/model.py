import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GraphSAGE
except:
    import warnings
    warnings.warn("PyTorch Geometric is not installed. Installing it now.")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    from torch_geometric.nn import GraphSAGE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.gcn = GraphSAGE(3, d_model, num_layers=2)

    def forward(self, src, x, edge_index):
        pe = self.gcn(x, edge_index)
        return src + pe

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.linear = nn.Linear(3, d_model)

    def forward(self, x, t_pose, edge_index, mask=None):
        src = self.linear(x)
        src = self.pos_encoder(src, t_pose, edge_index)
        output = self.transformer_encoder(src, mask)
        return output.mean(1)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.linear = nn.Linear(d_model, 32)
        self.linear2 = nn.Linear(32, 3)

    def forward(self, x, t_pose, edge_index, N=10, mask=None):
        src = x.unsqueeze(1).repeat(1, N, 1)
        src = self.pos_encoder(src, t_pose, edge_index)
        output = self.linear(src)
        output = nn.ReLU()(output)
        output = self.linear2(output)
        return output

class TransformerAutoEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerAutoEncoder, self).__init__()
        self.encoder = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.decoder = TransformerDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers)

    def forward(self, x, t_pose, edge_index, mask=None):
        output = self.encoder(x, t_pose, edge_index)
        decoded = self.decoder(output, t_pose, edge_index, N=t_pose.shape[1])
        return decoded, output