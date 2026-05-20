import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionLayer(nn.Module):
   
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d = x.shape
        x = x.reshape(b * n, d)
        x = self.net(x)
        return x.reshape(b, n, -1)

class BlockwiseAttention(nn.Module):
   
    def __init__(self, dim, heads=8, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        b, n, d = x.shape
        pad_len = (self.block_size - n % self.block_size) % self.block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        num_blocks = x.shape[1] // self.block_size
        x_blocked = x.reshape(b * num_blocks, self.block_size, d)
        attn_out, _ = self.attn(x_blocked, x_blocked, x_blocked)
        
        out = attn_out.reshape(b, -1, d)
        return out[:, :n, :]

class AMFModule(nn.Module):
  
    def __init__(self, dim, temperature=0.07):
        super().__init__()
        self.temp = temperature
        self.phi = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, v, a, t):
        modalities = torch.stack([v, a, t], dim=1) # [B, 3, N, D]
        s = self.phi(modalities) # [B, 3, N, 1]
        w = F.softmax(s / self.temp, dim=1) # Eq 6
        
        rectified = modalities * w # Eq 7
        return rectified.unbind(dim=1)

class TemporalSignificanceFilter(nn.Module):
   
    def __init__(self, dim):
        super().__init__()
        self.alpha_net = nn.Linear(dim, 1)

    def forward(self, x):
        weights = F.softmax(self.alpha_net(x), dim=1) # [B, N, 1]
        g = torch.sum(weights * x, dim=1) # Eq 3
        return g