import torch
import torch.nn as nn
from modules import *

class CMTBlock(nn.Module):
   
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.temp_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.chan_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
       
        tx, _ = self.temp_attn(x, x, x)
        x = self.norm1(x + tx)
        
      
        cx = x.transpose(1, 2) 
        cx_out, _ = self.chan_attn(cx, cx, cx)
        x = self.norm2(x + cx_out.transpose(1, 2))
        
        x = self.norm3(x + self.mlp(x))
        return x

class AMCModel(nn.Module):
    def __init__(self, v_dim, a_dim, t_dim, hid_dim=512):
        super().__init__()
        self.v_enc = ProjectionLayer(v_dim, hid_dim)
        self.a_enc = ProjectionLayer(a_dim, hid_dim)
        self.t_enc = ProjectionLayer(t_dim, hid_dim)
        
        self.block_attn = BlockwiseAttention(hid_dim)
        self.amf = AMFModule(hid_dim)
        self.cmt = nn.Sequential(*[CMTBlock(hid_dim) for _ in range(4)])
        self.tsf = TemporalSignificanceFilter(hid_dim)

    def forward(self, v_feat, a_feat, t_feat):
      
        v = self.v_enc(v_feat)
        a = self.block_attn(self.a_enc(a_feat))
        t = self.t_enc(t_feat)
        
      
        v_r, a_r, t_r = self.amf(v, a, t)
        x = self.cmt(v_r + a_r + t_r)
        
       
        g_music = self.tsf(x)
        return x, g_music