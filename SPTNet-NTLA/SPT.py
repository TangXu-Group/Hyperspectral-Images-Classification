import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from torch import nn,einsum

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    def forward(self,x,**kwargs):
        return x + self.fn(x,**kwargs)

class Layernorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)

class Feedforward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Spatial_pool(nn.Module):
    def __init__(self,pool_size):
        super().__init__()
        self.sp_pool = nn.Identity()
        self.pool_size = pool_size
    def forward(self,x):
        s = self.pool_size
        return self.sp_pool(x[:,:,s:-s,s:-s])


class Encoder(nn.Module):
    def __init__(self,patch_size,dim,head_dim,heads,dropout=0.1):
        super().__init__()
        
        self.dim = dim
        
        self.scale = (head_dim/heads) ** -0.5  # 1/sqrt(dim)
        self.heads = heads
        
        self.to_qkv = nn.Linear(dim,3*head_dim)
        self.mlp = nn.Linear(dim,dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
            
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.mlp(out)
        out = self.dropout(out)
        return out

class Transformer(nn.Module):
    def __init__(self,patch_size=[9,7,5,3], dim=64, hidden_dim=8, head_dim=64, heads=8, depth=4, dropout=0.1):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([])
        self.sp_pool = Spatial_pool(1)

        self.patch_size = patch_size
        for i in range(int(depth)):
            self.layers.append(nn.ModuleList([
                Residual(Layernorm(dim,Encoder(patch_size[i],dim,head_dim,heads,dropout))),
                Residual(Layernorm(dim,Feedforward(dim,hidden_dim,dropout)))
            ]))
    def forward(self, x, mask=None):
        result = []
        p = self.patch_size
        for i,(attention, mlp) in enumerate(self.layers):
            # res = self.sp_pool(x)
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block

            x = rearrange(x,'b (h w) d -> b d h w',h=p[i],w=p[i])
            if i!=self.depth-1:
                x = self.sp_pool(x)
                x = rearrange(x,'b d h w -> b (h w) d')

        return x

class SPT(nn.Module):
    def __init__(self, patch_size=9,input_dim=200, num_classes=16, dim=64, depth=4, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SPT, self).__init__()

        patch_size=[patch_size-2*i for i in range(depth)]
        
        self.dim = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1,8,(5, 1, 1),1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(8*(input_dim-4), dim, 1,1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.pos_embedding = nn.Parameter(torch.empty(1, patch_size[0]**2, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.transformer = Transformer(patch_size, dim, mlp_dim, dim, heads, depth, dropout)

        self.pool = nn.AvgPool2d(patch_size[-1])
        self.nn1 = nn.Linear(dim, num_classes)

        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x,'b c h w -> b (h w) c')
        
        x = x+self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game


        x = self.pool(x)
        x = rearrange(x,'b d h w -> b (d h w)')
        x = self.nn1(x)

        return x
