import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from torch import nn,einsum
import math

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    def forward(self,x,l,**kwargs):
        x_,l_,att = self.fn(x,l,**kwargs)
        return x + x_,l + l_,att

class Layernorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.fn = fn
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self,x,l,**kwargs):
        return self.fn(self.norm1(x),self.norm2(l),**kwargs)

class Feedforward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x,l):
        return self.net1(x),self.net2(l),None
        

class Atten(nn.Module):
    def __init__(self,num_atte,heads) -> None:
        super().__init__()
        
        self.heads = heads
        self.layers = nn.ModuleList([])
        
        if num_atte!=1:
            for i in range(heads):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(2,num_atte,1),
                        nn.BatchNorm2d(num_atte),
                        nn.LeakyReLU(),
                        nn.Conv2d(num_atte,1,1)
                    ))

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            if i==0:
                out = layer(x[:,i])
            else:
                out = torch.cat([out,layer(x[:,i])],1) 
        return out

def cc(img1, img2):
    N, C, _,_ = img1.shape

    KLloss = torch.nn.KLDivLoss(reduction="batchmean")
    img1 = img1.reshape(N, -1)
    img2 = img2.reshape(N, -1)
    img1 = F.log_softmax(img1, dim=1)
    img2 = F.softmax(img2, dim=1)
    return KLloss(img1,img2)
     
    
    
    
class Encoder(nn.Module):
    def __init__(self,dim,head_dim,heads,num_atte,dropout=0.1):
        super().__init__()
        
        self.dim = dim
        
        self.scale = (head_dim/heads) ** -0.5  # 1/sqrt(dim)
        self.heads = heads
        
        self.to_qkv = nn.Linear(dim,3*head_dim)
        self.to_qkv1 = nn.Linear(dim,3*head_dim)

        self.to_cls_token = nn.Identity()

        self.mlp = nn.Linear(dim,dim)
        self.mlp1 = nn.Linear(dim,dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.atte = Atten(num_atte,heads)

    def forward(self,x,l,mask):
        b, n, _, h = *x.shape, self.heads
        p_size = int(math.sqrt(n-1))

        q,k,v = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q,k,v])  # split into multi head attentions
        dots = (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale)  
        
        q1,k1,v1 = self.to_qkv1(l).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3  
        q1,k1,v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q1,k1,v1])  # split into multi head attentions
        dots1 = (torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale)

    
        sup = torch.stack([dots,dots1],2)
        sup = self.atte(sup)


        dots = (dots+sup)
        dots1 = (dots1+sup)

        att_loss = cc(dots,dots1)
        
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        attn1 = dots1.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v1)  # product of v times whatever inside softmax

        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out1 = rearrange(out1, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block

        out = self.mlp(out)
        out1 = self.mlp1(out1)

        out = self.dropout(out)
        out1 = self.dropout1(out1)

        return out,out1,att_loss


class Transformer_(nn.Module):
    def __init__(self, dim=64, hidden_dim=8, head_dim=64, heads=8, num_atte=8, depth=4, dropout=0.1):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([])

        for i in range(int(depth)):
            self.layers.append(nn.ModuleList([
                Residual(Layernorm(dim,Encoder(dim,head_dim,heads,num_atte,dropout))),
                Residual(Layernorm(dim,Feedforward(dim,hidden_dim,dropout)))
            ]))
    def forward(self, x,l, mask=None):
        att_loss = 0
        for i,(attention, mlp) in enumerate(self.layers):
            x,l, att_loss_ = attention(x,l, mask=mask)  # go to attention
            x,l,_ = mlp(x,l)  # go to MLP_Block
            att_loss += att_loss_

        return x,l, att_loss

