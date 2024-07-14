import torch
from einops import rearrange
from torch import nn


class Fuse(nn.Module):
    def __init__(self, dim, num_heads, num_classes, dropout):
        super(Fuse, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.scale = (dim/num_heads) ** -0.5  # 1/sqrt(dim)

        self.q = nn.Linear(num_classes, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        self.q1 = nn.Linear(num_classes, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)
        self.output = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, l, score1,score2):
        h = self.num_heads

        q = self.q(score2)[:,None].repeat(1,x.shape[1],1) 
        k = self.k(x)  
        v = self.v(x)   
       
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q,k,v])

        dots = (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale)  
        attn = dots.softmax(dim=-1)  
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  
        out = rearrange(out, 'b h n d -> b n (h d)') 


        q1 = self.q1(score1)[:,None].repeat(1,x.shape[1],1)  # [1, batch_size, hidden_size]
        k1 = self.k1(l)  # [1, batch_size, hidden_size]
        v1 = self.v1(l)  # [1, batch_size, hidden_size]

        
        q1,k1,v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q1,k1,v1])

        dots1 = (torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale)  
        attn1 = dots1.softmax(dim=-1)  
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v1)  
        out1 = rearrange(out1, 'b h n d -> b n (h d)')  

        out = torch.cat((out+x, out1+l), dim=-1) 
        
        return out   