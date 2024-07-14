import torch
from einops import rearrange
import torch.nn.functional as F
from torch import nn
import math
from Transformer_ import Transformer_
from fuse import Fuse

def modu(score1,score2,label,alpha=0.1):
    tanh = nn.Tanh()
    softmax = nn.Softmax(dim=-1)
    
    score_1 = sum([softmax(score1)[i][label[i]] for i in range(label.size(0))])
    score_2 = sum([softmax(score2)[i][label[i]] for i in range(label.size(0))])

    ratio_1 = score_1 / score_2
    ratio_2 = 1 / ratio_1
    if ratio_1 > 1:
        coeff_1 = 1-tanh(alpha * ratio_1).detach()
        coeff_2 = 1
    else:
        coeff_2 = 1-tanh(alpha * ratio_2).detach()
        coeff_1 = 1
    return coeff_1,coeff_2


class MICF_Net(nn.Module):
    def __init__(self, patch_size=9, dim=64, input_dim=144, num_classes=15, dep=2,heads=8, num_atte=8, mlp_dim=8, alpha=0.1, dropout=0.1, emb_dropout=0.1):
        super(MICF_Net, self).__init__()

        self.dim = dim
        self.alpha = alpha
        
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
        self.conv2d_lidar = nn.Sequential(
            nn.Conv2d(1, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.conv2d_lidar2 = nn.Sequential(
            nn.Conv2d(32, dim, 1,1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
        )

        self.dropout1 = nn.Dropout(emb_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)

        self.pos_embedding1 = nn.Parameter(torch.empty(1, 1+patch_size**2, dim))
        torch.nn.init.normal_(self.pos_embedding1, std=.02)
        self.pos_embedding2 = nn.Parameter(torch.empty(1, 1+patch_size**2, dim))
        torch.nn.init.normal_(self.pos_embedding2, std=.02)
        

        self.transformer1 = Transformer_(dim, mlp_dim, dim, heads, num_atte, dep, dropout)
        
        self.fuse = Fuse(dim,heads,num_classes,dropout)  ########语义引导的融合
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.nn = nn.Linear(2*dim, num_classes)
        self.to_cls_token = nn.Identity()

    def forward(self, x, lidar, x_proto, l_proto, label, epoch, mask=None):

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x,'b c h w -> b (h w) c')
        
        lidar = self.conv2d_lidar(lidar)
        lidar = self.conv2d_lidar2(lidar)
        
        lidar = rearrange(lidar,'b c h w -> b (h w) c')

        cls_tokens1 = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_tokens1,x],1)
        cls_tokens2 = self.cls_token.expand(lidar.shape[0],-1,-1)
        lidar = torch.cat([cls_tokens2,lidar],1)
        
        x = x+self.pos_embedding1
        lidar = lidar+self.pos_embedding2

        x = self.dropout1(x)
        lidar = self.dropout2(lidar)

        x, lidar, att_loss = self.transformer1(x, lidar, mask)  

        
        ############# 预分类
        cls1 = self.to_cls_token(x[:,0])
        cls2 = self.to_cls_token(lidar[:,0])
        

        score1 = (-torch.cdist(cls1, x_proto)).softmax(-1)
        score2 = (-torch.cdist(cls2, l_proto)).softmax(-1)

        score = torch.stack([score1,score2],1)
        score_ = score.mean(1)

        ##################################
        b,n,d = x.shape
        
        x_masked = x.clone()
        lidar_masked = lidar.clone()
        if label is not None:
            pro1,pro2 = modu(score1,score2,label,self.alpha)
            
            mask1 = (torch.rand(d).cuda() <= pro1)
            mask2 = (torch.rand(d).cuda() <= pro2)

            x_masked[:,:,(mask1 == 0)] *= pro1
            lidar_masked[:,:,(mask2 == 0)] *= pro2
                
                
        x, lidar = x_masked, lidar_masked
        fuse = self.fuse(x,lidar,score1,score2)
        fuse = self.to_cls_token(fuse[:,0])

        fuse = self.nn(fuse)

        return fuse, att_loss, (cls1, cls2)
        

    

