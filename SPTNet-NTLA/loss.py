import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Loss functions
def CEloss(logits, labels, soft=0.1):
    N, C = logits.shape
    given_labels = torch.full(size=(N, C), fill_value=soft/(C - 1)).cuda()
    given_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-soft)
    softmax_logits = F.softmax(logits,dim=1)
    log_softmax_logits = torch.log(softmax_logits)
    losses = -torch.sum(log_softmax_logits * given_labels, dim=1)  # (N)
    return losses

def Simi_loss(pros_simi, mask,logits_mask):

    loss_simi_1 = -(mask)*torch.log(pros_simi+1e-7)-(1-mask)*torch.log(1-pros_simi+1e-7)

    loss_simi_1 = logits_mask*loss_simi_1

    loss_simi = loss_simi_1.mean()
    return loss_simi

def loss_coteaching(pred1, pred2, y_1, y_2, labels, ind, noise_or_not, s_labels):
    

    bsz = len(y_1)
    mask = np.zeros((bsz,bsz),dtype=int)
    s_num = set(s_labels)
    for i in s_num:
        index = np.where(s_labels==i)[0]
        mask[index[:,np.newaxis],index[np.newaxis,:]] = 1
    mask = torch.tensor(mask).cuda()
    mask = mask.repeat(2,2).cuda()
    mask[torch.eye(2*bsz)==1] = 0

    predsA = F.softmax(y_1,-1)
    predsB = F.softmax(y_2,-1)
    pros_batch = torch.cat([predsA,predsB],dim=0)
    prob_simi = torch.mm(pros_batch,pros_batch.t())

    logits_mask = (torch.ones_like(mask) - torch.eye(2*bsz).cuda())
    simi_loss = Simi_loss(prob_simi, mask, logits_mask)

    
    pred1 = pred1[ind]
    pred2 = pred2[ind]

    idx1 = np.nonzero(pred1)[0]
    idx2 = np.nonzero(pred2)[0] ##shape: N
    
    num_remember1 = pred1.sum()
    num_remember2 = pred2.sum()

    pure_ratio_1 = np.sum(noise_or_not[ind[idx1]])/float(num_remember1)
    pure_ratio_2 = np.sum(noise_or_not[ind[idx2]])/float(num_remember2)

    soft=0.1
    loss_1_update = CEloss(y_1[idx2], labels[idx2], soft).mean()
    loss_2_update = CEloss(y_2[idx1], labels[idx1], soft).mean()

    return simi_loss, loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2
