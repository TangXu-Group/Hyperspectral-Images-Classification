import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import os
import time
from tqdm import tqdm
from utils import HyperX, get_dataset, acc_reports, seed_torch, createCubes
from model import MICF_Net

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("Our")
parser.add_argument('--gpuid', type=int, default=0, help='gpu id')
parser.add_argument('--lam1', default=0.1, type=float, help='balance coefficient') # 
parser.add_argument('--alpha', default=0.1, type=float, help='mask') 
parser.add_argument('--mode', type=str, default='MICF-Net')

parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--iters', type=int, default=1, help='number of iters')

parser.add_argument('--dim', type=int, default=64, help='dimension of feature')
parser.add_argument('--atte', type=int, default=8, help='number of atten fuse layer')
parser.add_argument('--heads', type=int, default=8, help='number of head')

parser.add_argument('--momentum_coef', type=float, default=0.9, help='monment')  

parser.add_argument('--epoches', type=int, default=120, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')  
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Houston', help='dataset to use')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=9, help='number1 of patches')
parser.add_argument('--dep', type=int, default=2, help='number1 of patches')
parser.add_argument('--data_path', type=str, default='../', help='dataset path')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum / self.cnt

def build_dataset():
    hsi,lidar,tr_label,ts_label,num_class, hsi_c, label_values = get_dataset(args.dataset,args.data_path) ###  hsi,lidar,tr_label,ts_label,num_class,label_values
    Patch_X, Patch_l, Y = createCubes(hsi, lidar, tr_label, patch_size=args.patch_size)
    print('Train data: ',Patch_X.shape, Patch_l.shape, Y.shape)
    Patch_X_test, Patch_l_test, Y_test = createCubes(hsi, lidar, ts_label, patch_size=args.patch_size)
    print('Test data: ',Patch_X_test.shape, Patch_l_test.shape, Y_test.shape)
    
    # for test
    train_dataset =  HyperX(Patch_X, Patch_l, Y, conv3d=True)
    train_loader = DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=2,
                                        shuffle=True)

    test_dataset =  HyperX(Patch_X_test, Patch_l_test, Y_test, conv3d=True)
    test_loader = DataLoader(test_dataset,
                                        batch_size=8*args.batch_size,
                                        num_workers=2,
                                        shuffle=False)

    return train_loader, test_loader, num_class, hsi_c

def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""


    pred = torch.max(logit,1)[1]
    correct_num = pred.eq(target.view(1, -1)).sum()            

    return correct_num/len(target)

def val(loader,model,x_proto,l_proto,epoch):
    model.eval()
    test_acc = AvgrageMeter()
    with torch.no_grad():
        for hsi, lidar, label, idx in tqdm(loader):
            batch_len = len(label)
            hsi, lidar, label = hsi.to(device), lidar.to(device), label.to(device)
            logits,_,_ = model(hsi,lidar, x_proto, l_proto,None,epoch)
            acc = accuracy(logits,label)
            test_acc.update(acc, batch_len)
             

    model.train()
    return test_acc

def test(loader,model,x_proto,l_proto,epoch):
    model.eval()

    preds_all = []
    labels_all = []
    with torch.no_grad():
        for hsi, lidar, label, idx in tqdm(loader):
            hsi, lidar, label = hsi.to(device), lidar.to(device), label.to(device)
            logits,_,_ = model(hsi,lidar, x_proto, l_proto,None,epoch)

            preds = torch.max(logits,1)[1].detach().cpu().numpy().tolist()
            preds_all += preds
            labels_all += label.detach().cpu().numpy().tolist()

    model.train()

    return preds_all, labels_all



def calculate_prototype(feat1, feat2,label, num_class, dim, x_proto=None, l_proto=None):
    unique_labels = label.unique() 
    
    hsi_proto = torch.zeros(num_class, dim).to(device)
    lidar_proto = torch.zeros(num_class, dim).to(device)
    for label_i in unique_labels:
        idx = label == label_i


        feat1_selected = feat1[idx]
        feat2_selected = feat2[idx]

        hsi = torch.mean(feat1_selected, dim=0)
        lidar = torch.mean(feat2_selected, dim=0)
        hsi_proto[label_i, :] = hsi
        lidar_proto[label_i, :] += lidar
    hsi_proto = (1 - args.momentum_coef) * hsi_proto + args.momentum_coef * x_proto
    lidar_proto = (1 - args.momentum_coef) * lidar_proto + args.momentum_coef * l_proto
 
    return hsi_proto, lidar_proto


def train(train_loader, test_loader, num_class, hsi_c):
    path = './checkpoint/'+args.dataset
    if not os.path.isdir(path):
        os.makedirs(path)
    print(args)

    ce_loss = nn.CrossEntropyLoss()
    model = MICF_Net(patch_size=args.patch_size, dim=args.dim, input_dim=hsi_c, num_classes=num_class, heads=args.heads, num_atte=args.atte, dep=args.dep).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr) ##
    model.train()
    best = 0

    x_proto = torch.empty(num_class, args.dim).to(device)
    torch.nn.init.normal_(x_proto, mean=0, std=0.2)

    l_proto = torch.empty(num_class, args.dim).to(device)  ######### 生成类别的原型
    torch.nn.init.normal_(l_proto, mean=0, std=0.2)

    t_start = time.time()
    
    for epoch in range(args.epoches):
        OA = AvgrageMeter()
        Loss = AvgrageMeter()
        for batch_idx, (hsi, lidar, label, idx) in tqdm(enumerate(train_loader)):
            hsi, lidar, label = hsi.to(device), lidar.to(device), label.to(device)
            batch_len = len(label)

            pred, att_loss,(cls1,cls2) = model(hsi,lidar,x_proto, l_proto,label,epoch)

            acc = accuracy(pred,label)   

            OA.update(acc, batch_len)

            model.zero_grad()
            loss = ce_loss(pred,label)+args.lam1*att_loss
            Loss.update(loss, batch_len)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                x_proto, l_proto = calculate_prototype(cls1,cls2, label, num_class, args.dim, x_proto, l_proto)

        val_acc = val(test_loader, model, x_proto, l_proto,epoch)
        if val_acc.avg>best:
            best = val_acc.avg
            torch.save({
                'state_dict': model.state_dict(),
                'HSI_proto': x_proto,
                'Lidar_proto': l_proto,
            },path+'/'+args.mode+'.pkl')
        print('Epoch %d Train acc:%.2f%% Loss:%.2f Val acc:%.2f%% Best acc:%.2f%%'%(epoch+1,OA.avg*100,Loss.avg,val_acc.avg*100,best*100))
        
    t_end = time.time()
    time_train = t_end - t_start
    print('Train time:{}'.format(time_train))


    return model


    
def main():
    # -------------------------------------------------------------------------------
    # prepare data
    train_loader, test_loader, num_class, hsi_c = build_dataset()
    train(train_loader,test_loader, num_class, hsi_c)


if __name__ == '__main__':
    seed_torch(args.seed)

    main()
    


