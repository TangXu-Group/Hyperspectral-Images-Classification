import torch
import time
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import random
import numpy as np
from torch.utils.data import DataLoader
import argparse
from SPT import SPT

from tools import HyperX,sample_gt,get_dataset
from loss import loss_coteaching, CEloss

parser = argparse.ArgumentParser(description='SPT')

parser.add_argument('--Object', default='', type=str, help='') 
parser.add_argument('--mode', default='SPT', type=str)

parser.add_argument('--lr', default=2e-3, type=float, help='learning_rate ')
parser.add_argument('--gamma', default=0.8, type=float, help='learning_rate adjust')
parser.add_argument('--num_adjust', default=20, type=int, help='number of adjust')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--batch_size', default=64, type=int) 

parser.add_argument('--noise_type', default='symmetric')
parser.add_argument('--noise_ratio', default=0.40, type=float, help='ratio of random perturbations')
parser.add_argument('--lambda_s', default=0.8, type=float, help='simi_loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='GMM')

parser.add_argument('--warmup', default=30, type=int, help='warm up')
parser.add_argument('--iters', default=5, type=int, help='num of experiment')
                    
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpuid', default=0, type=int)

parser.add_argument('--Dataset', default='Houston', type=str)
parser.add_argument('--num_class', default=15, type=int)

parser.add_argument('--head', default=8, type=int)
parser.add_argument('--Patch_size', default=9, type=int)

parser.add_argument('--data_path', default='../data')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
device = torch.device("cuda")

def seed_torch(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    
seed_torch(args.seed)

use_cuda = torch.cuda.is_available()

def build_dataset():

    img, gt, LABEL_VALUES, in_c = get_dataset(args.Dataset, args.data_path)
    N_CLASSES = len(LABEL_VALUES) - 1
    # Sample random training spectra
    train_gt, test_gt = sample_gt(gt, args.Dataset)

    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                    np.count_nonzero(gt)))
    dataset_name = args.Dataset
    
    train_dataset = HyperX(img, train_gt, dataset_name, args.Patch_size, True, noise_type='symmetric', noise_ratio=args.noise_ratio, nb_classes=args.num_class,conv3d=True)
    noise_or_not = train_dataset.noise_or_not
    train_loader = DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers)

    test_dataset = HyperX(img, test_gt, dataset_name, args.Patch_size, False, noise_type = 'clean', noise_ratio=args.noise_ratio, nb_classes=args.num_class,conv3d=True)
    test_loader = DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=False)
    
    all_dataset = HyperX(img, gt, dataset_name, args.Patch_size, False, noise_type = 'clean', noise_ratio=args.noise_ratio, nb_classes=args.num_class,conv3d=True)
    all_loader = DataLoader(all_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,         
                                        shuffle=False)

    return LABEL_VALUES, in_c, train_loader, test_loader, all_loader, noise_or_not

def accuracy(logit, target):
    pred = torch.max(logit,1)[1]
    correct_num = pred.eq(target.view(1, -1)).sum()            

    return correct_num

def warmup(epoch, net ,optimizer_net):
    net.train()
    
    train_correct=0 
    train_correct2=0 
    train_total=0
    for batch_idx, (images, labels, _, indexes, s_labels) in enumerate(train_loader):      
       
        images, labels = images.cuda(), labels.cuda() 
        
        train_total += labels.size(0)   
        logits = net(images)
        correct = accuracy(logits, labels)
        train_correct += correct
        prec = 100.*train_correct/train_total
        loss = CEloss(logits, labels).mean()
        optimizer_net.zero_grad()
        loss.backward()
        optimizer_net.step()
    
    train_acc=100.*float(train_correct)/float(train_total)
    print('Epoch:{} Net acc:{}'.format(epoch,train_acc))    


def train(epoch):
    net1.train()
    net2.train()
    
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    train_correct=0 
    train_correct2=0 
    train_total=0

    for i, (images, labels, _, indexes, s_labels) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        train_total += labels.size(0)

        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        # Forward + Backward + Optimize

        logits1=net1(images)
        correct1 = accuracy(logits1, labels)
        train_correct += correct1
        prec1 = 100.*train_correct/train_total

        logits2 = net2(images)
        correct2 = accuracy(logits2, labels)
        train_correct2 += correct2
        prec2 = 100.*train_correct2/train_total
        
        sim_loss, loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(pred1, pred2, logits1, logits2, labels, ind, noise_or_not, s_labels)

        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)

        l_1 = args.lambda_s*sim_loss+loss_1*(1-args.lambda_s)
        optimizer_net1.zero_grad()
        l_1.backward(retain_graph=True)

        l_2 = args.lambda_s*sim_loss+loss_2*(1-args.lambda_s)
        optimizer_net2.zero_grad()
        l_2.backward()

        optimizer_net1.step()
        optimizer_net2.step()
        
    print ('Epoch [%d/%d], Acc1:%.4F, Acc2:%.4f,Loss1:%.4f,Loss2:%.4f,similoss:%.4f,Pure Ratio1:%.4f,Pure Ratio2:%.4f' 
            %(epoch, args.num_epochs, prec1, prec2, loss_1.item(), loss_2.item(),sim_loss.item(),np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))
    
    train_acc1=100.*float(train_correct)/float(train_total)
    train_acc2=100.*float(train_correct2)/float(train_total)

    return train_acc1, train_acc2
def last_test():
    net1.eval()
    net2.eval()

    train_correct = 0

    total = 0
    count=0
    with torch.no_grad():
        for batch_idx, (images, labels, _, _, _) in enumerate(test_loader):
            if use_cuda:
                images, targets = images.cuda(), labels.cuda()
                
            total += targets.size(0)
            logits1=net1(images)
            logits2 = net2(images)

            out = (logits1+logits2)/2
            correct = accuracy(out, targets)
            train_correct += correct
            
            preds = np.argmax(out.detach().cpu().numpy(), axis=1)
            
            if count == 0:
                y_pred_test = preds  
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, preds))
                y_test = np.concatenate((y_test, labels))

    acc = 100.*train_correct/total

    print("| Test two \t\t\t Acc@2:[%d/%d] %.2f%%" %(train_correct,total,acc))

    return acc, y_test, y_pred_test

        
from sklearn.mixture import GaussianMixture

CE = nn.CrossEntropyLoss(reduction='none')

def eval_train(model):  
    model.eval()
    losses = torch.zeros(len(train_loader.dataset))   
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, index, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs)  
            loss = CE(outputs, targets)  
      
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    input_loss = losses.reshape(-1,1)
    

    gmm = GaussianMixture(n_components=2,max_iter=15,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)      
    prob = gmm.predict_proba(input_loss) 
    
    prob = prob[:,gmm.means_.argmin()]         
    return prob  #shape:N

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = LABEL_VALUES
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return oa*100, confusion, each_acc*100, aa*100, kappa*100




LABEL_VALUES, in_c, train_loader, test_loader, all_loader, noise_or_not = build_dataset()

print(args)

print("################################################\n")

net1 = SPT(patch_size=args.Patch_size, input_dim=in_c,num_classes=args.num_class,heads=args.head)
net2 = SPT(patch_size=args.Patch_size, input_dim=in_c,num_classes=args.num_class,heads=args.head)
if use_cuda:
    net1.cuda()
    net2.cuda()
    
optimizer_net1 = optim.Adam(net1.parameters(), lr=args.lr, weight_decay=1e-4)
optimizer_net2 = optim.Adam(net2.parameters(), lr=args.lr, weight_decay=1e-4)


scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer_net1, step_size=args.num_epochs//args.num_adjust, gamma=args.gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_net2, step_size=args.num_epochs//args.num_adjust, gamma=args.gamma)

t_pre = []

import time
start = time.time()
for epoch in range(1,1+args.num_epochs):

    if epoch<=args.warmup:
        warmup(epoch,net1,optimizer_net1)
        warmup(epoch,net2,optimizer_net2)

        scheduler1.step()
        scheduler2.step()

    else:
        prob1 = eval_train(net1)   
        prob2 = eval_train(net2)

        pred1 = (prob1 > args.p_threshold)   
        pred2 = (prob2 > args.p_threshold) 

        # print('Net1:',pred1.sum(),(pred1==noise_or_not).sum())
        # print('Net2:',pred2.sum(),(pred2==noise_or_not).sum())

        train_acc1, train_acc2 = train(epoch)
        scheduler1.step()
        scheduler2.step()
        
end = time.time()
total = end-start
print('training time: ',total)

test_acc,y_test,y_pre = last_test()
oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pre)
print('OA: {:.2f}%, AA: {:.2f}%, Kappa: {:.2f}%'.format(oa, aa, kappa))
