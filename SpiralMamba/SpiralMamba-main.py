import numpy as np
import torch
import torch.nn as nn
import os

import torch.optim as optim
import torch.utils.data
import scipy.io as sio
from configs.config import get_config
from models.spiralmamba import SpiralMamba
import random
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import time
import argparse
from data.hsidataloader import HyperX
import json
from timm.data import Mixup

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()

parser.add_argument('--mixup_prob', default=0.05, type=float, help='mixup_prob')
parser.add_argument('--tao', default=0.9, type=float, help='yuzhi')
parser.add_argument('--exp_name', default='test', type=str, help='experiment name')
parser.add_argument('--dataset', default='HUS18_100', type=str, help='dataset')
parser.add_argument('--epochs', default=500, type=int, help='epoch, default 500')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=0.05, type=float, help='weight_decay')  # 1e-3
parser.add_argument('--patch_size', type=int, default=17)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--experiment_time', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--is_pca', type=bool, default=False)
parser.add_argument('--seed', default=2024)
parser.add_argument('--output_path', default='./results', type=str, help='output path')
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file',
                        default='./configs/spiralmamba.yaml')
parser.add_argument('--mixup', type=bool, default=True)
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

args = parser.parse_args()
print(args)


output_dir = f"{args.output_path}/{args.dataset}/{args.exp_name}_b{args.batch_size}_lr{args.lr}_p{args.patch_size}_ T{args.experiment_time}_t{args.tao}_mixup{args.mixup_prob}"
os.makedirs(output_dir, exist_ok=True)

def save_log(log: dict, log_name='log.txt', mode='a', save_dir=output_dir):
    with open(os.path.join(save_dir, log_name), mode) as f:
        f.write(json.dumps(log) + '\n')

save_log(args.__dict__, log_name='total_result.txt')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

set_seed(args.seed)



#------read data------
root_path = "/media/ubuntu/YYX/DATA/"
path = root_path

if args.dataset == 'IP':
    path = root_path + "Indian/IndianPine50.mat"
elif args.dataset == 'PU':
    path = root_path + "PaviaU/Pavia_sf.mat"
elif args.dataset == 'HUS':
    path = root_path + "Houston/Houston_sf.mat"
elif args.dataset == 'HUS18_100':
    path = root_path + "Houston2018/hus2018_100.mat"

dict = sio.loadmat(path)
tr_gt = dict['TR'].astype(np.float32)
te_gt = dict['TE'].astype(np.float32)
orig_data = dict['input'].astype(np.float32)
height, width, band = orig_data.shape
classes = int(np.max(np.unique(tr_gt)))


# if (args.is_pca):
#     orig_data = apply_pca(orig_data, dim=80)

# normalize data by band norm
data_norm = np.zeros(orig_data.shape)
for i in range(orig_data.shape[2]):
    spatial_max = np.max(orig_data[:, :, i])
    spatial_min = np.min(orig_data[:, :, i])
    data_norm[:, :, i] = (orig_data[:, :, i]-spatial_min) / (spatial_max-spatial_min)
data_norm = data_norm.astype(np.float32)

#  # norm
# mean = np.mean(orig_data)
# std = np.std(orig_data)
# data_norm = (orig_data - mean) / std


train_dataset = HyperX(data_norm, tr_gt, 'Indian', patch_size=args.patch_size, nb_classes=classes)
test_dataset = HyperX(data_norm, te_gt, 'Indian', patch_size=args.patch_size, nb_classes=classes)
print(len(train_dataset))
print(len(test_dataset))


if args.mixup:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=24*args.batch_size, num_workers=4)



def train_test(exp_time=0):
    output_dir_one_time = os.path.join(output_dir, f'{exp_time + 1}')
    os.makedirs(output_dir_one_time, exist_ok=True)

    config = get_config(args)
    config.DATA.IMG_SIZE = args.patch_size
    config.MODEL.NUM_CLASSES = classes
    config.MODEL.VSSM.IN_CHANS = band
    config.MODEL.VSSM.EMBED_DIM = band
    config.MODEL.TAO = args.tao
    config.AUG.MIXUP_PROB = args.mixup_prob
    # Instantiate the model
    model = SpiralMamba(config).cuda()

    # Print the model architecture
    # print(model)
    epochs = args.epochs
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.wd)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr,  weight_decay=args.wd)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0*epochs*len(train_loader),
    #                                             num_training_steps=epochs*len(train_loader)
                                                # )
    # 创建 ReduceLROnPlateau 对象，当验证误差在 10 个 epoch 内没有下降时，将学习率减小为原来的 0.1 倍
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9)

    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    # setup mixup / cutmix label smooth
    if args.mixup:
        mixup_fn = None
        mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    else:
        mixup_fn = None

    best_epoch = 0
    best_val_acc = 0

    start_time = time.time()

    for epoch in range(epochs):
        # 训练
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        time_train_start = time.time()
        step = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            current_batch_size = inputs.shape[0]


            if mixup_fn is not None:
                inputs, labels_sm = mixup_fn(inputs, labels)#labels(b,16)
            else:
                labels_sm = labels


            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            loss = criterion(outputs, labels_sm)

            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            step += 1
            # print(f'step: {step}, lr: {current_lr}')

            # scheduler.step()
            # scheduler.step(loss)

            running_train_loss += loss.item() * current_batch_size
            _, predict = torch.max(outputs, dim=1)
            train_correct += torch.sum(predict == labels.data)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        time_train_epoch = time.time()


        # 验证
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                current_batch_size = inputs.shape[0]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * current_batch_size
                _, predict = torch.max(outputs, dim=1)
                val_correct += torch.sum(predict == labels.data)

            epoch_val_loss = val_running_loss / len(test_loader.dataset)
            val_acc = val_correct.double() / len(test_loader.dataset)

        time_val_epoch = time.time()
        train_time = time_train_epoch - time_train_start
        val_time = time_val_epoch - time_train_epoch

        print(f'Epoch [{epoch+1}/{epochs}], [Train Loss: {epoch_train_loss:.6f}] [OA: {train_acc:.2%}] [time: {train_time:.2f}s]     \
        [Val Loss: {epoch_val_loss:.6f}] [OA: {val_acc:.2%}] [time: {val_time:.2f}s]')


        if val_acc > best_val_acc:
            model_save_path = os.path.join(output_dir_one_time, 'best.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'best epoch: {epoch + 1}, loss: {epoch_val_loss:.6f} acc:{val_acc:.2%} \t Saving The Model')
            best_epoch = epoch
            best_val_acc = val_acc

        logs = {
            'Epo': f'{epoch+1}/{epochs}',
            'best_epo': f'{best_epoch + 1}',
            'best_acc': f'{best_val_acc:.2%}',
            'Train Loss': f'{epoch_train_loss:.6f}',
            'OA': f'{train_acc:.2%}',
            'train time': f'{train_time:.2f}s',
            'Val Loss': f'{epoch_val_loss:.6f}',
            'Val OA': f'{val_acc:.2%}',
            'Val time': f'{val_time:.2f}s',
        }
        save_log(logs, save_dir=output_dir_one_time)

    end_time = time.time()  # Step 3: Record the end time
    total_time = end_time - start_time  # Step 4: Calculate total training time

    print(f'Finished training. Total training time: {total_time:.2f} seconds')  # Print the total training time



    # 测试
    # Load the model (make sure to initialize the model architecture first)
    model.load_state_dict(torch.load(model_save_path))
    model.to(args.device)

    # Ensure the model is in evaluation mode
    model.eval()

    # Store predictions and actual labels
    predictions = []
    actual_labels = []

    start_time = time.time()  # Start timing

    with torch.no_grad():
        for hsi_patches, labels in test_loader:
            # Move data to the appropriate device
            hsi_patches = hsi_patches.to(args.device)

            # Forward pass
            outputs = model(hsi_patches)

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    end_time = time.time()  # End timing
    test_time = end_time - start_time  # Calculate the test time

    # Optionally, calculate accuracy or other metrics using predictions and actual_labels

    # Convert lists to NumPy arrays for easier manipulation
    predictions_array = np.array(predictions)
    actual_labels_array = np.array(actual_labels)

    # Overall Accuracy
    oa = accuracy_score(actual_labels_array, predictions_array)

    # Confusion Matrix
    cm = confusion_matrix(actual_labels_array, predictions_array)
    # Calculate per-class accuracy from the confusion matrix
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    # Average Accuracy
    aa = np.mean(class_accuracy)

    # Kappa Coefficient
    kappa = cohen_kappa_score(actual_labels_array, predictions_array)

    print(f'Overall Accuracy (OA): {oa:.2%}')
    print(f'Average Accuracy (AA): {aa:.2%}')
    print(f'Kappa Coefficient: {kappa:.2%}')
    print(f'Test time: {test_time:.2f} seconds')  # Print the test time

    one_time_res = {
        'OA': f'{oa:.2%}',
        'AA': f'{aa:.2%}',
        'Kappa': f'{kappa:.2%}',
    }

    for i, acc in enumerate(class_accuracy):
        print(f'Class {i+1} Accuracy: {acc:.2%}')
        one_time_res[f'C{i+1}'] = f'{acc:.2%}'

    save_log(one_time_res, save_dir=output_dir_one_time)

    return oa, aa, kappa, class_accuracy


if __name__ == '__main__':
    oa_list = []
    aa_list = []
    kappa_list = []
    class_acc_list = []
    for i in range(args.experiment_time):
        oa, aa, kappa, class_accuracy = train_test(exp_time=i)
        oa_list.append(oa)
        aa_list.append(aa)
        kappa_list.append(kappa)
        class_acc_list.append(class_accuracy)

    oa_mean = np.mean(oa_list)
    oa_std = np.std(oa_list)
    aa_mean = np.mean(aa_list)
    aa_std = np.std(aa_list)
    kappa_mean = np.mean(kappa_list)
    kappa_std = np.std(kappa_list)
    class_acc_np = np.array(class_acc_list)
    class_acc_mean = np.mean(class_acc_np, axis=0)
    class_acc_std = np.std(class_acc_np, axis=0)

    print('-----------total result----------')
    print(f'Overall Accuracy (OA): {oa_mean:.2%} +- {oa_std:.2%}')
    print(f'Average Accuracy (AA): {aa_mean:.2%} +- {aa_std:.2%}')
    print(f'Kappa Coefficient: {kappa_mean:.2%} +- {kappa_std:.2%}')

    log_text_res = {'OA': f'{oa_mean:.2%} +- {oa_std:.2%}',
                    'AA': f'{aa_mean:.2%} +- {aa_std:.2%}',
                    'Kappa': f'{kappa_mean:.2%} +- {kappa_std:.2%}'}

    for i, acc in enumerate(class_acc_mean):
        print(f'Class {i + 1} Accuracy: {acc:.2%} +- {class_acc_std[i]:.2%}')
        log_text_res[f'C{i + 1}'] = f'{acc:.2%} +- {class_acc_std[i]:.2%}'

    save_log(log_text_res, log_name='total_result.txt')





