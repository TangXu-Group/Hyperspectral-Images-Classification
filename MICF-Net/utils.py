from scipy import io
import numpy as np
import os
import random
import torch
import tqdm
import tifffile
################### get data set

def get_dataset(dataset_name,path):
    palette = None

    if dataset_name == 'Houston':
        num_class = 15
        #load the image
        DataPath1 = path+'dataset/hu2013/Houston2013_HSI.mat'
        DataPath2 = path+'dataset/hu2013/Houston2013_DSM.mat'
        train_path = path+'dataset/hu2013/Houston_train_gt.mat'
        test_path = path+'dataset/hu2013/Houston_test_gt.mat'

        hsi = io.loadmat(DataPath1)['HSI']
        lidar = io.loadmat(DataPath2)['DSM']      
        TrLabel = io.loadmat(train_path)['train']  # 349*1905
        TsLabel = io.loadmat(test_path)['test']  # 349*1905
    
        label_values = ["Undefined", "Healthy grass", "Stressed grass", "Synthetic grass", 
                        "Trees","Soil","Water","Residential","Commercial","Road",
                        "Highway","Railway","Parking Lot 1","Parking Lot 2",
                        "Tennis Court","Running Track"]
        rgb_bands = (43, 21, 11) #AVIRIS sensor
        ignored_labels = [0]
    elif dataset_name == 'Trento':
        num_class = 6
        DataPath1 = path+'dataset/Trento/HSI.mat'
        DataPath2 = path+'dataset/Trento/LiDAR.mat'
        train_path = path+'dataset/Trento/Trento_train_gt.mat'
        test_path = path+'dataset/Trento/Trento_test_gt.mat'

        hsi = io.loadmat(DataPath1)['HSI']
        lidar = io.loadmat(DataPath2)['LiDAR']      

        TrLabel = io.loadmat(train_path)['train']  # 166*600
        TsLabel = io.loadmat(test_path)['test']  # 166*600
        
        label_values = ['Undefined', 'Apple trees','Buildings','Ground','Woods','Vineyard',
                        'Roads']
        ignored_labels = [0]
    elif dataset_name == 'Muufl':
        # Load the image
        num_class = 11
        DataPath1 = path+'dataset/muufl/muulf_hsi.tif'
        DataPath2 = path+'dataset/muufl/muulf_lidar.tif'
        train_path = path+'dataset/muufl/Muufl_train_gt.mat'
        test_path = path+'dataset/muufl/Muufl_test_gt.mat'

        hsi = tifffile.imread(DataPath1)
        lidar = tifffile.imread(DataPath2)    

        TrLabel = io.loadmat(train_path)['train']  # 325*220
        TsLabel = io.loadmat(test_path)['test']  # 325*220
        
        
        label_values = ['Undefined', 'Trees','Mostly grass','Mixed ground surface','Dirt and sand', 'Road','Water',"Building shadow",
                    'Building','Sidewalk','Yellow curb','Cloth panels']

        ignored_labels = [0]
    else:
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    
    hsi = np.asarray(hsi, dtype='float32')
    H, W, C = hsi.shape
    
    lidar = np.asarray(lidar, dtype='float32').reshape([H, W, -1])
    print('HSI shape: {}x{}x{}'.format(*hsi.shape))
    print('DSM shape: {}x{}x{}'.format(*lidar.shape))
    tr_size = len(np.where(TrLabel>0)[0])
    ts_size = len(np.where(TsLabel>0)[0])
    print('Train size:{} Test size:{}'.format(tr_size,ts_size))
    n_bands = hsi.shape[-1]

    for band in range(n_bands):
        min_val = np.min(hsi[:,:,band])
        max_val = np.max(hsi[:,:,band])
        hsi[:,:,band] = (hsi[:,:,band] - min_val) / (max_val - min_val)
    for band in range(lidar.shape[-1]):
        min_val = np.min(lidar[:,:,band])
        max_val = np.max(lidar[:,:,band])
        lidar[:,:,band] = (lidar[:,:,band] - min_val) / (max_val - min_val)

    return hsi, lidar, TrLabel, TsLabel, num_class, C, label_values

def createCubes(X, lidar, label, patch_size=9):
    p = patch_size // 2
    X = np.pad(X, ((p,p),(p,p),(0,0)), mode='symmetric')
    lidar = np.pad(lidar, ((p,p),(p,p),(0,0)), mode='symmetric')

    label = np.pad(label, p, mode='constant')
    Patch_X = []
    Patch_l = []

    Y = []
    [ind1, ind2] = np.where(label>0)
    for i in range(len(ind1)):
        x1, y1 = ind1[i] - p, ind2[i] - p
        x2, y2 = x1 + patch_size, y1 + patch_size
        Patch_X.append(X[x1:x2, y1:y2])
        Patch_l.append(lidar[x1:x2, y1:y2])

        Y.append(label[ind1[i],ind2[i]]-1)
        

    Patch_X = torch.tensor(np.array(Patch_X)).permute((0, 3, 1, 2))
    Patch_l = torch.tensor(np.array(Patch_l)).permute((0, 3, 1, 2))
    

    Y = np.array(Y,dtype=np.int64)
    
    return Patch_X, Patch_l, Y

class HyperX(torch.utils.data.Dataset):
    def __init__(self, X, l, Y, conv3d=True):
        super(HyperX, self).__init__()
        self.X = X
        self.l = l
        self.Y = torch.from_numpy(Y)
        self.patch_size = X.shape[2]
        if conv3d:
            self.X = self.X.unsqueeze(1)
        
        
    def __len__(self):
        
        return len(self.Y)
 
    
    def __getitem__(self, i):

        return self.X[i],self.l[i], self.Y[i], i

############################################################ save model
def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir) #dengbin:20181011
    if isinstance(model, torch.nn.Module):
        filename = "non_augmentation_sample{sample_size}_run{run}_epoch{epoch}_{metric:.2f}".format(**kwargs)
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + '.pth')
        filename2 = "non_augmentation_sample{}_run{}".format(kwargs['sample_size'], kwargs['run'])
        torch.save(model.state_dict(), model_dir + filename2 + '.pth')
############################################################ save and get samples/results         
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return oa*100, confusion, each_acc*100, aa*100, kappa*100


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

if __name__ == '__main__':
    img, gt, _ = get_dataset('Houston')



