import torch.nn.functional as F
import torch
from scipy import io
import numpy as np
import os
import random
from tqdm import tqdm
from numpy.testing import assert_array_almost_equal
from sklearn import preprocessing
from copy import deepcopy
from sklearn.decomposition import PCA


############### Indian Pines
IP_num = {
    '1': 35,
    '2': 150,
    '3': 150,
    '4': 100,
    '5': 150,
    '6': 150,
    '7': 20,
    '8': 150,
    '9': 15,
    '10': 150,
    '11': 150,
    '12': 150,
    '13': 150,
    '14': 150,
    '15': 50,
    '16': 50,
}
############## Houston
HU_num = {
    '1': 150,
    '2': 150,
    '3': 150,
    '4': 150,
    '5': 150,
    '6': 150,
    '7': 150,
    '8': 150,
    '9': 150,
    '10': 150,
    '11': 150,
    '12': 150,
    '13': 150,
    '14': 150,
    '15': 150,
}

###################### PaviaU
UP_num = {
    '1': 548,
    '2': 540,
    '3': 392,
    '4': 542,
    '5': 256,
    '6': 532,
    '7': 375,
    '8': 514,
    '9': 231,
}

############## Houston2018
HU18_num = {
    '1': 100,
    '2': 100,
    '3': 100,
    '4': 100,
    '5': 100,
    '6': 100,
    '7': 100,
    '8': 100,
    '9': 100,
    '10': 100,
    '11': 100,
    '12': 100,
    '13': 100,
    '14': 100,
    '15': 100,
    '16': 100,
    '17': 100,
    '18': 100,
    '19': 100,
    '20': 100,
}
train_num = {'Houston':HU_num,'IndianPines':IP_num,'PaviaU':UP_num,'Houston2018':HU18_num}


######################  Noisy Label
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i], 1)[0]
        
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def generate_noise_matrix(noise_type, closeset_noise_ratio, nb_classes=10):
    """

    Example of the noise transition matrix (closeset_ratio = 0.3):
        - Symmetric:
            -                               -
            | 0.7  0.1  0.1  0.1  0.0  0.0  |
            | 0.1  0.7  0.1  0.1  0.0  0.0  |
            | 0.1  0.1  0.7  0.1  0.0  0.0  |
            | 0.1  0.1  0.1  0.7  0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            -                               -
        - Pairflip
            -                               -
            | 0.7  0.3  0.0  0.0  0.0  0.0  |
            | 0.0  0.7  0.3  0.0  0.0  0.0  |
            | 0.0  0.0  0.7  0.3  0.0  0.0  |
            | 0.3  0.0  0.0  0.7  0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            -                               -

    """
#     assert closeset_noise_ratio > 0.0, 'noise rate must be greater than 0.0'
    
    if noise_type == 'symmetric':
        P = np.ones((nb_classes, nb_classes))
        P = (closeset_noise_ratio / (nb_classes - 1)) * P
        for i in range(nb_classes):
            P[i, i] = 1.0 - closeset_noise_ratio
    
        
    elif noise_type == 'pairflip':
        P = np.eye(nb_classes)
        P[0, 0], P[0, 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        P[nb_classes - 1, nb_classes - 1] = 1.0 - closeset_noise_ratio
        P[nb_classes - 1, 0] = closeset_noise_ratio
    
    else:
        raise AssertionError("noise type must be either symmetric or pairflip")
    return P


def noisify(y_train, noise_transition_matrix, random_state=None):
    y_train_noisy = multiclass_noisify(y_train, P=noise_transition_matrix, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
#     assert actual_noise > 0.0
    return y_train_noisy, actual_noise


def noisify_dataset(nb_classes=10, train_labels=None, noise_type=None,
                    closeset_noise_ratio=0.0, random_state=0, verbose=True):
    noise_transition_matrix = generate_noise_matrix(noise_type, closeset_noise_ratio, nb_classes)
    train_noisy_labels, actual_noise_rate = noisify(train_labels, noise_transition_matrix, random_state)
    if verbose:
#         print(f'Noise Transition Matrix: \n {noise_transition_matrix}')
        print(f'Noise Type: {noise_type} (close set: {closeset_noise_ratio})\n'
              f'Actual Total Noise Ratio: {actual_noise_rate:.3f}')
    return train_noisy_labels, actual_noise_rate

################### get data set
def standartizeData(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX

def get_dataset(dataset_name, target_folder='./data'):
    palette = None
    folder = target_folder + '/'
    if dataset_name == 'IndianPines':
        #load the image
        img = io.loadmat(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']        
        gt = io.loadmat(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        rgb_bands = (43, 21, 11) #AVIRIS sensor
        ignored_labels = [0]
        in_c = 200
    elif dataset_name == 'Houston':
        #load the image
        img = io.loadmat(folder + 'houston.mat')
        img = img['data']        
        gt = io.loadmat(folder + 'houston_gt.mat')['label']
        label_values = ["Undefined", "Healthy grass", "Stressed grass", "Synthetic grass", 
                        "Trees","Soil","Water","Residential","Commercial","Road",
                        "Highway","Railway","Parking Lot 1","Parking Lot 2",
                        "Tennis Court","Running Track"]
        rgb_bands = (43, 21, 11) #AVIRIS sensor
        ignored_labels = [0]
        in_c = 144
    elif dataset_name == 'PaviaU':  
        # load the image
        img = io.loadmat(folder + 'PaviaU.mat')['paviaU']
        gt = io.loadmat(folder + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        rgb_bands = (55, 41, 12)
        ignored_labels = [0]
        in_c = 103

    else:
        raise ValueError("{} dataset is unknown.".format(dataset_name))
        
    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    n_bands = img.shape[-1]
    img = standartizeData(img)

    return img, gt, label_values, in_c


def sample_gt(gt,dataset):
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    print(len(X))
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    
    train_size = train_num[dataset]
    train_indices = []
    test_gt = np.copy(gt)
    for c in np.unique(gt):
        if c == 0:
            continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices)) # x,y features

        train_indices += random.sample(X, train_size[str(c)])

    index = tuple(zip(*train_indices))        
    train_gt[index] = gt[index]
    test_gt[index] = 0

    # print(len(np.nonzero(train_gt)[0]),len(np.nonzero(test_gt)[0]))
    return train_gt, test_gt



class HyperX(torch.utils.data.Dataset):
    def __init__(self, data, gt, dataset_name, patch_size=9, data_augmentation=True, 
                                            noise_type='symmetric', noise_ratio=0.4, nb_classes=16, conv3d = True):
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.noise_type = noise_type
        self.nb_classes = nb_classes
        self.conv3d = conv3d
        p = self.patch_size // 2

        segment = np.load('./segment/'+dataset_name+'.npy')
        
        
        # add padding
        if self.patch_size > 1:
            self.data = np.pad(self.data, ((p,p),(p,p),(0,0)), mode='constant').transpose((2, 0, 1))
            self.label = np.pad(self.label, p, mode='constant')
            segment = np.pad(segment, p, mode='constant')

        else:
            self.flip_argument = False
            self.rotated_argument = False
        self.indices = []
        
        print(np.unique(self.label))
        
        for c in np.unique(self.label):
            
            if c == 0:
                continue
            c_indices = np.nonzero(self.label == c)
            X = list(zip(*c_indices))
            self.indices += X
        # random.shuffle(self.indices)
        self.indices = np.asarray(self.indices)
        self.xy1 = self.indices - self.patch_size // 2
        self.xy2 = self.xy1 + self.patch_size
        
        self.train_labels = np.zeros(len(self.indices))
        self.s_labels = np.zeros(len(self.indices))

        for i in range(len(self.indices)):
            self.train_labels[i] = self.label[self.indices[i][0],self.indices[i][1]]-1
            self.s_labels[i] = segment[self.indices[i][0],self.indices[i][1]]

        self.train_labels = np.int64(self.train_labels)
        if (noise_type != 'clean'):
            self.noisy_labels, self.actual_noise_rate = noisify_dataset(self.nb_classes, self.train_labels, noise_type, noise_ratio)
            self.noise_or_not = self.noisy_labels==self.train_labels


    def random_crop(self,img,crop_size=(27,27),pad=4):
        img = np.pad(img, ((0,0),(pad,pad),(pad,pad)), mode='constant')
        w,h = img.shape[1:]
        x,y = np.random.randint(w-crop_size[0]),np.random.randint(h-crop_size[1])
        img = img[:,x:x+crop_size[0],y:y+crop_size[1]]
        return img
    
    def flip(self, data):
        horizontal = np.random.random() > 0.5
        data = self.random_crop(data,crop_size=(self.patch_size,self.patch_size),pad=4)
        if horizontal:
            data = np.fliplr(data)

        return data

    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, i):

        xy1 = self.xy1[i]
        xy2 = self.xy2[i]
        data = self.data[:,xy1[0]:xy2[0], xy1[1]:xy2[1]]

        if self.data_augmentation and self.patch_size > 1:
            # Perform data augmentation 
            data = self.flip(data)

        data = data.copy()
        data = torch.from_numpy(data)
        
        if self.conv3d:
            data = data.unsqueeze(0)

        if self.noise_type!='clean':
            label = self.noisy_labels[i]
            
            return data, label, self.train_labels[i], i, self.s_labels[i]
            
        else:
            label = self.train_labels[i]

            return data, label, self.train_labels[i], i, self.s_labels[i]


############################################################ save and get samples/results         

def get_sample(dataset_name, sample_size, run):
    sample_file = './trainTestSplit/' + dataset_name + '/sample' + str(sample_size) + '_run' + str(run) + '.mat'
    data = io.loadmat(sample_file)
    train_gt = data['train_gt']
    test_gt = data['test_gt']
    return train_gt, test_gt

def save_sample(train_gt, test_gt, dataset_name, ):
    sample_dir = './trainTestSplit/' + dataset_name + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + '.mat'
    io.savemat(sample_file, {'train_gt':train_gt, 'test_gt':test_gt})
    
def get_result(dataset_name, sample_size, run):
    scores_dir = './results/' + dataset_name + '/'
    scores_file = scores_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    scores = io.loadmat(scores_file)
    return scores

def save_result(result, dataset_name, sample_size, run):
    scores_dir = './results/' + dataset_name + '/'
    if not os.path.isdir(scores_dir):
        os.makedirs(scores_dir)
    scores_file = scores_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(scores_file,result)


