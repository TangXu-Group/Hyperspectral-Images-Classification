import scipy.io as io
import numpy as np
import torch
import torch.utils.data as dataf


###################################### torch datasets

class HyperX(torch.utils.data.Dataset):
    """
    以中心像素切patch
    """
    def __init__(self, data, gt, dataset_name, patch_size=9, nb_classes=16):
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.nb_classes = nb_classes
        p = self.patch_size // 2

        # 填充padding
        self.data = np.pad(self.data, ((p, p), (p, p), (0, 0)), mode='constant').transpose((2, 0, 1))
        self.label = np.pad(self.label, p, mode='constant')
        self.indices = []
        self.data = torch.from_numpy(self.data)

        # 去除其中重复的元素
        for c in np.unique(self.label):
            if c == 0:
                continue
            # 非零元素的索引
            c_indices = np.nonzero(self.label == c)
            # 坐标（x,y）的列表
            X = list(zip(*c_indices))
            self.indices += X
        self.indices = np.asarray(self.indices)
        self.xy1 = self.indices - self.patch_size // 2
        self.xy2 = self.xy1 + self.patch_size
        self.train_labels = np.zeros(len(self.indices))
        for i in range(len(self.indices)):
            self.train_labels[i] = self.label[self.indices[i][0], self.indices[i][1]] - 1
        self.train_labels = np.int64(self.train_labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        xy1 = self.xy1[i]
        xy2 = self.xy2[i]
        data = self.data[:, xy1[0]:xy2[0], xy1[1]:xy2[1]]
        # data = data.reshape(data.shape[0], -1)
        # data = np.transpose(data, (1, 2, 0))
        # data = data.reshape(-1, data.shape[0], data.shape[1], data.shape[2])
        label = self.train_labels[i]

        return data, label #(200, 9, 9)




def read_data():
    root_path = "/media/admin1/DLdata/YYX/DATA/"
    tr_gt = io.loadmat(root_path + "Indian_30_split.mat")['TR'].astype(np.float32)
    te_gt = io.loadmat(root_path + "Indian_30_split.mat")['TE'].astype(np.float32)
    data = io.loadmat(root_path + "Indian_30_split.mat")['input'].astype(np.float32)
    train_dataset = HyperX(data, tr_gt, 'Indian', patch_size=13, nb_classes=16)
    test_dataset = HyperX(data, te_gt, 'Indian', patch_size=13, nb_classes=16)
    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = dataf.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    for step, (b_x1, b_y, i) in enumerate(train_loader):
        print(b_x1.shape)
        print(b_y.shape)

    # return train_dataset, test_dataset

def get_img_shape():
    return (200, 9, 9)


if __name__ == '__main__':
    read_data()






