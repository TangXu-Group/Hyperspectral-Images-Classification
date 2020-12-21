import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler


load_path1 = './PaviaU (1).mat'
load_data = sio.loadmat(load_path1)
data_load = load_data["paviaU"]
load_path2 = './PaviaU_gt(1).mat'
load_labels = sio.loadmat(load_path2)
labels_load = load_labels["paviaU_gt"]


data_normal = (data_load-data_load.min())/(data_load.max()-data_load.min())
data_reshape = data_normal.reshape(-1, 103)
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_reshape)
data_stand = data_standardized.reshape(610, 340, 103)


data_pad = np.pad(data_stand, ((6, 6), (6, 6), (0, 0)), mode="symmetric")
print("data_pad"+str(data_pad.shape))
print("labels_load"+str(labels_load.shape))
print("###########################################################")


data_list = []
labels_list = []
hang_list = []
lie_list = []

for i in range(610):
    for j in range(340):
        data_block = data_pad[i:i+13, j:j+13, ...]
        labels_block = labels_load[i, j, ...]
        if labels_block > 0:
            data_list.append(data_block)
            labels_list.append(labels_block)
            hang_list.append(i)
            lie_list.append(j)

data_all = np.array(data_list)
labels_all = np.array(labels_list)
hang = np.array(hang_list)
lie = np.array(lie_list)
print("data_all"+str(data_all.shape))
print("labels_all"+str(labels_all.shape))
print("hang"+str(hang))
print("lie"+str(lie))
print("========================================================")


data_train_list = []
data_test_list = []
labels_train_list = []
labels_test_list = []

for c in range(1, 10):
    index = np.array(np.where(labels_all == c))
    index_block = np.squeeze(index)
    data_chose = data_all[index_block, ...]
    labels_chose = labels_all[index_block, ...]
    print("c" + str(c))
    print("data_chose" + str(data_chose.shape))
    print("labels_chose" + str(labels_chose.shape))

    aa = np.arange(0, data_chose.shape[0])
    np.random.shuffle(aa)
    data_shuffle = data_chose[aa, ...]
    labels_shuffle = labels_chose[aa, ...]
    print("data_shuffle" + str(data_shuffle.shape))
    print("labels_shuffle" + str(labels_shuffle.shape))

    if c == 1:
        mid = 548
    if c == 2:
        mid = 540
    if c == 3:
        mid = 392
    if c == 4:
        mid = 542
    if c == 5:
        mid = 256
    if c == 6:
        mid = 532
    if c == 7:
        mid = 375
    if c == 8:
        mid = 514
    if c == 9:
        mid = 231

    data_train_list.extend(data_shuffle[: mid, ...])
    data_test_list.extend(data_shuffle[mid:, ...])
    labels_train_list.extend(labels_shuffle[: mid, ...])
    labels_test_list.extend(labels_shuffle[mid:, ...])

data_train = np.array(data_train_list).astype("float32")
data_test = np.array(data_test_list).astype("float32")
labels_train_scalar = np.array(labels_train_list)
labels_test_scalar = np.array(labels_test_list)
labels_train = np.eye(9)[labels_train_scalar - 1].astype("float32")
labels_test = np.eye(9)[labels_test_scalar - 1].astype("float32")

print("data_train"+str(data_train.shape))
print("data_test"+str(data_test.shape))
print("labels_train"+str(labels_train.shape))
print("labels_test"+str(labels_test.shape))
print("*********************************************************")

np.save("data_train.npy", data_train)
np.save("data_test.npy", data_test)
np.save("labels_train.npy", labels_train)
np.save("labels_test.npy", labels_test)