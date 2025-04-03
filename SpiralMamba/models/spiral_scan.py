import numpy as np
import torch

def get_hui_path1(size): # (0, 0)出发
    p = 0
    q = size - 1
    path1d = []
    while p < q:
        for i in range(p, q):
            path1d.append(p * size + i)
        for i in range(p, q):
            path1d.append(i * size + q)
        for i in range(q, p, -1):# 8,7,6,5,4,3,2,1
            path1d.append(q * size + i)
        for i in range(q, p, -1):
            path1d.append(i * size + p)
        p += 1
        q -= 1
    if q == p:
        path1d.append(p * size + p)
    return path1d

def get_hui_path2(size): # (0, 8)出发
    p = 0
    q = size - 1
    path1d = []
    while p < q:
        for i in range(p, q):
            path1d.append(i * size + q)
        for i in range(q, p, -1):# 8,7,6,5,4,3,2,1
            path1d.append(q * size + i)
        for i in range(q, p, -1):
            path1d.append(i * size + p)
        for i in range(p, q):
            path1d.append(p * size + i)
        p += 1
        q -= 1
    if q == p:
        path1d.append(p * size + p)
    return path1d

def get_hui_path3(size): # (8, 8)出发
    p = 0
    q = size - 1
    path1d = []
    while p < q:
        for i in range(q, p, -1):# 8,7,6,5,4,3,2,1
            path1d.append(q * size + i)
        for i in range(q, p, -1):
            path1d.append(i * size + p)
        for i in range(p, q):
            path1d.append(p * size + i)
        for i in range(p, q):
            path1d.append(i * size + q)
        p += 1
        q -= 1
    if q == p:
        path1d.append(p * size + p)
    return path1d

def get_hui_path4(size): # (8, 0)出发
    p = 0
    q = size - 1
    path1d = []
    while p < q:
        for i in range(q, p, -1):
            path1d.append(i * size + p)
        for i in range(p, q):
            path1d.append(p * size + i)
        for i in range(p, q):
            path1d.append(i * size + q)
        for i in range(q, p, -1):# 8,7,6,5,4,3,2,1
            path1d.append(q * size + i)
        p += 1
        q -= 1
    if q == p:
        path1d.append(p * size + p)
    return path1d


def get_hui_path_origin_path(spatial_size):
    path1d1 = get_hui_path1(spatial_size)
    path1d2 = get_hui_path2(spatial_size)
    path1d3 = get_hui_path3(spatial_size)
    path1d4 = get_hui_path4(spatial_size)
    # hui路径1d
    hui_path1ds = [path1d1, path1d2, path1d3, path1d4]
    # 回溯路径
    origin_path1ds = []
    for hui_path1d in hui_path1ds:
        # hui1d与原1d的映射关系
        hui_map_origin = {}  # 矩阵直接展平后对应的一维数组的下标与回形遍历的一维数组下标之间的关系
        origin_path1d = []  # 矩阵直接展平后对应的一维数组的每个元素在回形遍历序列中的位置
        for origin_id, hui_id in enumerate(hui_path1d):
            hui_map_origin[hui_id] = origin_id
        for id in range(spatial_size ** 2):
            origin_path1d.append(hui_map_origin[id])
        origin_path1ds.append(origin_path1d)
    return hui_path1ds, origin_path1ds


def hui_scan(data, hui_path1ds):
    batch, c, h, w = data.shape
    origin_seq = torch.flatten(data, start_dim=2)
    hui_seq = origin_seq.new_empty((batch, 4, c, h*w))
    hui_seq[:, 0] = origin_seq[:, :, hui_path1ds[0]]
    hui_seq[:, 1] = origin_seq[:, :, hui_path1ds[1]]
    hui_seq[:, 2] = origin_seq[:, :, hui_path1ds[2]]
    hui_seq[:, 3] = origin_seq[:, :, hui_path1ds[3]]
    hui_seq2d = hui_seq.view(batch * 4, c, h, w)
    return hui_seq2d



def hui_merge(hui_seq2d, origin_path1ds):
    batch, c, h, w = hui_seq2d.shape
    b = batch // 4
    hui_seq = hui_seq2d.view(b, 4, c, h*w)
    origin_seq = hui_seq.new_empty((b, 4, c, h*w))
    origin_seq[:, 0] = hui_seq[:, 0][:, :, origin_path1ds[0]]
    origin_seq[:, 1] = hui_seq[:, 1][:, :, origin_path1ds[1]]
    origin_seq[:, 2] = hui_seq[:, 2][:, :, origin_path1ds[2]]
    origin_seq[:, 3] = hui_seq[:, 3][:, :, origin_path1ds[3]]
    origin_seq2d = origin_seq.sum(1)
    origin_seq2d = origin_seq2d.view(b, c, h, w)
    return origin_seq2d



if __name__ == '__main__':

    # 设置矩阵大小
    spatial_size = 9

    # 最初先得到hui路径和回溯路径
    hui_path1ds, origin_path1ds = get_hui_path_origin_path(spatial_size)

    # 原2d数据
    data = np.array(range(0, 81))
    data = data.reshape(1, 1, 9, 9)
    data = data.repeat(200, axis=1)
    data = torch.from_numpy(data)

    hui_seq2d = hui_scan(data, hui_path1ds)


    origin_seq2d = hui_merge(hui_seq2d, origin_path1ds)

    print(hui_seq2d.flatten(2, 3)[0, 0, :])
    print(origin_seq2d.flatten(2, 3)[0, 0, :] // 4)
    print(data.flatten(2, 3)[0, 0, :])




























