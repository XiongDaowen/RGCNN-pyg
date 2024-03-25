import numpy as np
import os
import torch
from scipy import sparse
import itertools
# from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import config
import scipy
import scipy.io as scio
import random

def adjacency():
    row_ = np.array(
        [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
         13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25, 25, 26, 26,
         27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40,
         41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54,
         54, 55, 55, 56, 57, 58, 59,
         60, 1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12,
         20, 13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26,
         34, 27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47,
         40, 48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58,
         54, 59, 55, 60, 56, 61, 61, 58, 59, 60, 61])

    col_ = np.array(
        [1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20,
         13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26, 34,
         27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47, 40,
         48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58, 54,
         59, 55, 60, 56, 61, 61, 58,
         59, 60, 61, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
         11, 12, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25,
         25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38,
         39, 39, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52,
         53, 53, 54, 54, 55, 55, 56, 57, 58, 59, 60])

    # te_r = np.array([1,2,3])
    # tr_c = np.array([4,5,6])
    # data_ = np.ones(3).astype('float32')
    # B = scipy.sparse.csr_matrix((data_, (row_, col_)), shape=(62, 62))

    weight_ = np.ones(236).astype('float32')
    # A = scipy.sparse.csr_matrix((weight_, (row_, col_)), shape=(62, 62))
    
    return row_, col_, weight_


def adjacency1():

    row_ = []
    col_ = []
    num = 0
    for i in range(62):
        # for j in range(i+1,62):
        for j in range(62):
            row_.append(i)
            col_.append(j)
            num = num + 1

    row_ = np.array(row_)
    col_ = np.array(col_)

    weight_ = np.ones(num).astype('float32')

    return row_, col_, weight_ 
 #PYG 对图的零阶矩阵的输入分两部分 一个是2xN的索引，一个是N的权重

def create_graph(data, label, shuffle=False, batch_size=128, drop_last=True):
    row_, col_, weight_ = adjacency1()
    edge_index = torch.from_numpy(np.vstack((row_, col_))).long()
    #edge_attr = weight_
    edge_attr = torch.from_numpy(weight_)
    graph = []

    for i in range(data.shape[0]):
        x = data[i]
        x = torch.from_numpy(x).type(torch.float32)
        
        y = torch.tensor(label[i], dtype=torch.long)
        graph.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return DataLoader(graph, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)#, num_workers=config.num_workers)


def load_eegdata(file_path):
    raw = scio.loadmat(file_path)  #生成字典
    DE_theta = raw['DE_theta']
    DE_alpha = raw['DE_alpha']
    DE_beta = raw['DE_beta']
    DE_gamma = raw['DE_gamma']
    label = raw['label']

    index1 = np.where(label == 1)[0]   #条件满足 返回满足条件的坐标  （以元组的形式，里面是数组）   https://www.jb51.net/article/260293.htm  原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。例如二维，一一组合对应坐标几行几列
    index2 = np.where(label == 2)[0]
    index3 = np.where(label == 3)[0]
###去掉0标签对应的数据
    DE_theta = np.concatenate([DE_theta[index1, :], DE_theta[index2, :], DE_theta[index3, :]], axis=0)
    DE_alpha = np.concatenate([DE_alpha[index1, :], DE_alpha[index2, :], DE_alpha[index3, :]], axis=0)
    DE_beta = np.concatenate([DE_beta[index1, :], DE_beta[index2, :], DE_beta[index3, :]], axis=0)
    DE_gamma = np.concatenate([DE_gamma[index1, :], DE_gamma[index2, :], DE_gamma[index3, :]], axis=0)
    label = np.concatenate([np.zeros([len(index1)]), np.zeros([len(index2)])+1, np.zeros([len(index3)])+2], axis=0)

    ###x = np.zeros([len(index1)])   一维数组是“列向量”
    DE_theta = np.expand_dims(DE_theta, axis=2)
    DE_alpha = np.expand_dims(DE_alpha, axis=2)
    DE_beta = np.expand_dims(DE_beta, axis=2)
    DE_gamma = np.expand_dims(DE_gamma, axis=2)




##########打乱
    index = np.random.permutation(DE_theta.shape[0])   #随机排列一个序列，或者数组。
                                                        #如果x是多维数组，则沿其第一个坐标轴的索引随机排列数组。
    DE_theta = DE_theta[index, :, :]
    DE_alpha = DE_alpha[index, :, :]
    DE_beta = DE_beta[index, :, :]
    DE_gamma = DE_gamma[index, :, :]
    label = label[index]

    data = np.concatenate([DE_theta, DE_alpha, DE_beta, DE_gamma], axis=2)
    label = label.flatten()

    return data, label


def load_SEED_DE_feature(file_path):

    clip_label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]

    raw = scio.loadmat(file_path)

    for clip_i in range(1, 16):# trails

        DE = raw['de_LDS' + str(clip_i)] 
        DE = DE.transpose([1,0,2])# 改变原数据的视图 不是拷贝  #样本x电极x微分熵
        ll = np.zeros([DE.shape[0], ]) + clip_label[clip_i-1] + 1

        if clip_i == 1:

            data = DE
            label = ll

        else:

            data = np.concatenate([data, DE], axis=0)
            label = np.concatenate([label, ll], axis=0)

    label = label.flatten()

    return data, label
