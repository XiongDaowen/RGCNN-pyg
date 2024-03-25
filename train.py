import torch
import torch.nn as nn
import numpy as np
import config
from dataset import create_graph, load_eegdata, load_SEED_DE_feature
from model import DGCNN
import os
from itertools import cycle
import time
import pdb
from sklearn.model_selection import KFold
import scipy.io as scio 



load_path = 'E:/数据集汇总/SEED/ExtractedFeatures1/session1/'


batch_size = config.batch_size
epochs = config.epochs
lr = config.lr
weight_decay = config.weight_decay
device = config.device
DATASET = config.DATASET



def writeEachEpoch(people, epoch, batchsize, lr, temperature, acc):
    import model

    if not os.path.exists(f'./xxx/'):
        os.makedirs(f'./xxx/')

    log = []
    log.append(f'{DATASET}\t{people}\t{temperature}\t'
               f'{batchsize}\t{epoch}\t{lr}\t{model.drop_rate}\t{acc:.4f}\n')
    with open(
            f'./xxx/{DATASET}_All_log.txt',
            'a') as f:
        f.writelines(log)


def updatelog(people, epoch, acc):

    if not os.path.exists(f'./xxx/'):
        os.makedirs(f'./xxx/')

    log = []
    log.append(f'{DATASET}\t{people}\t{epoch}\t{lr}\t{batch_size}\t{acc:.4f}\n')
    with open('./xxx/{DATASET}_UPDATE_LOG.txt', 'a') as f:
        f.writelines(log)


def test(net, test_data, test_label, people, highest_acc, epoch):
    criterion = nn.CrossEntropyLoss().to(device)

    gloader = create_graph(test_data, test_label, shuffle=True, batch_size=batch_size, drop_last=True)
    net.testmode = True
    net.eval()
    epoch_loss = 0.0
    correct_pred = 0
    for ind, data in enumerate(gloader):


        data = data.to(device)
        out = net(data)
        y = data.y
        _, pre = torch.max(out, dim=1)

        correct_pred += sum([1 for a, b in zip(pre, y) if a == b])
        loss = criterion(out, y)

        epoch_loss += float(loss.item())

    ACC = correct_pred / ((ind + 1) * batch_size)
    if ACC > highest_acc:
        updatelog(people, epoch, ACC)
        highest_acc = ACC
        ck = {}
        ck['epoch'] = epoch
        ck['model'] = net.state_dict()
        ck['ACC'] = ACC
        
        torch.save(ck, f'{DATASET}_DGCNN_checkpoint/checkpoint_{people}.pkl')

    net.train()
    net.testmode=False
    return highest_acc, ACC


def train(train_data, train_label, test_data, test_label, people):
    highest_acc = 0.0
    
    if not os.path.exists(f'{DATASET}_DGCNN_checkpoint'):
        os.makedirs(f'{DATASET}_DGCNN_checkpoint')
    
    if os.path.exists(f'{DATASET}_DGCNN_checkpoint/checkpoint_{people}.pkl'):
        check = torch.load(f'{DATASET}_DGCNN_checkpoint/checkpoint_{people}.pkl')
        highest_acc = check['ACC']
    
    n_class = config.n_class
    n_channel = config.n_channel
    n_freq = config.n_freq
  
    net = DGCNN(n_freq, 32, batch_size, n_class, testmode=False)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8)

    
    gloader = create_graph(train_data, train_label, shuffle=True, batch_size=batch_size)
    timeseed = time.time()

    net_best = net

    for epoch in range(epochs):

        if device == 'cuda:0':
            print('empty cuda cache...')
            torch.cuda.empty_cache()


        epoch_loss = 0.0
        correct_pred = 0

        for ind, datas in enumerate(gloader):

            

            gdata = datas
            gdata = gdata.to(device)
            x = net(gdata)
            y = gdata.y
            _, pred = torch.max(x, dim=1)
            
            correct_pred += sum([1 for a,b in zip(pred, y) if a==b])
            
            loss = criterion(x, y)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            

        highest_acc, current_acc = test(net, test_data, test_label, people, highest_acc, epoch)
        writeEachEpoch(people, epoch, batch_size, lr, 0.25, current_acc)

        scheduler.step(epoch_loss)

        denominator = (ind+1)*batch_size
        if epoch % 5 == 0:
            print()
            print(f'-----highest_acc {highest_acc:.4f} current_acc {current_acc:.4f}-----')
            print('Dataset: ', DATASET)
            print(f'batch {batch_size}, lr {lr}')
            print()
            
        print(f'Epoch [{epoch}/{epochs}] \n'
              f'Loss eLoss[{epoch_loss/(ind+1):.4f}] \n'
              f'ACC@1 gACC[{correct_pred/denominator:.4f}] \n')

        # 使得模型具有继承性
        if current_acc >= highest_acc:
            net_best = net

        
        if current_acc < highest_acc-0.1:
            net = net_best

    return highest_acc



def intra_subject_Kold(people):
    print(f'load object {people}\'s data.....')
    data, label = load_SEED_DE_feature(load_path + people)

    index = np.random.permutation(data.shape[0])
    data = data[index,:,:]
    label = label[index]

    print('loaded!')
    Kfold_n = 10
    kf = KFold(n_splits=Kfold_n)
    idx = 1
    for train_index, test_index in kf.split(data):

        print('> ' + str(idx) + 'fold...')
        people_Kfold = people + '_' + str(idx) + 'fold'

        train_data = data[train_index, :, :]
        test_data = data[test_index, :, :]

        train_label = label[train_index]
        test_label = label[test_index]

        highest_acc = train(train_data, train_label, test_data, test_label, people_Kfold)

        idx = idx + 1

    return highest_acc


def inter_subject_LOSV(people, train_list):
    print(f'load test {people}\'s data.....')
    data_test, label_test = load_SEED_DE_feature(load_path + people)

    index = np.random.permutation(data_test.shape[0])
    data_test = data_test[index,:,:]
    label_test = label_test[index]

    print('loaded train {people}\'s data.....')

    flag = 0
    
    for train_index in train_list:

        print('> ' + train_index)

        data, label = load_SEED_DE_feature(load_path + train_index)
        
        if flag==0:

            data_train = data
            label_train = label
        else:        

            data_train = np.concatenate([data_train, data], axis=0)
            label_train = np.concatenate([label_train, label], axis=0)

    highest_acc = train(data_train, label_train, data_test, label_test, people)

        # idx = idx + 1

    return highest_acc


if __name__ == '__main__':
    
    file_list = os.listdir(load_path)

    # intra_subject_Kold(file_list[2])

    # train_list = file_list
    # train_list.remove(file_list[2])
    # inter_subject_LOSV(file_list[2], train_list)

    acc_all = []

    for file_i in range(len(file_list)):

        train_list = file_list
        train_list.remove(file_list[file_i])
        acc = inter_subject_LOSV(file_list[file_i], train_list)
        acc_all.append(acc)

    scio.savemat('1',{'acc':acc_all})
