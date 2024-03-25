import torch
import torch.nn as nn
import numpy as np
import config
from dataset import create_graph, load_eegdata, load_SEED_DE_feature
from model import DGCNN
from model_RGCNN import SymSimGCNNet
import os
from itertools import cycle
import time
import pdb
from sklearn.model_selection import KFold
import scipy.io as scio 
from scipy.stats import zscore


load_path = './ExtractedFeatures/Session1/'


batch_size = config.batch_size
epochs = config.epochs
lr = config.lr
weight_decay = config.weight_decay
device = config.device
DATASET = config.DATASET





def test(net, test_data, test_label, people, highest_acc, epoch):
    criterion = nn.CrossEntropyLoss().to(device)

    gloader = create_graph(test_data, test_label, shuffle=True, batch_size=batch_size, drop_last=True)
    net.testmode = True
    net.eval()
    epoch_loss = 0.0
    correct_pred = 0
    for ind, data in enumerate(gloader):


        data = data.to(device)
        out,_ = net(data)
        y = data.y
        _, pre = torch.max(out, dim=1)

        correct_pred += sum([1 for a, b in zip(pre, y) if a == b])
        loss = criterion(out, y)

        epoch_loss += float(loss.item())

    ACC = correct_pred / ((ind + 1) * batch_size)
    if ACC > highest_acc:
        highest_acc = ACC
        ck = {}
        ck['epoch'] = epoch
        ck['model'] = net.state_dict()
        ck['ACC'] = ACC
        
        torch.save(ck, f'{DATASET}_RGCNN_checkpoint/checkpoint_{people}.pkl')

    net.train()
    net.testmode=False
    return highest_acc, ACC


def train(train_data, train_label, test_data, test_label, people):
    highest_acc = 0.0
    current_highest_acc = 0
    if not os.path.exists(f'{DATASET}_RGCNN_checkpoint'):
        os.makedirs(f'{DATASET}_RGCNN_checkpoint')
    
    if os.path.exists(f'{DATASET}_RGCNN_checkpoint/checkpoint_{people}.pkl'):
        check = torch.load(f'{DATASET}_RGCNN_checkpoint/checkpoint_{people}.pkl')
        highest_acc = check['ACC']
    
    n_class = config.n_class
    n_channel = config.n_channel
    n_freq = config.n_freq
    num_hiddens = config.num_hiddens
    edge_weight = torch.ones([n_channel*n_channel],device=device)
  
    net = SymSimGCNNet(num_nodes=62, learn_edge_weight=True, edge_weight=edge_weight,
        num_features=5, num_classes=3, num_hiddens=num_hiddens,K=config.K,dropout=config.drop_rate, droprate=config.drop_rate_pooling, domain_adaptation="RevGrad")

    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True,
    #                                                        threshold=0.0001,
    #                                                        threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8)

    
    gloader_train = create_graph(train_data, train_label, shuffle=True, batch_size=batch_size)  #47516个样本
    gloader_test = create_graph(test_data, test_label, shuffle=True, batch_size=batch_size)
    timeseed = time.time()

    # source_iter = iter(gloader_train)
    target_iter = iter(gloader_test)

    net_best = net

    for epoch in range(epochs):

        if device == 'cuda:0':
            print('empty cuda cache...')
            torch.cuda.empty_cache()

        len_dataloader = len(gloader_train)
        ind = 0
        n =0
        epoch_loss = 0.0
        correct_pred = 0
        for source_data in gloader_train:

            p = float(ind + epoch * len_dataloader) / epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            #net.train()
        # try:
        #     source_data = next(source_iter)
        # except Exception as err:
        #     source_iter = iter(gloader_train)
        #     source_data = next(source_iter)
        # try:
        #     target_data = next(target_iter)
        # except Exception as err:
        #     target_iter = iter(gloader_test)
        #     target_data = next(target_iter)
        
            try:
                target_data = next(target_iter)
            #except Exception as err:
            except:
                target_iter = iter(gloader_test)
                target_data = next(target_iter)

            source_data = source_data.to(device)
            target_data = target_data.to(device)

            

            x_source, x_s_domain = net(source_data, alpha=alpha) #256*3   1*2
            y_source = source_data.y
            y_s_domain = torch.zeros(x_s_domain.shape,device=device,dtype=torch.float32)
            _, pred = torch.max(x_source, dim=1)
            
            current_correct = sum([1 for a,b in zip(pred, y_source) if a==b])
            correct_pred += current_correct
            
            loss_class = criterion(x_source, y_source)

            x_target, x_t_domain = net(target_data,alpha=alpha)
            y_target = target_data.y
            y_t_domain = torch.ones(x_t_domain.shape,device=device,dtype=torch.float32)

            loss_s_domain = criterion(x_s_domain,y_s_domain)
            loss_t_domain = criterion(x_t_domain,y_t_domain)

            loss = loss_class+loss_s_domain+loss_t_domain

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            #scheduler.step(epoch_loss)

            highest_acc, current_acc = test(net, test_data, test_label, people, highest_acc, epoch)

            

            denominator = (ind+1)*batch_size

                        # 使得模型具有继承性

            if current_acc < current_highest_acc-0.1 :
                n += 1
                if n>50:
                    net = net_best
                    n =0
                    print("模型覆盖")
            else:
                n =0
            if current_acc >= current_highest_acc:
                net_best = net
                current_highest_acc = current_acc
                n =0
                print(f"current_highest_acc_updating:{current_highest_acc}")
                if current_acc >= highest_acc:
                     highest_acc = current_acc
                     print(f"highest_acc_updating:{highest_acc}")


            if ind % 5 == 0:
                print()
                print(f'-----highest_acc {highest_acc:.4f} current_acc {current_acc:.4f} current_highest_acc {current_highest_acc:.4f}-----')
                print('Dataset: ', DATASET)
                print(f'batch {batch_size}, lr {lr}')
                print()
                
            print(f'Epoch [{epoch}/{epochs}] \n'
                f'Loss [{float(loss.item()):.4f}]\n'
                f'ACC@1 gACC[{current_correct/batch_size:.4f}] \n'
                f'totalACC[{correct_pred/denominator:.4f}] \n')
            print(f'alpha[{alpha:.4f}] \n')
            ind +=1

            


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

    index = np.random.permutation(data_test.shape[0])# 随机排列序列
    data_test = data_test[index,:,:]
    label_test = label_test[index]

    for freq_i in range(5):
        data_test[:,:,freq_i] = zscore(np.squeeze(data_test[:,:,freq_i]))# np.squeeze()删除指定单维度，默认删除所有单维度 。这里或许不需要；zscore（）默认0维计算
        

    print(f'loaded train {people}\'s data.....')

    flag = 0
    
    for train_index in train_list:

        print('> ' + train_index)

        data, label = load_SEED_DE_feature(load_path + train_index) 

        for freq_i in range(5):
            data[:,:,freq_i] = zscore(np.squeeze(data[:,:,freq_i]))
        
        if flag==0:

            data_train = data
            label_train = label
            flag += 1
        else:        

            data_train = np.concatenate([data_train, data], axis=0)  #np.concatenate()沿着现有的轴进行拼接
            label_train = np.concatenate([label_train, label], axis=0)

    highest_acc = train(data_train, label_train, data_test, label_test, people)

        # idx = idx + 1

    return highest_acc

if __name__ == '__main__':
    meanAcc=[]
    for  load_path  in ['./ExtractedFeatures/Session1/','./ExtractedFeatures/Session2/','./ExtractedFeatures/Session3/']:
        file_list = os.listdir(load_path) #返回指定的文件夹包含的文件或文件夹的名字的列表
        file_list.sort(key=lambda a:  int(a.split('_')[0]))
        # intra_subject_Kold(file_list[2])
        # train_list = file_list
        # train_list.remove(file_list[2])
        # inter_subject_LOSV(file_list[2], train_list)

        acc_all = []
        train_list = []
        for file_i in range(len(file_list)):

            train_list[:] = file_list[:]
            train_list.remove(file_list[file_i])
            acc = inter_subject_LOSV(file_list[file_i], train_list)
            acc_all.append(acc)
        
        meanAcc.append(np.mean(acc_all))
        scio.savemat('RGCNN-pyg2.0/results/session'+load_path[-2]+'.mat',{'acc':acc_all})
    print(meanAcc,np.mean(meanAcc))