import torch

num_hiddens=(64,128,32)
epochs = 5
batch_size = 64
lr = 0.001
weight_decay = 8e-5
drop_rate = 0.5
drop_rate_pooling = 0.5
num_workers = 3
device = ('cuda:0' )#if torch.cuda.is_available() else 'cpu' )

K = 3

DATASET = 'SEED'

if DATASET == 'SEED':
    
    n_channel = 62
    n_freq = 5
    n_class = 3