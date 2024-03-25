import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
import config

drop_rate = config.drop_rate

# DGCNN

class DGCNN(nn.Module):
    def __init__(self, inchannel, outchannel, batch, n_class, testmode=False):
        super(DGCNN, self).__init__()
        self.batch = batch
        self.testmode = testmode
        linearsize = 512


        self.conv1 = gnn.ChebConv(inchannel, outchannel, K=config.K)
        self.conv2 = nn.Conv1d(outchannel, 1, kernel_size=(1, 1))
    
        # self.full_connection = nn.Sequential(
        #     nn.Linear(32*62, 8),
        #     nn.Linear(8, n_class)

        # )

        self.full_connection = nn.Sequential(
            nn.Linear(outchannel * 62, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize//2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize // 2,n_class)
        )

    def forward(self, g):
        x, e = g.x, g.edge_index
        x = self.conv1(x, e)
        # x = self.conv2(x)
        x = F.relu(x)
        x = x.view(self.batch, -1)
        x = self.full_connection(x)

        return x
