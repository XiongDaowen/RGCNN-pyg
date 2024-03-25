import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool, global_max_pool




class ReadoutLayer(torch.nn.Module):
    def __init__(self):
        super(ReadoutLayer, self).__init__()
        #self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, batch, size):
        a = global_mean_pool(x, batch, size) #a(64,64)
        b = global_max_pool(x, batch, size)  #b(64*64)
        return torch.cat((a,b),1)

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col
    # inv_mask = 1 - mask
    inv_mask = ~mask
    loop_weight = torch.full(
        (num_nodes, ),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    # allow negative edge weights
    @staticmethod  #staticmethod用于修饰类中的方法,使其可以在不创建类实例的情况下调用方法
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        return norm.view(-1, 1) * x_j


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SymSimGCNNet(torch.nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout=0, droprate=0,domain_adaptation=""):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0) # 生成下三角矩阵索引
        
        # tensor 可以用列表进行索引
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys] # strict lower triangular values   # tensor 可以用列表进行索引
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight) 
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens[0], K=K)
        #self.sagP = SAGPool(num_hiddens[0])
        self.fc = nn.Linear(num_hiddens[0]*2, num_hiddens[1])
        self.fc1 = nn.Linear(num_hiddens[1], num_hiddens[2])
        self.fc2 = nn.Linear(num_hiddens[2], num_classes)
        self.BN1 = nn.BatchNorm1d(num_hiddens[1])
        self.BN2 = nn.BatchNorm1d(num_hiddens[2])
        self.sagP = SAGPool(num_hiddens[0],ratio=droprate)
        self.readOut = ReadoutLayer()
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = nn.Linear(num_hiddens[0]*2, 2)

    def forward(self, data, alpha=0, tLabel=1):
        batch_size = len(data.y)
        x, edge_index = data.x, data.edge_index
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal()) # copy values from lower tri to upper tri
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = F.relu(self.conv1(x, edge_index, edge_weight))    #62*62*256
        x, edge_index, edge_attr, batch, perm = self.sagP(x, edge_index, edge_weight,data.batch)
        #before shape x(3968,64) edge_index(2,246016) edge_weight(246016) batch(3968)
        #before shape x(1984,64) edge_index(2,61504) edge_weight(61504) batch(1984)


        # domain classification
        domain_output = None
        x = self.readOut(x, batch, batch_size)#(64,128)
        #if self.domain_adaptation in ["RevGrad"]:
        if tLabel:
            reverse_x = ReverseLayerF.apply(x, alpha)
            #reverse_x = global_add_pool(x, data.batch, size=batch_size)
            domain_output = self.domain_classifier(reverse_x)
        # else :
        #     #x = global_add_pool(x, data.batch, size=batch_size)
        #     domain_output = self.domain_classifier(x)
        #x = global_add_pool(x, data.batch, size=batch_size)
        
        x = F.relu(self.BN1(self.fc(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.BN2(self.fc1(x)))
        x = self.fc2(x)
        return x, domain_output



class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm