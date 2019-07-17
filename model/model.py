from base import BaseModel
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.inits import uniform

from torch_geometric.utils import scatter_

class SGC_LL(MessagePassing, BaseModel):
    def __init__(self, in_channels, out_channels, K, alpha, bias=True, **kwargs):
        super(SGC_LL, self).__init__(aggr='add', **kwargs)

        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        self.M_weight = Parameter(torch.Tensor(in_channels, in_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)
        uniform(size, self.M_weight)

    @staticmethod
    def norm(edge_index, x, num_nodes, dtype=None):
        edge_weight = torch.ones((edge_index.size(1), ),
                                  dtype=dtype,
                                  device=edge_index.device)

        fill_value = -1
        edge_index, edge_weight = add_remaining_self_loops(
                                    edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        residue_edge_weight = torch.zeros((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        for i in range(edge_index.size(1)):
            xij = x[row[i]] - x[col[i]]
            dij = torch.sqrt(xij.view(1, -1).mm(self.M_weight.mm(xij.view(-1, 1))))[0, 0]
            residue_edge_weight[i] = torch.exp(-dij/2)
        residue_deg = scatter_add(residue_edge_weight, row, dim=0, dim_size=num_nodes)
        residue_deg_inv_sqrt = residue_deg.pow(-0.5)
        residue_deg_inv_sqrt[residue_deg_inv_sqrt == float('inf')] = 0

        return edge_index, (1+self.alpha)*(-deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]\
                           -residue_deg_inv_sqrt[row] * residue_edge_weight * residue_deg_inv_sqrt[col])

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = self.norm(edge_index, x, x.size(0), x.dtype)

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


class Graph_max_pooling():
    pass
