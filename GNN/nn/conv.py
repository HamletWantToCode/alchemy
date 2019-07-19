from ..base import BaseModel
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from torch_geometric.nn.inits import uniform


class SGC_LL(MessagePassing, BaseModel):
    def __init__(self, in_channels, out_channels, K, alpha, root_weight=True, bias=True, **kwargs):
        super(SGC_LL, self).__init__(aggr='add', **kwargs)

        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        self.W_weight = Parameter(torch.Tensor(in_channels, in_channels))
        
        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.M_weight = self.W_weight.mm(torch.transpose(self.W_weight, 1, 0))

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)
        uniform(size, self.W_weight)
        uniform(size, self.root)

    @staticmethod
    def norm(edge_index, num_nodes, dtype=None):
        edge_weight = torch.ones((edge_index.size(1), ),
                                  dtype=dtype,
                                  device=edge_index.device)

        edge_index_, edge_weight_ = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=0
        )

        # compute graph Laplacian 
        row, col = edge_index_
        deg = scatter_add(edge_weight_, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        L = -deg_inv_sqrt[row] * edge_weight_ * deg_inv_sqrt[col]
        L[L==0] = 1

        return edge_index_, L
    
    @staticmethod
    def residue_norm(x, M_weight, alpha, num_nodes, dtype=None):
        I = torch.arange(num_nodes)
        ii, jj = torch.meshgrid(I, I)
        edge_index = torch.stack([ii.flatten(), jj.flatten()])

        edge_index, _ = remove_self_loops(edge_index)

        row, col = edge_index
        edge_weight = torch.zeros(edge_index.size(1))
        for i in range(edge_index.size(1)):
            dxij = x[row[i]] - x[col[i]]
            d = torch.sqrt(dxij.view(1, -1).mm(M_weight.mm(dxij.view(-1, 1))))[0, 0]
            edge_weight[i] = torch.exp(-d/2.0)

        edge_index_, edge_weight_ = add_remaining_self_loops(edge_index, edge_weight, fill_value=0)

        row_, col_ = edge_index_
        deg = scatter_add(edge_weight_, row_, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        Lres = -deg_inv_sqrt[row_] * edge_weight_ * deg_inv_sqrt[col_]
        Lres[Lres==0] = 1

        return edge_index_, alpha*Lres


    def forward(self, x, edge_index):
        edge_index1, norm1 = self.norm(edge_index, x.size(0), x.dtype)
        edge_index2, norm2 = self.residue_norm(x, self.M_weight, self.alpha, x.size(0), x.dtype)

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index1, x=x, norm=norm1) + self.propagate(edge_index2, x=x, norm=norm2)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * (self.propagate(edge_index1, x=Tx_1, norm=norm1)\
                        + self.propagate(edge_index2, x=Tx_1, norm=norm2)) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.root is not None:
            out = out.mm(self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))