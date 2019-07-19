import torch
# from ..nn import SGC_LL, graph_max_pool
from ..nn import graph_max_pool
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool
from torch.nn import BatchNorm1d
import torch.nn.functional as F

class AGCN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 node_hidden_dim=15,
                 K=3,
                 alpha=0.1,
                 num_step_combo=5,
                 fcc_hidden_dim=20,
                 output_dim=12,
                 ):

        super(AGCN, self).__init__()

        self.num_step_combo = num_step_combo
        self.conv = ChebConv(node_input_dim, node_hidden_dim, K)
        # self.conv = SGC_LL(node_input_dim, node_hidden_dim, K, alpha)
        self.batch_norm = BatchNorm1d(node_hidden_dim)

        self.lin1 = torch.nn.Linear(node_hidden_dim, fcc_hidden_dim)
        self.lin2 = torch.nn.Linear(fcc_hidden_dim, output_dim)

    def forward(self, data):
        out = data.x
        batch = data.batch
        edge_index = data.edge_index

        for i in range(self.num_step_combo):
            tmp = F.relu(self.conv(out, edge_index))
            tmp = self.batch_norm(tmp)
            out = graph_max_pool(tmp, edge_index)
        
        out = global_add_pool(out, batch)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out