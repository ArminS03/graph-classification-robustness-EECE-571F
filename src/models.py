import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, arch='GCN', dropout=0.5, num_layers=3):
        super(GraphClassifier, self).__init__()
        self.arch = arch
        self.dropout = dropout
        self.num_layers = num_layers

        ConvLayer = GCNConv if arch == 'GCN' else GATConv
        if arch not in ('GCN', 'GAT'):
            raise ValueError("Architecture must be 'GCN' or 'GAT'")

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(ConvLayer(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(ConvLayer(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Message Passing Layers with BatchNorm and Dropout
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Graph-level Readout (concat mean + sum pooling for robustness)
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)

        # 3. Classifier head with Dropout
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x