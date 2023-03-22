"""Module that defines the GNN models for the odor prediction task."""

import torch 
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.models import GAT, GIN
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from xgboost import XGBClassifier

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pooling_type='mean'):
        super().__init__()
        self.mlp = torch.nn.Linear(in_channels, out_channels)
        self.clf = XGBClassifier(tree_method='gpu_hist',
                             num_class = 6,objective = 'multi:softprob')
        if pooling_type == 'mean':
          self.pool = global_mean_pool
        elif pooling_type == 'max':
          self.pool = global_max_pool
        elif pooling_type == 'add':
          self.pool = global_add_pool

    def forward(self, x, batch):
        x = self.pool(x, batch)
        preds = self.clf.predict(x.detach().cpu().numpy())
        preds = torch.tensor(preds)
        return preds
        return torch.sigmoid(self.mlp(x))

    def reset_parameters(self):
      self.mlp.reset_parameters()


class ScentClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0, pooling_type='mean'):
        super().__init__()
        self.gnn = GIN(
            in_channels, 
            hidden_channels, 
            num_layers,
            dropout=dropout,
        )
        self.classifier = GraphClassifier(hidden_channels, out_channels, pooling_type=pooling_type)
        self.gnn.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index, batch=torch.tensor([0])):
        h = self.gnn(x, edge_index)
        return self.classifier(h, batch)

class PretrainingGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0, pooling_type='max'):
        super().__init__()
        self.gnn = GIN(
            in_channels, 
            hidden_channels, 
            num_layers,
            dropout=dropout,
        )
        self.gnn.reset_parameters()

        if pooling_type == 'mean':
          self.pool = global_mean_pool
        elif pooling_type == 'max':
          self.pool = global_max_pool
        elif pooling_type == 'add':
          self.pool = global_add_pool

    def forward(self, x, edge_index, batch=torch.tensor([0])):
        batch = batch.cuda()
        h = self.gnn(x, edge_index)
        return self.pool(h, batch)