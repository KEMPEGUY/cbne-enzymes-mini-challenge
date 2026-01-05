# models/hybrid_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class HybridClassifier(nn.Module):
    def __init__(self, gcn_out_dim, moment_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, gcn_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(gcn_out_dim + moment_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, data, moment_vecs):
        x = data.x if data.x is not None else torch.ones((data.num_nodes, 3), device=data.edge_index.device)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        graph_ids = torch.tensor([d.graph_id for d in data.to_data_list()], device=x.device)
        moments = moment_vecs[graph_ids]
        return self.classifier(torch.cat([x, moments], dim=1))

