from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

from .common import model_embeddings


class DegreePredictionTask(nn.Module):
    name = "degree"

    def __init__(self, hidden_dim: int, num_bins: int = 6):
        super().__init__()
        self.num_bins = int(num_bins)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_bins),
        )

    def _degree_bins(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        if deg.max() <= 0:
            return torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        quantiles = torch.quantile(deg, torch.linspace(0, 1, self.num_bins + 1, device=deg.device))
        return torch.bucketize(deg, quantiles[1:-1], right=False).long()

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h = model_embeddings(model, feat, edge_index, edge_weight)
        labels = self._degree_bins(edge_index, h.shape[0])
        logits = self.mlp(h)
        return F.cross_entropy(logits, labels, reduction="none")

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
