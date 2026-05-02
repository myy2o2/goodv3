from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import model_embeddings, neighbor_mean


class NeighborReconstructionTask(nn.Module):
    name = "neighbor"

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h = model_embeddings(model, feat, edge_index, edge_weight)
        pred = self.mlp(h)
        target = neighbor_mean(h, edge_index)
        return F.mse_loss(pred, target.detach(), reduction="none").mean(dim=-1)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
