from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import model_embeddings


class EmbeddingConsistencyTask(nn.Module):
    name = "consistency"

    def __init__(self, hidden_dim: int, proj_dim: int = 0, dropout: float = 0.2):
        super().__init__()
        proj_dim = int(proj_dim) if int(proj_dim) > 0 else int(hidden_dim)
        self.dropout = float(dropout)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, hidden_dim),
        )

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h = model_embeddings(model, feat, edge_index, edge_weight)
        z1 = self.projector(F.dropout(h, p=self.dropout, training=self.training))
        z2 = self.projector(F.dropout(h, p=self.dropout, training=self.training))
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return 1.0 - (z1 * z2).sum(dim=-1)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
