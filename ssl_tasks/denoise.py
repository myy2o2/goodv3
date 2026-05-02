from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import model_embeddings


class FeatureDenoisingTask(nn.Module):
    name = "denoise"

    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h = model_embeddings(model, feat, edge_index, edge_weight)
        x_rec = self.decoder(h)
        return F.mse_loss(x_rec, feat.detach(), reduction="none").mean(dim=-1)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
